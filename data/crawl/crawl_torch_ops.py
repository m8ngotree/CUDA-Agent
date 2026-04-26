"""Stage 1a — crawl operators from `torch.nn` and `torch.nn.functional`.

Faithful reproduction of the seed-problem crawl described in §3.1 of the
CUDA Agent paper. We walk the public namespaces, extract class source via
`inspect.getsource`, classify each op into a coarse tag, and apply the
exclusion rules listed in the spec. Output is `data/raw/torch_ops.jsonl`
plus a yield-funnel `stats.json` next to it.

Every filtered entry is logged with a reason so we can audit the funnel
later when comparing against the paper's reported numbers.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from data.common import load_config, write_jsonl, write_stats
from data.crawl.op_schema import OperatorEntry

SKIP_BASE_NAMES = {
    "Module",
    "ModuleList",
    "ModuleDict",
    "Sequential",
    "ParameterList",
    "ParameterDict",
    "Container",
    "_ConvNd",
    "_BatchNorm",
    "_LazyNormBase",
    "_NormBase",
    "_InstanceNorm",
    "LazyModuleMixin",
    "Identity",
    "RNNBase",
    "RNNCellBase",
}

DISTRIBUTED_HOOKS = ("torch.distributed", "DistributedDataParallel", "all_reduce")
NON_DETERMINISTIC_HINTS = (
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "AlphaDropout",
    "FeatureAlphaDropout",
)


def _classify(name: str, cls: type) -> list[str]:
    """Coarse taxonomy used to bias synthesis sampling toward diverse fusions."""
    tags: list[str] = []
    n = name.lower()
    if any(t in n for t in ("linear", "bilinear")):
        tags.append("linear")
    if any(t in n for t in ("conv",)):
        tags.append("conv")
    if any(t in n for t in ("norm", "groupnorm", "layernorm", "rmsnorm", "batchnorm")):
        tags.append("normalization")
    if any(
        t in n
        for t in (
            "relu",
            "gelu",
            "silu",
            "elu",
            "selu",
            "tanh",
            "sigmoid",
            "softmax",
            "softplus",
            "mish",
            "hardswish",
            "hardsigmoid",
        )
    ):
        tags.append("activation")
    if any(t in n for t in ("pool",)):
        tags.append("pool")
    if any(t in n for t in ("embedding",)):
        tags.append("embedding")
    if any(t in n for t in ("attention", "multihead")):
        tags.append("attention")
    if not tags:
        tags.append("misc")
    return tags


def _init_signature_dict(cls: type) -> dict[str, Any]:
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}
    out: dict[str, Any] = {}
    for n, p in sig.parameters.items():
        if n == "self":
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        out[n] = None if p.default is inspect.Parameter.empty else _safe_default(p.default)
    return out


def _safe_default(v: Any) -> Any:
    """Default values may include callables / tensors which aren't JSON-serialisable.
    Coerce them to a stable string representation so the JSONL stays valid."""
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_safe_default(x) for x in v]
    if isinstance(v, dict):
        return {k: _safe_default(x) for k, x in v.items()}
    return repr(v)


def _forward_signature_dict(cls: type) -> dict[str, Any]:
    fn = getattr(cls, "forward", None)
    if fn is None:
        return {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    out: dict[str, Any] = {}
    for n, p in sig.parameters.items():
        if n == "self":
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        ann = p.annotation
        out[n] = "Tensor" if ann is inspect.Parameter.empty else repr(ann)
    return out


def _class_source(cls: type) -> str | None:
    try:
        return inspect.getsource(cls)
    except (OSError, TypeError):
        return None


def _is_excluded(name: str, cls: type, src: str) -> str | None:
    """Return a string fail-reason or None if op should be kept."""
    if cls.__name__ in SKIP_BASE_NAMES:
        return "abstract_or_container"
    if inspect.isabstract(cls):
        return "abstract"
    if any(h in src for h in DISTRIBUTED_HOOKS):
        return "uses_distributed"
    if any(h in cls.__name__ for h in NON_DETERMINISTIC_HINTS):
        return "non_deterministic"
    if "register_parameter" in src and "Lazy" in cls.__name__:
        return "lazy_module"
    return None


def _walk_module(mod, source_lib: str) -> Iterator[tuple[str, type]]:
    """Yield (qualified_name, class) for every public nn.Module subclass."""
    seen: set[int] = set()
    for attr_name in dir(mod):
        if attr_name.startswith("_"):
            continue
        try:
            obj = getattr(mod, attr_name)
        except Exception:
            continue
        if not inspect.isclass(obj):
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if not issubclass(obj, nn.Module):
            continue
        if obj is nn.Module:
            continue
        # Only include classes actually defined in torch packages.
        objmod = getattr(obj, "__module__", "")
        if not objmod.startswith("torch"):
            continue
        yield f"{mod.__name__}.{obj.__name__}", obj


def _walk_functional(mod) -> Iterator[tuple[str, Any]]:
    """`torch.nn.functional` is functions, not classes. We wrap each public
    function in a synthesised nn.Module-style stub so synthesis treats it
    uniformly. We emit only the function source — synthesis composes them
    inside the generated `Model.forward()`."""
    for attr_name in dir(mod):
        if attr_name.startswith("_"):
            continue
        try:
            fn = getattr(mod, attr_name)
        except Exception:
            continue
        if not inspect.isfunction(fn):
            continue
        if getattr(fn, "__module__", "") != mod.__name__:
            continue
        yield f"{mod.__name__}.{attr_name}", fn


def crawl_torch(modules: list[str]) -> tuple[list[OperatorEntry], dict[str, int]]:
    """Walk each requested module and produce a list of entries plus funnel stats."""
    stats: Counter[str] = Counter()
    out: list[OperatorEntry] = []

    for modpath in modules:
        if modpath == "torch.nn":
            for name, cls in _walk_module(nn, "torch"):
                stats["seen"] += 1
                src = _class_source(cls)
                if src is None:
                    stats["skip_no_source"] += 1
                    continue
                reason = _is_excluded(name, cls, src)
                if reason:
                    stats[f"skip_{reason}"] += 1
                    continue
                out.append(
                    OperatorEntry(
                        name=name,
                        source_lib="torch",
                        class_def=src,
                        init_signature=_init_signature_dict(cls),
                        forward_signature=_forward_signature_dict(cls),
                        tags=_classify(name, cls),
                    )
                )
                stats["kept"] += 1

        elif modpath == "torch.nn.functional":
            # We wrap each public function as a "pseudo-class" entry so the
            # synthesis prompt can include its source verbatim.
            for name, fn in _walk_functional(F):
                stats["seen"] += 1
                try:
                    src = inspect.getsource(fn)
                except (OSError, TypeError):
                    stats["skip_no_source"] += 1
                    continue
                if any(h in src for h in DISTRIBUTED_HOOKS):
                    stats["skip_uses_distributed"] += 1
                    continue
                if any(h in name for h in ("dropout",)):
                    stats["skip_non_deterministic"] += 1
                    continue
                tags = _classify(name, fn)
                tags.append("functional")
                out.append(
                    OperatorEntry(
                        name=name,
                        source_lib="torch",
                        class_def=src,
                        init_signature={},
                        forward_signature=_forward_signature_dict(fn),
                        tags=tags,
                    )
                )
                stats["kept"] += 1
        else:
            logger.warning(f"Unknown torch module: {modpath}")

    return out, dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    parser.add_argument(
        "--out", default=None, help="Override output JSONL (defaults to <output_dir>/torch_ops.jsonl)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    crawl_cfg = cfg["crawl"]
    out_dir = Path(crawl_cfg["output_dir"])
    out_path = Path(args.out) if args.out else out_dir / "torch_ops.jsonl"
    stats_path = out_dir / "torch_ops_stats.json"

    if out_path.exists():
        out_path.unlink()

    logger.info(f"Crawling torch modules: {crawl_cfg['torch_modules']}")
    entries, stats = crawl_torch(crawl_cfg["torch_modules"])

    write_jsonl(out_path, entries)
    write_stats(
        stats_path,
        {
            "torch_version": torch.__version__,
            "modules": crawl_cfg["torch_modules"],
            "funnel": stats,
            "total_kept": len(entries),
        },
    )

    logger.info(
        f"Crawled {stats.get('seen', 0)} ops, kept {len(entries)} -> {out_path}. "
        f"Funnel: {stats}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
