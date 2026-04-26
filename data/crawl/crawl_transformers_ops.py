"""Stage 1b — crawl module-level operators from `transformers`.

Extension beyond the paper. We walk `transformers.models.*.modeling_*` and
extract self-contained, deterministic forward passes (pure MLP, RMSNorm,
LayerNorm, simple activation wrappers). These give the synthesis stage
harder fusion targets at the module-block boundary — the kind of fusion
that real-world LLM serving cares about.

Filtering rules (from spec):
  - exclude anything with `past_key_values` in forward signature
  - exclude anything with `attention_mask` requiring variable-length inputs
  - exclude anything calling `torch.distributed` or `accelerate`
  - keep: pure MLP blocks, normalization layers, activation functions

We log a full yield funnel (this is a primary artifact).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from loguru import logger

from data.common import load_config, write_jsonl, write_stats
from data.crawl.op_schema import OperatorEntry

# Heuristic name patterns. We deliberately keep the keep-list narrow so
# synthesised problems stay tractable for the agent. Adding more architectures
# is a one-liner here; the rest of the pipeline doesn't care.
INCLUDE_NAME_PATTERNS = (
    "MLP",
    "RMSNorm",
    "LayerNorm",
    "RotaryEmbedding",   # constant-shape, deterministic, fusable
    "SwiGLU",
    "GELUActivation",
    "FFN",
    "Mlp",
)

# Reject anything that names attention/cache state in its forward.
EXCLUDE_FORWARD_PARAMS = (
    "past_key_values",
    "past_key_value",
    "attention_mask",
    "encoder_hidden_states",
    "cross_attn_head_mask",
    "head_mask",
    "use_cache",
    "cache_position",
)

EXCLUDE_SOURCE_HINTS = (
    "torch.distributed",
    "all_reduce",
    "from accelerate",
    "import accelerate",
    "deepspeed",
    "PreTrainedModel",
)


def _classify(name: str) -> list[str]:
    n = name.lower()
    tags: list[str] = []
    if "mlp" in n or "ffn" in n:
        tags.append("mlp")
    if "norm" in n:
        tags.append("normalization")
    if "rotary" in n:
        tags.append("rotary")
    if "swiglu" in n or "gelu" in n or "silu" in n or "activation" in n:
        tags.append("activation")
    if not tags:
        tags.append("misc")
    tags.append("transformers")
    return tags


def _safe_default(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_safe_default(x) for x in v]
    if isinstance(v, dict):
        return {k: _safe_default(x) for k, x in v.items()}
    return repr(v)


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


def _iter_modeling_modules(stats: Counter[str]) -> Iterator[Any]:
    """Yield every `transformers.models.*.modeling_*` module that imports cleanly.

    Skipped modules increment `stats["module_import_failed"]` so the crawl
    funnel is honest about how much of the corpus we actually scanned.
    """
    try:
        import transformers.models as models_pkg
    except Exception as e:
        logger.error(f"Failed to import transformers.models: {e}")
        return

    for _finder, name, ispkg in pkgutil.iter_modules(models_pkg.__path__):
        if not ispkg:
            continue
        try:
            sub = importlib.import_module(f"transformers.models.{name}")
        except Exception:
            stats["package_import_failed"] += 1
            continue
        sub_path = getattr(sub, "__path__", None)
        if not sub_path:
            continue
        for _, modname, _ in pkgutil.iter_modules(sub_path):
            if not modname.startswith("modeling_"):
                continue
            stats["modeling_modules_seen"] += 1
            full = f"transformers.models.{name}.{modname}"
            try:
                yield importlib.import_module(full)
                stats["modeling_modules_loaded"] += 1
            except Exception:
                # Some models depend on optional deps (sentencepiece, etc).
                stats["module_import_failed"] += 1
                continue


def _is_kept(cls: type, src: str, fwd_sig: dict[str, Any], cfg: dict[str, Any]) -> str | None:
    """Return None to keep, or a fail-reason."""
    name = cls.__name__

    if not any(p in name for p in INCLUDE_NAME_PATTERNS):
        return "name_not_in_include_list"

    if cfg.get("exclude_with_past_key_values", True):
        if any(p in fwd_sig for p in EXCLUDE_FORWARD_PARAMS if "past" in p):
            return "has_past_key_values"

    if cfg.get("exclude_with_attention_mask", True):
        if "attention_mask" in fwd_sig:
            return "has_attention_mask"

    for hint in EXCLUDE_SOURCE_HINTS:
        if hint in src:
            return f"source_hint:{hint}"

    return None


def crawl_transformers(filter_cfg: dict[str, Any]) -> tuple[list[OperatorEntry], dict[str, int]]:
    stats: Counter[str] = Counter()
    seen_ids: set[int] = set()
    out: list[OperatorEntry] = []

    try:
        import torch.nn as nn
    except ImportError:
        logger.error("torch is required to inspect transformers nn.Module classes")
        return [], dict(stats)

    for mod in _iter_modeling_modules(stats):
        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            try:
                obj = getattr(mod, attr)
            except Exception:
                continue
            if not inspect.isclass(obj):
                continue
            if not issubclass(obj, nn.Module):
                continue
            if id(obj) in seen_ids:
                continue
            seen_ids.add(id(obj))
            stats["seen"] += 1

            try:
                src = inspect.getsource(obj)
            except (OSError, TypeError):
                stats["skip_no_source"] += 1
                continue

            fwd_sig = _forward_signature_dict(obj)
            reason = _is_kept(obj, src, fwd_sig, filter_cfg)
            if reason:
                stats[f"skip_{reason}"] += 1
                continue

            qual_name = f"{obj.__module__}.{obj.__name__}"
            out.append(
                OperatorEntry(
                    name=qual_name,
                    source_lib="transformers",
                    class_def=src,
                    init_signature=_init_signature_dict(obj),
                    forward_signature=fwd_sig,
                    tags=_classify(obj.__name__),
                )
            )
            stats["kept"] += 1

    return out, dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    crawl_cfg = cfg["crawl"]
    filter_cfg = crawl_cfg.get("transformers_filter", {})
    out_dir = Path(crawl_cfg["output_dir"])
    out_path = Path(args.out) if args.out else out_dir / "transformers_ops.jsonl"
    stats_path = out_dir / "transformers_ops_stats.json"

    if out_path.exists():
        out_path.unlink()

    try:
        import transformers

        tv = transformers.__version__
    except Exception:
        tv = "not_installed"
        logger.warning("transformers not importable; emitting empty crawl")
        write_jsonl(out_path, [])
        write_stats(stats_path, {"transformers_version": tv, "funnel": {}, "total_kept": 0})
        return 0

    logger.info(f"Crawling transformers={tv}, filter={filter_cfg}")
    entries, stats = crawl_transformers(filter_cfg)

    write_jsonl(out_path, entries)
    write_stats(
        stats_path,
        {
            "transformers_version": tv,
            "filter_cfg": filter_cfg,
            "funnel": stats,
            "total_kept": len(entries),
        },
    )

    logger.info(f"Kept {len(entries)} of {stats.get('seen', 0)} -> {out_path}. Funnel: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
