"""Push the assembled dataset to the HuggingFace Hub.

Expects `data.dataset.build_dataset` to have run first. Reads the saved
`DatasetDict`, attaches a dataset card with the yield funnel from
`data/filtered/filter_stats.json`, and pushes via `Dataset.push_to_hub`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi
from loguru import logger

from data.common import load_config

CARD_TEMPLATE = """\
---
language: en
license: apache-2.0
tags:
- cuda
- triton
- kernel-generation
- reinforcement-learning
- code-generation
size_categories:
- 1K<n<10K
---

# {title}

Synthesised PyTorch fused-operator problems for training kernel-generation
agents. Reproduces and extends the dataset construction from the **CUDA
Agent** paper (ByteDance / Tsinghua, Feb 2026,
[arXiv:2602.24286](https://arxiv.org/abs/2602.24286)).

- **Target**: `{target}`
- **Splits**: train={n_train}, validation={n_val}
- **Source libraries**: `torch` (paper §3.1) + `transformers` (extension)
- **Fusion depth**: 2..5 operators per problem
- **Filtering**: executability + determinism + anti-trivial + workload range
  (1..100 ms eager) + KernelBench MinHash decontamination

## Yield funnel

```
{funnel}
```

## Schema

| field                   | type             | meaning                            |
|-------------------------|------------------|------------------------------------|
| `problem_id`            | string           | stable hash                        |
| `pytorch_source`        | string           | full module: `class Model` + `get_init_inputs()` + `get_inputs()` (KernelBench format) |
| `component_ops`         | list[string]     | ops composed to form this problem  |
| `source_libs`           | list[string]     | one of `torch` / `transformers`    |
| `fusion_depth`          | int              | number of ops fused                |
| `eager_time_ms`         | float            | reference eager runtime            |
| `compile_time_ms`       | float            | `torch.compile` runtime            |
| `decontamination_score` | float            | max MinHash sim vs KernelBench     |
| `input_shapes`          | list[list[int]]  | one shape per tensor in `get_inputs()` |
| `target`                | string           | `cuda` or `triton`                 |

## Citation

```bibtex
@article{{cuda_agent_2026,
  title  = {{CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA
            Kernel Generation}},
  author = {{ByteDance Seed Team and Tsinghua University}},
  journal= {{arXiv:2602.24286}},
  year   = {{2026}}
}}
```
"""


def _format_funnel(stats_path: Path) -> str:
    if not stats_path.exists():
        return "(no filter_stats.json available)"
    return json.dumps(json.loads(stats_path.read_text()), indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_id = cfg["dataset"]["hf_repo_id"]
    if not repo_id or "${" in repo_id:
        raise ValueError(f"hf_repo_id not configured (got {repo_id!r})")

    out_dir = Path(cfg["dataset"]["output_dir"])
    ds = load_from_disk(str(out_dir))

    funnel = _format_funnel(Path(cfg["filter"]["output_dir"]) / "filter_stats.json")
    card = CARD_TEMPLATE.format(
        title=repo_id,
        target=cfg.get("target", "cuda"),
        n_train=len(ds["train"]),
        n_val=len(ds["validation"]),
        funnel=funnel,
    )

    if args.dry_run:
        logger.info(f"--dry-run: would push to {repo_id}")
        print(card)
        return 0

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    ds.push_to_hub(repo_id, private=args.private)

    # Push the README.md (dataset card) separately.
    card_path = out_dir / "README.md"
    card_path.write_text(card)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info(f"Pushed {repo_id} ({len(ds['train'])} train + {len(ds['validation'])} val)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
