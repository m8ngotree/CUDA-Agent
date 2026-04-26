"""Stage 7 — assemble the final HuggingFace dataset from filtered problems.

Reads `{filter.output_dir}/filtered_problems.jsonl`, casts to a typed
`datasets.Dataset` with the schema described in the spec, applies a
deterministic train/validation split, and saves to disk.

The resulting `DatasetDict` is what you would push to the Hub via
`upload_dataset.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value
from loguru import logger

from data.common import load_config, read_jsonl
from data.filter.filter_schema import FilteredProblem

FEATURES = Features(
    {
        "problem_id": Value("string"),
        "pytorch_source": Value("string"),
        "component_ops": Sequence(Value("string")),
        "source_libs": Sequence(Value("string")),
        "fusion_depth": Value("int32"),
        "eager_time_ms": Value("float32"),
        "compile_time_ms": Value("float32"),
        "decontamination_score": Value("float32"),
        "input_shapes": Sequence(Sequence(Value("int32"))),
        "target": Value("string"),
    }
)


def _records(path: Path) -> list[dict]:
    rows: list[dict] = []
    for r in read_jsonl(path, FilteredProblem):
        # Pydantic -> plain dict so the Dataset typecast doesn't choke on
        # custom classes.
        rows.append(r.model_dump())  # type: ignore[union-attr]
    if not rows:
        raise FileNotFoundError(f"No filtered problems in {path}")
    return rows


def build(cfg: dict) -> DatasetDict:
    fcfg = cfg["filter"]
    dcfg = cfg["dataset"]
    inp = Path(fcfg["output_dir"]) / "filtered_problems.jsonl"
    rows = _records(inp)
    logger.info(f"Loaded {len(rows)} filtered problems from {inp}")

    ds = Dataset.from_list(rows, features=FEATURES)
    train_frac = float(dcfg.get("splits", {}).get("train", 0.95))
    seed = 0
    split = ds.train_test_split(test_size=1 - train_frac, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["dataset"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = build(cfg)
    ds.save_to_disk(str(out_dir))
    logger.info(
        f"Saved dataset to {out_dir} — train={len(ds['train'])}, val={len(ds['validation'])}"
    )

    # Card-friendly summary, written next to the dataset.
    summary = {split: len(ds[split]) for split in ds}
    summary["target"] = cfg.get("target", "cuda")
    summary["repo_id"] = cfg["dataset"].get("hf_repo_id")
    (out_dir / "dataset_summary.json").write_text(__import__("json").dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
