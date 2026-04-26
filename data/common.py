"""Shared utilities for the data pipeline: config loading and resumable JSONL I/O.

Every stage uses these helpers so behavior is uniform: configs come from YAML,
intermediate state lives in JSONL files (one record per line), and any stage
can be safely re-run — the loaders skip records that are already present.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import yaml
from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML config and expand ${VAR} environment-variable references."""
    with open(path) as f:
        text = f.read()
    text = os.path.expandvars(text)
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg).__name__}")
    return cfg


def write_jsonl(path: str | os.PathLike, records: Iterable[BaseModel | dict]) -> int:
    """Append records to JSONL. Returns count written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("a") as f:
        for r in records:
            if isinstance(r, BaseModel):
                f.write(r.model_dump_json() + "\n")
            else:
                f.write(json.dumps(r) + "\n")
            n += 1
    return n


def append_jsonl(path: str | os.PathLike, record: BaseModel | dict) -> None:
    """Append a single record. Cheap enough that we open/close per call so a
    crash in the middle of a long synthesis still leaves a valid file."""
    write_jsonl(path, [record])


def read_jsonl(path: str | os.PathLike, model: type[T] | None = None) -> Iterator[T | dict]:
    """Stream a JSONL file. Yields parsed pydantic models if `model` given,
    otherwise plain dicts. Silently skips malformed lines and logs them."""
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"{path}:{i + 1} malformed JSON: {e}")
                continue
            if model is not None:
                try:
                    yield model.model_validate(obj)
                except Exception as e:
                    logger.warning(f"{path}:{i + 1} schema mismatch: {e}")
                    continue
            else:
                yield obj


def existing_ids(path: str | os.PathLike, key: str = "problem_id") -> set[str]:
    """Set of `key` values already present in a JSONL file. Used by resumable
    stages to avoid recomputing finished entries."""
    seen: set[str] = set()
    for rec in read_jsonl(path):
        if isinstance(rec, dict) and key in rec:
            seen.add(rec[key])
    return seen


def write_stats(path: str | os.PathLike, stats: dict[str, Any]) -> None:
    """Write a stats.json next to a stage output. Always overwritten."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    logger.info(f"Wrote stats to {p}")
