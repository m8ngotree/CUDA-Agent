"""Aggregate metrics shared by all eval scripts.

Each eval script writes one `EvalRecord` per problem to a JSONL file;
`summarise_results` ingests that file and produces the headline numbers
the paper reports — pass-rate, faster-than-compile rate, mean reward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class EvalRecord(BaseModel):
    problem_id: str
    correctness: bool
    eager_time_ms: float = 0.0
    compile_time_ms: float = 0.0
    generated_time_ms: float = 0.0
    reward: int = -1
    n_turns: int = 0
    fail_reason: str | None = None


@dataclass
class Summary:
    n: int
    pass_rate: float
    faster_than_eager_rate: float
    faster_than_compile_rate: float
    reward_mean: float
    median_speedup_vs_compile: float
    reward_distribution: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "pass_rate": self.pass_rate,
            "faster_than_eager_rate": self.faster_than_eager_rate,
            "faster_than_compile_rate": self.faster_than_compile_rate,
            "reward_mean": self.reward_mean,
            "median_speedup_vs_compile": self.median_speedup_vs_compile,
            "reward_distribution": self.reward_distribution,
        }


def _safe_speedup(t_baseline: float, t_gen: float) -> float:
    if t_gen <= 0:
        return 0.0
    return t_baseline / t_gen


def summarise_results(jsonl_path: str | Path, threshold: float = 0.05) -> Summary:
    rows: list[EvalRecord] = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                rows.append(EvalRecord.model_validate_json(line))
            except Exception:
                continue
    if not rows:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

    n = len(rows)
    n_pass = sum(1 for r in rows if r.correctness)

    def _is_faster(r: EvalRecord, t_base_field: str) -> bool:
        if not r.correctness:
            return False
        t_base = getattr(r, t_base_field)
        if t_base <= 0 or r.generated_time_ms <= 0:
            return False
        return (t_base - r.generated_time_ms) / t_base > threshold

    fte = sum(1 for r in rows if _is_faster(r, "eager_time_ms"))
    ftc = sum(1 for r in rows if _is_faster(r, "compile_time_ms"))
    speedups = sorted(
        _safe_speedup(r.compile_time_ms, r.generated_time_ms) for r in rows if r.correctness
    )
    median_sp = speedups[len(speedups) // 2] if speedups else 0.0

    dist: dict[int, int] = {}
    for r in rows:
        dist[r.reward] = dist.get(r.reward, 0) + 1

    return Summary(
        n=n,
        pass_rate=n_pass / n,
        faster_than_eager_rate=fte / n,
        faster_than_compile_rate=ftc / n,
        reward_mean=sum(r.reward for r in rows) / n,
        median_speedup_vs_compile=median_sp,
        reward_distribution=dist,
    )


def write_summary(jsonl_path: str | Path, out_path: str | Path, threshold: float = 0.05) -> None:
    summary = summarise_results(jsonl_path, threshold=threshold)
    Path(out_path).write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
