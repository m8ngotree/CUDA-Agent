"""Stage 3 — execution-based filtering (paper §3.1).

For each synthesised problem we spawn `python -m data.filter._filter_worker`
in a subprocess and apply a hard timeout (default 30 s). The subprocess
runs the four-criterion check on a real GPU; the parent only handles
bookkeeping. This isolates kernel hangs / OOMs / segfaults — none of which
can be caught from a Python try/except in-process.

After all problems are checked, we MinHash-LSH against KernelBench and
drop any problem with similarity > `decontamination_threshold`.

Outputs:
  - `{output_dir}/per_problem_results.jsonl`   (one FilterResult per input)
  - `{output_dir}/filtered_problems.jsonl`     (passing FilteredProblem records)
  - `{output_dir}/filter_stats.json`           (yield funnel — primary artifact)

Resumable: re-running skips problems already present in per_problem_results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from data.common import (
    append_jsonl,
    existing_ids,
    load_config,
    read_jsonl,
    write_jsonl,
    write_stats,
)
from data.filter.decontaminate import KernelBenchIndex, load_kernelbench_sources
from data.filter.filter_schema import FilteredProblem, FilterResult
from data.synthesis.synthesize_fused_ops import FusedProblemEntry


def _run_worker(entry: dict, cfg: dict, timeout_s: float) -> FilterResult:
    """Run one problem through the worker subprocess with a hard timeout."""
    pid = entry["problem_id"]
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "in.json"
        outp = Path(td) / "out.json"
        inp.write_text(json.dumps({"entry": entry, "cfg": cfg}))

        cmd = [sys.executable, "-m", "data.filter._filter_worker", str(inp), str(outp)]
        env = os.environ.copy()
        env.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        env.setdefault("TORCH_INDUCTOR_CACHE_DIR", str(Path(td) / "inductor_cache"))
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return FilterResult(
                problem_id=pid, passed=False, fail_reason=f"timeout_{timeout_s:.0f}s"
            )

        if not outp.exists():
            tail = (proc.stderr or "")[-300:].strip().replace("\n", " | ")
            return FilterResult(
                problem_id=pid,
                passed=False,
                fail_reason=f"worker_no_output rc={proc.returncode} stderr={tail!r}",
            )

        try:
            return FilterResult.model_validate_json(outp.read_text())
        except Exception as e:
            return FilterResult(
                problem_id=pid, passed=False, fail_reason=f"result_parse_error: {e}"
            )


def _bucket_failure(reason: str | None) -> str:
    """Map an arbitrary fail reason to one of the funnel buckets the spec defines."""
    if reason is None:
        return "passed"
    r = reason
    if r.startswith("eager_execution_error") or r.startswith("compile_execution_error"):
        return "failed_execution"
    if r.startswith("class_compile_error") or r.startswith("instantiation_error"):
        return "failed_execution"
    if r.startswith("input_alloc_error") or r.startswith("benchmark_error"):
        return "failed_execution"
    if r == "non_deterministic" or r.startswith("determinism_check_error"):
        return "failed_determinism"
    if r in ("output_constant", "zero_output") or r.startswith("anti_trivial_check_error"):
        return "failed_anti_trivial"
    if r.startswith("out_of_time_range_too_fast"):
        return "failed_time_range_too_fast"
    if r.startswith("out_of_time_range_too_slow"):
        return "failed_time_range_too_slow"
    if r.startswith("timeout_"):
        return "failed_execution"  # timeouts are typically hangs / OOMs
    if r.startswith("worker_") or r.startswith("result_parse_error"):
        return "failed_execution"
    if r == "decontamination":
        return "failed_decontamination"
    return "failed_execution"


def _source_lib_bucket(libs: list[str]) -> str:
    s = set(libs)
    if s == {"torch"}:
        return "torch_only"
    if s == {"transformers"}:
        return "transformers_only"
    return "mixed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default=None, help="Override input synth JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N problems")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fcfg: dict[str, Any] = cfg["filter"]
    out_dir = Path(fcfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    syn_dir = Path(cfg["synthesis"]["output_dir"])
    inp_path = Path(args.input) if args.input else syn_dir / "fused_problems.jsonl"
    if not inp_path.exists():
        raise FileNotFoundError(f"Synthesis output not found: {inp_path}")

    per_problem = out_dir / "per_problem_results.jsonl"
    filtered = out_dir / "filtered_problems.jsonl"
    stats_path = out_dir / "filter_stats.json"

    # Resume: never re-run a problem we already have a result for.
    done = existing_ids(per_problem)
    if done:
        logger.info(f"Resuming filter: {len(done)} problems already evaluated")

    timeout_s = float(fcfg.get("subprocess_timeout_s", 30))
    worker_cfg = {
        k: fcfg[k]
        for k in (
            "determinism_atol",
            "anti_trivial_atol",
            "zero_output_atol",
            "eager_time_min_ms",
            "eager_time_max_ms",
            "bench_warmup",
            "bench_iters",
            "device",
        )
        if k in fcfg
    }

    # Pass 1: execution-based filter (subprocess per problem).
    funnel: Counter[str] = Counter()
    funnel.update({k: 0 for k in (
        "total_synthesized",
        "failed_execution",
        "failed_determinism",
        "failed_anti_trivial",
        "failed_time_range_too_fast",
        "failed_time_range_too_slow",
        "failed_decontamination",
        "passed",
    )})
    by_source: dict[str, Counter[str]] = {
        "torch_only": Counter(),
        "transformers_only": Counter(),
        "mixed": Counter(),
    }

    n_processed = 0
    iter_entries = list(read_jsonl(inp_path, FusedProblemEntry))
    funnel["total_synthesized"] = len(iter_entries)
    pbar = tqdm(iter_entries, desc="filter")
    for entry in pbar:
        if args.limit is not None and n_processed >= args.limit:
            break
        n_processed += 1
        e: FusedProblemEntry = entry  # type: ignore[assignment]
        if e.problem_id in done:
            continue

        result = _run_worker(e.model_dump(), worker_cfg, timeout_s=timeout_s)
        append_jsonl(per_problem, result)
        bucket = _bucket_failure(result.fail_reason)
        funnel[bucket] += 1
        by_source[_source_lib_bucket(e.source_libs)][bucket] += 1

    # Pass 2: decontamination on the survivors.
    target_n = int(fcfg.get("target_n", 0))
    threshold = float(fcfg.get("decontamination_threshold", 0.7))
    kernelbench_path = fcfg.get("kernelbench_path")
    kb_index: KernelBenchIndex | None = None
    if kernelbench_path:
        sources = load_kernelbench_sources(kernelbench_path)
        if sources:
            # We index at LSH-threshold = 0.5 to widen recall; the actual
            # filter uses jaccard >= `threshold` from `max_similarity`.
            kb_index = KernelBenchIndex.from_sources(sources, threshold=0.5)
        else:
            logger.warning("KernelBench index empty — decontamination disabled")

    # Build the final filtered set. Re-stream synthesis + per-problem results
    # to keep memory bounded for 100k+ entries.
    syn_by_id: dict[str, FusedProblemEntry] = {
        e.problem_id: e  # type: ignore[union-attr]
        for e in iter_entries
    }
    sim_dist: list[float] = []
    final: list[FilteredProblem] = []
    for r in read_jsonl(per_problem, FilterResult):
        r: FilterResult  # type: ignore[no-redef]
        if not r.passed:
            continue
        e = syn_by_id.get(r.problem_id)
        if e is None:
            continue
        sim = 0.0
        if kb_index is not None:
            sim = kb_index.max_similarity(e.module_source)
            sim_dist.append(sim)
            if sim > threshold:
                funnel["failed_decontamination"] += 1
                continue
        final.append(
            FilteredProblem(
                problem_id=r.problem_id,
                pytorch_source=e.module_source,
                component_ops=e.component_ops,
                source_libs=e.source_libs,
                fusion_depth=e.fusion_depth,
                eager_time_ms=float(r.eager_time_ms or 0.0),
                compile_time_ms=float(r.compile_time_ms or 0.0),
                decontamination_score=sim,
                input_shapes=[list(s) for s in (r.input_shapes or [])],
                target=cfg.get("target", "cuda"),
            )
        )

    # Truncate to target_n, preferring shorter eager_time_ms (broader coverage).
    if target_n and len(final) > target_n:
        final.sort(key=lambda p: p.eager_time_ms)
        final = final[:target_n]

    # Atomically rewrite filtered_problems.jsonl.
    if filtered.exists():
        filtered.unlink()
    write_jsonl(filtered, final)

    funnel["passed"] = len(final)

    stats: dict[str, Any] = dict(funnel)
    stats["by_source_lib"] = {k: dict(v) for k, v in by_source.items()}
    if sim_dist:
        sim_sorted = sorted(sim_dist)
        stats["decontamination_distribution"] = {
            "p50": sim_sorted[len(sim_sorted) // 2],
            "p90": sim_sorted[int(0.9 * len(sim_sorted))],
            "p99": sim_sorted[int(0.99 * (len(sim_sorted) - 1))],
            "max": sim_sorted[-1],
            "n": len(sim_sorted),
        }
    write_stats(stats_path, stats)

    logger.info(
        f"Filter done. Funnel: { {k: v for k, v in stats.items() if not isinstance(v, dict)} }"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
