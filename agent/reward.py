"""Reward function from §3.2 (Eq. 1) of the CUDA Agent paper.

Reward levels:
    -1 : correctness check failed
     1 : correct, but not significantly faster than eager or compile
     2 : correct, significantly faster than eager (>5%) but not compile
     3 : correct, significantly faster than both eager and compile (>5% each)

"Significantly faster" means the new kernel reduces runtime by more than
`speedup_threshold` (default 0.05 = 5%) over the baseline.

This module is intentionally tiny and pure-Python so it can be imported
from any context (training loop, eval, agent loop).
"""

from __future__ import annotations

from dataclasses import dataclass


def _significantly_faster(t_gen: float, t_baseline: float, threshold: float) -> bool:
    if t_baseline <= 0:
        return False
    return (t_baseline - t_gen) / t_baseline > threshold


def compute_reward(
    generated_kernel_time_ms: float,
    eager_time_ms: float,
    compile_time_ms: float,
    correctness_passed: bool,
    speedup_threshold: float = 0.05,
) -> int:
    """Eq. 1 from §3.2. See module docstring for the level definitions."""
    if not correctness_passed:
        return -1

    faster_than_eager = _significantly_faster(
        generated_kernel_time_ms, eager_time_ms, speedup_threshold
    )
    faster_than_compile = _significantly_faster(
        generated_kernel_time_ms, compile_time_ms, speedup_threshold
    )

    if faster_than_eager and faster_than_compile:
        return 3
    if faster_than_eager:
        return 2
    return 1


@dataclass
class RewardBreakdown:
    """Diagnostic record returned by `compute_reward_with_breakdown`. Useful
    for W&B logging — we want per-bucket rates, not just mean reward."""

    reward: int
    correctness_passed: bool
    faster_than_eager: bool
    faster_than_compile: bool
    speedup_vs_eager: float | None
    speedup_vs_compile: float | None


def compute_reward_with_breakdown(
    generated_kernel_time_ms: float,
    eager_time_ms: float,
    compile_time_ms: float,
    correctness_passed: bool,
    speedup_threshold: float = 0.05,
) -> RewardBreakdown:
    speedup_vs_eager = None
    speedup_vs_compile = None
    fte = ftc = False
    if correctness_passed:
        if eager_time_ms > 0:
            speedup_vs_eager = eager_time_ms / max(generated_kernel_time_ms, 1e-9)
        if compile_time_ms > 0:
            speedup_vs_compile = compile_time_ms / max(generated_kernel_time_ms, 1e-9)
        fte = _significantly_faster(generated_kernel_time_ms, eager_time_ms, speedup_threshold)
        ftc = _significantly_faster(
            generated_kernel_time_ms, compile_time_ms, speedup_threshold
        )

    r = compute_reward(
        generated_kernel_time_ms,
        eager_time_ms,
        compile_time_ms,
        correctness_passed,
        speedup_threshold,
    )
    return RewardBreakdown(
        reward=r,
        correctness_passed=correctness_passed,
        faster_than_eager=fte,
        faster_than_compile=ftc,
        speedup_vs_eager=speedup_vs_eager,
        speedup_vs_compile=speedup_vs_compile,
    )
