#!/usr/bin/env python3
"""Performance profiling for Triton-based ModelNew implementations.

Mirrors the existing CUDA `profiling.py` but uses extra warmup iterations to
absorb Triton JIT compile time on the first launches. Output format is
identical so the agent harness's regex parser works for both targets.
"""

from __future__ import annotations

import argparse
import time

import torch

from model import Model, get_inputs, get_init_inputs
from model_new import ModelNew


def transform_tensors(tensors, fn):
    if isinstance(tensors, torch.Tensor):
        return fn(tensors)
    if isinstance(tensors, (list, tuple)):
        return [transform_tensors(x, fn) for x in tensors]
    if isinstance(tensors, dict):
        return {k: transform_tensors(v, fn) for k, v in tensors.items()}
    return tensors


def initialize_models() -> tuple[torch.nn.Module, torch.nn.Module, list, list]:
    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]

    torch_model = Model(*init_inputs).eval().cuda()
    triton_model = ModelNew(*init_inputs).eval().cuda()
    triton_model.load_state_dict(torch_model.state_dict())

    inputs = get_inputs()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    inputs = transform_tensors(inputs, lambda x: x.cuda())
    return torch_model, triton_model, inputs, [t.clone() for t in inputs]


def benchmark(model, inputs, warmup_iters: int, run_iters: int) -> float:
    """Returns mean kernel time per call in microseconds.

    We intentionally use the wall clock (with cuda.synchronize bookends)
    rather than torch.profiler here because the latter doesn't always pick
    up Triton-launched kernels on every torch+triton combo. The wall clock
    after a long warmup is good enough — Triton dispatch overhead is in
    the microsecond range either way."""
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(*inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(run_iters):
            _ = model(*inputs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    # Convert seconds -> microseconds, then divide by iters.
    return elapsed * 1e6 / run_iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="More warmup than CUDA — Triton JITs on first run",
    )
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    torch_model, triton_model, torch_inputs, triton_inputs = initialize_models()
    torch_compile_model = torch.compile(torch_model)

    torch_us = benchmark(torch_model, torch_inputs, args.warmup, args.iters)
    compile_us = benchmark(torch_compile_model, torch_inputs, args.warmup, args.iters)
    if args.baseline_only:
        print(f"Torch Baseline: {torch_us:.3f}us, Torch Compile: {compile_us:.3f}us")
        return

    # We prepend 3 dedicated Triton-only warmup runs to make sure JIT is done
    # before the timing window starts, on top of the regular warmup.
    with torch.no_grad():
        for _ in range(3):
            _ = triton_model(*triton_inputs)
    torch.cuda.synchronize()

    triton_us = benchmark(triton_model, triton_inputs, args.warmup, args.iters)
    # Same line format as profiling.py so the agent harness can parse both.
    print(
        f"Torch Baseline: {torch_us:.3f}us, "
        f"Torch Compile: {compile_us:.3f}us, "
        f"CUDA Extension: {triton_us:.3f}us"
    )


if __name__ == "__main__":
    main()
