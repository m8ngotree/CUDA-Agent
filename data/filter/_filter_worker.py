"""Worker subprocess for the filter stage.

Run as: `python -m data.filter._filter_worker <input.json> <output.json>`

Reads a single FusedProblemEntry as JSON, executes the four-criterion check
(executability, determinism, anti-trivial, workload range), and writes a
FilterResult JSON. We run in a separate process so the parent can apply a
hard 30 s timeout via `subprocess.run(timeout=...)` — this kills hanging
or OOM kernels without poisoning the parent.

Sample format
-------------
The synthesised module follows the KernelBench convention (paper §3.1):

    class Model(nn.Module):
        def __init__(self, *args): ...
        def forward(self, *tensors): ...

    def get_init_inputs(): ...   # constructor args
    def get_inputs():       ...  # tensors for forward()

We instantiate via `Model(*get_init_inputs())` and call
`model(*get_inputs())`. The same convention is used by
`agent/workdir/utils/{verification,profiling}.py`, so synthesised problems
drop into the agent sandbox unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Module loading + the three required hooks (Model / get_init_inputs / get_inputs)
# ---------------------------------------------------------------------------


def _exec_module(module_source: str) -> dict:
    """Execute the synthesised module in a fresh namespace with `torch` and
    `nn` available, and return the namespace. Raises if any required symbol
    is missing."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: F401  (often used by generated code)

    g: dict = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "__builtins__": __builtins__,
        "__name__": "synthesized_module",
    }
    exec(module_source, g)

    if "Model" not in g or not isinstance(g["Model"], type):
        raise RuntimeError("class Model not defined")
    if "get_init_inputs" not in g or not callable(g["get_init_inputs"]):
        raise RuntimeError("get_init_inputs() not defined")
    if "get_inputs" not in g or not callable(g["get_inputs"]):
        raise RuntimeError("get_inputs() not defined")
    return g


def _move_to_device(obj, device: str):
    """Recursively move tensors to device. Anything else passes through."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _randomise_tensors(obj):
    """Return a new structure where every tensor is replaced by torch.randn_like."""
    import torch

    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return torch.randn_like(obj)
        # Integer tensors (e.g. embedding ids): re-roll within the same range
        # using randint so the anti-trivial check stays meaningful.
        if obj.numel() > 0 and obj.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            lo = int(obj.min().item())
            hi = max(lo + 1, int(obj.max().item()) + 1)
            return torch.randint(lo, hi, obj.shape, dtype=obj.dtype, device=obj.device)
        return obj.clone()
    if isinstance(obj, list):
        return [_randomise_tensors(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_randomise_tensors(x) for x in obj)
    return obj


def _input_shapes(inputs) -> list[list[int]]:
    """Best-effort shape extraction for the funnel record. Non-tensor inputs
    are skipped."""
    import torch

    shapes: list[list[int]] = []
    for t in inputs if isinstance(inputs, (list, tuple)) else [inputs]:
        if isinstance(t, torch.Tensor):
            shapes.append(list(t.shape))
    return shapes


# ---------------------------------------------------------------------------
# Benchmark + criterion helpers
# ---------------------------------------------------------------------------


def _bench(model, inputs, n_warmup: int, n_iters: int) -> float:
    """Mean per-call wall time in milliseconds. `inputs` is a list passed via *."""
    import torch

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(*inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model(*inputs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_iters
    return elapsed_ms


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(entry: dict, cfg: dict) -> dict:
    import torch

    pid = entry["problem_id"]
    out: dict = {
        "problem_id": pid,
        "passed": False,
        "fail_reason": None,
        "eager_time_ms": None,
        "compile_time_ms": None,
        "input_shapes": None,
    }

    if not torch.cuda.is_available():
        out["fail_reason"] = "cuda_not_available"
        return out

    # Accept both new ("module_source") and legacy ("fused_class_source") keys
    # so older synthesis runs still load. Prefer the new one.
    module_src = entry.get("module_source") or entry.get("fused_class_source")
    if module_src is None:
        out["fail_reason"] = "missing_module_source"
        return out

    try:
        ns = _exec_module(module_src)
    except Exception as e:
        out["fail_reason"] = f"class_compile_error: {type(e).__name__}: {e}"
        return out

    Model = ns["Model"]
    get_init_inputs = ns["get_init_inputs"]
    get_inputs = ns["get_inputs"]
    device = cfg.get("device", "cuda")

    try:
        init_args = get_init_inputs()
        if not isinstance(init_args, (list, tuple)):
            init_args = [init_args]
        model = Model(*init_args).to(device).eval()
    except Exception as e:
        out["fail_reason"] = f"instantiation_error: {type(e).__name__}: {e}"
        return out

    try:
        raw_inputs = get_inputs()
        if not isinstance(raw_inputs, (list, tuple)):
            raw_inputs = [raw_inputs]
        inputs = _move_to_device(list(raw_inputs), device)
    except Exception as e:
        out["fail_reason"] = f"input_alloc_error: {type(e).__name__}: {e}"
        return out

    out["input_shapes"] = _input_shapes(inputs)

    # ---- Criterion 1: Executability (eager + compile) ---------------------
    try:
        with torch.no_grad():
            out_eager = model(*inputs)
    except Exception as e:
        out["fail_reason"] = f"eager_execution_error: {type(e).__name__}: {e}"
        return out

    try:
        compiled = torch.compile(model)
        with torch.no_grad():
            _ = compiled(*inputs)
    except Exception as e:
        out["fail_reason"] = f"compile_execution_error: {type(e).__name__}: {e}"
        return out

    # ---- Criterion 2: Determinism -----------------------------------------
    try:
        with torch.no_grad():
            o1 = model(*inputs)
            o2 = model(*inputs)
        det_atol = float(cfg.get("determinism_atol", 1e-5))

        # Outputs may be a tensor, list, tuple or dict; reduce to a flat
        # list of (a,b) tensor pairs and check every pair.
        def _flatten(x, y, acc):
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                acc.append((x, y))
            elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
                for a, b in zip(x, y, strict=False):
                    _flatten(a, b, acc)
            elif isinstance(x, dict) and isinstance(y, dict):
                for k in x:
                    _flatten(x[k], y.get(k), acc)

        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        _flatten(o1, o2, pairs)
        if not pairs:
            out["fail_reason"] = "no_tensor_output"
            return out
        if not all(torch.allclose(a, b, atol=det_atol) for a, b in pairs):
            out["fail_reason"] = "non_deterministic"
            return out
    except Exception as e:
        out["fail_reason"] = f"determinism_check_error: {type(e).__name__}: {e}"
        return out

    # ---- Criterion 3: Anti-trivial ----------------------------------------
    try:
        inputs2 = _randomise_tensors(inputs)
        with torch.no_grad():
            out_x2 = model(*inputs2)
        anti_atol = float(cfg.get("anti_trivial_atol", 1e-3))
        zero_atol = float(cfg.get("zero_output_atol", 1e-4))

        # Use the first tensor of each output structure for the anti-trivial
        # comparison — sufficient to catch constant / zero outputs.
        def _first_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            if isinstance(x, (list, tuple)):
                for v in x:
                    t = _first_tensor(v)
                    if t is not None:
                        return t
            if isinstance(x, dict):
                for v in x.values():
                    t = _first_tensor(v)
                    if t is not None:
                        return t
            return None

        a = _first_tensor(out_eager)
        b = _first_tensor(out_x2)
        if a is None:
            out["fail_reason"] = "no_tensor_output"
            return out
        if torch.allclose(a, b, atol=anti_atol):
            out["fail_reason"] = "output_constant"
            return out
        if torch.allclose(a, torch.zeros_like(a), atol=zero_atol):
            out["fail_reason"] = "zero_output"
            return out
    except Exception as e:
        out["fail_reason"] = f"anti_trivial_check_error: {type(e).__name__}: {e}"
        return out

    # ---- Criterion 4: Workload range --------------------------------------
    try:
        torch.cuda.empty_cache()
        eager_ms = _bench(
            model, inputs,
            n_warmup=int(cfg.get("bench_warmup", 5)),
            n_iters=int(cfg.get("bench_iters", 20)),
        )
        torch.cuda.empty_cache()
        compile_ms = _bench(
            compiled, inputs,
            n_warmup=int(cfg.get("bench_warmup", 5)),
            n_iters=int(cfg.get("bench_iters", 20)),
        )
    except Exception as e:
        out["fail_reason"] = f"benchmark_error: {type(e).__name__}: {e}"
        return out

    out["eager_time_ms"] = eager_ms
    out["compile_time_ms"] = compile_ms

    lo = float(cfg.get("eager_time_min_ms", 1.0))
    hi = float(cfg.get("eager_time_max_ms", 100.0))
    if eager_ms < lo:
        out["fail_reason"] = f"out_of_time_range_too_fast:{eager_ms:.3f}ms"
        return out
    if eager_ms > hi:
        out["fail_reason"] = f"out_of_time_range_too_slow:{eager_ms:.3f}ms"
        return out

    out["passed"] = True
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: _filter_worker.py <input.json> <output.json>", file=sys.stderr)
        return 2
    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])

    payload = json.loads(inp.read_text())
    entry = payload["entry"]
    cfg = payload["cfg"]

    try:
        result = run(entry, cfg)
    except Exception:  # last-ditch
        result = {
            "problem_id": entry.get("problem_id", "?"),
            "passed": False,
            "fail_reason": f"worker_uncaught: {traceback.format_exc().splitlines()[-1]}",
            "eager_time_ms": None,
            "compile_time_ms": None,
            "input_shapes": None,
        }

    outp.write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
    sys.exit(main())
