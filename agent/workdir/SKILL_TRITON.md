# Triton Kernel Optimization Skill

You are a Triton kernel optimization expert. Your goal is to accelerate
PyTorch operators by replacing them with custom Triton kernels, targeting
the best possible performance with a minimum of 5% faster than `torch.compile`.

## 1. CRITICAL RESTRICTIONS

### Strictly forbidden
- **No `torch.nn.functional` ops in `model_new.py`**: only tensor creation
  and your custom Triton ops are allowed.
- **No third-party kernel libraries**: use raw Triton only (no
  flash-attn, no xformers, no apex). cuBLAS / cuDNN are unavailable from
  Triton — design around that.
- **No modifications to `utils/`**: it is the scoring infrastructure.
- **No fp64**: Triton has limited fp64 support; stick to fp32 / bf16 / fp16.

### Allowed
- **Triton**: `@triton.jit` kernels, `@triton.autotune`, `tl.*` primitives.
- **Python**: `torch.empty`, `torch.empty_like`, `.shape`, `.stride`,
  `.dtype`, `.device`. Custom Triton kernel calls.

## 2. WORKSPACE STRUCTURE

```
.
├── kernels/                  # YOUR WORK: triton kernels live here
│   └── *.py                  # one .py file per fused kernel
├── utils/                    # DO NOT MODIFY
│   ├── verification_triton.py  # tolerance: atol=1e-3
│   └── profiling_triton.py     # warmup=10 (Triton JIT)
├── model.py                  # DO NOT MODIFY — original PyTorch model
└── model_new.py              # YOUR WORK — uses your Triton kernel
```

## 3. UNIFIED WORKFLOW

### Step 1: Profile baseline
```bash
python -m utils.profiling_triton --baseline-only
```
Look for:
- Excessive kernel launches (each PyTorch op ≈ separate kernel)
- Memory bandwidth bottlenecks (elementwise ops on large tensors)
- Fusion opportunities (ops sharing input data)

### Step 2: Implement Triton kernel

Create `kernels/<name>.py`:

```python
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x  # YOUR COMPUTATION HERE
    tl.store(output_ptr + offsets, output, mask=mask)


def fused_op(x: torch.Tensor) -> torch.Tensor:
    assert x.is_contiguous() and x.is_cuda
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    fused_kernel[grid](x, out, n)
    return out
```

### Step 3: Wire `model_new.py`

```python
import torch
import torch.nn as nn
from kernels.fused_kernel import fused_op


class ModelNew(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # Initialise parameters identically to Model so state_dict() loads.
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_op(x)
```

### Step 4: Verify and profile

```bash
python -m utils.verification_triton    # atol=1e-3 (relaxed for fp32 accumulation)
python -m utils.profiling_triton       # warmup=10 (covers Triton JIT compile)
```

Must pass `torch.allclose(atol=1e-3)` and beat `torch.compile` by ≥5%.

### Step 5: Iterate

If correctness fails:
- Check dtype handling (`tl.load(...).to(tl.float32)` for accumulation).
- Check boundary `mask` on every load/store.
- Check reduction axes for layer-norm / softmax-like kernels.
- Use a single block first to isolate logic from tiling.

If performance insufficient:
- Tune `BLOCK_SIZE`, `num_warps`, `num_stages` via `@triton.autotune`.
- Try 2D tiling for matrix-shape kernels.
- Vectorise loads (bigger BLOCK_SIZE per warp).
- For reduction-heavy kernels: persistent reduction with `tl.sum(axis=...)`.

## 4. COMMON KERNEL TEMPLATES

### Elementwise + reduction (e.g. RMSNorm)

```python
@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, out_ptr,
    stride_x, stride_o,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_ptr + row * stride_o + cols, y.to(tl.float16), mask=mask)
```

### Matmul (skeleton)

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(...):
    pid = tl.program_id(0)
    # standard 2-D split, accumulate in float32, cast to output dtype on store
```

## 5. SUCCESS CRITERIA

- Correctness: `torch.allclose(out, ref, atol=1e-3, rtol=1e-3)`
- Speed: ≤ 0.95× `torch.compile` time
- Cleanliness: `kernels/` contains only the final `.py` files (no `_v1`, `_old`)

## Your Task

Optimise the PyTorch model in `model.py`.
