# CUDA Agent

A reimplementation of the data, reward, and training pipeline from the
**CUDA Agent** paper (ByteDance / Tsinghua, Feb 2026,
[arXiv:2602.24286](https://arxiv.org/abs/2602.24286)) and a parallel
**Triton kernel** pipeline.

The official release ships only the agent workdir (`SKILL.md`,
`verification.py`, `profiling.py`, one example kernel). Everything else
(seed crawl, combinatorial synthesis, execution-based filtering, the reward
function, and the multi-stage RL training pipeline) is missing. This repo
fills in those gaps and adds a Triton target.

## Structure

```
kernel-agent/
├── configs/                # YAML configs for each pipeline
├── data/
│   ├── crawl/              # Stage 1: crawl torch + transformers ops
│   ├── synthesis/          # Stage 2: LLM combinatorial synthesis
│   ├── filter/             # Stage 3: execution-based filtering + decontam
│   └── dataset/            # Stage 7: HF dataset assembly
├── agent/
│   ├── workdir/            # Agent sandbox (SKILL.md, model.py, ...)
│   ├── loop.py             # Minimal ReAct agent loop
│   └── reward.py           # Eq. 1 reward
├── train/
│   ├── single_turn_warmup.py   # Stage 8a: single-turn PPO warmup
│   ├── rft.py                  # Stage 8b: rejection fine-tuning
│   ├── value_pretrain.py       # Stage 8c: critic pretraining
│   └── agentic_rl.py           # Stage 8d: full agentic PPO
├── eval/
│   ├── eval_kernelbench.py
│   ├── eval_tritonbench.py
│   └── metrics.py
└── scripts/
    ├── run_data_pipeline.sh
    ├── run_training.sh
    └── run_eval.sh
```

## Pipeline overview

```
crawl -> synthesize -> filter -> dataset -> warmup -> rft -> value-pretrain -> agentic-rl
```

Each stage is independently runnable, resumable, config-driven, and emits
its own `stats.json` with a yield funnel. There is no implicit state
between stages. Every stage reads JSONL/parquet from disk and writes JSONL
to disk.

## Quickstart

```bash
pip install -e .[train,triton,eval]

# Build the dataset (CPU + 1 GPU)
bash scripts/run_data_pipeline.sh configs/cuda_pipeline.yaml

# Train (multi-GPU recommended)
bash scripts/run_training.sh configs/cuda_pipeline.yaml

# Eval
bash scripts/run_eval.sh configs/cuda_pipeline.yaml
```

## Stages

### 1. Crawl

`data/crawl/crawl_torch_ops.py` walks `torch.nn` and `torch.nn.functional`
and extracts operator class source. `crawl_transformers_ops.py` extracts
self-contained MLP/norm blocks from `transformers.models.*` (extension
beyond the paper). Both write JSONL conforming to `op_schema.OperatorEntry`.

### 2. Synthesize

`data/synthesis/synthesize_fused_ops.py` samples 2–5 ops, asks an LLM (any
OpenAI-compatible client) to fuse them into a self-contained PyTorch module
(`class Model(nn.Module)` + `get_init_inputs()` + `get_inputs()`, the
KernelBench format), and writes JSONL of `FusedProblemEntry`.

### 3. Filter

`data/filter/filter_pipeline.py` applies the four-criterion filter from
§3.1 (executability, determinism, anti-trivial, workload range) in a 30-s
subprocess to handle hangs. `decontaminate.py` MinHash-LSH-checks against
KernelBench. Yields `data/filtered/filter_stats.json` — the **primary
artifact**.

### 4. Reward

`agent/reward.py` implements Eq. 1 verbatim: r ∈ {-1, 1, 2, 3}.

### 5. Triton extension

`agent/workdir/SKILL_TRITON.md` mirrors the CUDA SKILL with Triton
templates. `agent/workdir/utils/profiling_triton.py` and
`utils/verification_triton.py` use the same harness with relaxed atol
(1e-3) and longer warmup (10 iters) to absorb Triton JIT compile time.

### 6. Train

Four scripts under `train/`, one per stage of §3.3:
single-turn PPO warmup → RFT → value pretraining → agentic PPO.
Asymmetric clip (ε_low=0.20, ε_high=0.28), GAE γ=1, λ=0.95.

### 7. Eval

`eval/eval_kernelbench.py` runs the trained agent on KernelBench L1/L2/L3.
`eval/eval_tritonbench.py` runs on TritonBench. `metrics.py` computes the
faster-than-compile rate, pass-rate, and reward statistics.

## What this repo does NOT include

- The full OpenHands runtime: we use a minimal ReAct loop and call
  OpenHands externally if available.
- A 128-GPU sandbox pool: we isolate per-rollout via `subprocess` with
  a 30 s timeout on a single GPU.
- The Seed1.6 base model: we default to `Qwen/Qwen2.5-Coder-7B-Instruct`.

## Citation

```bibtex
@article{cuda_agent_2026,
  title  = {CUDA Agent: Large-Scale Agentic RL for High-Performance CUDA
            Kernel Generation},
  author = {ByteDance Seed Team and Tsinghua University},
  journal= {arXiv:2602.24286},
  year   = {2026}
}
```
