"""Shared utilities for the training scripts.

Centralises wandb init, dataset loading, and the policy adapter that
wraps a HuggingFace causal-LM as the `PolicyFn` expected by `agent.loop`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


def init_wandb(project: str, run_name: str, config: dict[str, Any]) -> None:
    """Initialise W&B if available and configured. Silent no-op otherwise."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping logging")
        return
    if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return
    wandb.init(project=project, name=run_name, config=config)


def wandb_log(metrics: dict[str, Any], step: int | None = None) -> None:
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


def load_filtered_dataset(cfg: dict[str, Any]):
    """Load the dataset built by `data.dataset.build_dataset`."""
    out_dir = Path(cfg["dataset"]["output_dir"])
    if not (out_dir / "dataset_dict.json").exists():
        # Could be a single Dataset (no DatasetDict) — try fallback.
        if (out_dir / "dataset_info.json").exists():
            from datasets import load_from_disk

            return load_from_disk(str(out_dir))
        raise FileNotFoundError(
            f"No HF dataset at {out_dir}; run data.dataset.build_dataset first."
        )
    return load_from_disk(str(out_dir))


@dataclass
class HFPolicy:
    """Wraps a HuggingFace causal-LM into the `PolicyFn` shape used by
    `agent.loop.run_agent_loop`. Greedy-by-default; pass a sampler in for
    rollouts."""

    model: Any
    tokenizer: Any
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95

    def __call__(self, messages: list[dict]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        import torch

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


def system_prompt_for_target(target: str, skill_md: str) -> str:
    """The single-turn warmup uses a shorter system prompt than the agent
    loop because it has to fit the answer in one shot."""
    if target == "triton":
        return (
            "You are a Triton kernel optimisation expert. Given a PyTorch "
            "`Model` class (with `get_init_inputs()` and `get_inputs()`), "
            "write a single Python file containing a `ModelNew` class that "
            "uses Triton kernels to accelerate the model. "
            "Follow this skill specification verbatim:\n\n"
            f"{skill_md}\n\n"
            "Output ONLY the Python module source for `model_new.py`, "
            "no markdown fences."
        )
    return (
        "You are a CUDA kernel optimisation expert. Given a PyTorch "
        "`Model` class (with `get_init_inputs()` and `get_inputs()`), "
        "write a CUDA C++ extension and a `model_new.py` that uses it to "
        "accelerate the model. Follow this skill specification verbatim:\n\n"
        f"{skill_md}\n\n"
        "Output the kernel source, the binding source, and `model_new.py` "
        "as separate code blocks tagged with their target file paths "
        "(`kernels/...cu`, `kernels/..._binding.cpp`, `model_new.py`)."
    )
