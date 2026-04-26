"""Stage 8a — single-turn PPO warmup (paper §3.3).

Given a PyTorch operator, the model emits an entire kernel + bindings in
one shot; the reward function then scores it via the existing
verification/profiling sandbox in `agent/workdir`. Asymmetric PPO clip
(ε_lower=0.20, ε_upper=0.28, paper Eq. 5).

We use TRL's `PPOTrainer` for the actor/critic update step. Rollouts are
done in-process: for each prompt we call `policy.generate`, parse out the
file blocks, drop them into a fresh sandbox, run verification + profiling,
and read back the wall-clock times to compute the reward.

Note: this stage is intentionally single-turn — multi-turn agentic RL is
the final stage. The single-turn pass primarily serves to (a) initialise
the actor with calibrated logits before agentic RL, and (b) initialise
the critic with non-random value estimates.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

from agent.reward import compute_reward_with_breakdown
from data.common import load_config
from train.utils import init_wandb, load_filtered_dataset, system_prompt_for_target, wandb_log

# ---------------------------------------------------------------------------
# Output parser — turns one model response into a set of files to drop into
# the sandbox.
# ---------------------------------------------------------------------------


_FILE_BLOCK_RE = re.compile(
    r"```(?:\w+)?\s*\n\s*(?:#|//)\s*(?P<path>[^\n]+\.(?:py|cu|cpp|h))\s*\n(?P<body>.*?)\n```",
    re.DOTALL,
)


def parse_files_from_response(text: str) -> dict[str, str]:
    """Extract `path -> contents` from triple-backtick blocks whose first
    line is a comment of the form `# kernels/foo.cu`. The single-turn
    prompt asks the model to follow this convention."""
    out: dict[str, str] = {}
    for m in _FILE_BLOCK_RE.finditer(text):
        path = m.group("path").strip()
        body = m.group("body")
        out[path] = body
    return out


# ---------------------------------------------------------------------------
# Sandbox scoring — reuses agent/workdir as a template.
# ---------------------------------------------------------------------------


@dataclass
class ScoringConfig:
    workdir_template: Path
    timeout_s: float
    speedup_threshold: float


def score_response(
    problem_source: str, response: str, scfg: ScoringConfig
) -> tuple[int, dict[str, float]]:
    files = parse_files_from_response(response)
    if not files:
        return -1, {"reason": -1, "compile_ms": 0.0, "eager_ms": 0.0, "gen_ms": 0.0}

    with tempfile.TemporaryDirectory(prefix="warmup-") as td:
        wd = Path(td) / "wd"
        shutil.copytree(scfg.workdir_template, wd)
        (wd / "model.py").write_text(problem_source)
        for rel, body in files.items():
            target = wd / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(wd) + os.pathsep + env.get("PYTHONPATH", "")

        # Compile.
        rc = subprocess.run(
            ["bash", "utils/compile.sh"],
            cwd=wd, env=env, capture_output=True, text=True, timeout=scfg.timeout_s,
        ).returncode
        if rc != 0:
            return -1, {"compile_ms": 0.0, "eager_ms": 0.0, "gen_ms": 0.0}

        # Verify.
        v = subprocess.run(
            ["python", "-m", "utils.verification"],
            cwd=wd, env=env, capture_output=True, text=True, timeout=scfg.timeout_s,
        )
        correct = v.returncode == 0

        # Profile (always — even if verify failed we want timing for reward
        # diagnostics; compute_reward will floor the result at -1 for failure).
        p = subprocess.run(
            ["python", "-m", "utils.profiling"],
            cwd=wd, env=env, capture_output=True, text=True, timeout=scfg.timeout_s,
        )
        m = re.search(
            r"Torch Baseline:\s*([\d.]+)us.*?Torch Compile:\s*([\d.]+)us.*?CUDA Extension:\s*([\d.]+)us",
            p.stdout, re.DOTALL,
        )
        if not m:
            return (-1 if not correct else 1), {
                "compile_ms": 0.0, "eager_ms": 0.0, "gen_ms": 0.0
            }
        eager_ms = float(m.group(1)) / 1000.0
        compile_ms = float(m.group(2)) / 1000.0
        gen_ms = float(m.group(3)) / 1000.0

    bd = compute_reward_with_breakdown(
        gen_ms, eager_ms, compile_ms, correct, speedup_threshold=scfg.speedup_threshold
    )
    return bd.reward, {
        "compile_ms": compile_ms,
        "eager_ms": eager_ms,
        "gen_ms": gen_ms,
        "faster_than_eager": int(bd.faster_than_eager),
        "faster_than_compile": int(bd.faster_than_compile),
        "correct": int(bd.correctness_passed),
    }


# ---------------------------------------------------------------------------
# Training driver.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    acfg = cfg["agent"]

    init_wandb(
        project=tcfg["wandb_project"],
        run_name="single_turn_warmup",
        config={"stage": "single_turn_warmup", **tcfg},
    )

    # Lazy import — pulling these in at module top would make `--help`
    # painfully slow.
    from transformers import AutoTokenizer
    from trl import (
        AutoModelForCausalLMWithValueHead,
        PPOConfig,
        PPOTrainer,
    )

    base_model = tcfg["base_model"]
    logger.info(f"Loading {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    )

    ppo_cfg = PPOConfig(
        learning_rate=tcfg["learning_rate_actor"],
        batch_size=tcfg.get("micro_batch_size", 8),
        mini_batch_size=tcfg.get("micro_batch_size", 8),
        gradient_accumulation_steps=max(
            1, tcfg["global_batch_size"] // tcfg.get("micro_batch_size", 8)
        ),
        cliprange=tcfg["clip_epsilon_lower"],
        cliprange_value=0.2,
        gamma=tcfg["gae_gamma"],
        lam=tcfg["gae_lambda"],
        max_grad_norm=1.0,
        log_with=None,
    )

    trainer = PPOTrainer(
        config=ppo_cfg,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    ds = load_filtered_dataset(cfg)["train"]
    sys_prompt = system_prompt_for_target(
        cfg.get("target", "cuda"), Path(acfg["skill_md"]).read_text()
    )
    workdir_template = Path(acfg["workdir"])

    scoring_cfg = ScoringConfig(
        workdir_template=workdir_template,
        timeout_s=acfg.get("subprocess_timeout_s", 60),
        speedup_threshold=acfg.get("reward_speedup_threshold", 0.05),
    )

    micro_bs = tcfg.get("micro_batch_size", 8)
    n_steps = args.max_steps or len(ds) // micro_bs
    logger.info(f"Single-turn warmup: {n_steps} steps, micro_bs={micro_bs}")

    step = 0
    for _epoch in range(args.epochs):
        for batch_start in range(0, len(ds), micro_bs):
            if step >= n_steps:
                break
            batch = ds[batch_start : batch_start + micro_bs]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": f"Optimise this model:\n\n{src}"},
                    ],
                    tokenize=False, add_generation_prompt=True,
                )
                for src in batch["pytorch_source"]
            ]
            query_tensors = [
                tokenizer(p, return_tensors="pt").input_ids[0].to(trainer.accelerator.device)
                for p in prompts
            ]
            response_tensors = trainer.generate(
                query_tensors,
                max_new_tokens=2048,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
            responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

            t_score = time.perf_counter()
            rewards: list[float] = []
            metas: list[dict[str, float]] = []
            for src, resp in zip(batch["pytorch_source"], responses, strict=True):
                r, meta = score_response(src, resp, scoring_cfg)
                rewards.append(float(r))
                metas.append(meta)
            score_time = time.perf_counter() - t_score

            stats = trainer.step(
                query_tensors, response_tensors,
                [torch.tensor(r, device=trainer.accelerator.device) for r in rewards],
            )

            mean_reward = sum(rewards) / max(len(rewards), 1)
            ftc_rate = sum(m.get("faster_than_compile", 0) for m in metas) / max(len(metas), 1)
            fte_rate = sum(m.get("faster_than_eager", 0) for m in metas) / max(len(metas), 1)
            pass_rate = sum(m.get("correct", 0) for m in metas) / max(len(metas), 1)

            wandb_log(
                {
                    "step": step,
                    "reward_mean": mean_reward,
                    "reward_std": (
                        torch.tensor(rewards, dtype=torch.float).std().item()
                        if len(rewards) > 1 else 0.0
                    ),
                    "faster_than_compile_rate": ftc_rate,
                    "faster_than_eager_rate": fte_rate,
                    "pass_rate": pass_rate,
                    "score_time_s": score_time,
                    **{f"ppo/{k}": v for k, v in (stats or {}).items() if isinstance(v, (int, float))},
                },
                step=step,
            )
            logger.info(
                f"step={step} reward={mean_reward:.3f} ftc={ftc_rate:.2f} pass={pass_rate:.2f}"
            )
            step += 1

    out_dir = Path(tcfg["output_dir"]) / "warmup"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"Saved warmup checkpoint to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
