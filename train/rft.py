"""Stage 8b — Rejection Fine-Tuning (paper §3.3, Eq. 2).

We:
  1. Run the warmed-up actor in the agent loop on `rft_max_samples` problems
     and collect full trajectories (including all observations).
  2. Apply the rejection criteria from the spec:
       - outcome filter: final reward > 0
       - pattern filter:
           * redundant multi-turn loops (same action repeated >3x)
           * invalid tool call schema (hallucinated action types)
           * calls to torch.nn.functional in model_new.py (reward hacking)
           * trajectory length > 150 turns
  3. SFT the actor on the surviving trajectories with a standard
     causal-LM loss masked to assistant tokens only (Eq. 2).

Trajectories are checkpointed to JSONL so the rollout phase is resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from agent.loop import Trajectory, run_agent_loop
from data.common import append_jsonl, existing_ids, load_config
from train.utils import (
    HFPolicy,
    init_wandb,
    load_filtered_dataset,
    wandb_log,
)

# ---------------------------------------------------------------------------
# Pattern filters
# ---------------------------------------------------------------------------


def _has_redundant_loop(actions: list[dict], max_repeats: int = 3) -> bool:
    """Same action payload appearing >max_repeats times in a row."""
    if len(actions) <= max_repeats:
        return False
    last = None
    run = 0
    for a in actions:
        sig = (a.get("kind"), (a.get("payload") or "")[:200])
        if sig == last:
            run += 1
            if run > max_repeats:
                return True
        else:
            last = sig
            run = 1
    return False


def _has_invalid_actions(actions: list[dict]) -> bool:
    return any(a.get("kind") not in {"bash", "write", "finish"} for a in actions)


def _calls_torch_functional(traj: Trajectory) -> bool:
    """Reward-hacking guard. Look at every <write> targeting model_new.py
    and check whether the body imports `torch.nn.functional` or invokes any
    `F.*` call. We keep this rule deliberately conservative — false
    positives are cheaper than letting reward hacks into the SFT corpus."""
    for a in traj.actions:
        if a.get("kind") != "write":
            continue
        if not a.get("path", "").endswith("model_new.py"):
            continue
        body = a.get("payload", "")
        if "torch.nn.functional" in body or "import torch.nn.functional" in body:
            return True
        if "F." in body and "import torch.nn.functional as F" in body:
            return True
    return False


def trajectory_passes_filters(traj: Trajectory, max_turns: int = 150) -> tuple[bool, str]:
    if traj.final_reward is None or traj.final_reward <= 0:
        return False, "outcome_filter"
    if traj.n_turns > max_turns:
        return False, "too_long"
    if _has_redundant_loop(traj.actions):
        return False, "redundant_loop"
    if _has_invalid_actions(traj.actions):
        return False, "invalid_actions"
    if _calls_torch_functional(traj):
        return False, "reward_hack_torch_functional"
    return True, "ok"


# ---------------------------------------------------------------------------
# Phase A: trajectory collection
# ---------------------------------------------------------------------------


def collect_trajectories(cfg: dict[str, Any], n_target: int, ckpt_dir: Path) -> Path:
    """Roll out the warmed-up policy across the train split, save every trajectory
    (passed or not) to JSONL, and return the path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tcfg = cfg["train"]
    acfg = cfg["agent"]

    model_path = ckpt_dir if ckpt_dir.exists() else tcfg["base_model"]
    logger.info(f"Loading actor from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    policy = HFPolicy(model=model, tokenizer=tokenizer, max_new_tokens=2048)

    ds = load_filtered_dataset(cfg)["train"]
    out_path = Path(tcfg["output_dir"]) / "rft_trajectories.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = existing_ids(out_path)
    if seen:
        logger.info(f"Resume RFT collection — already {len(seen)} trajectories")

    n_collected = 0
    for ex in ds:
        if n_collected >= n_target:
            break
        if ex["problem_id"] in seen:
            continue
        traj = run_agent_loop(
            problem_id=ex["problem_id"],
            problem_source=ex["pytorch_source"],
            skill_md_path=acfg["skill_md"],
            workdir_template=acfg["workdir"],
            policy=policy,
            max_turns=acfg.get("max_turns_train", 150),
            score_timeout_s=acfg.get("subprocess_timeout_s", 60),
            speedup_threshold=acfg.get("reward_speedup_threshold", 0.05),
        )
        # JSONL-friendly serialisation of the trajectory.
        record = {
            "problem_id": traj.problem_id,
            "messages": traj.messages,
            "actions": traj.actions,
            "observations": traj.observations,
            "n_turns": traj.n_turns,
            "finished": traj.finished,
            "fail_reason": traj.fail_reason,
            "final_reward": traj.final_reward,
        }
        append_jsonl(out_path, record)
        n_collected += 1

    logger.info(f"Collected {n_collected} new trajectories to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Phase B: filter + SFT
# ---------------------------------------------------------------------------


def _filter_to_sft_corpus(traj_path: Path, out_path: Path, max_turns: int) -> dict:
    """Read all trajectories, drop those that fail the filters, write a
    chat-format JSONL ready for `SFTTrainer`."""
    funnel = {"total": 0, "ok": 0}
    with traj_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            traj_dict = json.loads(line)
            traj = Trajectory(
                problem_id=traj_dict["problem_id"],
                messages=traj_dict["messages"],
                actions=traj_dict["actions"],
                observations=traj_dict["observations"],
                n_turns=traj_dict.get("n_turns", 0),
                finished=traj_dict.get("finished", False),
                fail_reason=traj_dict.get("fail_reason"),
                final_reward=traj_dict.get("final_reward"),
            )
            funnel["total"] += 1
            ok, reason = trajectory_passes_filters(traj, max_turns=max_turns)
            funnel[reason] = funnel.get(reason, 0) + 1
            if not ok:
                continue
            fout.write(json.dumps({"messages": traj.messages}) + "\n")
    logger.info(f"RFT funnel: {funnel}")
    return funnel


def run_sft(corpus_path: Path, cfg: dict[str, Any]) -> Path:
    """Standard SFT loop using TRL's SFTTrainer (Eq. 2)."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tcfg = cfg["train"]
    base_model = (Path(tcfg["output_dir"]) / "warmup")
    if not base_model.exists():
        base_model = Path(tcfg["base_model"])

    tokenizer = AutoTokenizer.from_pretrained(str(base_model))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model), torch_dtype=torch.bfloat16
    )

    rows = []
    with corpus_path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(
            f"No trajectories survived RFT filtering at {corpus_path}; cannot train."
        )
    ds = Dataset.from_list(rows)

    out_dir = Path(tcfg["output_dir"]) / "rft"
    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=tcfg.get("micro_batch_size", 4),
        gradient_accumulation_steps=max(
            1, tcfg["global_batch_size"] // (tcfg.get("micro_batch_size", 4) * 8)
        ),
        learning_rate=tcfg["learning_rate_actor"],
        num_train_epochs=tcfg.get("rft_epochs", 2),
        bf16=True,
        logging_steps=10,
        save_steps=500,
        max_seq_length=tcfg.get("single_turn_context", 32768),
        report_to="wandb",
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-rollouts", type=int, default=None)
    parser.add_argument("--skip-rollouts", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    init_wandb(
        project=cfg["train"]["wandb_project"],
        run_name="rft",
        config={"stage": "rft", **cfg["train"]},
    )

    tcfg = cfg["train"]
    warmup_ckpt = Path(tcfg["output_dir"]) / "warmup"

    if not args.skip_rollouts:
        n_target = args.n_rollouts or tcfg.get("rft_max_samples", 50000)
        traj_path = collect_trajectories(cfg, n_target, warmup_ckpt)
    else:
        traj_path = Path(tcfg["output_dir"]) / "rft_trajectories.jsonl"

    sft_corpus = Path(tcfg["output_dir"]) / "rft_sft_corpus.jsonl"
    sft_corpus.parent.mkdir(parents=True, exist_ok=True)
    funnel = _filter_to_sft_corpus(
        traj_path, sft_corpus, max_turns=cfg["agent"].get("max_turns_train", 150)
    )
    wandb_log({"rft/funnel/" + k: v for k, v in funnel.items()})

    out_dir = run_sft(sft_corpus, cfg)
    logger.info(f"Saved RFT checkpoint to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
