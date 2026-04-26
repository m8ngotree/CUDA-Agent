"""Stage 8d — Full agentic PPO (paper §3.3, Eq. 5).

The actor is initialised from the RFT checkpoint, the critic from the
value-pretrain checkpoint. Asymmetric clip:

    L_PPO = E[ min(ρ Â, clip(ρ, 1-ε_low, 1+ε_high) Â) ]
    ε_low = 0.20, ε_high = 0.28   (paper)

Per training step we:
  1. Sample a batch of problems from the train split.
  2. Run each through the agent loop with the current policy, recording
     log-probs at sampling time and final reward via `agent.reward`.
  3. Compute GAE returns over assistant-turn boundaries (γ=1, λ=0.95).
  4. Apply the asymmetric-clip PPO update on the actor and an MSE update
     on the critic.

This is the *only* stage where the actor and critic are jointly updated.
We keep TRL out of the loop here because TRL's PPOTrainer has only
symmetric clipping; rolling our own keeps us faithful to the paper.

Logged per step (paper §3.3 instability analysis):
  reward_mean / reward_std
  faster_than_compile_rate / faster_than_eager_rate / pass_rate
  mean_trajectory_length
  actor_entropy   (collapse signal)
  policy_loss / value_loss / explained_variance
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from agent.loop import run_agent_loop
from data.common import load_config
from train.utils import init_wandb, load_filtered_dataset, wandb_log

# ---------------------------------------------------------------------------
# Sample storage — one record per assistant turn we will update on.
# ---------------------------------------------------------------------------


@dataclass
class TurnSample:
    input_ids: torch.Tensor      # full conversation up to & including assistant tokens
    response_mask: torch.Tensor  # 1 on assistant-generated tokens, else 0
    old_logprobs: torch.Tensor   # log p(response | prefix) under the policy at sampling
    advantage: float
    return_: float
    value_target: float


# ---------------------------------------------------------------------------
# GAE — assistant turns are the timesteps. Outcome reward at terminal step.
# ---------------------------------------------------------------------------


def gae_advantages(
    rewards: list[float], values: list[float], gamma: float, lam: float
) -> tuple[list[float], list[float]]:
    """Standard GAE. Bootstrapped value at terminal = 0.
    Returns (advantages, returns)."""
    n = len(rewards)
    adv = [0.0] * n
    last = 0.0
    for t in reversed(range(n)):
        v_next = values[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * v_next - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    ret = [a + v for a, v in zip(adv, values, strict=True)]
    return adv, ret


# ---------------------------------------------------------------------------
# Policy that records sampling-time logprobs for PPO.
# ---------------------------------------------------------------------------


class RecordingPolicy:
    """Like `HFPolicy` but also records, for each generation, the prompt
    tokens, response tokens, and their per-token log-probs. We expose a
    list of records that the outer loop drains after each rollout."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 1024) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.records: list[dict] = []

    def __call__(self, messages: list[dict]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        device = self.model.pretrained_model.device
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].size(1)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        full_ids = out.sequences[0]                       # [prompt_len + gen_len]
        gen_ids = full_ids[prompt_len:]
        # Per-token log-prob at sampling time (using the scores we got back).
        # `out.scores` is a tuple of length gen_len, each [1, vocab].
        scores = torch.stack(out.scores, dim=1)[0]        # [gen_len, vocab]
        logprobs = F.log_softmax(scores, dim=-1)
        sampled_logp = logprobs.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)

        self.records.append(
            {
                "input_ids": full_ids.cpu(),
                "prompt_len": prompt_len,
                "old_logprobs": sampled_logp.cpu(),
            }
        )
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# PPO update with asymmetric clip.
# ---------------------------------------------------------------------------


def ppo_update(
    model,
    samples: list[TurnSample],
    optim,
    eps_low: float,
    eps_high: float,
    vf_coef: float = 1.0,
    ent_coef: float = 0.0,
) -> dict:
    """Single gradient update over `samples`. Caller decides batching."""
    device = model.pretrained_model.device

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl_old_new": 0.0}
    total_resp = 0
    pred_values: list[float] = []
    target_values: list[float] = []

    optim.zero_grad()
    for s in samples:
        ids = s.input_ids.to(device).unsqueeze(0)
        mask = s.response_mask.to(device).unsqueeze(0)
        old_lp = s.old_logprobs.to(device)
        adv = torch.tensor(s.advantage, device=device, dtype=old_lp.dtype)
        target_v = torch.tensor(s.value_target, device=device, dtype=torch.float32)

        outputs = model(input_ids=ids, attention_mask=(ids != 0).long())
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        values = outputs[2] if isinstance(outputs, tuple) and len(outputs) > 2 else outputs.value

        # Per-token log-probs aligned with response tokens. logits[t] predicts ids[t+1].
        shift_logits = logits[:, :-1, :]
        shift_ids = ids[:, 1:]
        shift_mask = mask[:, 1:]
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        new_lp = log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)[shift_mask.bool()]

        # Align old logprobs: they cover only the generated portion (response).
        new_lp = new_lp[: old_lp.numel()]
        old_lp_a = old_lp[: new_lp.numel()]

        ratio = torch.exp(new_lp - old_lp_a)
        clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high)
        # Asymmetric clip — a tighter lower bound than upper.
        pg = -torch.min(ratio * adv, clipped * adv).mean()

        # Critic — value of the last response token.
        v_at_end = values[0, -1] if values.dim() == 2 else values[0]
        vf_loss = 0.5 * (v_at_end - target_v) ** 2
        pred_values.append(float(v_at_end.detach().float().cpu()))
        target_values.append(float(target_v.detach().float().cpu()))

        # Entropy on the response tokens — useful for the collapse-monitor.
        full_log_probs = log_probs[shift_mask.bool()]
        entropy = -(full_log_probs.exp() * full_log_probs).sum(dim=-1).mean()

        loss = pg + vf_coef * vf_loss - ent_coef * entropy
        loss.backward()

        metrics["policy_loss"] += float(pg.detach())
        metrics["value_loss"] += float(vf_loss.detach())
        metrics["entropy"] += float(entropy.detach())
        metrics["kl_old_new"] += float((old_lp_a - new_lp).mean().detach())
        total_resp += new_lp.numel()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()

    n = max(1, len(samples))
    for k in metrics:
        metrics[k] /= n
    if pred_values:
        p, t = np.array(pred_values), np.array(target_values)
        var_t = t.var()
        ev = 1.0 - ((t - p) ** 2).mean() / var_t if var_t > 1e-8 else float("nan")
        metrics["explained_variance"] = ev if not math.isnan(ev) else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Main loop.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    acfg = cfg["agent"]

    init_wandb(
        project=tcfg["wandb_project"],
        run_name="agentic_rl",
        config={"stage": "agentic_rl", **tcfg},
    )

    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead

    rft_path = Path(tcfg["output_dir"]) / "rft"
    val_path = Path(tcfg["output_dir"]) / "value_pretrain"
    actor_init = val_path if val_path.exists() else (rft_path if rft_path.exists() else Path(tcfg["base_model"]))

    logger.info(f"Loading actor+critic from {actor_init}")
    tokenizer = AutoTokenizer.from_pretrained(str(actor_init))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        str(actor_init), torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.train()

    optim = torch.optim.AdamW(
        [
            {"params": model.pretrained_model.parameters(), "lr": tcfg["learning_rate_actor"]},
            {"params": model.v_head.parameters(), "lr": tcfg["learning_rate_critic"]},
        ]
    )

    ds = load_filtered_dataset(cfg)["train"]
    micro_bs = tcfg.get("micro_batch_size", 4)
    n_steps = int(tcfg.get("agentic_rl_steps", 150))
    eps_low = tcfg["clip_epsilon_lower"]
    eps_high = tcfg["clip_epsilon_upper"]
    gamma = tcfg["gae_gamma"]
    lam = tcfg["gae_lambda"]

    rng = np.random.default_rng(0)

    for step in range(n_steps):
        # ---- Rollout phase --------------------------------------------------
        idxs = rng.choice(len(ds), size=micro_bs, replace=False)
        batch = [ds[int(i)] for i in idxs]
        recorder = RecordingPolicy(model, tokenizer, max_new_tokens=2048)
        trajs = []
        for ex in batch:
            recorder.records.clear()
            traj = run_agent_loop(
                problem_id=ex["problem_id"],
                problem_source=ex["pytorch_source"],
                skill_md_path=acfg["skill_md"],
                workdir_template=acfg["workdir"],
                policy=recorder,
                max_turns=acfg.get("max_turns_train", 150),
                speedup_threshold=acfg.get("reward_speedup_threshold", 0.05),
            )
            trajs.append({"traj": traj, "records": list(recorder.records)})

        rewards = [t["traj"].final_reward if t["traj"].final_reward is not None else -1 for t in trajs]
        traj_lens = [t["traj"].n_turns for t in trajs]
        passes = sum(1 for t in trajs if (t["traj"].breakdown and t["traj"].breakdown.correctness_passed))
        ftc = sum(1 for t in trajs if (t["traj"].breakdown and t["traj"].breakdown.faster_than_compile))
        fte = sum(1 for t in trajs if (t["traj"].breakdown and t["traj"].breakdown.faster_than_eager))

        # ---- Build per-turn samples with GAE -------------------------------
        samples: list[TurnSample] = []
        for tinfo in trajs:
            recs = tinfo["records"]
            traj = tinfo["traj"]
            if not recs:
                continue
            # Per-turn rewards: 0 except final.
            n_turns = len(recs)
            r = [0.0] * n_turns
            r[-1] = float(traj.final_reward if traj.final_reward is not None else -1)

            # Critic forward pass — get V φ at each turn boundary.
            # (Single batched forward would be faster; keep simple here.)
            with torch.no_grad():
                values: list[float] = []
                for rec in recs:
                    ids = rec["input_ids"].unsqueeze(0).to(model.pretrained_model.device)
                    out = model(input_ids=ids, attention_mask=(ids != 0).long())
                    v = out[2] if isinstance(out, tuple) and len(out) > 2 else out.value
                    v_at_end = float(v[0, -1]) if v.dim() == 2 else float(v[0])
                    values.append(v_at_end)
            adv, ret = gae_advantages(r, values, gamma=gamma, lam=lam)

            # Normalise advantages within trajectory (paper section uses
            # per-batch normalisation; we keep it per-trajectory for stability
            # at small micro-batch sizes).
            adv_arr = np.array(adv)
            if adv_arr.std() > 1e-6:
                adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-6)

            for i, rec in enumerate(recs):
                full = rec["input_ids"]
                prompt_len = rec["prompt_len"]
                resp_mask = torch.zeros_like(full)
                resp_mask[prompt_len:] = 1
                samples.append(
                    TurnSample(
                        input_ids=full,
                        response_mask=resp_mask,
                        old_logprobs=rec["old_logprobs"],
                        advantage=float(adv_arr[i]),
                        return_=float(ret[i]),
                        value_target=float(ret[i]),
                    )
                )

        if not samples:
            logger.warning(f"step={step}: no samples produced, skipping update")
            continue

        # ---- Update --------------------------------------------------------
        update_metrics = ppo_update(
            model, samples, optim, eps_low=eps_low, eps_high=eps_high
        )

        # ---- Log -----------------------------------------------------------
        ftc_rate = ftc / len(trajs)
        fte_rate = fte / len(trajs)
        pass_rate = passes / len(trajs)
        wandb_log(
            {
                "step": step,
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "pass_rate": pass_rate,
                "faster_than_eager_rate": fte_rate,
                "faster_than_compile_rate": ftc_rate,
                "mean_trajectory_length": float(np.mean(traj_lens)),
                **{f"actor/{k}": v for k, v in update_metrics.items()},
            },
            step=step,
        )
        logger.info(
            f"step={step} reward={np.mean(rewards):.3f} pass={pass_rate:.2f} "
            f"ftc={ftc_rate:.2f} entropy={update_metrics.get('entropy', 0):.3f} "
            f"ev={update_metrics.get('explained_variance', 0):.2f}"
        )

    out_dir = Path(tcfg["output_dir"]) / "agentic_rl"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"Saved final agentic-RL checkpoint to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
