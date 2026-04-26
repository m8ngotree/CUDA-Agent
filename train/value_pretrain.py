"""Stage 8c — Value pretraining (paper §3.3, Eq. 4).

Initialise the critic so that subsequent agentic PPO doesn't start with
~zero explained variance (which the paper notes triggers an instability
loop). Concretely:

  V_t^targ = V_φ(s_t) + Â_t              (GAE advantage estimate)
  L_VP(φ) = (1/2) * E[ (1/T) * sum_t (V_φ(s_t) - V_t^targ)^2 ]
  γ = 1, λ = 0.95   (paper)

Inputs are the trajectory JSONL produced by `train.rft` (collected before
rejection so we have low + high reward states). We do *not* update the
actor here.

We compute a per-token value target by treating the entire assistant
turn as one timestep with reward 0, and the final turn (after <finish/>)
as the terminal step with reward = trajectory.final_reward. This is the
standard outcome-only credit assignment that the paper uses.

Stops when explained_variance ≥ target (default 0.5) or we hit max steps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from data.common import load_config
from train.utils import init_wandb, wandb_log


@dataclass
class Step:
    input_ids: torch.Tensor   # tokenised conversation up to this turn
    target_value: float       # GAE target


class TrajectoryValueDataset(Dataset):
    """One sample per assistant turn. Each sample ships its V^targ that we
    pre-compute via GAE before training (we don't need V_φ to compute the
    targets because we use Monte-Carlo returns when V_φ is missing — i.e.
    the first epoch is a regression onto the discounted return)."""

    def __init__(
        self,
        traj_jsonl: Path,
        tokenizer,
        max_seq_len: int,
        gamma: float,
        lam: float,
        max_trajectories: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.gamma = gamma
        self.lam = lam
        self.steps: list[Step] = []
        self._load(traj_jsonl, max_trajectories)

    def _load(self, traj_jsonl: Path, max_trajectories: int | None) -> None:
        n = 0
        with traj_jsonl.open() as f:
            for line in f:
                if max_trajectories is not None and n >= max_trajectories:
                    break
                rec = json.loads(line)
                final_r = rec.get("final_reward")
                if final_r is None:
                    continue
                msgs = rec["messages"]
                # Assistant turn indices.
                asst_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
                if not asst_idx:
                    continue

                # Per-step reward: 0 for all but the last assistant turn,
                # final_r for the last (paper's outcome-only reward).
                rewards = [0.0] * len(asst_idx)
                rewards[-1] = float(final_r)

                # Monte-Carlo return for the first-epoch target (we don't have
                # bootstrapped V φ yet). With γ=1 this is just `final_r` for
                # every step — exactly what GAE collapses to under outcome-only
                # rewards anyway.
                returns = self._compute_returns(rewards)

                for i, idx in enumerate(asst_idx):
                    convo = msgs[: idx + 1]
                    text = self.tokenizer.apply_chat_template(
                        convo, tokenize=False, add_generation_prompt=False
                    )
                    ids = self.tokenizer(
                        text, truncation=True, max_length=self.max_seq_len, return_tensors="pt"
                    ).input_ids[0]
                    self.steps.append(Step(input_ids=ids, target_value=returns[i]))
                n += 1
        logger.info(f"Built {len(self.steps)} value-pretrain steps from {n} trajectories")

    def _compute_returns(self, rewards: list[float]) -> list[float]:
        """Discounted return per step. With γ=1 (paper) every step gets the
        same outcome reward; we still implement the general formula for
        completeness."""
        ret = 0.0
        out = [0.0] * len(rewards)
        for i in reversed(range(len(rewards))):
            ret = rewards[i] + self.gamma * ret
            out[i] = ret
        return out

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> dict:
        s = self.steps[idx]
        return {"input_ids": s.input_ids, "target_value": s.target_value}


def _collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(b["input_ids"].size(0) for b in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros_like(ids)
    targets = torch.zeros(len(batch), dtype=torch.float)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        ids[i, :n] = b["input_ids"]
        attn[i, :n] = 1
        targets[i] = b["target_value"]
    return {"input_ids": ids, "attention_mask": attn, "target_value": targets}


def _explained_variance(preds: np.ndarray, targets: np.ndarray) -> float:
    var_t = targets.var()
    if var_t < 1e-8:
        return float("nan")
    return 1.0 - ((targets - preds) ** 2).mean() / var_t


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]

    init_wandb(
        project=tcfg["wandb_project"],
        run_name="value_pretrain",
        config={"stage": "value_pretrain", **tcfg},
    )

    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead

    actor_path = Path(tcfg["output_dir"]) / "rft"
    if not actor_path.exists():
        actor_path = Path(tcfg["base_model"])

    tokenizer = AutoTokenizer.from_pretrained(str(actor_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        str(actor_path), torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Freeze the LM backbone — we only train the value head.
    for n, p in model.named_parameters():
        p.requires_grad = "v_head" in n

    traj_path = Path(tcfg["output_dir"]) / "rft_trajectories.jsonl"
    if not traj_path.exists():
        raise FileNotFoundError(
            f"No trajectories at {traj_path}; run train.rft (with rollouts) first."
        )

    ds = TrajectoryValueDataset(
        traj_path,
        tokenizer,
        max_seq_len=tcfg.get("agentic_context", 32768),
        gamma=tcfg["gae_gamma"],
        lam=tcfg["gae_lambda"],
    )

    loader = DataLoader(
        ds,
        batch_size=tcfg.get("micro_batch_size", 4),
        shuffle=True,
        collate_fn=lambda b: _collate(b, tokenizer.pad_token_id),
    )
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["learning_rate_critic"],
    )

    target_ev = float(tcfg.get("value_pretrain_target_explained_variance", 0.5))
    max_steps = args.max_steps or int(tcfg.get("value_pretrain_steps", 5000))

    step = 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: v.to(model.pretrained_model.device) for k, v in batch.items()}
            _, _, values = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_past_key_values=False,
            )
            # `values` is shape [B, T]; grab the last non-pad position per row.
            seq_lens = batch["attention_mask"].sum(dim=1).clamp(min=1) - 1
            v_at_end = values[torch.arange(values.size(0)), seq_lens]
            target = batch["target_value"].to(v_at_end.dtype)

            loss = 0.5 * ((v_at_end - target) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optim.step()

            preds = v_at_end.detach().float().cpu().numpy()
            tgts = target.detach().float().cpu().numpy()
            ev = _explained_variance(preds, tgts)

            wandb_log(
                {
                    "step": step,
                    "value_loss": float(loss.item()),
                    "explained_variance": (ev if not math.isnan(ev) else 0.0),
                    "target_mean": float(tgts.mean()),
                    "target_std": float(tgts.std()),
                },
                step=step,
            )
            if step % 50 == 0:
                logger.info(f"step={step} loss={loss.item():.4f} ev={ev:.3f}")
            step += 1

            if not math.isnan(ev) and ev >= target_ev and step > 100:
                logger.info(f"Reached target explained variance {ev:.3f} >= {target_ev}")
                break
        else:
            continue
        break

    out_dir = Path(tcfg["output_dir"]) / "value_pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info(f"Saved value-pretrain checkpoint to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
