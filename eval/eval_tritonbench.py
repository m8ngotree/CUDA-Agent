"""TritonBench evaluation driver.

Same shape as `eval_kernelbench.py`, but expects problems under
`data/tritonbench/<category>/*.py`. We use the Triton SKILL.md and the
Triton workdir template (which routes to `verification_triton.py` /
`profiling_triton.py` automatically).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

from agent.loop import run_agent_loop
from data.common import append_jsonl, load_config
from eval.metrics import EvalRecord, write_summary
from train.utils import HFPolicy


def _problem_paths(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"TritonBench root missing: {root}")
    return sorted(root.rglob("*.py"))


def _load_policy(checkpoint: str | Path | None, base_model: str) -> HFPolicy:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = str(checkpoint) if checkpoint else base_model
    tokenizer = AutoTokenizer.from_pretrained(src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return HFPolicy(model=model, tokenizer=tokenizer, max_new_tokens=2048, do_sample=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--root", default="data/tritonbench")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    acfg = cfg["agent"]
    ecfg = cfg["eval"]
    out_dir = Path(ecfg.get("output_dir", "eval_results_triton"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tritonbench.jsonl"
    summary_path = out_dir / "tritonbench_summary.json"

    if out_path.exists():
        out_path.unlink()

    files = _problem_paths(Path(args.root))
    if args.limit is not None:
        files = files[: args.limit]
    if ecfg.get("n_problems"):
        files = files[: ecfg["n_problems"]]

    policy = _load_policy(args.checkpoint, cfg["train"]["base_model"])

    for fp in files:
        src = fp.read_text()
        traj = run_agent_loop(
            problem_id=str(fp.relative_to(args.root)),
            problem_source=src,
            skill_md_path=acfg["skill_md"],   # uses SKILL_TRITON.md when triton config is loaded
            workdir_template=acfg["workdir"],
            policy=policy,
            max_turns=acfg.get("max_turns_eval", 200),
            speedup_threshold=acfg.get("reward_speedup_threshold", 0.05),
        )
        bd = traj.breakdown
        rec = EvalRecord(
            problem_id=str(fp.relative_to(args.root)),
            correctness=bool(bd.correctness_passed) if bd else False,
            reward=traj.final_reward if traj.final_reward is not None else -1,
            n_turns=traj.n_turns,
            fail_reason=traj.fail_reason,
        )
        append_jsonl(out_path, rec)
        logger.info(f"{rec.problem_id}: r={rec.reward} pass={rec.correctness}")

    write_summary(out_path, summary_path, threshold=acfg.get("reward_speedup_threshold", 0.05))
    logger.info(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
