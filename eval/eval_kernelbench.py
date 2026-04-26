"""KernelBench evaluation driver.

Loads KernelBench problems from `data/kernelbench/<level>/*.py`, runs the
agent loop on each, scores via reward, and writes one `EvalRecord` per
problem to `<eval_results_dir>/<benchmark>.jsonl`. A summary JSON sits
next to it.

The KernelBench layout we expect:
    data/kernelbench/
      level1/
        1_<name>.py
        2_<name>.py
        ...
      level2/...
      level3/...

Each problem file defines `class Model`, `get_init_inputs`, `get_inputs`
(same convention as the agent workdir) so we can drop it directly into
the sandbox's `model.py`.
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


def _problem_paths(kbroot: Path, benchmark: str) -> list[Path]:
    """Map `kernelbench_l1` -> `<root>/level1/*.py`, etc."""
    suffix = benchmark.split("_")[-1]
    level_dir = {
        "l1": "level1",
        "l2": "level2",
        "l3": "level3",
    }.get(suffix)
    if level_dir is None:
        raise ValueError(f"Unknown KernelBench benchmark: {benchmark}")
    files = sorted((kbroot / level_dir).glob("*.py"))
    if not files:
        raise FileNotFoundError(f"No problems under {kbroot / level_dir}")
    return files


def _load_policy(checkpoint: str | Path | None, base_model: str) -> HFPolicy:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = str(checkpoint) if checkpoint else base_model
    logger.info(f"Loading actor from {src}")
    tokenizer = AutoTokenizer.from_pretrained(src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return HFPolicy(model=model, tokenizer=tokenizer, max_new_tokens=2048, do_sample=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark", required=True, help="kernelbench_l1 / l2 / l3")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    acfg = cfg["agent"]
    ecfg = cfg["eval"]
    out_dir = Path(ecfg.get("output_dir", "eval_results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.benchmark}.jsonl"
    summary_path = out_dir / f"{args.benchmark}_summary.json"

    if out_path.exists():
        out_path.unlink()

    kbroot = Path(cfg["filter"]["kernelbench_path"])
    files = _problem_paths(kbroot, args.benchmark)
    if args.limit is not None:
        files = files[: args.limit]
    n_take = ecfg.get("n_problems")
    if n_take is not None:
        files = files[:n_take]

    policy = _load_policy(args.checkpoint, cfg["train"]["base_model"])

    for fp in files:
        src = fp.read_text()
        traj = run_agent_loop(
            problem_id=fp.stem,
            problem_source=src,
            skill_md_path=acfg["skill_md"],
            workdir_template=acfg["workdir"],
            policy=policy,
            max_turns=acfg.get("max_turns_eval", 200),
            speedup_threshold=acfg.get("reward_speedup_threshold", 0.05),
        )
        bd = traj.breakdown
        rec = EvalRecord(
            problem_id=fp.stem,
            correctness=bool(bd.correctness_passed) if bd else False,
            eager_time_ms=0.0,
            compile_time_ms=0.0,
            generated_time_ms=0.0,
            reward=traj.final_reward if traj.final_reward is not None else -1,
            n_turns=traj.n_turns,
            fail_reason=traj.fail_reason,
        )
        append_jsonl(out_path, rec)
        logger.info(
            f"{fp.stem}: r={rec.reward} pass={rec.correctness} turns={rec.n_turns}"
        )

    write_summary(out_path, summary_path, threshold=acfg.get("reward_speedup_threshold", 0.05))
    logger.info(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
