"""Minimal ReAct agent loop, OpenHands-compatible at the action level.

The full OpenHands runtime is overkill for our use case (and isn't available
on every machine). Instead, we run a small loop that:

  1. Materialises a fresh per-rollout workdir (copy of `agent/workdir`).
  2. Drops the Model source (class Model + get_init_inputs + get_inputs)
     into `model.py`.
  3. Lets the policy emit `<bash>...</bash>`, `<write path="...">...</write>`,
     or `<finish/>` actions.
  4. Executes each action in the workdir, captures stdout/stderr, and feeds
     the observation back to the policy.
  5. After `<finish/>` (or max-turns), runs verification.py + profiling.py
     to compute the reward via `agent/reward.py`.

The policy is any callable `policy(messages: list[dict]) -> str`. Plug in
TRL's `AutoModelForCausalLMWithValueHead`, an OpenAI client, or a stub.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agent.reward import RewardBreakdown, compute_reward_with_breakdown

PolicyFn = Callable[[list[dict]], str]


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------


_BASH_RE = re.compile(r"<bash>(.*?)</bash>", re.DOTALL)
_WRITE_RE = re.compile(r'<write\s+path="([^"]+)">(.*?)</write>', re.DOTALL)
_FINISH_RE = re.compile(r"<finish\s*/?>")


@dataclass
class Action:
    kind: str          # "bash" | "write" | "finish" | "noop"
    payload: str = ""
    path: str = ""


def parse_action(text: str) -> Action:
    """Parse the *first* recognised action out of an LLM message. The model
    is told to emit exactly one action per turn; if it emits more we keep
    the earliest, matching OpenHands' single-action contract."""

    matches: list[tuple[int, Action]] = []
    for m in _BASH_RE.finditer(text):
        matches.append((m.start(), Action(kind="bash", payload=m.group(1).strip())))
    for m in _WRITE_RE.finditer(text):
        matches.append(
            (m.start(), Action(kind="write", path=m.group(1), payload=m.group(2)))
        )
    for m in _FINISH_RE.finditer(text):
        matches.append((m.start(), Action(kind="finish")))

    if not matches:
        return Action(kind="noop")
    matches.sort(key=lambda t: t[0])
    return matches[0][1]


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """Full record of one rollout — used as RFT / value-pretrain input."""

    problem_id: str
    messages: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    final_reward: int | None = None
    breakdown: RewardBreakdown | None = None
    n_turns: int = 0
    finished: bool = False
    fail_reason: str | None = None


class Sandbox:
    """Per-rollout copy of `agent/workdir`. We never let the agent write
    outside this directory. Every shell action runs in this cwd."""

    def __init__(self, template_dir: Path, problem_source: str):
        self.template_dir = template_dir
        self._tmp = tempfile.mkdtemp(prefix="kernel-agent-")
        self.workdir = Path(self._tmp) / "workdir"
        shutil.copytree(template_dir, self.workdir)

        # Replace model.py with the synthesised problem.
        (self.workdir / "model.py").write_text(problem_source)

    def cleanup(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def run_bash(self, cmd: str, timeout_s: float = 60.0) -> tuple[int, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.workdir) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=self.workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return 124, f"[timeout after {timeout_s:.0f}s]"
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out[-4000:]  # cap observation length

    def write_file(self, rel_path: str, contents: str) -> str:
        # Reject absolute paths and parent traversal.
        rp = Path(rel_path)
        if rp.is_absolute() or ".." in rp.parts:
            return f"[error] illegal path: {rel_path}"
        target = self.workdir / rp
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
        return f"[wrote {rel_path} ({len(contents)} bytes)]"


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are a kernel optimization agent. You have a Linux shell sandbox with
NVCC and Python. Each turn you emit exactly one action:

  <bash>SHELL COMMAND</bash>      — run a shell command in the sandbox
  <write path="rel/path">CONTENT</write>   — overwrite a file in the sandbox
  <finish/>                       — declare you are done

After <finish/> the harness will run verification.py and profiling.py to
score you. Read SKILL.md before doing anything.
"""


def _initial_user_message(skill_md: str, problem_source: str) -> str:
    return textwrap.dedent(
        f"""\
        Here is the SKILL specification you must follow:
        ---
        {skill_md}
        ---
        Here is the PyTorch model you must accelerate (it is at model.py in your workdir):
        ```python
        {problem_source}
        ```
        Begin.
        """
    )


def _score_workdir(
    sandbox: Sandbox, speedup_threshold: float, timeout_s: float
) -> tuple[bool, float, float, float, str]:
    """Run verification + profiling in the sandbox. Returns
    (correct, gen_ms, eager_ms, compile_ms, log)."""
    # Compile.
    rc, log_compile = sandbox.run_bash("bash utils/compile.sh", timeout_s=timeout_s)
    if rc != 0:
        return False, 0.0, 0.0, 0.0, f"compile failed:\n{log_compile}"

    rc, log_verify = sandbox.run_bash("python -m utils.verification", timeout_s=timeout_s)
    if rc != 0:
        return False, 0.0, 0.0, 0.0, f"verify failed:\n{log_verify}"

    rc, log_prof = sandbox.run_bash("python -m utils.profiling", timeout_s=timeout_s)
    if rc != 0:
        return False, 0.0, 0.0, 0.0, f"profile failed:\n{log_prof}"

    # Parse "Torch Baseline: 12.345us, Torch Compile: 6.789us, CUDA Extension: 3.210us"
    m = re.search(
        r"Torch Baseline:\s*([\d.]+)us.*?Torch Compile:\s*([\d.]+)us.*?CUDA Extension:\s*([\d.]+)us",
        log_prof,
        re.DOTALL,
    )
    if not m:
        return False, 0.0, 0.0, 0.0, f"could not parse profile line:\n{log_prof}"

    eager_ms = float(m.group(1)) / 1000.0
    compile_ms = float(m.group(2)) / 1000.0
    gen_ms = float(m.group(3)) / 1000.0
    return True, gen_ms, eager_ms, compile_ms, log_prof


def run_agent_loop(
    *,
    problem_id: str,
    problem_source: str,
    skill_md_path: str | Path,
    workdir_template: str | Path,
    policy: PolicyFn,
    max_turns: int = 150,
    bash_timeout_s: float = 60.0,
    score_timeout_s: float = 120.0,
    speedup_threshold: float = 0.05,
) -> Trajectory:
    """Run a single rollout end-to-end. Returns a `Trajectory`."""
    skill_md = Path(skill_md_path).read_text()
    sandbox = Sandbox(Path(workdir_template), problem_source)
    traj = Trajectory(problem_id=problem_id)

    try:
        traj.messages.append({"role": "system", "content": SYSTEM_PROMPT})
        traj.messages.append(
            {"role": "user", "content": _initial_user_message(skill_md, problem_source)}
        )

        for turn in range(max_turns):
            traj.n_turns = turn + 1
            try:
                response = policy(traj.messages)
            except Exception as e:
                traj.fail_reason = f"policy_error: {e}"
                break

            traj.messages.append({"role": "assistant", "content": response})
            action = parse_action(response)

            if action.kind == "noop":
                obs = "[error] no action recognised — emit <bash>...</bash>, <write>, or <finish/>."
            elif action.kind == "bash":
                rc, out = sandbox.run_bash(action.payload, timeout_s=bash_timeout_s)
                obs = f"[bash rc={rc}]\n{out}"
            elif action.kind == "write":
                obs = sandbox.write_file(action.path, action.payload)
            elif action.kind == "finish":
                traj.finished = True
                traj.actions.append({"kind": action.kind, "path": action.path})
                traj.observations.append("[finish]")
                break
            else:
                obs = f"[error] unknown action kind {action.kind}"

            traj.actions.append(
                {"kind": action.kind, "path": action.path, "payload": action.payload[:500]}
            )
            traj.observations.append(obs)
            traj.messages.append({"role": "user", "content": obs})

        if not traj.finished:
            traj.fail_reason = traj.fail_reason or "max_turns_exceeded"
            return traj

        ok, gen_ms, eager_ms, compile_ms, log = _score_workdir(
            sandbox, speedup_threshold, score_timeout_s
        )
        bd = compute_reward_with_breakdown(
            generated_kernel_time_ms=gen_ms if ok else 0.0,
            eager_time_ms=eager_ms,
            compile_time_ms=compile_ms,
            correctness_passed=ok,
            speedup_threshold=speedup_threshold,
        )
        traj.breakdown = bd
        traj.final_reward = bd.reward
        if not ok:
            traj.fail_reason = log[:500]

    finally:
        sandbox.cleanup()

    return traj


# ---------------------------------------------------------------------------
# CLI for one-shot debugging.
# ---------------------------------------------------------------------------


def _stub_policy(messages: list[dict]) -> str:
    """Trivial policy that immediately gives up — handy for harness tests."""
    return "<finish/>"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir-template", default="agent/workdir")
    parser.add_argument("--skill-md", default="agent/workdir/SKILL.md")
    parser.add_argument("--problem-source", required=True, help="Path to a .py with class Model")
    parser.add_argument("--problem-id", default="cli-debug")
    parser.add_argument("--max-turns", type=int, default=5)
    args = parser.parse_args()

    src = Path(args.problem_source).read_text()
    traj = run_agent_loop(
        problem_id=args.problem_id,
        problem_source=src,
        skill_md_path=args.skill_md,
        workdir_template=args.workdir_template,
        policy=_stub_policy,
        max_turns=args.max_turns,
    )
    print(json.dumps(
        {
            "problem_id": traj.problem_id,
            "n_turns": traj.n_turns,
            "finished": traj.finished,
            "fail_reason": traj.fail_reason,
            "final_reward": traj.final_reward,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
