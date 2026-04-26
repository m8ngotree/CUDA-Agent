"""Stage 2 — LLM combinatorial synthesis (paper §3.1).

Given the unioned op pool from Stage 1 (torch + transformers crawls), sample
2..5 operators uniformly at random and ask an LLM to compose them into a
self-contained module of the KernelBench shape: `class Model(nn.Module)`,
`def get_init_inputs()`, and `def get_inputs()`. Each successful generation
is appended to `data/synthesized/fused_problems.jsonl`. The stage is fully
resumable — restarting it picks up from the existing JSONL.

Provider abstraction:
  - `provider: "anthropic"` -> uses `anthropic.Anthropic`
  - `provider: "openai"`    -> uses `openai.OpenAI`, optionally with `base_url`

Adding a new provider is a one-method extension to `LLMClient`.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from data.common import append_jsonl, existing_ids, load_config, read_jsonl, write_stats
from data.crawl.op_schema import OperatorEntry
from data.synthesis.synthesis_prompts import (
    SYSTEM_PROMPT,
    FusionPromptInputs,
    render_fusion_prompt,
)


class FusedProblemEntry(BaseModel):
    """One synthesised problem ready for filtering.

    `module_source` is a full Python module containing `class Model`,
    `get_init_inputs()`, and `get_inputs()` — the KernelBench format that
    the agent environment expects. It is NOT just a single class.
    """

    problem_id: str = Field(..., description="Stable hash of component ops + seed")
    component_ops: list[str]
    source_libs: list[str]
    module_source: str = Field(
        ..., description="Full module text (class Model + get_init_inputs + get_inputs)"
    )
    fusion_depth: int
    synthesis_model: str
    synthesis_seed: int = 0


# ---------------------------------------------------------------------------
# LLM client abstraction
# ---------------------------------------------------------------------------


class LLMClient:
    """Tiny wrapper around either anthropic.Anthropic or openai.OpenAI.
    Both expose a common .complete(system, user, ...) method returning text."""

    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        if provider == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic()
        elif provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(base_url=base_url) if base_url else OpenAI()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def complete(self, system: str, user: str) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in resp.content if hasattr(b, "text"))
        else:
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Synthesis core
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"^```(?:python)?\n(.*?)\n```\s*$", re.DOTALL)


def _strip_markdown_fence(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1) if m else text


def _extract_module(text: str) -> str | None:
    """Validate that the LLM emitted the three required symbols and return
    the cleaned module source. Returns None if anything is missing.

    We do a cheap structural check (string-match) rather than ast.parse so
    that minor syntax errors still pass to the filter stage, where they
    will be rejected with a precise `class_compile_error` reason — that
    information is more valuable in the funnel than silently dropping
    them at synthesis time.
    """
    text = _strip_markdown_fence(text).strip()
    if not text:
        return None

    required = ("class Model", "def get_init_inputs", "def get_inputs", "def forward")
    if any(tok not in text for tok in required):
        return None

    # Ensure the module imports torch even if the LLM forgot.
    if "import torch" not in text:
        text = "import torch\nimport torch.nn as nn\n\n\n" + text
    elif "import torch.nn" not in text and "torch.nn" in text:
        text = "import torch.nn as nn\n" + text
    return text


def _problem_id(op_names: list[str], seed: int) -> str:
    h = hashlib.sha256()
    h.update("|".join(op_names).encode())
    h.update(b"|")
    h.update(str(seed).encode())
    return h.hexdigest()[:16]


def synthesize_fused_problem(
    op_pool: list[OperatorEntry],
    n_ops: int,
    llm: LLMClient,
    rng: random.Random,
    seed: int,
) -> FusedProblemEntry | None:
    sample = rng.sample(op_pool, n_ops)
    inputs = FusionPromptInputs(
        op_names=[o.name for o in sample],
        op_source_libs=[o.source_lib for o in sample],
        op_tags=[o.tags for o in sample],
        op_class_defs=[o.class_def for o in sample],
    )
    prompt = render_fusion_prompt(inputs)

    try:
        raw = llm.complete(SYSTEM_PROMPT, prompt)
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None

    module_src = _extract_module(raw)
    if module_src is None:
        logger.debug(
            f"Response missing one of class Model / get_init_inputs / "
            f"get_inputs / forward (n_ops={n_ops})"
        )
        return None

    return FusedProblemEntry(
        problem_id=_problem_id([o.name for o in sample], seed),
        component_ops=[o.name for o in sample],
        source_libs=[o.source_lib for o in sample],
        module_source=module_src,
        fusion_depth=n_ops,
        synthesis_model=llm.model,
        synthesis_seed=seed,
    )


def load_op_pool(crawl_output_dir: Path) -> list[OperatorEntry]:
    pool: list[OperatorEntry] = []
    for fname in ("torch_ops.jsonl", "transformers_ops.jsonl"):
        p = crawl_output_dir / fname
        if not p.exists():
            logger.warning(f"Missing crawl output: {p}")
            continue
        for entry in read_jsonl(p, OperatorEntry):
            pool.append(entry)  # type: ignore[arg-type]
    if not pool:
        raise FileNotFoundError(
            f"No crawl output under {crawl_output_dir}; run data.crawl.* first."
        )
    return pool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-problems", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    syn_cfg: dict[str, Any] = cfg["synthesis"]
    crawl_cfg = cfg["crawl"]

    out_dir = Path(syn_cfg["output_dir"])
    out_path = out_dir / "fused_problems.jsonl"
    stats_path = out_dir / "synthesis_stats.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    op_pool = load_op_pool(Path(crawl_cfg["output_dir"]))
    logger.info(f"Loaded {len(op_pool)} operators from crawl output")

    seen = existing_ids(out_path) if syn_cfg.get("resume", True) else set()
    if seen:
        logger.info(f"Resuming: {len(seen)} problems already in {out_path}")

    target = args.n_problems if args.n_problems is not None else syn_cfg["n_problems_target"]
    n_remaining = max(0, target - len(seen))
    logger.info(f"Target {target}, need {n_remaining} more")

    if n_remaining == 0:
        logger.info("Nothing to do.")
        return 0

    llm = LLMClient(
        provider=syn_cfg.get("provider", "anthropic"),
        model=syn_cfg["model"],
        base_url=syn_cfg.get("base_url"),
        max_tokens=syn_cfg.get("max_tokens", 2048),
        temperature=syn_cfg.get("temperature", 0.8),
    )

    rng = random.Random(args.seed)
    min_ops = syn_cfg.get("min_ops", 2)
    max_ops = syn_cfg.get("max_ops", 5)

    n_attempts = 0
    n_success = 0
    n_extract_fail = 0
    n_llm_fail = 0
    n_dup = 0

    pbar = tqdm(total=n_remaining, desc="synth")
    while n_success < n_remaining:
        n_attempts += 1
        n_ops = rng.randint(min_ops, max_ops)
        seed = rng.randint(0, 2**31 - 1)

        entry = synthesize_fused_problem(op_pool, n_ops, llm, rng, seed)
        if entry is None:
            n_extract_fail += 1  # we lump LLM-error and parse-error here
            # Light backoff to avoid tight failure loops on rate limit.
            time.sleep(0.2)
            continue

        if entry.problem_id in seen:
            n_dup += 1
            continue
        seen.add(entry.problem_id)
        append_jsonl(out_path, entry)
        n_success += 1
        pbar.update(1)

    pbar.close()

    stats = {
        "model": syn_cfg["model"],
        "provider": syn_cfg.get("provider", "anthropic"),
        "target": target,
        "attempts": n_attempts,
        "success": n_success,
        "duplicates": n_dup,
        "extract_or_llm_failures": n_extract_fail,
        "llm_failures": n_llm_fail,
        "total_in_file": len(seen),
    }
    write_stats(stats_path, stats)
    logger.info(f"Synthesis done. {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
