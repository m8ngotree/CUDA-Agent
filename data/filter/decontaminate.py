"""KernelBench decontamination via MinHash LSH (Appendix A of the paper).

We k-shingle the source code (k=5 token windows over a coarse tokenisation),
hash each shingle into a MinHash signature with 128 permutations, and query
an LSH index for any KernelBench problem with Jaccard ≥ 0.7.

This file exposes a single function:
    score_against_kernelbench(source: str, index) -> float

It returns the maximum estimated Jaccard similarity vs. any indexed problem.
A small CLI is included for ad-hoc inspection.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from loguru import logger

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[+\-*/=<>!]+|\d+|\S")


def _tokens(src: str) -> list[str]:
    return _TOKEN_RE.findall(src)


def _shingles(tokens: list[str], k: int = 5) -> set[str]:
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def make_minhash(src: str, num_perm: int = 128, k: int = 5) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(_tokens(src), k=k):
        m.update(sh.encode())
    return m


class KernelBenchIndex:
    """Wraps a `MinHashLSH` plus the per-key MinHashes so we can score
    against the closest match (LSH only tells us if anything is over the
    threshold, not what the actual score is)."""

    def __init__(self, threshold: float = 0.5, num_perm: int = 128) -> None:
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._mh_by_key: dict[str, MinHash] = {}

    @classmethod
    def from_sources(
        cls, items: Iterable[tuple[str, str]], threshold: float = 0.5, num_perm: int = 128
    ) -> KernelBenchIndex:
        idx = cls(threshold=threshold, num_perm=num_perm)
        n = 0
        for key, src in items:
            mh = make_minhash(src, num_perm=num_perm)
            idx.lsh.insert(key, mh)
            idx._mh_by_key[key] = mh
            n += 1
        logger.info(f"Built KernelBench LSH index from {n} sources")
        return idx

    def max_similarity(self, src: str) -> float:
        if not self._mh_by_key:
            return 0.0
        mh = make_minhash(src, num_perm=self.num_perm)
        candidates = self.lsh.query(mh)
        if not candidates:
            # No LSH bucket hit — true similarity is below `threshold`.
            return 0.0
        return max(mh.jaccard(self._mh_by_key[k]) for k in candidates)


def load_kernelbench_sources(root: str | Path) -> list[tuple[str, str]]:
    """Walk a KernelBench checkout and pull every reference source file. We
    accept both the new `levelN/N_<idx>_<name>.py` layout and a flat
    `*.py` dump."""
    rootp = Path(root)
    if not rootp.exists():
        logger.warning(f"KernelBench path missing: {rootp} — decontamination disabled")
        return []
    items: list[tuple[str, str]] = []
    for p in rootp.rglob("*.py"):
        try:
            items.append((str(p.relative_to(rootp)), p.read_text()))
        except Exception:
            continue
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernelbench", required=True, help="Path to KernelBench root")
    parser.add_argument("--input", required=True, help="JSONL of FusedProblemEntry to score")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    items = load_kernelbench_sources(args.kernelbench)
    idx = KernelBenchIndex.from_sources(items, threshold=args.threshold)

    n = 0
    with open(args.input) as fin, open(args.out, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            src = entry.get("module_source") or entry.get("fused_class_source", "")
            score = idx.max_similarity(src)
            fout.write(
                json.dumps({"problem_id": entry["problem_id"], "score": score}) + "\n"
            )
            n += 1
    logger.info(f"Scored {n} problems -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
