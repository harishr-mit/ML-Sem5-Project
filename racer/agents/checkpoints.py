"""Which checkpoints of one run race together in the Evolution Race (T126).

Pure functions over registry rows (``models.csv`` as typed dicts), so the picker, the command line
and the AI Sandbox choose the same cars.
"""
from __future__ import annotations

from typing import Any, Sequence

STRATEGIES = {
    "every": "every N-th checkpoint",
    "spaced": "N checkpoints spread evenly over the run (first and last included)",
    "top": "the N best by registry score (shown in training order)",
}


def run_checkpoints(rows: Sequence[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    """The run's checkpoints that have a model folder, in training order."""
    return sorted((r for r in rows if r.get("run_id") == run_id and r.get("path")), key=lambda r: r.get("steps") or 0)


def _score(r: dict[str, Any]) -> float:
    for key in ("score", "overall_completion"):
        v = r.get(key)
        if isinstance(v, (int, float)) and v == v:
            return float(v)
    return float("-inf")


def choose(checkpoints: Sequence[dict[str, Any]], strategy: str, n: int) -> list[dict[str, Any]]:
    """Pick checkpoints (rows) with ``strategy``; ``checkpoints`` must be in training order."""
    n = max(int(n), 1)
    cps = list(checkpoints)
    if strategy == "every":
        return cps[::n]
    if strategy == "spaced":
        if n >= len(cps):
            return cps
        if n == 1:
            return cps[-1:]
        idx = sorted({round(i * (len(cps) - 1) / (n - 1)) for i in range(n)})
        return [cps[i] for i in idx]
    if strategy == "top":
        best = sorted(range(len(cps)), key=lambda i: (-_score(cps[i]), i))[:n]
        return [cps[i] for i in sorted(best)]
    raise ValueError(f"unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}")
