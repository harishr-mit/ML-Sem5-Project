"""Learning curves from a run's TensorBoard event files (T68).

train.py logs to ``runs/<run_id>_<tag>_<n>/`` (trial runs: ``outputs/scratch/tb/``). Each
event file is parsed once and re-read only when it changes; curves are downsampled for
plotting. Uses tensorboard's ``EventAccumulator`` (already a dependency).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from racer.sandbox.data import STORE, Store, _cached

# The curves the Explorer shows by default, with a plain-language title.
MAIN_CURVES = {
    "rollout/ep_rew_mean": "Episode reward (mean)",
    "race/lap_time": "Lap time (s)",
    "race/progress_m": "Progress per episode (m)",
    "race/laps": "Laps per episode",
    "rollout/ep_len_mean": "Episode length (decisions)",
}
END_PREFIX = "end/"          # termination reasons: share of episodes ending each way
TRAIN_PREFIX = "train/"      # losses and optimiser statistics


def _load_events(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        ev = ea.Scalars(tag)
        out[tag] = (np.array([e.step for e in ev], dtype=np.int64), np.array([e.value for e in ev], dtype=float))
    return out


def run_scalars(run_id: str, store: Store = STORE) -> dict[str, pd.Series]:
    """Every scalar of a run (all its event files merged), as step-indexed Series."""
    merged: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for d in store.tensorboard_dirs(run_id):
        for ev in sorted(d.glob("events.out.tfevents.*")):
            for tag, xy in _cached(ev, "events", _load_events).items():
                merged.setdefault(tag, []).append(xy)
    out = {}
    for tag, parts in merged.items():
        steps = np.concatenate([p[0] for p in parts])
        vals = np.concatenate([p[1] for p in parts])
        s = pd.Series(vals, index=steps, name=tag).sort_index()
        out[tag] = s[~s.index.duplicated(keep="last")]
    return out


def downsample(s: pd.Series, max_points: int = 500) -> pd.Series:
    """At most ``max_points`` points: the mean of equal-width step bins."""
    if len(s) <= max_points:
        return s
    bins = np.linspace(s.index.min(), s.index.max(), max_points + 1)
    idx = np.clip(np.searchsorted(bins, s.index, side="right") - 1, 0, max_points - 1)
    g = s.groupby(idx)
    return pd.Series(g.mean().to_numpy(), index=g.apply(lambda x: int(x.index[len(x) // 2])).to_numpy(), name=s.name)


def smooth(s: pd.Series, weight: float) -> pd.Series:
    """TensorBoard-style exponential smoothing (0 = raw, 0.99 = very smooth)."""
    if weight <= 0 or s.empty:
        return s
    return s.ewm(alpha=1 - weight, adjust=False).mean()


def table(run_ids: Iterable[str], tag: str, labels: Iterable[str] | None = None, weight: float = 0.0,
          max_points: int = 500, store: Store = STORE) -> pd.DataFrame:
    """One column per run (step index), for overlaying a curve of several runs."""
    cols = {}
    for run_id, label in zip(run_ids, labels or run_ids):
        s = run_scalars(run_id, store).get(tag)
        if s is not None and not s.empty:
            cols[label] = downsample(smooth(s, weight), max_points)
    return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()
