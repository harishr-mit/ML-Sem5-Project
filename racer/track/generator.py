"""Random track generator (PROJECT_CONTEXT §6).

Recipe: n points around a circle with random radius and angle jitter → spline → reject
invalid tracks (self-intersection / sections too close, tight corners, length) → random
driving direction and mirroring.

Seeds are split into *streams* with ``SeedSequence(seed, spawn_key=(stream,))``. Training
uses stream 0 and the held-out set uses its own stream (``configs/track/default.yaml``),
so training can never generate a held-out track.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from racer.config import load_preset
from racer.track.io import TrackSpec
from racer.track.track import Track

TRAIN_STREAM = 0


def stream_rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream,)))


def random_spec(rng: np.random.Generator, gen: Mapping[str, Any], name: str) -> TrackSpec:
    """One candidate track (not validated)."""
    n_lo, n_hi = gen["n_points"]
    n = int(rng.integers(n_lo, n_hi + 1))
    base = rng.uniform(*gen["radius_m"])
    spacing = 2.0 * math.pi / n
    angles = np.arange(n) * spacing + rng.uniform(-1, 1, n) * gen["angle_jitter"] * spacing
    radii = base * rng.uniform(*gen["radius_jitter"], n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    if rng.random() < 0.5:                       # mirroring
        pts[:, 0] = -pts[:, 0]
    pts = np.roll(pts, -int(rng.integers(n)), axis=0)   # random start/finish point
    direction = "ccw" if rng.random() < 0.5 else "cw"
    return TrackSpec(name, pts.round(3).tolist(), gen["width_m"], gen["runoff_m"], gen["kerb_m"], direction)


def generate(seed: int, stream: int = TRAIN_STREAM, name: str | None = None,
             track_cfg: Mapping[str, Any] | None = None, build_field: bool = True) -> tuple[Track, int]:
    """First valid track from this seed's stream. Returns (track, number of attempts)."""
    cfg = track_cfg if track_cfg is not None else load_preset("track", "default")
    gen = cfg["generator"]
    rng = stream_rng(seed, stream)
    name = name or f"gen_s{stream}_{seed}"
    for attempt in range(1, gen["max_attempts"] + 1):
        spec = random_spec(rng, gen, name)
        track = Track(spec, cfg, build_field=False)
        if not track.validate():
            spec.meta = {"generator": {"seed": seed, "stream": stream, "attempt": attempt}}
            if build_field:
                track.build_field()
            return track, attempt
    raise RuntimeError(f"no valid track after {gen['max_attempts']} attempts (seed {seed}, stream {stream})")
