"""Hand-coded baseline driver (PROJECT_CONTEXT §11, E8).

Follows the centreline with pure pursuit and plans speed from the curvature ahead. It
reads the track directly (privileged map knowledge), so it doubles as an env sanity check:
if it can't lap a track, the env is broken, not PPO.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from racer.config import from_dict


@dataclass(frozen=True)
class BaselineParams:
    algorithm: str
    lookahead_min_m: float
    lookahead_time_s: float
    lat_acc_target: float
    brake_decel: float
    plan_horizon_m: float
    v_max: float
    speed_gain: float


class BaselineController:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.p = from_dict(BaselineParams, cfg, where="baseline config")
        self._track = None

    def _prepare(self, track) -> None:
        """Per-track speed limit from curvature (cached)."""
        self._track = track
        k = np.abs(track.centreline.curvature_smooth)
        self._v_corner = np.minimum(np.sqrt(self.p.lat_acc_target / np.maximum(k, 1e-6)), self.p.v_max)
        self._horizon = int(self.p.plan_horizon_m / track.ds)

    def act(self, env) -> np.ndarray:
        """Action for the env's current state (uses env.track, env.tracker, env.car)."""
        track, tracker, car = env.track, env.tracker, env.car
        if track is not self._track:
            self._prepare(track)
        p = self.p
        cl = track.centreline
        n, ds = cl.n, cl.ds
        i = tracker.idx
        speed = car.speed

        # Pure pursuit towards a centreline point ahead.
        look = p.lookahead_min_m + p.lookahead_time_s * speed
        j = (i + int(round(look / ds))) % n
        tx, ty = cl.points[j]
        alpha = math.atan2(ty - car.y, tx - car.x) - car.theta
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        dist = max(math.hypot(tx - car.x, ty - car.y), 1e-3)
        delta = math.atan2(2.0 * env.params.wheelbase_m * math.sin(alpha), dist)
        steer = max(min(delta / env.params.max_steer(speed), 1.0), -1.0)

        # Speed plan: slowest of (corner speed at d ahead, reachable by braking over d).
        idx = (i + np.arange(self._horizon)) % n
        d = np.arange(self._horizon) * ds
        v_target = float(np.min(np.sqrt(self._v_corner[idx] ** 2 + 2.0 * p.brake_decel * d)))
        gas = max(min(p.speed_gain * (v_target - speed), 1.0), -1.0)
        return np.array([steer, gas], dtype=np.float32)
