"""Reward functions (PROJECT_CONTEXT §7.3), selected by ``type`` in configs/reward/*.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from racer.config import from_dict


@dataclass(frozen=True)
class ProgressReward:
    """r = progress·Δs − time·dt − steer_change·|Δsteer| − offroad·frac·dt + lap_bonus·laps − crash."""

    type: str
    progress: float
    time: float
    steer_change: float
    offroad: float
    lap_bonus: float
    crash: float

    def __call__(self, ds: float, dt: float, dsteer: float, offroad_frac: float,
                 new_laps: int, terminated: bool, speed: float) -> float:
        r = (self.progress * ds - self.time * dt - self.steer_change * abs(dsteer)
             - self.offroad * offroad_frac * dt + self.lap_bonus * new_laps)
        if terminated:
            r -= self.crash
        return r


REWARDS = {"progress": ProgressReward}


def build_reward(cfg: Mapping[str, Any]):
    kind = cfg.get("type")
    if kind not in REWARDS:
        raise ValueError(f"unknown reward type {kind!r} (known: {sorted(REWARDS)})")
    return from_dict(REWARDS[kind], cfg, where=f"reward config ({kind})")
