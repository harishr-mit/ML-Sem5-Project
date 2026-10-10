"""Action layouts: what a policy's outputs mean (PROJECT_CONTEXT §7.2, D22).

A layout turns the policy's output vector (each value in [−1, 1]) into the car's controls
``(steer, gas)``: steer in [−1, 1], gas in [−1, 1] with negative = brake (the physics
interface, §5.4). The layout is part of every model's snapshot (``env.action_spec``);
an old snapshot without it means ``steer_gas``.

NEVER change what an existing layout means: saved models depend on it. A new way of
driving the car gets a new layout name, registered in ``LAYOUTS``.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

DEFAULT_ACTION_SPEC = {"layout": "steer_gas"}


class SteerGas:
    """[steer, gas]; negative gas brakes. The layout of every model before D22."""

    name = "steer_gas"
    labels = ["steer", "gas"]
    description = "2 outputs: steering, and one pedal where negative means brake."
    dim = 2

    def to_controls(self, a: np.ndarray) -> tuple[float, float]:
        return float(a[0]), float(a[1])

    def neutral(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)


class SteerThrottleBrake:
    """[steer, throttle, brake]; pedals map [−1, 1] → [0, 1], the car gets gas = throttle − brake."""

    name = "steer_throttle_brake"
    labels = ["steer", "throttle", "brake"]
    description = ("3 outputs: steering, throttle pedal and brake pedal (each pedal 0–1). "
                   "The car receives throttle − brake.")
    dim = 3

    def to_controls(self, a: np.ndarray) -> tuple[float, float]:
        throttle = 0.5 * (float(a[1]) + 1.0)
        brake = 0.5 * (float(a[2]) + 1.0)
        return float(a[0]), throttle - brake

    def neutral(self) -> np.ndarray:
        return np.array([0.0, -1.0, -1.0], dtype=np.float32)   # no steering, both pedals released


LAYOUTS = {cls.name: cls for cls in (SteerGas, SteerThrottleBrake)}


def action_spec_of(env_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The env config's action spec; snapshots from before D22 have none and mean steer_gas."""
    return dict(env_cfg.get("action_spec") or DEFAULT_ACTION_SPEC)


def build_action_layout(spec: Mapping[str, Any]):
    unknown = set(spec) - {"layout"}
    if unknown:
        raise ValueError(f"action_spec: unknown keys {sorted(unknown)}")
    name = spec.get("layout")
    if name not in LAYOUTS:
        raise ValueError(f"unknown action layout {name!r} (known: {sorted(LAYOUTS)})")
    return LAYOUTS[name]()
