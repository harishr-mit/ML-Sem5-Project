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
    description = "2 outputs: steering, and one pedal (above 0 = gas, below 0 = brake). What every model so far uses."
    dim = 2

    def to_controls(self, a: np.ndarray) -> tuple[float, float]:
        return float(a[0]), float(a[1])

    def neutral(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)


class SteerThrottleBrake:
    """[steer, throttle, brake]; pedals map [−1, 1] → [0, 1], the car gets gas = throttle − brake."""

    name = "steer_throttle_brake"
    labels = ["steer", "throttle", "brake"]
    description = ("3 outputs: steering, a gas pedal and a separate brake pedal. "
                   "The car receives gas minus brake, so pressing both cancels out.")
    dim = 3

    def to_controls(self, a: np.ndarray) -> tuple[float, float]:
        throttle = 0.5 * (float(a[1]) + 1.0)
        brake = 0.5 * (float(a[2]) + 1.0)
        return float(a[0]), throttle - brake

    def neutral(self) -> np.ndarray:
        return np.array([0.0, -1.0, -1.0], dtype=np.float32)   # no steering, both pedals released


class SteerRate(SteerGas):
    """[steer change, gas] (D30): the first output turns the wheel *from where it is* (each decision by up to
    ``STEP`` of full lock) instead of naming an angle: smooth by design. The layout remembers the wheel
    target; ``neutral()`` (called at every reset) centres it."""

    name = "steer_rate"
    labels = ["steer_change", "gas"]
    description = ("2 outputs: how much to turn the wheel from where it is (not an angle), and one pedal "
                   "(above 0 = gas, below 0 = brake). Smooth steering by design.")
    STEP = 0.3          # shortcut: fixed, not in a config (a layout has no config); change = a new layout

    def __init__(self) -> None:
        self.target = 0.0

    def to_controls(self, a: np.ndarray) -> tuple[float, float]:
        self.target = min(max(self.target + self.STEP * float(a[0]), -1.0), 1.0)
        return self.target, float(a[1])

    def neutral(self) -> np.ndarray:
        self.target = 0.0
        return super().neutral()


class SteerThrottleBrakeAbs(SteerThrottleBrake):
    """[steer, throttle, brake] like ``steer_throttle_brake``, but the brake is capped just inside the grip
    limit (ABS, as for the keyboard player): the wheels never lock, whatever the network asks. The cap is
    applied by the env, which knows the surfaces (``abs_margin``)."""

    name = "steer_throttle_brake_abs"
    description = ("3 outputs: steering, a gas pedal and a separate brake pedal, with ABS: the brake is limited to "
                   "just under the grip limit so the wheels never lock.")
    abs_margin = 0.97


LAYOUTS = {cls.name: cls for cls in (SteerGas, SteerThrottleBrake, SteerRate, SteerThrottleBrakeAbs)}


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
