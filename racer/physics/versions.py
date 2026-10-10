"""Physics versions (PROJECT_CONTEXT §5.2, D23).

A physics preset may carry ``version`` (a preset or snapshot without it is version 1).
Each version maps to a parameter dataclass and a car class, so a future change to the
physics *equations* (e.g. real gears, a tyre model) becomes version 2 with its own car,
and every model keeps driving with the physics it was trained on. Different cars with the
same equations (mass, power, grip) are just presets, not versions.
"""
from __future__ import annotations

from typing import Any, Mapping

from racer.physics.car import Car, PhysicsParams

PHYSICS: dict[int, tuple[type, type]] = {1: (PhysicsParams, Car)}


def physics_version(cfg: Mapping[str, Any]) -> int:
    return int(cfg.get("version", 1))


def build_physics(cfg: Mapping[str, Any]):
    """(params, car class) for a physics config of any supported version."""
    version = physics_version(cfg)
    if version not in PHYSICS:
        raise ValueError(f"unknown physics version {version} (known: {sorted(PHYSICS)})")
    params_cls, car_cls = PHYSICS[version]
    params = params_cls.from_config({k: v for k, v in cfg.items() if k != "version"})
    return params, car_cls
