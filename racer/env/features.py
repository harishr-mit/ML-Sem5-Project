"""Observation features for ``obs_spec`` v2 (PROJECT_CONTEXT §7.1, D21).

A v2 observation is an ordered list of features, each with its own parameters::

    features:
      - {feature: rays, rays: [{angle_deg: -20, range_m: 120}, {angle_deg: 0, range_m: 200}, ...]}
      - {feature: heading_sincos}
      - {feature: v_long, scale: 70.0}
      - {feature: prev_action}

Each feature class declares its parameters (``params``: name → default, ``REQUIRED`` if
none), its size, bounds, labels (for the brain panel) and a short description (for the AI
Sandbox). NEVER change what a registered feature computes once a model uses it: a new
behaviour gets a new feature name. Adding a feature = a new class in ``FEATURES``.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from racer.env.sensors import cast_rays

REQUIRED = object()


class ObsState:
    """What the env knows at observation time (one reusable object per observer)."""

    __slots__ = ("track", "car", "tracker", "heading_error", "onroad_frac", "prev_action")


class Feature:
    name = ""
    description = ""
    params: dict[str, Any] = {}

    def __init__(self, cfg: Mapping[str, Any], ctx: Mapping[str, Any]) -> None:
        unknown = set(cfg) - {"feature"} - set(self.params)
        if unknown:
            raise ValueError(f"feature {self.name}: unknown parameters {sorted(unknown)}")
        self.cfg: dict[str, Any] = {}
        for k, default in self.params.items():
            if k not in cfg and default is REQUIRED:
                raise ValueError(f"feature {self.name}: missing parameter {k!r}")
            self.cfg[k] = cfg.get(k, default)
        self.setup(ctx)

    def setup(self, ctx: Mapping[str, Any]) -> None:
        self.dim = 1
        self.labels = [self.name]
        self.low = np.full(self.dim, -1.0)

    def write(self, st: ObsState, out: np.ndarray, k: int) -> None:
        raise NotImplementedError


class Rays(Feature):
    name = "rays"
    description = ("Distance from the car centre to the road edge along each ray (angle counter-clockwise "
                   "from the heading, −90 = right), divided by the ray's range and capped at 1. "
                   "Shallow angles see bends early; wide angles see the road edges beside the car.")
    params = {"rays": REQUIRED}

    def setup(self, ctx):
        rays = self.cfg["rays"]
        if not rays:
            raise ValueError("feature rays: needs at least one ray")
        self.angles = np.radians([float(r["angle_deg"]) for r in rays])
        self.ranges = np.array([float(r["range_m"]) for r in rays])
        self.raycast_params = ctx["raycast_params"]
        self.dist = np.empty(len(rays))
        self.dim = len(rays)
        self.labels = [f"ray{float(r['angle_deg']):+.0f}" for r in rays]
        self.low = np.zeros(self.dim)

    def write(self, st, out, k):
        f = st.track.field
        c = st.car
        cast_rays(f.dist, f.idx, f.geom, c.x, c.y, c.theta, self.angles, self.ranges, self.raycast_params, self.dist)
        out[k:k + self.dim] = np.minimum(self.dist / self.ranges, 1.0)


class HeadingSinCos(Feature):
    name = "heading_sincos"
    description = "sin and cos of the angle between the car and the track direction (0 = pointing along the track)."

    def setup(self, ctx):
        self.dim, self.labels, self.low = 2, ["sin_head", "cos_head"], np.full(2, -1.0)

    def write(self, st, out, k):
        out[k] = math.sin(st.heading_error)
        out[k + 1] = math.cos(st.heading_error)


class _Scaled(Feature):
    """One car value divided by ``scale`` and clipped to [−1, 1]."""

    attr = ""
    params = {"scale": REQUIRED}

    def setup(self, ctx):
        super().setup(ctx)
        self.inv = 1.0 / float(self.cfg["scale"])

    def write(self, st, out, k):
        out[k] = min(max(getattr(st.car, self.attr) * self.inv, -1.0), 1.0)


class VLong(_Scaled):
    name, attr = "v_long", "v_long"
    description = "Forward speed (m/s) / scale. How fast the car is going."


class VLat(_Scaled):
    name, attr = "v_lat", "v_lat"
    description = "Sideways speed (m/s) / scale. Non-zero when the car slides: lets the agent feel a skid."


class YawRate(_Scaled):
    name, attr = "yaw_rate", "yaw_rate"
    description = "Turning rate (rad/s) / scale. How fast the car is rotating."


class SteerAngle(Feature):
    name = "steer_angle"
    description = ("Current front-wheel angle / scale_deg. The wheels move at a limited rate, so this tells the "
                   "agent where the steering actually is, not just what it asked for.")
    params = {"scale_deg": 30.0}

    def setup(self, ctx):
        super().setup(ctx)
        self.inv = 1.0 / math.radians(float(self.cfg["scale_deg"]))

    def write(self, st, out, k):
        out[k] = min(max(st.car.steer * self.inv, -1.0), 1.0)


class OnroadFrac(Feature):
    name = "onroad_frac"
    description = "Fraction of the 4 wheels on the road (kerbs count as road): 1 = on track, 0 = on the grass."

    def setup(self, ctx):
        self.dim, self.labels, self.low = 1, ["onroad_frac"], np.zeros(1)

    def write(self, st, out, k):
        out[k] = st.onroad_frac


class PrevAction(Feature):
    name = "prev_action"
    description = ("The policy's previous outputs (one value per output of the action layout). "
                   "Helps smooth control: the agent knows what it just did.")

    def setup(self, ctx):
        layout = ctx["action_layout"]
        self.dim = layout.dim
        self.labels = [f"prev_{label}" for label in layout.labels]
        self.low = np.full(self.dim, -1.0)

    def write(self, st, out, k):
        out[k:k + self.dim] = st.prev_action


class CurvatureAhead(Feature):
    name = "curvature_ahead"
    description = ("Track curvature at the given distances ahead along the centreline, × radius_m and clipped "
                   "to [−1, 1] (positive = bends left; 1 = a corner of radius radius_m or tighter). "
                   "This is map knowledge: the car no longer relies on its rays alone (E6).")
    params = {"distances_m": [25.0, 50.0, 100.0], "radius_m": 15.0}

    def setup(self, ctx):
        self.distances = [float(d) for d in self.cfg["distances_m"]]
        if not self.distances:
            raise ValueError("feature curvature_ahead: needs at least one distance")
        self.radius = float(self.cfg["radius_m"])
        self.dim = len(self.distances)
        self.labels = [f"curv+{d:.0f}m" for d in self.distances]
        self.low = np.full(self.dim, -1.0)

    def write(self, st, out, k):
        cl = st.track.centreline
        kappa = cl.curvature_smooth
        s = st.tracker.s
        for j, d in enumerate(self.distances):
            i = int(round((s + d) / cl.ds)) % cl.n
            out[k + j] = min(max(kappa[i] * self.radius, -1.0), 1.0)


FEATURES: dict[str, type[Feature]] = {cls.name: cls for cls in (
    Rays, HeadingSinCos, VLong, VLat, YawRate, SteerAngle, OnroadFrac, PrevAction, CurvatureAhead)}


def build_feature(cfg: Mapping[str, Any], ctx: Mapping[str, Any]) -> Feature:
    name = cfg.get("feature")
    if name not in FEATURES:
        raise ValueError(f"unknown observation feature {name!r} (known: {sorted(FEATURES)})")
    return FEATURES[name](cfg, ctx)


def describe_features() -> list[dict[str, Any]]:
    """Every registered feature with its description and parameters (for docs and the sandbox)."""
    return [{"feature": n, "description": c.description,
             "params": {k: (None if v is REQUIRED else v) for k, v in c.params.items()}}
            for n, c in FEATURES.items()]
