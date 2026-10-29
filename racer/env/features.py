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

from racer.env.sensors import cast_rays, cast_rays_multi

REQUIRED = object()


class ObsState:
    """What the env knows at observation time (one reusable object per observer)."""

    __slots__ = ("track", "car", "tracker", "heading_error", "onroad_frac", "prev_action", "wheels")


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

    def reset(self) -> None:
        """Called at the start of every episode (only stateful features need it)."""


class Rays(Feature):
    name = "rays"
    description = ("The car's eyes: how far away the road edge is along each ray, as a share of the ray's range "
                   "(1 = nothing within reach). Angles are measured from straight ahead, + to the left. "
                   "Shallow rays see bends early; wide rays see the road beside the car.")
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
        self.labels = [f"ray{float(r['angle_deg']):+g}" for r in rays]
        self.low = np.zeros(self.dim)

    def write(self, st, out, k):
        f = st.track.field
        c = st.car
        cast_rays(f.dist, f.idx, f.geom, c.x, c.y, c.theta, self.angles, self.ranges, self.raycast_params, self.dist)
        out[k:k + self.dim] = np.minimum(self.dist / self.ranges, 1.0)


class Rays2(Rays):
    name = "rays2"
    description = ("Rays that see through the road edge (D33). Each ray reports where it first meets the road edge "
                   "and where it meets an edge again (an exit, then the way back onto the road across the infield: "
                   "the shape of the bend ahead), each as a share of the ray's range (1 = none within reach). "
                   "'penetrations' is how many edges beyond the first a ray passes through.")
    params = {"rays": REQUIRED, "penetrations": 1}

    def setup(self, ctx):
        super().setup(ctx)
        self.n_cross = int(self.cfg["penetrations"]) + 1
        if self.n_cross < 2:
            raise ValueError("feature rays2: penetrations must be at least 1 (use 'rays' for a single crossing)")
        n = len(self.angles)
        self.cross = np.empty((n, self.n_cross))
        self.dist = self.cross[:, 0]                      # the first crossing: what renderers draw as the ray
        self.dim = n * self.n_cross
        self.labels = [f"{base}{'' if j == 0 else f'#{j + 1}'}"
                       for base in (f"ray{math.degrees(a):+g}" for a in self.angles) for j in range(self.n_cross)]
        self.low = np.zeros(self.dim)

    def write(self, st, out, k):
        f = st.track.field
        c = st.car
        cast_rays_multi(f.dist, f.idx, f.geom, c.x, c.y, c.theta, self.angles, self.ranges, self.raycast_params,
                        self.n_cross, self.cross)
        out[k:k + self.dim] = np.minimum(self.cross / self.ranges[:, None], 1.0).ravel()


class HeadingSinCos(Feature):
    name = "heading_sincos"
    description = ("Which way the car points compared with the road (as sin and cos, two numbers). "
                   "Straight along the road is 0° — it tells the car it is turned or sliding sideways.")

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
    description = "How fast the car is going forwards (divided by the scale so it is about −1 to 1). Needed to brake in time."


class VLat(_Scaled):
    name, attr = "v_lat", "v_lat"
    description = "How fast the car is sliding sideways. Zero when gripping; non-zero in a skid, so the car can feel one."


class YawRate(_Scaled):
    name, attr = "yaw_rate", "yaw_rate"
    description = "How fast the car is spinning around (turning rate). Shows a spin starting before the rays notice."


class SteerAngle(Feature):
    name = "steer_angle"
    description = ("Where the front wheels actually point right now. The wheels turn slowly, so this can differ "
                   "from what the car asked for.")
    params = {"scale_deg": 30.0}

    def setup(self, ctx):
        super().setup(ctx)
        self.inv = 1.0 / math.radians(float(self.cfg["scale_deg"]))

    def write(self, st, out, k):
        out[k] = min(max(st.car.steer * self.inv, -1.0), 1.0)


class OnroadFrac(Feature):
    name = "onroad_frac"
    description = "How many of the 4 wheels are on the road (kerbs count as road): 1 = all on track, 0 = all on the grass."

    def setup(self, ctx):
        self.dim, self.labels, self.low = 1, ["onroad_frac"], np.zeros(1)

    def write(self, st, out, k):
        out[k] = st.onroad_frac


class PrevAction(Feature):
    name = "prev_action"
    description = ("What the car did on the last decision (steering and pedals). Helps it drive smoothly, because "
                   "it knows what it just did.")

    def setup(self, ctx):
        layout = ctx["action_layout"]
        self.dim = layout.dim
        self.labels = [f"prev_{label}" for label in layout.labels]
        self.low = np.full(self.dim, -1.0)

    def write(self, st, out, k):
        out[k:k + self.dim] = st.prev_action


class CurvatureAhead(Feature):
    name = "curvature_ahead"
    description = ("How sharply the track bends at set distances ahead (positive = left, 1 = a very tight corner). "
                   "This is map knowledge: the car no longer relies on its rays alone (experiment E6).")
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


class _Clipped(Feature):
    """Shared set-up: ``dim`` values in [low, 1] (labels and low declared by the subclass)."""

    def setup(self, ctx):
        self.dim = len(self.labels_)
        self.labels = list(self.labels_)
        self.low = np.full(self.dim, self.low_)


class AccelG(_Clipped):
    name = "accel_g"
    description = ("How hard the car is accelerating or braking (forwards) and cornering (sideways), divided by the "
                   "scale in m/s². Shows how close the car is to the grip limit.")
    params = {"scale_ms2": 12.0}
    labels_, low_ = ("accel_long", "accel_lat"), -1.0

    def setup(self, ctx):
        super().setup(ctx)
        self.inv = 1.0 / float(self.cfg["scale_ms2"])

    def write(self, st, out, k):
        out[k] = min(max(st.car.a_long * self.inv, -1.0), 1.0)
        out[k + 1] = min(max(st.car.a_lat * self.inv, -1.0), 1.0)


class GripUse(_Clipped):
    name = "grip_use"
    description = ("How much of the tyres' grip is being used: 0 = none, 0.5 = right at the limit, "
                   "above that = sliding (capped at 2× the limit).")
    params = {"cap": 2.0}
    labels_, low_ = ("grip_use",), 0.0

    def write(self, st, out, k):
        out[k] = min(max(st.car.grip_use / float(self.cfg["cap"]), 0.0), 1.0)


class WheelSurface(_Clipped):
    name = "wheel_surface"
    description = ("What each of the 4 wheels is on: 1 = road, 0.5 = kerb, 0 = grass. "
                   "More detail than the on-road fraction: which wheels are off.")
    labels_, low_ = ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"), 0.0

    def write(self, st, out, k):
        for i in range(4):
            sid = int(st.wheels[i])
            out[k + i] = 1.0 if sid == 1 else 0.5 if sid == 2 else 0.0


class RoadPosition(_Clipped):
    name = "road_position"
    description = ("Where the car is across the road: its offset from the centre line (−1 right edge … +1 left edge) "
                   "and the gap to the left and right road edge in metres ÷ scale. Lets it use the full width.")
    params = {"scale_m": 10.0}
    labels_, low_ = ("road_offset", "gap_left", "gap_right"), -1.0

    def write(self, st, out, k):
        hw, lat = st.track.half_width, st.tracker.lateral
        inv = 1.0 / float(self.cfg["scale_m"])
        out[k] = min(max(lat / hw, -1.0), 1.0)
        out[k + 1] = min(max((hw - lat) * inv, -1.0), 1.0)
        out[k + 2] = min(max((hw + lat) * inv, -1.0), 1.0)


class History(Feature):
    """Memory (D30): the values of the features listed before it, as they were 1, 2, … ``steps`` decisions ago."""

    name = "history"
    description = ("Memory: the car's earlier readings (as they were 1, 2, … n decisions ago, 20 per second), so it can "
                   "feel trends such as speeding up or a bend coming closer. Put it last. At the start of an "
                   "episode the past is the current state.")
    params = {"steps": REQUIRED, "stride": 1, "of": None}      # of: feature names to remember (None = all but rays)

    def setup(self, ctx):
        self.n, self.stride = int(self.cfg["steps"]), int(self.cfg["stride"])
        if self.n < 1 or self.stride < 1:
            raise ValueError("feature history: steps and stride must be at least 1")
        of = self.cfg["of"]
        earlier = [(f, off) for f, off in ctx.get("earlier", []) if (f.name != "rays" if of is None else f.name in of)]
        if of is not None and {f.name for f, _ in earlier} != set(of):
            raise ValueError(f"feature history: 'of' names features that don't come before it: {sorted(of)}")
        if not earlier:
            raise ValueError("feature history: nothing before it to remember")
        self.idx = np.concatenate([np.arange(off, off + f.dim) for f, off in earlier])
        names = [lab for f, _ in earlier for lab in f.labels]
        low = np.concatenate([f.low for f, _ in earlier])
        self.m = len(self.idx)
        self.dim = self.m * self.n
        self.labels = [f"{lab} (-{j * self.stride})" for j in range(1, self.n + 1) for lab in names]
        self.low = np.tile(low, self.n)
        self.buf: list[np.ndarray] = []

    def reset(self) -> None:
        self.buf = []

    def write(self, st, out, k):
        cur = out[self.idx].copy()
        if not self.buf:                                 # first decision of an episode: no past yet
            self.buf = [cur] * (self.n * self.stride)
        for j in range(self.n):
            out[k + j * self.m:k + (j + 1) * self.m] = self.buf[(j + 1) * self.stride - 1]
        self.buf = [cur] + self.buf[:self.n * self.stride - 1]


FEATURES: dict[str, type[Feature]] = {cls.name: cls for cls in (
    Rays, Rays2, HeadingSinCos, VLong, VLat, YawRate, SteerAngle, OnroadFrac, PrevAction, CurvatureAhead, AccelG, GripUse, WheelSurface, RoadPosition, History)}


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
