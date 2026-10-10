"""Versioned observation specs (PROJECT_CONTEXT §7.1, §9).

NEVER change what an existing version means: saved models depend on it. A new feature
gets a new version with its own observer class, registered in ``OBSERVERS``.

v1 (frozen, D2): rays from the car centre to the first road-edge crossing (/range, clipped
to [0, 1]), then the extras in the order the spec lists them:

    heading_sincos  sin, cos of (car heading − track tangent)            2
    v_long          forward speed / scales.v_long, clipped to [−1, 1]    1
    v_lat           sideways speed / scales.v_lat, clipped to [−1, 1]    1
    yaw_rate        yaw rate / scales.yaw_rate, clipped to [−1, 1]       1
    onroad_frac     fraction of the 4 wheels on road or kerb             1
    prev_action     previous steer and gas actions                       2

v2 (D21): an ordered list of named features (``racer.env.features``), so inputs can be
added, removed or reordered by editing the list; ``prev_action`` follows the model's
action layout (``racer.env.actions``).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from racer.env.sensors import cast_rays, pack_raycast_params

V1_EXTRAS = {"heading_sincos": 2, "v_long": 1, "v_lat": 1, "yaw_rate": 1, "onroad_frac": 1, "prev_action": 2}


class ObserverV1:
    version = 1

    def __init__(self, spec: Mapping[str, Any], raycast_cfg: Mapping[str, Any]) -> None:
        self.raycast_params = pack_raycast_params(raycast_cfg)
        unknown = set(spec) - {"version", "name", "rays", "extras", "scales"}
        if unknown:
            raise ValueError(f"obs_spec v1: unknown keys {sorted(unknown)}")
        self.name = spec["name"]
        self.angles = np.radians([float(r["angle_deg"]) for r in spec["rays"]])
        self.ranges = np.array([float(r["range_m"]) for r in spec["rays"]])
        self.extras = list(spec["extras"])
        bad = [e for e in self.extras if e not in V1_EXTRAS]
        if bad or len(set(self.extras)) != len(self.extras):
            raise ValueError(f"obs_spec v1: bad or repeated extras {self.extras}")
        sc = spec["scales"]
        self.s_vlong = float(sc["v_long"])
        self.s_vlat = float(sc["v_lat"])
        self.s_yaw = float(sc["yaw_rate"])
        self.n_rays = len(self.angles)
        self.dim = self.n_rays + sum(V1_EXTRAS[e] for e in self.extras)
        self.ray_dist = np.empty(self.n_rays)
        self.labels = [f"ray{float(r['angle_deg']):+.0f}" for r in spec["rays"]]
        for e in self.extras:
            self.labels += {"heading_sincos": ["sin_head", "cos_head"], "prev_action": ["prev_steer", "prev_gas"]}.get(e, [e])

    def observe(self, track, car, heading_error: float, onroad_frac: float,
                prev_action: np.ndarray, out: np.ndarray, tracker=None) -> np.ndarray:
        f = track.field
        cast_rays(f.dist, f.idx, f.geom, car.x, car.y, car.theta, self.angles, self.ranges,
                  self.raycast_params, self.ray_dist)
        n = self.n_rays
        out[:n] = np.minimum(self.ray_dist / self.ranges, 1.0)
        k = n
        for e in self.extras:
            if e == "heading_sincos":
                out[k] = math.sin(heading_error)
                out[k + 1] = math.cos(heading_error)
                k += 2
            elif e == "v_long":
                out[k] = min(max(car.v_long / self.s_vlong, -1.0), 1.0)
                k += 1
            elif e == "v_lat":
                out[k] = min(max(car.v_lat / self.s_vlat, -1.0), 1.0)
                k += 1
            elif e == "yaw_rate":
                out[k] = min(max(car.yaw_rate / self.s_yaw, -1.0), 1.0)
                k += 1
            elif e == "onroad_frac":
                out[k] = onroad_frac
                k += 1
            else:  # prev_action
                out[k] = prev_action[0]
                out[k + 1] = prev_action[1]
                k += 2
        return out

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Box bounds: rays and onroad_frac in [0, 1], everything else in [−1, 1]."""
        low = -np.ones(self.dim, dtype=np.float32)
        low[: self.n_rays] = 0.0
        k = self.n_rays
        for e in self.extras:
            if e == "onroad_frac":
                low[k] = 0.0
            k += V1_EXTRAS[e]
        return low, np.ones(self.dim, dtype=np.float32)


    def describe(self) -> list[dict[str, Any]]:
        """Inputs as feature-like entries (same shape as v2) for the brain panel and the sandbox."""
        rays = {"feature": "rays", "dims": self.n_rays, "labels": self.labels[: self.n_rays],
                "params": {"rays": [{"angle_deg": round(math.degrees(a), 3), "range_m": float(r)}
                                    for a, r in zip(self.angles, self.ranges)]}}
        out, k = [rays], self.n_rays
        for e in self.extras:
            n = V1_EXTRAS[e]
            out.append({"feature": e, "dims": n, "labels": self.labels[k:k + n], "params": {}})
            k += n
        return out


class ObserverV2:
    """v2 (D21): the concatenation of named features from ``racer.env.features``, in list order."""

    version = 2

    def __init__(self, spec: Mapping[str, Any], raycast_cfg: Mapping[str, Any], action_layout) -> None:
        from racer.env.features import ObsState, build_feature

        unknown = set(spec) - {"version", "name", "features"}
        if unknown:
            raise ValueError(f"obs_spec v2: unknown keys {sorted(unknown)}")
        self.name = spec["name"]
        self.raycast_params = pack_raycast_params(raycast_cfg)
        ctx = {"raycast_params": self.raycast_params, "action_layout": action_layout}
        self.features = [build_feature(f, ctx) for f in spec["features"]]
        if not self.features:
            raise ValueError("obs_spec v2: needs at least one feature")
        rays = [f for f in self.features if f.name == "rays"]
        if len(rays) > 1:
            raise ValueError("obs_spec v2: at most one 'rays' feature (list every ray in it)")
        # Ray geometry for renderers (same attributes as v1); empty when the spec has no rays.
        self.rays = rays[0] if rays else None
        self.angles = self.rays.angles if rays else np.zeros(0)
        self.ranges = self.rays.ranges if rays else np.ones(0)
        self.n_rays = len(self.angles)
        self.ray_dist = self.rays.dist if rays else np.zeros(0)
        self.offsets = np.cumsum([0] + [f.dim for f in self.features])
        self.dim = int(self.offsets[-1])
        self.labels = [label for f in self.features for label in f.labels]
        self._st = ObsState()

    def observe(self, track, car, heading_error: float, onroad_frac: float,
                prev_action: np.ndarray, out: np.ndarray, tracker=None) -> np.ndarray:
        st = self._st
        st.track, st.car, st.tracker = track, car, tracker
        st.heading_error, st.onroad_frac, st.prev_action = heading_error, onroad_frac, prev_action
        for f, k in zip(self.features, self.offsets):
            f.write(st, out, int(k))
        return out

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.concatenate([f.low for f in self.features]).astype(np.float32)
        return low, np.ones(self.dim, dtype=np.float32)

    def describe(self) -> list[dict[str, Any]]:
        return [{"feature": f.name, "dims": f.dim, "labels": list(f.labels), "params": dict(f.cfg)}
                for f in self.features]


OBSERVERS = {1: ObserverV1, 2: ObserverV2}


def build_observer(spec: Mapping[str, Any], raycast_cfg: Mapping[str, Any], action_layout=None):
    """Observer for a spec; ``raycast_cfg`` tunes ray marching only (not what a ray means).

    ``action_layout`` (``racer.env.actions``) sizes ``prev_action``; v1 only supports
    ``steer_gas`` (the default when it is None).
    """
    from racer.env.actions import SteerGas

    version = spec.get("version")
    if version not in OBSERVERS:
        raise ValueError(f"unknown obs_spec version {version!r} (known: {sorted(OBSERVERS)})")
    layout = action_layout or SteerGas()
    if version == 1:
        if layout.name != "steer_gas":
            raise ValueError(f"obs_spec v1 only supports the steer_gas action layout (got {layout.name}); use v2")
        return ObserverV1(spec, raycast_cfg)
    return OBSERVERS[version](spec, raycast_cfg, layout)
