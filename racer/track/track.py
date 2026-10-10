"""A built track: centreline geometry, kerbs, field and validation (PROJECT_CONTEXT §6)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from racer.config import load_preset
from racer.track import field as F
from racer.track.io import TrackSpec, load_track_spec, resolve_track_path
from racer.track.spline import Centreline, build_centreline, signed_area


def _kerb_mask(curv_smooth: np.ndarray, ds: float, cfg: Mapping[str, Any]) -> np.ndarray:
    """Kerbs where |curvature| > threshold, extended before/after, short runs dropped."""
    n = len(curv_smooth)
    core = np.abs(curv_smooth) > cfg["curvature_min"]
    ext = int(round(cfg["extend_m"] / ds))
    mask = core.copy()
    for k in range(1, ext + 1):
        mask |= np.roll(core, k) | np.roll(core, -k)
    if mask.all() or not mask.any():
        return mask
    # Drop circular runs shorter than min_length.
    min_len = int(round(cfg["min_length_m"] / ds))
    start = int(np.argmin(mask))          # rotate so index 0 is a non-kerb sample
    rot = np.roll(mask, -start)
    padded = np.concatenate([[False], rot, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < min_len:
            rot[a:b] = False
    return np.roll(rot, start)


class Track:
    """Everything the simulation needs to know about one track."""

    def __init__(self, spec: TrackSpec, track_cfg: Mapping[str, Any] | None = None,
                 build_field: bool = True) -> None:
        """Build the centreline and kerbs; ``build_field=False`` skips the field (validation only)."""
        cfg = track_cfg if track_cfg is not None else load_preset("track", "default")
        self.spec = spec
        self.cfg = cfg
        self.name = spec.name
        self.key = spec.name          # file stem when loaded from tracks/ (set by load); used as an id
        self.half_width = spec.width_m / 2.0
        self.runoff = spec.runoff_m
        self.kerb_m = spec.kerb_m

        ctrl = np.asarray(spec.control_points, dtype=np.float64)
        sp = cfg["spline"]
        cl = build_centreline(ctrl, sp["alpha"], sp["dense_per_segment"], sp["resample_m"], sp["curvature_smooth_m"])
        want_ccw = spec.direction == "ccw"
        if (signed_area(cl.points) > 0) != want_ccw:
            # Reverse the driving direction but keep control point 0 as the start/finish line.
            ctrl = np.vstack([ctrl[:1], ctrl[:0:-1]])
            cl = build_centreline(ctrl, sp["alpha"], sp["dense_per_segment"], sp["resample_m"], sp["curvature_smooth_m"])
        self.centreline: Centreline = cl
        self.kerb_mask = _kerb_mask(cl.curvature_smooth, cl.ds, cfg["kerb"])

        self.field: F.TrackField | None = None
        if build_field:
            self.build_field()

    def build_field(self) -> None:
        fc = self.cfg["field"]
        cl = self.centreline
        self.field = F.TrackField(cl.points, self.half_width, self.runoff, self.kerb_m,
                                  fc["cell_m"], fc["margin_m"], fc["territory_jump_m"], cl.ds)

    # ------------------------------------------------------------ construction
    @classmethod
    def load(cls, name_or_path: str | Path, track_cfg: Mapping[str, Any] | None = None) -> "Track":
        path = resolve_track_path(str(name_or_path))
        track = cls(load_track_spec(path), track_cfg)
        track.key = path.stem
        return track

    # -------------------------------------------------------------- geometry
    @property
    def length(self) -> float:
        return self.centreline.length

    @property
    def n(self) -> int:
        return self.centreline.n

    @property
    def ds(self) -> float:
        return self.centreline.ds

    def pose_at(self, s: float, lateral: float = 0.0) -> tuple[float, float, float]:
        """World (x, y, heading) at arc length ``s`` (nearest sample) and a lateral offset (left +)."""
        i = int(round(s / self.ds)) % self.n
        p = self.centreline.points[i] + lateral * self.centreline.normal[i]
        t = self.centreline.tangent[i]
        return float(p[0]), float(p[1]), math.atan2(t[1], t[0])

    def edges(self, offset: float) -> np.ndarray:
        """Polyline at a lateral offset from the centreline (left +), e.g. ±half_width."""
        return self.centreline.points + offset * self.centreline.normal

    # ------------------------------------------------------------- field I/O
    def sample(self, x: float, y: float) -> tuple[float, int]:
        """(distance to centreline, nearest sample index) at a point."""
        return F.sample(self.field.dist, self.field.idx, self.field.geom, x, y)

    def surface_at(self, x: float, y: float) -> int:
        d, i = self.sample(x, y)
        return F.classify(d, i, self.field.geom, self.kerb_mask)

    def surfaces_at(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        xs = np.ascontiguousarray(xs, dtype=np.float64)
        ys = np.ascontiguousarray(ys, dtype=np.float64)
        out = np.empty(xs.shape[0], dtype=np.int64)
        F.surfaces_at(self.field.dist, self.field.idx, self.field.geom, self.kerb_mask, xs, ys, out)
        return out

    # ------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        """Problems that make the track unusable (empty list = valid)."""
        v = self.cfg["validation"]
        cl = self.centreline
        problems = []
        if not v["min_length_m"] <= cl.length <= v["max_length_m"]:
            problems.append(f"length {cl.length:.0f} m outside [{v['min_length_m']}, {v['max_length_m']}]")
        kmax = float(np.max(np.abs(cl.curvature_smooth)))
        if kmax > 1.0 / v["min_corner_radius_m"]:
            problems.append(f"tightest corner radius {1.0 / kmax:.1f} m < {v['min_corner_radius_m']} m")
        if self.half_width >= 1.0 / max(kmax, 1e-9):
            problems.append("road half-width exceeds the tightest corner radius")
        clash = self.closest_sections()
        min_gap = self.spec.width_m + v["clearance_m"]
        if clash is not None and clash[0] < min_gap:
            d, i, j = clash
            problems.append(f"sections at s={i * cl.ds:.0f} m and s={j * cl.ds:.0f} m are {d:.1f} m apart "
                            f"(need ≥ {min_gap:.1f} m: road width + clearance)")
        return problems

    def closest_sections(self) -> tuple[float, int, int] | None:
        """Smallest distance between two centreline samples of *different* sections.

        Samples less than ``adjacency_arc_m`` apart along the track count as one section.
        Returns (distance, i, j) or None if no pair is within the search radius.
        """
        v = self.cfg["validation"]
        cl = self.centreline
        radius = self.spec.width_m + v["clearance_m"] + 2.0 * self.runoff
        adj = int(round(v["adjacency_arc_m"] / cl.ds))
        pairs = cKDTree(cl.points).query_pairs(r=radius, output_type="ndarray")
        if len(pairs) == 0:
            return None
        k = np.abs(pairs[:, 0] - pairs[:, 1])
        k = np.minimum(k, cl.n - k)
        pairs = pairs[k > adj]
        if len(pairs) == 0:
            return None
        d = np.hypot(*(cl.points[pairs[:, 0]] - cl.points[pairs[:, 1]]).T)
        m = int(np.argmin(d))
        return float(d[m]), int(pairs[m, 0]), int(pairs[m, 1])
