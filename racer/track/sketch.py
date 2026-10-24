"""Draw-a-Track: a drawing becomes a track (PROJECT_CONTEXT §6, §10.1).

The drawing (freehand or clicked points, in metres) is a closed loop: the last point joins the
first, which is the start/finish line, and the driving direction is the way it was drawn. It is
smoothed against hand jitter (a window that is a fraction of the loop, so the size it was drawn at
doesn't matter), scaled into the allowed length and thinned to control points. A corner that is
still too tight gets a wider smoothing window. The result is checked with the generator's validation;
problems that can't be fixed this way (e.g. a figure-eight) are returned, not hidden.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from racer.config import load_preset
from racer.track.io import TrackSpec
from racer.track.spline import circular_smooth, signed_area
from racer.track.track import Track


def _closed_resample(pts: np.ndarray, n: int | None = None, step: float | None = None) -> np.ndarray:
    """``n`` points (or one every ``step`` metres) along the closed polyline, starting at ``pts[0]``."""
    loop = np.vstack([pts, pts[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(loop, axis=0).T))])
    at = np.linspace(0.0, s[-1], n, endpoint=False) if n else np.arange(0.0, s[-1], step)
    return np.column_stack([np.interp(at, s, loop[:, 0]), np.interp(at, s, loop[:, 1])])


def _smooth(pts: np.ndarray, n: int, frac: float) -> np.ndarray:
    dense = _closed_resample(pts, n=n)
    window = int(round(frac * n))
    return np.column_stack([circular_smooth(dense[:, 0], window), circular_smooth(dense[:, 1], window)])


def sketch_track(points: Sequence[Sequence[float]] | np.ndarray, name: str, sketch_cfg: Mapping[str, Any],
                 width_m: float | None = None, track_cfg: Mapping[str, Any] | None = None
                 ) -> tuple[Track | None, list[str]]:
    """(track without field, problems). ``track`` is None when there is too little drawing."""
    cfg = track_cfg if track_cfg is not None else load_preset("track", "default")
    gen, v = cfg["generator"], cfg["validation"]
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts):
        keep = np.concatenate([[True], np.hypot(*np.diff(pts, axis=0).T) > 1e-6])
        pts = pts[keep]
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    perimeter = float(np.sum(np.hypot(*(np.roll(pts, -1, axis=0) - pts).T))) if len(pts) else 0.0
    if len(pts) < sketch_cfg["min_points"] or perimeter < sketch_cfg["min_drawn_m"]:
        return None, ["Draw a closed loop: the drawing is too short."]
    spread = np.sqrt(np.clip(np.linalg.eigvalsh(np.cov(pts.T)), 0.0, None))      # across, along
    if spread[0] < sketch_cfg["min_thickness"] * spread[1]:
        return None, ["Draw a closed loop, not a line: the drawing has almost no width."]

    direction = "ccw" if signed_area(pts) > 0 else "cw"
    width = float(width_m if width_m is not None else gen["width_m"])
    centre = pts.mean(axis=0)
    scale = float(np.clip(perimeter, v["min_length_m"], v["max_length_m"])) / perimeter
    frac = float(sketch_cfg["smooth_frac"])
    for _ in range(sketch_cfg["fix_steps"]):
        shape = _smooth(pts, sketch_cfg["samples"], frac)
        ctrl = _closed_resample(centre + (shape - centre) * scale, step=sketch_cfg["control_spacing_m"])
        spec = TrackSpec(name, ctrl.round(3).tolist(), width, gen["runoff_m"], gen["kerb_m"], direction)
        track = Track(spec, cfg, build_field=False)
        problems = track.validate()
        if not problems:
            break
        length = track.length
        radius = 1.0 / max(float(np.max(np.abs(track.centreline.curvature_smooth))), 1e-9)
        clash = track.closest_sections()
        # Scaling up widens corners and gaps too (the shape is fixed); the 2-5 % margins absorb
        # the small change in length that the spline and smoothing add.
        needs = [1.02 * v["min_length_m"] / length,
                 1.05 * max(v["min_corner_radius_m"], width / 2.0) / radius,
                 1.05 * (width + v["clearance_m"]) / max(clash[0], 1e-9) if clash else 0.0]
        most = 0.98 * v["max_length_m"] / length
        if needs[1] > most:            # corners too sharp for any allowed size
            frac = min(2.0 * frac, sketch_cfg["max_smooth_frac"])
        scale *= min(max(*needs, 1.0), most) if length <= v["max_length_m"] else most
    return track, problems
