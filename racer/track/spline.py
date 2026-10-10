"""Closed Catmull-Rom splines resampled by arc length (PROJECT_CONTEXT §6)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def catmull_rom_closed(points: np.ndarray, alpha: float, samples_per_segment: int) -> np.ndarray:
    """Dense points on the closed Catmull-Rom curve through ``points`` (n, 2).

    ``alpha`` = 0.5 is the centripetal variant, which never forms cusps or self-loops
    within a segment. Returns (n * samples_per_segment, 2); the curve closes back on itself.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        raise ValueError("a closed track needs at least 3 control points")
    u = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
    out = np.empty((n, samples_per_segment, 2))
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-9) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-9) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-9) ** alpha
        t = t1 + u * (t2 - t1)
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out[i] = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
    return out.reshape(-1, 2)


def resample_closed(dense: np.ndarray, spacing: float) -> tuple[np.ndarray, float]:
    """Resample a closed polyline at (almost exactly) ``spacing`` metres.

    Returns (points (N, 2), total length). The spacing is adjusted to length / N so
    that the last sample is one spacing away from the first.
    """
    closed = np.vstack([dense, dense[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(cum[-1])
    n = max(int(round(length / spacing)), 8)
    s = np.arange(n) * (length / n)
    x = np.interp(s, cum, closed[:, 0])
    y = np.interp(s, cum, closed[:, 1])
    return np.column_stack([x, y]), length


def signed_area(points: np.ndarray) -> float:
    """Shoelace area: positive for counter-clockwise polygons."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average with wrap-around (window rounded up to odd)."""
    if window <= 1:
        return values.copy()
    window |= 1
    half = window // 2
    padded = np.concatenate([values[-half:], values, values[:half]])
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


@dataclass
class Centreline:
    """Uniformly sampled closed centreline with per-sample geometry."""

    points: np.ndarray      # (N, 2)
    tangent: np.ndarray     # (N, 2) unit, driving direction
    normal: np.ndarray      # (N, 2) unit, pointing left of the driving direction
    curvature: np.ndarray   # (N,) signed 1/m, positive = turning left
    curvature_smooth: np.ndarray
    s: np.ndarray           # (N,) arc length of each sample from the start/finish line
    length: float
    ds: float               # sample spacing (length / N)

    @property
    def n(self) -> int:
        return len(self.points)


def build_centreline(control_points: np.ndarray, alpha: float, samples_per_segment: int,
                     spacing: float, smooth_m: float) -> Centreline:
    """Spline through the control points, resampled, with tangents, normals and curvature."""
    dense = catmull_rom_closed(control_points, alpha, samples_per_segment)
    pts, length = resample_closed(dense, spacing)
    n = len(pts)
    ds = length / n
    nxt, prv = np.roll(pts, -1, axis=0), np.roll(pts, 1, axis=0)
    d1 = (nxt - prv) / (2.0 * ds)
    d2 = (nxt - 2.0 * pts + prv) / (ds * ds)
    speed = np.hypot(d1[:, 0], d1[:, 1])
    tangent = d1 / speed[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    curvature = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed ** 3
    smooth = circular_smooth(curvature, int(round(smooth_m / ds)))
    return Centreline(pts, tangent, normal, curvature, smooth, np.arange(n) * ds, length, ds)
