"""The track field: a grid of distance-to-centreline and nearest-sample index (§6, D3).

Grid nodes sit at ``(x0 + c * cell, y0 + r * cell)``. Distances are bilinearly interpolated;
the nearest-sample index comes from the closest node. Nodes farther than ``reach`` from
the centreline hold ``FAR`` and index −1 (outside the world).

All numba kernels take raw arrays plus a small float64 ``geom`` vector, so the same
compiled code serves every track:

    geom = [x0, y0, inv_cell, half_width, runoff, kerb_m, n_samples, jump_samples]
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from racer.physics.surfaces import GRASS_ID, KERB_ID, OUTSIDE_ID, ROAD_ID

FAR = 1.0e6

G_X0, G_Y0, G_INV_CELL, G_HW, G_RUNOFF, G_KERB, G_N, G_JUMP = range(8)


@njit(cache=True)
def _build(px, py, x0, y0, cell, reach, dist, idx):
    n = px.shape[0]
    h, w = dist.shape
    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        ax, ay, bx, by = px[i], py[i], px[j], py[j]
        sx, sy = bx - ax, by - ay
        l2 = sx * sx + sy * sy
        c0 = max(int(math.floor((min(ax, bx) - reach - x0) / cell)), 0)
        c1 = min(int(math.ceil((max(ax, bx) + reach - x0) / cell)), w - 1)
        r0 = max(int(math.floor((min(ay, by) - reach - y0) / cell)), 0)
        r1 = min(int(math.ceil((max(ay, by) + reach - y0) / cell)), h - 1)
        for r in range(r0, r1 + 1):
            cy = y0 + r * cell
            for c in range(c0, c1 + 1):
                cx = x0 + c * cell
                t = ((cx - ax) * sx + (cy - ay) * sy) / l2 if l2 > 0.0 else 0.0
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                dx = cx - (ax + t * sx)
                dy = cy - (ay + t * sy)
                d = math.sqrt(dx * dx + dy * dy)
                if d < dist[r, c] and d <= reach:
                    dist[r, c] = d
                    idx[r, c] = i if t < 0.5 else j


@njit(cache=True)
def sample(dist, idx, geom, x, y):
    """(distance to centreline, nearest sample index) at a point; (FAR, −1) outside the grid."""
    h, w = dist.shape
    fx = (x - geom[0]) * geom[2]
    fy = (y - geom[1]) * geom[2]
    if not (fx >= 0.0 and fy >= 0.0):   # also catches NaN
        return FAR, -1
    c = int(fx)
    r = int(fy)
    if c >= w - 1 or r >= h - 1:
        return FAR, -1
    tx = fx - c
    ty = fy - r
    d = ((dist[r, c] * (1.0 - tx) + dist[r, c + 1] * tx) * (1.0 - ty)
         + (dist[r + 1, c] * (1.0 - tx) + dist[r + 1, c + 1] * tx) * ty)
    i = idx[r + (1 if ty > 0.5 else 0), c + (1 if tx > 0.5 else 0)]
    return d, i


@njit(cache=True)
def classify(d, i, geom, kerb_mask):
    """Surface id for a field sample (D1: kerbs are a band just inside the road edge)."""
    hw = geom[3]
    if i < 0 or d > hw + geom[4]:
        return OUTSIDE_ID
    if d <= hw:
        if kerb_mask[i] and d >= hw - geom[5]:
            return KERB_ID
        return ROAD_ID
    return GRASS_ID


@njit(cache=True)
def arc_jump(i, j, n):
    """Circular index distance between samples i and j."""
    k = abs(i - j)
    return k if k <= n - k else n - k


@njit(cache=True)
def surfaces_at(dist, idx, geom, kerb_mask, xs, ys, out):
    for k in range(xs.shape[0]):
        d, i = sample(dist, idx, geom, xs[k], ys[k])
        out[k] = classify(d, i, geom, kerb_mask)


@njit(cache=True)
def probe_car(dist, idx, geom, kerb_mask, x, y, theta, wheel_off, corner_off, tracked_idx, wheel_out):
    """Surface under each wheel (into ``wheel_out``) and whether the body hit the barrier.

    The barrier is hit when a body corner is beyond the run-off, outside the grid, or in
    another track section's territory (nearest sample > jump_samples from ``tracked_idx``).
    """
    c = math.cos(theta)
    s = math.sin(theta)
    for k in range(wheel_off.shape[0]):
        px = x + wheel_off[k, 0] * c - wheel_off[k, 1] * s
        py = y + wheel_off[k, 0] * s + wheel_off[k, 1] * c
        d, i = sample(dist, idx, geom, px, py)
        wheel_out[k] = classify(d, i, geom, kerb_mask)
    n = int(geom[6])
    jump = int(geom[7])
    for k in range(corner_off.shape[0]):
        px = x + corner_off[k, 0] * c - corner_off[k, 1] * s
        py = y + corner_off[k, 0] * s + corner_off[k, 1] * c
        d, i = sample(dist, idx, geom, px, py)
        if i < 0 or d > geom[3] + geom[4]:
            return True
        if tracked_idx >= 0 and arc_jump(i, tracked_idx, n) > jump:
            return True
    return False


@njit(cache=True)
def rasterise(dist, idx, geom, kerb_mask, x0, y0, step, out_surf, out_idx):
    """Surface id and nearest sample per pixel; pixel (r, c) is centred at
    (x0 + (c + 0.5) * step, y0 + (r + 0.5) * step). Row 0 is the lowest y (renderers flip)."""
    h, w = out_surf.shape
    for r in range(h):
        y = y0 + (r + 0.5) * step
        for c in range(w):
            d, i = sample(dist, idx, geom, x0 + (c + 0.5) * step, y)
            out_surf[r, c] = classify(d, i, geom, kerb_mask)
            out_idx[r, c] = i


class TrackField:
    """Grid arrays + ``geom`` vector for one track."""

    def __init__(self, points: np.ndarray, half_width: float, runoff: float, kerb_m: float,
                 cell: float, margin: float, jump_m: float, ds: float) -> None:
        self.cell = float(cell)
        self.reach = half_width + runoff + margin
        lo = points.min(axis=0) - self.reach - cell
        hi = points.max(axis=0) + self.reach + cell
        w = int(math.ceil((hi[0] - lo[0]) / cell)) + 1
        h = int(math.ceil((hi[1] - lo[1]) / cell)) + 1
        self.x0, self.y0 = float(lo[0]), float(lo[1])
        self.dist = np.full((h, w), FAR, dtype=np.float32)
        self.idx = np.full((h, w), -1, dtype=np.int32)
        px = np.ascontiguousarray(points[:, 0], dtype=np.float64)
        py = np.ascontiguousarray(points[:, 1], dtype=np.float64)
        _build(px, py, self.x0, self.y0, self.cell, self.reach, self.dist, self.idx)
        self.geom = np.array([self.x0, self.y0, 1.0 / cell, half_width, runoff, kerb_m,
                              len(points), max(int(round(jump_m / ds)), 1)], dtype=np.float64)

    @property
    def shape(self) -> tuple[int, int]:
        return self.dist.shape

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(x_min, x_max, y_min, y_max) of the grid nodes."""
        h, w = self.dist.shape
        return self.x0, self.x0 + (w - 1) * self.cell, self.y0, self.y0 + (h - 1) * self.cell

    @property
    def nbytes(self) -> int:
        return self.dist.nbytes + self.idx.nbytes
