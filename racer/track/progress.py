"""Progress along the centreline (PROJECT_CONTEXT §6).

The field gives the nearest sample in O(1); the position is then refined along that
sample's tangent, so ``s`` changes smoothly. ``Δs`` is signed (backwards = negative) and
wraps correctly at the start/finish line. A nearest sample far (in arc length) from the
tracked one means the car is in another section's territory: that is the barrier (D3).
"""
from __future__ import annotations

import math

from racer.track.track import Track


class ProgressTracker:
    def __init__(self, track: Track) -> None:
        self.track = track
        cl = track.centreline
        self._pts = cl.points
        self._tan = cl.tangent
        self._nrm = cl.normal
        self._n = cl.n
        self._ds = cl.ds
        self._len = cl.length
        self._jump = int(track.field.geom[7])
        self.idx = -1
        self.s = 0.0
        self.total = 0.0          # net progress since reset (m)
        self.crossings = 0        # net forward crossings of the start/finish line
        self.lateral = 0.0        # signed offset from the centreline (left +)

    def _refine(self, i: int, x: float, y: float) -> tuple[float, float]:
        px, py = self._pts[i]
        tx, ty = self._tan[i]
        dx, dy = x - px, y - py
        along = dx * tx + dy * ty
        lim = 1.5 * self._ds
        along = min(max(along, -lim), lim)
        s = (i * self._ds + along) % self._len
        nx, ny = self._nrm[i]
        return s, dx * nx + dy * ny

    def reset(self, x: float, y: float) -> None:
        _, i = self.track.sample(x, y)
        if i < 0:
            raise ValueError(f"start position ({x:.1f}, {y:.1f}) is outside the track field")
        self.idx = int(i)
        self.s, self.lateral = self._refine(self.idx, x, y)
        self.total = 0.0
        self.crossings = 0

    def update(self, x: float, y: float) -> tuple[float, int, bool]:
        """Move to (x, y). Returns (Δs, line crossing −1/0/+1, left_territory).

        On ``left_territory`` the state is not updated (the episode ends at the barrier).
        """
        _, i = self.track.sample(x, y)
        if i < 0:
            return 0.0, 0, True
        k = abs(i - self.idx)
        if min(k, self._n - k) > self._jump:
            return 0.0, 0, True
        s_new, self.lateral = self._refine(int(i), x, y)
        ds = s_new - self.s
        half = 0.5 * self._len
        crossing = 0
        if ds < -half:
            ds += self._len
            crossing = 1
        elif ds > half:
            ds -= self._len
            crossing = -1
        self.s = s_new
        self.idx = int(i)
        self.total += ds
        self.crossings += crossing
        return ds, crossing, False

    def heading_error(self, theta: float) -> float:
        """Car heading minus the track tangent at the tracked sample, wrapped to (−π, π]."""
        tx, ty = self._tan[self.idx]
        e = theta - math.atan2(ty, tx)
        return (e + math.pi) % (2.0 * math.pi) - math.pi
