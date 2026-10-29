"""Ray sensors: distance to the road edge, sphere-traced on the track field (§7.1, D2).

A ray returns the distance to its *first road-edge crossing*: leaving the road when it
starts on the road, entering it when it starts off-road. Rays ignore other cars and the
barrier. The signed distance used for marching is ``half_width − d_centreline``
(positive on the road). Marching steps by |sdf| (never past an edge), with a minimum step
that grows with distance so near-parallel rays stay cheap; a step that jumps over an
edge is refined by bisection. A ray stops only at a real crossing (a grazing touch is not one).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from numba import njit

from racer.track.field import sample

# Index of each raycast parameter in the packed float64 vector.
P_EPS, P_MIN_STEP, P_GROWTH, P_MAX_ITERS, P_BISECT = range(5)


def pack_raycast_params(cfg: Mapping[str, Any]) -> np.ndarray:
    return np.array([cfg["hit_eps_m"], cfg["min_step_m"], cfg["min_step_growth"],
                     cfg["max_iters"], cfg["bisect_iters"]], dtype=np.float64)


@njit(cache=True)
def _sdf(dist, idx, geom, x, y):
    d, _ = sample(dist, idx, geom, x, y)
    return geom[3] - d


@njit(cache=True)
def cast_ray(dist, idx, geom, ox, oy, dx, dy, max_range, params):
    """Distance from (ox, oy) along unit (dx, dy) to the first road-edge crossing (≤ max_range)."""
    s0 = _sdf(dist, idx, geom, ox, oy)
    if abs(s0) < params[0]:
        return 0.0
    return _march(dist, idx, geom, ox, oy, dx, dy, max_range, params, s0 >= 0.0)


@njit(cache=True)
def _march(dist, idx, geom, ox, oy, dx, dy, max_range, params, inside):
    """Distance to the first change of side from ``inside`` (True = on the road) along the ray."""
    min_step = params[1]
    growth = params[2]
    max_iters = int(params[3])
    bisect = int(params[4])
    # In unknown (FAR) territory the road is at least `runoff` away, so cap steps there.
    cap = geom[4]
    t = 0.0
    t_prev = 0.0
    for _ in range(max_iters):
        s = _sdf(dist, idx, geom, ox + dx * t, oy + dy * t)
        if (s >= 0.0) != inside:
            lo, hi = t_prev, t
            for _b in range(bisect):
                mid = 0.5 * (lo + hi)
                sm = _sdf(dist, idx, geom, ox + dx * mid, oy + dy * mid)
                if (sm >= 0.0) == inside:
                    lo = mid
                else:
                    hi = mid
            return min(0.5 * (lo + hi), max_range)
        # No early stop near an edge: a grazing ray that only touches the edge has not
        # crossed it. Close to an edge the floor step takes over and bisection finds the hit.
        a = abs(s)
        if t >= max_range:
            return max_range
        step = min(a, cap)
        floor_step = min_step + growth * t
        if step < floor_step:
            step = floor_step
        t_prev = t
        t = min(t + step, max_range)
    return max_range


@njit(cache=True)
def cast_rays(dist, idx, geom, x, y, theta, angles, ranges, params, out):
    """Cast one ray per (angle, range) relative to heading ``theta``; distances into ``out``."""
    for k in range(angles.shape[0]):
        a = theta + angles[k]
        out[k] = cast_ray(dist, idx, geom, x, y, math.cos(a), math.sin(a), ranges[k], params)


# How far past a crossing the next search starts (m): clear of the ``hit_eps_m`` band around the edge.
STEP_PAST = 0.1


@njit(cache=True)
def cast_rays_multi(dist, idx, geom, x, y, theta, angles, ranges, params, n_cross, out):
    """Like :func:`cast_rays`, but each ray goes on past its crossings (D33, ``rays2``).

    ``out[k, j]`` is the distance along ray ``k`` to its (j+1)-th road-edge crossing, or the ray's
    range when there is none within it. A ray from the road first leaves it, then re-enters it, and so on.
    """
    for k in range(angles.shape[0]):
        a = theta + angles[k]
        dx = math.cos(a)
        dy = math.sin(a)
        reach = ranges[k]
        t = 0.0                       # distance from the car to the start of the current search
        # The side the search starts on follows from the crossings so far; asking the field again just past
        # a crossing would see a shallow ray still inside the ``hit_eps_m`` band and report a false hit.
        inside = _sdf(dist, idx, geom, x, y) >= 0.0
        for j in range(n_cross):
            if t >= reach:
                out[k, j] = reach
                continue
            d = _march(dist, idx, geom, x + dx * t, y + dy * t, dx, dy, reach - t, params, inside)
            inside = not inside
            if t + d >= reach:
                out[k, j] = reach
                t = reach
            else:
                out[k, j] = t + d
                t = t + d + STEP_PAST
