"""Trackside scenery (PROJECT_CONTEXT §10.2): grandstands, tents, trees and tyre walls, placed from the
track geometry and never inside the barrier.

Placement is pure (:func:`place_props` returns where things go); :class:`racer.render.world.TrackWorld`
draws the sprites into the pre-rendered world. The layout is deterministic per track (seeded by name).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Prop:
    kind: str          # sprite file stem in assets/sprites/props
    x: float
    y: float
    theta: float       # radians; the sprite's "up" points along +theta
    width_m: float     # world width of the sprite (its short side for trees and tents)


def _dist_at(field, x: float, y: float) -> float:
    """Distance to the centreline at a point (nearest grid node); huge outside the field."""
    c = int(round((x - field.x0) / field.cell))
    r = int(round((y - field.y0) / field.cell))
    h, w = field.dist.shape
    if not (0 <= r < h and 0 <= c < w):
        return -1.0
    return float(field.dist[r, c])


def place_props(track, cfg: Mapping[str, Any], rng: np.random.Generator) -> list[Prop]:
    """Props around ``track``. ``cfg`` is ``render.yaml: debug.scenery``."""
    f = track.field
    cl = track.centreline
    hw = track.half_width
    barrier = hw + track.runoff                   # distance of the barrier from the centreline
    x0, x1, y0, y1 = f.extent
    taken: list[tuple[float, float, float]] = []         # x, y, radius

    def free(x: float, y: float, r: float) -> bool:
        if not (x0 + r < x < x1 - r and y0 + r < y < y1 - r):
            return False
        for qx, qy, qr in taken:
            if (qx - x) ** 2 + (qy - y) ** 2 < (qr + r) ** 2:
                return False
        # clear of the barrier of every section, including other parts of the track
        return _dist_at(f, x, y) >= barrier + float(cfg["barrier_gap_m"]) + r

    props: list[Prop] = []

    def add(kind: str, x: float, y: float, theta: float, width: float, radius: float) -> bool:
        if free(x, y, radius):
            taken.append((x, y, radius))
            props.append(Prop(kind, x, y, theta, width))
            return True
        return False

    n = cl.n
    step = max(int(float(cfg["spacing_m"]) / track.ds), 1)
    grand_span = int(float(cfg["grandstand_span_m"]) / track.ds)
    for i in range(0, n, step):
        p, t, nrm = cl.points[i], cl.tangent[i], cl.normal[i]
        theta = math.atan2(t[1], t[0])
        straight = abs(float(cl.curvature[i])) < float(cfg["straight_curvature"])
        dist_from_start = min(i, n - i) * track.ds
        for side in (1.0, -1.0):
            off = barrier + float(cfg["barrier_gap_m"])
            # grandstands on the straight by the start line, facing the road
            if dist_from_start < grand_span * track.ds and straight and rng.random() < 0.8:
                w = float(cfg["grandstand_width_m"])
                o = off + float(cfg["grandstand_depth_m"]) / 2 + float(cfg["grandstand_gap_m"])
                kind = "grandstand_full" if rng.random() < 0.7 else "grandstand_empty"
                add(kind, *(p + nrm * side * o), theta + (math.pi if side > 0 else 0.0) + math.pi / 2, w, w * 0.5)
                continue
            # tyre walls on the outside of corners
            curv = float(cl.curvature[i])
            if not straight and curv * side < 0 and rng.random() < float(cfg["tyre_wall_chance"]):
                o = barrier + 1.3
                kind = "tyres_red" if rng.random() < 0.5 else "tyres_white"
                for j in range(3):
                    q = p + nrm * side * o + t * (j * 1.7)
                    if _dist_at(f, q[0], q[1]) >= barrier + 0.8:
                        props.append(Prop(kind, float(q[0]), float(q[1]), float(rng.uniform(0, 6.28)), 1.6))
                continue
            r = rng.random()
            if r < float(cfg["tent_chance"]):
                kind = str(rng.choice(["tent_red", "tent_blue", "tent_red_large", "tent_blue_large"]))
                w = float(rng.uniform(5.0, 9.0))
                add(kind, *(p + nrm * side * (off + w * 0.8 + rng.uniform(0, 6))), theta + float(rng.choice([0, math.pi / 2])),
                    w, w * 0.8)
            elif r < float(cfg["tent_chance"]) + float(cfg["tree_chance"]):
                for _ in range(int(rng.integers(1, int(cfg["tree_clump"]) + 1))):
                    w = float(rng.uniform(5.0, 11.0))
                    q = p + nrm * side * (off + w * 0.6 + rng.uniform(1, float(cfg["tree_depth_m"]))) + t * rng.uniform(-8, 8)
                    add("tree_large" if w > 8 else "tree_small", float(q[0]), float(q[1]), float(rng.uniform(0, 6.28)), w, w * 0.45)
    return props
