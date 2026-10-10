"""Track world layer: the whole track pre-rendered once, then blitted through a camera.

The surface is rasterised from the same field the physics uses (D3), so road, kerbs,
grass, run-off walls and the barrier on screen are exactly what the cars feel. Week 3
replaces the flat colours with textures; the interface stays the same.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pygame

from racer.physics.surfaces import GRASS_ID, KERB_ID, OUTSIDE_ID, ROAD_ID
from racer.render.camera import Camera
from racer.track import field as F


class TrackWorld:
    def __init__(self, cfg: Mapping[str, Any], colours: Mapping[str, tuple[int, ...]]) -> None:
        self.cfg = cfg
        self.c = colours
        self.k = float(cfg["world_px_per_m"])
        self.track = None
        self.surface: pygame.Surface | None = None
        self._scaled_cache: tuple[Any, pygame.Surface] | None = None

    def set_track(self, track) -> None:
        """Pre-render the whole track to one surface."""
        self.track = track
        f = track.field
        k = self.k
        x_min, x_max, y_min, y_max = f.extent
        w, h = int((x_max - x_min) * k), int((y_max - y_min) * k)
        surf = np.empty((h, w), dtype=np.int32)
        idx = np.empty((h, w), dtype=np.int32)
        F.rasterise(f.dist, f.idx, f.geom, track.kerb_mask, x_min, y_min, 1.0 / k, surf, idx)

        c = self.c
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[:] = c["outside"]
        rgb[surf == GRASS_ID] = c["grass"]
        rgb[surf == ROAD_ID] = c["road"]
        kerb = surf == KERB_ID
        stripe = ((np.maximum(idx, 0) * track.ds) // self.cfg["kerb_stripe_m"]).astype(np.int64) % 2 == 0
        rgb[kerb & stripe] = c["kerb_red"]
        rgb[kerb & ~stripe] = c["kerb_white"]

        # Barrier: outside pixels next to the world; walls: territory jumps between neighbours.
        n, jump = int(f.geom[6]), int(f.geom[7])
        wall = np.zeros((h, w), dtype=bool)
        inside = surf != OUTSIDE_ID
        for axis in (0, 1):
            a = np.take(idx, range(1, idx.shape[axis]), axis=axis)
            b = np.take(idx, range(0, idx.shape[axis] - 1), axis=axis)
            ia = np.take(inside, range(1, idx.shape[axis]), axis=axis)
            ib = np.take(inside, range(0, idx.shape[axis] - 1), axis=axis)
            kk = np.abs(a - b)
            kk = np.minimum(kk, n - kk)
            edge = ((kk > jump) & (a >= 0) & (b >= 0)) | (ia != ib)
            if axis == 0:
                wall[1:, :] |= edge
                wall[:-1, :] |= edge
            else:
                wall[:, 1:] |= edge
                wall[:, :-1] |= edge
        rgb[wall & (surf != ROAD_ID) & (surf != KERB_ID)] = c["wall"]

        self.surface = pygame.surfarray.make_surface(np.ascontiguousarray(rgb[::-1].transpose(1, 0, 2)))
        self.x_min, self.y_max = x_min, y_max
        self._draw_track_marks()
        self._scaled_cache = None

    def wpx(self, x: float, y: float) -> tuple[float, float]:
        """World metres → pixel on the pre-rendered surface."""
        return (x - self.x_min) * self.k, (self.y_max - y) * self.k

    def _draw_track_marks(self) -> None:
        tr = self.track
        cl = tr.centreline
        # dashed centreline
        dash = max(int(self.cfg["centreline_dash_m"] / tr.ds), 1)
        for i in range(0, cl.n, 2 * dash):
            j = min(i + dash, cl.n - 1)
            pygame.draw.line(self.surface, self.c["centreline"], self.wpx(*cl.points[i]), self.wpx(*cl.points[j]), 1)
        # chequered start/finish line at sample 0
        sq = self.cfg["start_square_m"]
        p, t, nrm = cl.points[0], cl.tangent[0], cl.normal[0]
        n_sq = int(tr.spec.width_m / sq)
        for a in range(n_sq):
            for b in range(2):
                o = p + nrm * (tr.half_width - (a + 0.5) * sq) + t * ((b - 0.5) * sq)
                corners = [o + nrm * dx * sq / 2 + t * dy * sq / 2 for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                colour = self.c["start_light"] if (a + b) % 2 == 0 else self.c["start_dark"]
                pygame.draw.polygon(self.surface, colour, [self.wpx(*q) for q in corners])

    def draw(self, screen: pygame.Surface, cam: Camera) -> None:
        """Blit the part of the track the camera sees (the scaled crop is cached while it doesn't change)."""
        z = cam.view_zoom
        screen.fill(self.c["outside"])
        vw, vh = cam.w / z, cam.h / z
        sx, sy = self.wpx(cam.cx - vw / 2, cam.cy + vh / 2)
        src = pygame.Rect(int(math.floor(sx)), int(math.floor(sy)), int(math.ceil(vw * self.k)) + 2,
                          int(math.ceil(vh * self.k)) + 2)
        clip = src.clip(self.surface.get_rect())
        if clip.w > 0 and clip.h > 0:
            s = z / self.k
            dest_size = (max(int(clip.w * s), 1), max(int(clip.h * s), 1))
            key = (clip.topleft, clip.size, dest_size)
            if self._scaled_cache is None or self._scaled_cache[0] != key:
                sub = self.surface.subsurface(clip)
                scaled = pygame.transform.smoothscale(sub, dest_size) if s >= 0.5 else pygame.transform.scale(sub, dest_size)
                self._scaled_cache = (key, scaled)
            screen.blit(self._scaled_cache[1], ((clip.x - sx) * s, (clip.y - sy) * s))
