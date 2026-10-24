"""Track world layer: the whole track pre-rendered once, then blitted through a camera.

The surface is rasterised from the same field the physics uses (D3), so road, kerbs, grass,
run-off and the barrier on screen are exactly what the cars feel. On top of that raster the world
gets textures (asphalt, grass, a gravel band, edge lines), a striped barrier and trackside scenery;
with ``textures.enabled: false`` (or missing asset files) it falls back to flat colours.
"""
from __future__ import annotations

import math
import zlib
from typing import Any, Mapping

import numpy as np
import pygame

from racer.config import PROJECT_ROOT
from racer.physics.surfaces import GRASS_ID, KERB_ID, OUTSIDE_ID, ROAD_ID
from racer.render import textures as T
from racer.render.camera import Camera
from racer.render.scenery import Prop, place_props
from racer.track import field as F

PROPS = PROJECT_ROOT / "assets" / "sprites" / "props"


class TrackWorld:
    def __init__(self, cfg: Mapping[str, Any], colours: Mapping[str, tuple[int, ...]]) -> None:
        self.cfg = cfg
        self.c = colours
        self.k = float(cfg["world_px_per_m"])
        self.track = None
        self.surface: pygame.Surface | None = None
        self.props: list[Prop] = []
        self._scaled_cache: tuple[Any, pygame.Surface] | None = None
        self._sprites: dict[str, pygame.Surface | None] = {}
        self.outside_colour = tuple(colours["outside"])

    # ------------------------------------------------------------------ build
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
        self.x_min, self.y_max, self.y_min = x_min, y_max, y_min
        rng = np.random.default_rng(zlib.crc32(str(track.key).encode()))

        tcfg = self.cfg.get("textures") or {}
        rgb = self._textured(surf, idx, rng, tcfg) if tcfg.get("enabled") else self._flat(surf, idx)
        self._paint_barrier(rgb, self._wall(surf, idx), surf, idx)
        self.surface = pygame.surfarray.make_surface(np.ascontiguousarray(rgb[::-1].transpose(1, 0, 2)))
        del rgb
        self._draw_track_marks()
        self.props = []
        scfg = self.cfg.get("scenery") or {}
        if scfg.get("enabled"):
            self.props = place_props(track, scfg, rng)
            self._draw_props()
        self._scaled_cache = None

    def _flat(self, surf: np.ndarray, idx: np.ndarray) -> np.ndarray:
        c = self.c
        h, w = surf.shape
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[:] = c["outside"]
        rgb[surf == GRASS_ID] = c["grass"]
        rgb[surf == ROAD_ID] = c["road"]
        self._kerbs(rgb, surf, idx)
        return rgb

    def _kerbs(self, rgb: np.ndarray, surf: np.ndarray, idx: np.ndarray) -> None:
        c = self.c
        kerb = surf == KERB_ID
        stripe = ((np.maximum(idx, 0) * self.track.ds) // self.cfg["kerb_stripe_m"]).astype(np.int64) % 2 == 0
        rgb[kerb & stripe] = c["kerb_red"]
        rgb[kerb & ~stripe] = c["kerb_white"]

    def _textured(self, surf: np.ndarray, idx: np.ndarray, rng: np.random.Generator, tc: Mapping[str, Any]) -> np.ndarray:
        c, k, track = self.c, self.k, self.track
        f = track.field
        h, w = surf.shape
        gt = tc["ground"]

        def tile(name: str, fallback) -> np.ndarray:
            g = gt[name]
            return T.load_tile(g["file"], int(g["tile_m"] * k), g["darken"], g["desaturate"], fallback, rng)

        rows, cols = np.arange(h)[:, None], np.arange(w)[None, :]

        def layer(t: np.ndarray) -> np.ndarray:
            return t[rows % t.shape[0], cols % t.shape[1]]

        # distance to the centreline of every pixel, from the field the physics uses
        xs = self.x_min + (np.arange(w) + 0.5) / k
        ys = self.y_min + (np.arange(h) + 0.5) / k
        edge = T.bilinear(f.dist, (ys - f.y0) / f.cell, (xs - f.x0) / f.cell) - track.half_width   # > 0 off the road

        grass = layer(tile("grass", c["grass"])).astype(np.float32)
        var = tc.get("variation") or {}
        if var.get("strength", 0) > 0:
            grass *= T.low_frequency((h, w), max(int(var["cell_m"] * k), 4), float(var["strength"]), rng)[..., None]
        outside = surf == OUTSIDE_ID
        rgb = np.where(outside[..., None], grass * (1.0 - float(tc["outside_darken"])), grass)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if outside.any():
            self.outside_colour = tuple(int(v) for v in rgb[outside][::997].mean(axis=0))   # beyond the world's edge
        del grass

        near = (surf == GRASS_ID) & (edge <= float(tc["gravel_band_m"]))
        rgb = np.where(near[..., None], layer(tile("gravel", c["grass"])), rgb)
        road = (surf == ROAD_ID) | (surf == KERB_ID)
        asphalt = layer(tile("asphalt", c["road"])).astype(np.float32)
        if var.get("strength", 0) > 0:
            patchy = T.low_frequency((h, w), max(int(var["cell_m"] * k / 2), 4), float(var["strength"]), rng)
            asphalt *= (1.0 + 0.5 * (patchy - 1.0))[..., None]
        rgb = np.where(road[..., None], np.clip(asphalt, 0, 255).astype(np.uint8), rgb)
        del asphalt
        self._kerbs(rgb, surf, idx)
        line = (surf == ROAD_ID) & (edge > -float(tc["edge_line_m"]))
        rgb[line] = (0.2 * rgb[line] + 0.8 * np.asarray(tc["edge_line_colour"], dtype=np.float32)).astype(np.uint8)
        return rgb

    def _wall(self, surf: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Barrier pixels: outside pixels next to the world; walls: territory jumps between neighbours."""
        f = self.track.field
        h, w = surf.shape
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
        return wall & (surf != ROAD_ID) & (surf != KERB_ID)

    def _paint_barrier(self, rgb: np.ndarray, wall: np.ndarray, surf: np.ndarray, idx: np.ndarray) -> None:
        tc = self.cfg.get("textures") or {}
        if not tc.get("enabled"):
            rgb[wall] = self.c["wall"]
            return
        # a thicker barrier painted in alternating blocks, with a shadow on the grass side
        off_road = (surf != ROAD_ID) & (surf != KERB_ID)
        thick = wall.copy()
        thick[1:, :] |= wall[:-1, :]
        thick[:, 1:] |= wall[:, :-1]
        thick &= off_road
        shadow = np.zeros_like(thick)
        shadow[2:, 2:] = thick[:-2, :-2]
        shadow &= ~thick & off_road
        rgb[shadow] = (rgb[shadow] * 0.55).astype(np.uint8)
        block = ((np.maximum(idx, 0) * self.track.ds) // float(tc["barrier_block_m"])).astype(np.int64) % 2 == 0
        rgb[thick & block] = tc["barrier_a"]
        rgb[thick & ~block] = tc["barrier_b"]

    # ------------------------------------------------------------------ marks and props
    def wpx(self, x: float, y: float) -> tuple[float, float]:
        """World metres → pixel on the pre-rendered surface."""
        return (x - self.x_min) * self.k, (self.y_max - y) * self.k

    def _draw_track_marks(self) -> None:
        tr = self.track
        cl = tr.centreline
        dash = max(int(self.cfg["centreline_dash_m"] / tr.ds), 1)
        colour = self.c["centreline"]
        for i in range(0, cl.n, 2 * dash):
            j = min(i + dash, cl.n - 1)
            pygame.draw.line(self.surface, colour, self.wpx(*cl.points[i]), self.wpx(*cl.points[j]), 1)
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

    def _sprite(self, kind: str) -> pygame.Surface | None:
        if kind not in self._sprites:
            try:
                self._sprites[kind] = pygame.image.load(str(PROPS / f"{kind}.png")).convert_alpha()
            except (FileNotFoundError, OSError, pygame.error):
                self._sprites[kind] = None            # a missing asset never crashes the demo
        return self._sprites[kind]

    def _draw_props(self) -> None:
        sc = self.cfg["scenery"]
        for pr in sorted(self.props, key=lambda p: p.width_m):          # small things first, big ones on top
            spr = self._sprite(pr.kind)
            if spr is None:
                continue
            sw, sh = spr.get_size()
            s = pr.width_m * self.k / sw
            img = pygame.transform.smoothscale(spr, (max(int(sw * s), 2), max(int(sh * s), 2)))
            img = pygame.transform.rotate(img, math.degrees(pr.theta) - 90.0)
            cx, cy = self.wpx(pr.x, pr.y)
            shadow = img.copy()
            shadow.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MIN)     # black silhouette
            shadow.set_alpha(int(sc["shadow_alpha"]))
            off = float(sc["shadow_m"]) * self.k * (1.0 if pr.kind.startswith("tree") else 0.5)
            self.surface.blit(shadow, shadow.get_rect(center=(cx + off, cy + off)))
            self.surface.blit(img, img.get_rect(center=(cx, cy)))

    # ------------------------------------------------------------------ per frame
    def draw(self, screen: pygame.Surface, cam: Camera) -> None:
        """Blit the part of the track the camera sees (the scaled crop is cached while it doesn't change)."""
        z = cam.view_zoom
        screen.fill(self.outside_colour)
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
