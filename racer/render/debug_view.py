"""Minimal debug renderer (pygame-ce). It only *draws* simulation objects; it never steps them.

The track surface is rasterised once from the same field the physics uses (D3), so road,
kerbs, grass, run-off walls and the barrier on screen are exactly what the car feels.
"""
from __future__ import annotations

import colorsys
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pygame

from racer.config import PROJECT_ROOT
from racer.physics.surfaces import GRASS_ID, KERB_ID, OUTSIDE_ID, ROAD_ID
from racer.track import field as F

ASSETS = PROJECT_ROOT / "assets"
SURFACE_NAMES = {OUTSIDE_ID: "out", GRASS_ID: "grass", ROAD_ID: "road", KERB_ID: "kerb"}


def _col(c: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(v) for v in c)


def load_font(name: str, size: int) -> pygame.font.Font:
    path = ASSETS / "fonts" / name
    try:
        return pygame.font.Font(str(path), size)
    except (FileNotFoundError, OSError, pygame.error):
        return pygame.font.Font(None, int(size * 1.25))   # fallback: never crash on a missing asset


def speed_colours(n: int = 64) -> list[tuple[int, int, int]]:
    """Blue (slow) → green → yellow → red (fast)."""
    out = []
    for k in range(n):
        h = (1.0 - k / (n - 1)) * 0.66
        r, g, b = colorsys.hsv_to_rgb(h, 0.9, 1.0)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


class DebugView:
    def __init__(self, size: tuple[int, int], cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.c = {k: _col(v) for k, v in cfg["colours"].items()}
        self.w, self.h = size
        self.k = float(cfg["world_px_per_m"])
        self.zoom = float(cfg["zoom_px_per_m"])
        self.cx = self.cy = 0.0
        self.overview = False
        self.zoom_fit = self.zoom
        self.font = load_font(cfg["font"], int(cfg["font_size"]))
        self.font_big = load_font(cfg["font"], int(cfg["font_size"] * 2))
        self.speed_lut = speed_colours()
        self._scaled_cache: tuple[Any, pygame.Surface] | None = None
        self._sprite_src: pygame.Surface | None = None
        try:
            self._sprite_src = pygame.image.load(str(ASSETS / "sprites" / "cars" / cfg["car_sprite"])).convert_alpha()
        except (FileNotFoundError, OSError, pygame.error):
            self._sprite_src = None
        self._sprite_zoom = None
        self._sprite: pygame.Surface | None = None
        self.track = None

    # ------------------------------------------------------------- world
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

        self.world = pygame.surfarray.make_surface(np.ascontiguousarray(rgb[::-1].transpose(1, 0, 2)))
        self.x_min, self.y_max = x_min, y_max
        self._draw_track_marks()
        self._scaled_cache = None

    def _wpx(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x_min) * self.k, (self.y_max - y) * self.k

    def _draw_track_marks(self) -> None:
        tr = self.track
        cl = tr.centreline
        # dashed centreline
        dash = max(int(self.cfg["centreline_dash_m"] / tr.ds), 1)
        for i in range(0, cl.n, 2 * dash):
            j = min(i + dash, cl.n - 1)
            pygame.draw.line(self.world, self.c["centreline"], self._wpx(*cl.points[i]), self._wpx(*cl.points[j]), 1)
        # chequered start/finish line at sample 0
        sq = self.cfg["start_square_m"]
        p, t, nrm = cl.points[0], cl.tangent[0], cl.normal[0]
        n_sq = int(tr.spec.width_m / sq)
        for a in range(n_sq):
            for b in range(2):
                o = p + nrm * (tr.half_width - (a + 0.5) * sq) + t * ((b - 0.5) * sq)
                corners = [o + nrm * dx * sq / 2 + t * dy * sq / 2 for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                colour = self.c["start_light"] if (a + b) % 2 == 0 else self.c["start_dark"]
                pygame.draw.polygon(self.world, colour, [self._wpx(*q) for q in corners])

    # ------------------------------------------------------------ camera
    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.cx) * self.zoom + self.w / 2, (self.cy - y) * self.zoom + self.h / 2

    def zoom_by(self, factor: float) -> None:
        self.overview = False
        self.zoom = min(max(self.zoom * factor, self.cfg["zoom_min"]), self.cfg["zoom_max"])

    def update_camera(self, x: float, y: float, vx: float, vy: float, sim_speed: float = 1.0) -> None:
        """Follow the car. ``sim_speed`` = simulated physics steps shown per frame, so the
        camera follows per *simulated* time and keeps up at ×4 as well as at ×1."""
        if self.overview:
            x0, x1, y0, y1 = self.track.field.extent
            self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
            self.zoom_fit = min(self.w / (x1 - x0), self.h / (y1 - y0))
            return
        lead = self.cfg["camera_lead_s"]
        tx, ty = x + vx * lead, y + vy * lead
        a = 1.0 - (1.0 - self.cfg["camera_lerp"]) ** max(sim_speed, 0.0)
        self.cx += (tx - self.cx) * a
        self.cy += (ty - self.cy) * a

    def snap_camera(self, x: float, y: float) -> None:
        self.cx, self.cy = x, y

    @property
    def view_zoom(self) -> float:
        return self.zoom_fit if self.overview else self.zoom

    # ------------------------------------------------------------ drawing
    def draw_world(self, screen: pygame.Surface) -> None:
        z = self.view_zoom
        saved = self.zoom
        self.zoom = z
        screen.fill(self.c["outside"])
        vw, vh = self.w / z, self.h / z
        sx, sy = self._wpx(self.cx - vw / 2, self.cy + vh / 2)
        src = pygame.Rect(int(math.floor(sx)), int(math.floor(sy)), int(math.ceil(vw * self.k)) + 2, int(math.ceil(vh * self.k)) + 2)
        clip = src.clip(self.world.get_rect())
        if clip.w > 0 and clip.h > 0:
            s = z / self.k
            dest_size = (max(int(clip.w * s), 1), max(int(clip.h * s), 1))
            key = (clip.topleft, clip.size, dest_size)
            if self._scaled_cache is None or self._scaled_cache[0] != key:
                sub = self.world.subsurface(clip)
                scaled = pygame.transform.smoothscale(sub, dest_size) if s >= 0.5 else pygame.transform.scale(sub, dest_size)
                self._scaled_cache = (key, scaled)
            screen.blit(self._scaled_cache[1], ((clip.x - sx) * s, (clip.y - sy) * s))
        self.zoom = saved

    def _with_view_zoom(self):
        z = self.view_zoom
        saved, self.zoom = self.zoom, z
        return saved

    def draw_trail(self, screen: pygame.Surface, trail: Iterable[tuple[float, float, float] | None]) -> None:
        saved = self._with_view_zoom()
        vmax = self.cfg["trail_speed_max"]
        n = len(self.speed_lut)
        prev = None
        for p in trail:
            if p is None:
                prev = None
                continue
            q = self.to_screen(p[0], p[1])
            if prev is not None:
                col = self.speed_lut[min(int(p[2] / vmax * (n - 1)), n - 1)]
                pygame.draw.line(screen, col, prev, q, 2)
            prev = q
        self.zoom = saved

    def draw_rays(self, screen: pygame.Surface, x: float, y: float, theta: float,
                  angles: np.ndarray, dists: np.ndarray, ranges: np.ndarray) -> None:
        saved = self._with_view_zoom()
        o = self.to_screen(x, y)
        for a, d, r in zip(angles, dists, ranges):
            frac = float(d / r)
            col = (min(255, int(510 * (1 - frac))), min(255, int(510 * frac)), 60)
            hx, hy = x + d * math.cos(theta + a), y + d * math.sin(theta + a)
            h = self.to_screen(hx, hy)
            pygame.draw.line(screen, col, o, h, 1)
            if d < r:
                pygame.draw.circle(screen, col, (int(h[0]), int(h[1])), 4)
        self.zoom = saved

    def draw_car(self, screen: pygame.Surface, car, x: float, y: float, theta: float, braking: bool) -> None:
        saved = self._with_view_zoom()
        z = self.zoom
        p = car.p
        if self._sprite_src is not None:
            if self._sprite_zoom != z:
                sw, sh = self._sprite_src.get_size()
                f = p.car_length_m * z / sh
                self._sprite = pygame.transform.smoothscale(self._sprite_src, (max(int(sw * f), 2), max(int(sh * f), 2)))
                self._sprite_zoom = z
            img = pygame.transform.rotate(self._sprite, math.degrees(theta) - 90.0)
            sx, sy = self.to_screen(x, y)
            screen.blit(img, img.get_rect(center=(int(sx), int(sy))))
        else:
            c, s = math.cos(theta), math.sin(theta)
            pts = [self.to_screen(x + f * c - l * s, y + f * s + l * c) for f, l in
                   (car.corner_offsets[0], car.corner_offsets[1], car.corner_offsets[3], car.corner_offsets[2])]
            pygame.draw.polygon(screen, self.c["car"], pts)
            pygame.draw.polygon(screen, self.c["car_outline"], pts, 2)
        if braking:
            c, s = math.cos(theta), math.sin(theta)
            for l in (-0.6, 0.6):
                f = -p.car_length_m / 2
                bx, by = self.to_screen(x + f * c - l * s, y + f * s + l * c)
                pygame.draw.circle(screen, (255, 40, 40), (int(bx), int(by)), max(int(0.35 * z), 2))
        self.zoom = saved

    # --------------------------------------------------------------- HUD
    def text(self, screen: pygame.Surface, s: str, pos: tuple[int, int], colour=None, big: bool = False) -> int:
        font = self.font_big if big else self.font
        img = font.render(s, True, colour or self.c["hud_text"])
        screen.blit(img, pos)
        return img.get_height()

    def panel(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        bg = pygame.Surface(rect.size, pygame.SRCALPHA)
        bg.fill(self.c["hud_bg"])
        screen.blit(bg, rect.topleft)

    def draw_lines(self, screen: pygame.Surface, lines: Sequence[tuple[str, tuple[int, ...] | None]],
                   topleft: tuple[int, int], width: int) -> None:
        lh = self.font.get_linesize()
        self.panel(screen, pygame.Rect(topleft[0] - 8, topleft[1] - 6, width, lh * len(lines) + 12))
        y = topleft[1]
        for s, col in lines:
            self.text(screen, s, (topleft[0], y), col)
            y += lh

    def wrap(self, items: Sequence[str], max_width: int, sep: str = " · ") -> list[str]:
        """Greedily join ``items`` into lines no wider than ``max_width`` pixels."""
        lines: list[str] = []
        cur = ""
        for item in items:
            cand = item if not cur else cur + sep + item
            if cur and self.font.size(cand)[0] > max_width:
                lines.append(cur)
                cur = item
            else:
                cur = cand
        if cur:
            lines.append(cur)
        return lines

    def bar(self, screen: pygame.Surface, label: str, value: float, rect: pygame.Rect, colour, centred: bool) -> None:
        """Horizontal bar; ``centred`` bars grow left/right from the middle (value in [−1, 1])."""
        self.panel(screen, rect.inflate(8, 8))
        pygame.draw.rect(screen, self.c["hud_dim"], rect, 1)
        if centred:
            mid = rect.centerx
            w = int(abs(value) * rect.w / 2)
            r = pygame.Rect(mid - w if value > 0 else mid, rect.y + 2, w, rect.h - 4)
            pygame.draw.line(screen, self.c["hud_dim"], (mid, rect.y), (mid, rect.bottom - 1))
        else:
            r = pygame.Rect(rect.x, rect.y + 2, int(max(value, 0.0) * rect.w), rect.h - 4)
        pygame.draw.rect(screen, colour, r)
        self.text(screen, label, (rect.x + 4, rect.y - self.font.get_linesize() - 2), self.c["hud_dim"])

    def banner(self, screen: pygame.Surface, s: str, colour) -> None:
        img = self.font_big.render(s, True, colour)
        rect = img.get_rect(center=(self.w // 2, self.h // 3))
        self.panel(screen, rect.inflate(40, 20))
        screen.blit(img, rect)
