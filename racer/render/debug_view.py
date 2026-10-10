"""Minimal debug renderer (pygame-ce). It only *draws* simulation objects; it never steps them.

The track layer (``world.py``) is rasterised once from the same field the physics uses (D3),
so road, kerbs, grass, run-off walls and the barrier on screen are exactly what the car
feels. The camera lives in ``camera.py``; both are shared with the Model Arena.
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
from racer.render.camera import Camera
from racer.render.world import TrackWorld

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
    """Debug window drawing helpers: a :class:`TrackWorld` seen through a :class:`Camera`, plus HUD bits."""

    def __init__(self, size: tuple[int, int], cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.c = {k: _col(v) for k, v in cfg["colours"].items()}
        self.w, self.h = size
        self.cam = Camera(size, cfg)
        self.worldr = TrackWorld(cfg, self.c)
        self.font = load_font(cfg["font"], int(cfg["font_size"]))
        self.font_big = load_font(cfg["font"], int(cfg["font_size"] * 2))
        self.speed_lut = speed_colours()
        self._sprite_src: pygame.Surface | None = None
        try:
            self._sprite_src = pygame.image.load(str(ASSETS / "sprites" / "cars" / cfg["car_sprite"])).convert_alpha()
        except (FileNotFoundError, OSError, pygame.error):
            self._sprite_src = None
        self._sprite_zoom = None
        self._sprite: pygame.Surface | None = None

    # ------------------------------------------------------------- world
    @property
    def track(self):
        return self.worldr.track

    def set_track(self, track) -> None:
        """Pre-render the whole track to one surface."""
        self.worldr.set_track(track)

    def draw_world(self, screen: pygame.Surface) -> None:
        self.worldr.draw(screen, self.cam)

    # ------------------------------------------------------------ camera
    @property
    def overview(self) -> bool:
        return self.cam.overview

    @overview.setter
    def overview(self, on: bool) -> None:
        self.cam.overview = on

    @property
    def zoom(self) -> float:
        return self.cam.zoom

    @property
    def view_zoom(self) -> float:
        return self.cam.view_zoom

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return self.cam.to_screen(x, y)

    def zoom_by(self, factor: float) -> None:
        self.cam.zoom_by(factor)

    def update_camera(self, x: float, y: float, vx: float, vy: float, sim_speed: float = 1.0) -> None:
        """Follow the car (per simulated time), or fit the whole track in overview."""
        self.cam.update(x, y, vx, vy, sim_speed, self.track.field.extent if self.track else None)

    def snap_camera(self, x: float, y: float) -> None:
        self.cam.snap(x, y)

    # ------------------------------------------------------------ drawing
    def draw_trail(self, screen: pygame.Surface, trail: Iterable[tuple[float, float, float] | None]) -> None:
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

    def draw_rays(self, screen: pygame.Surface, x: float, y: float, theta: float,
                  angles: np.ndarray, dists: np.ndarray, ranges: np.ndarray) -> None:
        o = self.to_screen(x, y)
        for a, d, r in zip(angles, dists, ranges):
            frac = float(d / r)
            col = (min(255, int(510 * (1 - frac))), min(255, int(510 * frac)), 60)
            hx, hy = x + d * math.cos(theta + a), y + d * math.sin(theta + a)
            h = self.to_screen(hx, hy)
            pygame.draw.line(screen, col, o, h, 1)
            if d < r:
                pygame.draw.circle(screen, col, (int(h[0]), int(h[1])), 4)

    def draw_car(self, screen: pygame.Surface, car, x: float, y: float, theta: float, braking: bool) -> None:
        z = self.view_zoom
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

    def badge(self, screen: pygame.Surface, s: str, topleft: tuple[int, int], colour) -> None:
        """Small outlined status label, e.g. "ABS ON"."""
        img = self.font.render(s, True, colour)
        rect = img.get_rect(topleft=(topleft[0] + 8, topleft[1] + 4))
        self.panel(screen, rect.inflate(16, 8))
        pygame.draw.rect(screen, colour, rect.inflate(16, 8), 2, border_radius=4)
        screen.blit(img, rect)

    def banner(self, screen: pygame.Surface, s: str, colour) -> None:
        img = self.font_big.render(s, True, colour)
        rect = img.get_rect(center=(self.w // 2, self.h // 3))
        self.panel(screen, rect.inflate(40, 20))
        screen.blit(img, rect)
