"""Car sprites for many cars: Kenney tintable bodies in distinct colours (assets/LICENSES.md).

Car ``i`` gets shape ``i % 4 + 1`` and colour ``palette[i]``. Tinting multiplies
``car_N_tint.png`` by the colour and pastes the result back through the mask's alpha, so
only the body changes colour. A missing asset falls back to a coloured rectangle.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pygame

from racer.config import PROJECT_ROOT
from racer.render.camera import Camera

CARS_DIR = PROJECT_ROOT / "assets" / "sprites" / "cars"


def tint(base: pygame.Surface, mask: pygame.Surface, colour: Sequence[int]) -> pygame.Surface:
    """Multiply the masked body of ``base`` by ``colour`` (RGB 0–255)."""
    out = base.copy()
    rgb = pygame.surfarray.pixels3d(out)
    a = pygame.surfarray.pixels_alpha(mask).astype(np.float32)[..., None] / 255.0
    col = np.asarray(colour, dtype=np.float32) / 255.0
    rgb[:] = (rgb * (1.0 - a) + rgb * col * a).astype(np.uint8)
    del rgb
    return out


class CarSprites:
    def __init__(self, palette: Sequence[Sequence[int]], car_length_m: float, car_width_m: float) -> None:
        self.palette = [tuple(int(v) for v in c) for c in palette]
        self.length, self.width = car_length_m, car_width_m
        self._src: dict[int, pygame.Surface | None] = {}
        self._scaled: dict[tuple[int, int], pygame.Surface] = {}
        self._shadows: dict[tuple[int, int], pygame.Surface] = {}
        self.shadow_m = 0.45         # drop shadow offset (down-right on screen)
        self.shadow_alpha = 85

    def colour(self, i: int) -> tuple[int, int, int]:
        return self.palette[i % len(self.palette)]

    def _source(self, i: int) -> pygame.Surface | None:
        if i not in self._src:
            shape = i % 4 + 1
            try:
                base = pygame.image.load(str(CARS_DIR / f"car_{shape}_tint.png")).convert_alpha()
                mask = pygame.image.load(str(CARS_DIR / f"car_{shape}_tint_mask.png")).convert_alpha()
                self._src[i] = tint(base, mask, self.colour(i))
            except (FileNotFoundError, OSError, pygame.error, ValueError):
                self._src[i] = None          # never crash on a missing asset
        return self._src[i]

    def _sprite(self, i: int, zoom: float) -> pygame.Surface | None:
        key = (i, int(round(zoom * 20)))
        if key not in self._scaled:
            src = self._source(i)
            if src is None:
                return None
            sw, sh = src.get_size()
            f = self.length * zoom / sh
            self._scaled[key] = pygame.transform.smoothscale(src, (max(int(sw * f), 2), max(int(sh * f), 2)))
            if len(self._scaled) > 400:
                self._scaled.clear()
        return self._scaled[key]

    def _shadow(self, i: int, zoom: float, spr: pygame.Surface) -> pygame.Surface:
        key = (i, int(round(zoom * 20)))
        sh = self._shadows.get(key)
        if sh is None:
            sh = spr.copy()
            sh.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MIN)       # black silhouette
            sh.set_alpha(self.shadow_alpha)
            if len(self._shadows) > 400:
                self._shadows.clear()
            self._shadows[key] = sh
        return sh

    def draw(self, screen: pygame.Surface, cam: Camera, i: int, x: float, y: float, theta: float,
             braking: bool = False, ghost: bool = False) -> None:
        z = cam.view_zoom
        sx, sy = cam.to_screen(x, y)
        if sx < -50 or sy < -50 or sx > cam.w + 50 or sy > cam.h + 50:
            return
        spr = self._sprite(i, z)
        if spr is not None:
            img = pygame.transform.rotate(spr, math.degrees(theta) - 90.0)
            if ghost:
                img.set_alpha(90)
            else:
                shadow = pygame.transform.rotate(self._shadow(i, z, spr), math.degrees(theta) - 90.0)
                off = int(self.shadow_m * z)
                screen.blit(shadow, shadow.get_rect(center=(int(sx) + off, int(sy) + off)))
            screen.blit(img, img.get_rect(center=(int(sx), int(sy))))
        else:
            c, s = math.cos(theta), math.sin(theta)
            hl, hw = self.length / 2, self.width / 2
            pts = [cam.to_screen(x + f * c - l * s, y + f * s + l * c) for f, l in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
            pygame.draw.polygon(screen, self.colour(i), pts)
        if braking and not ghost:
            c, s = math.cos(theta), math.sin(theta)
            for l in (-0.6, 0.6):
                f = -self.length / 2
                bx, by = cam.to_screen(x + f * c - l * s, y + f * s + l * c)
                pygame.draw.circle(screen, (255, 40, 40), (int(bx), int(by)), max(int(0.35 * z), 2))
