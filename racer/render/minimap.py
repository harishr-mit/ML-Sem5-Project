"""Minimap (T131, PROJECT_CONTEXT §10.2): the track outline with a dot per car, in a corner."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pygame

from racer.render.hud import panel


class Minimap:
    def __init__(self, colours: Mapping[str, Any], size: tuple[int, int] = (250, 170), margin: int = 12) -> None:
        self.c = colours
        self.size = size
        self.margin = margin
        self._surf: pygame.Surface | None = None
        self._xf: tuple[float, float, float, float] | None = None     # scale, x offset, y offset, height
        self._track_key: Any = None

    def set_track(self, track) -> None:
        """Pre-render the outline of ``track`` (call again when the track changes)."""
        pts = np.asarray(track.centreline.points, dtype=np.float64)
        w, h = self.size
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        inner_w, inner_h = w - 2 * self.margin, h - 2 * self.margin
        s = min(inner_w / max(x1 - x0, 1e-6), inner_h / max(y1 - y0, 1e-6))
        ox = self.margin + (inner_w - (x1 - x0) * s) / 2 - x0 * s
        oy = self.margin + (inner_h - (y1 - y0) * s) / 2 + y1 * s        # y is flipped on screen
        self._xf = (s, ox, oy, h)
        surf = pygame.Surface(self.size, pygame.SRCALPHA)
        poly = [self.to_map(x, y) for x, y in pts]
        pygame.draw.lines(surf, (30, 32, 38), True, poly, 7)
        pygame.draw.lines(surf, (170, 172, 182), True, poly, 3)
        a, b = self.to_map(*pts[0]), self.to_map(*pts[1])
        pygame.draw.circle(surf, (240, 240, 240), (int(a[0]), int(a[1])), 4, 1)           # start / finish
        self._surf = surf
        self._track_key = getattr(track, "key", id(track))

    @property
    def track_key(self):
        return self._track_key

    def to_map(self, x: float, y: float) -> tuple[float, float]:
        """World metres → pixel inside the minimap."""
        s, ox, oy, _ = self._xf
        return x * s + ox, oy - y * s

    def draw(self, screen: pygame.Surface, pos: tuple[int, int], cars: Sequence[Any], selected: int,
             colour_of: Callable[[int], Sequence[int]]) -> pygame.Rect:
        rect = pygame.Rect(pos, self.size)
        panel(screen, rect, self.c["hud_bg"])
        if self._surf is None:
            return rect
        screen.blit(self._surf, rect.topleft)
        for car in sorted(cars, key=lambda k: k.index == selected):
            x, y, _ = car.pose
            mx, my = self.to_map(x, y)
            p = (int(rect.x + mx), int(rect.y + my))
            if car.index == selected:
                pygame.draw.circle(screen, (255, 255, 255), p, 6, 2)
            pygame.draw.circle(screen, colour_of(car.index), p, 4 if car.index == selected else 3)
        return rect
