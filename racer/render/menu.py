"""Mode menu (T124): the first screen of the demo. Up/Down or click to choose, Enter to open, Esc quits."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import math
import textwrap

import numpy as np
import pygame

from racer.render.hud import TextCache, panel

DOTS = ((230, 57, 70), (69, 123, 230), (42, 191, 108), (255, 183, 3))    # the arena's first palette colours


@dataclass
class Menu:
    modes: Sequence[Any]                              # objects with .key, .title, .blurb
    size: tuple[int, int]
    fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]
    colours: Mapping[str, Any]
    credits: str = ""                                 # attribution line (CC BY 4.0 engine sound needs it)
    index: int = 0
    done: str | None = None                           # the chosen mode's key
    quit: bool = False
    rects: list[pygame.Rect] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.text = TextCache(self.fonts[0])
        self.big = TextCache(self.fonts[1])
        self.small = TextCache(self.fonts[2])
        self._loop = self._make_loop()
        self._bg: pygame.Surface | None = None

    def _make_loop(self, n: int = 600) -> np.ndarray:
        """A closed track-like curve filling the window (decoration only), shape (n, 2) in pixels."""
        w, h = self.size
        th = np.linspace(0, 2 * math.pi, n, endpoint=False)
        x = 0.42 * w * (np.cos(th) + 0.22 * np.cos(3 * th + 0.6))
        y = 0.36 * h * (np.sin(th) + 0.18 * np.sin(2 * th + 1.2))
        return np.stack([w / 2 + x, h / 2 + y], axis=1)

    def _background(self, screen: pygame.Surface, now: float) -> None:
        w, h = self.size
        if self._bg is None or self._bg.get_size() != (w, h):
            bg = pygame.Surface((w, h))
            for yy in range(h):                                       # vertical gradient
                k = yy / h
                pygame.draw.line(bg, (int(18 - 8 * k), int(22 - 10 * k), int(32 - 14 * k)), (0, yy), (w, yy))
            pts = [tuple(p) for p in self._loop]
            pygame.draw.lines(bg, (24, 27, 36), True, pts, 34)       # the road
            pygame.draw.lines(bg, (44, 48, 60), True, pts, 2)
            self._bg = bg
        screen.blit(self._bg, (0, 0))
        n = len(self._loop)
        ov = pygame.Surface((w, h), pygame.SRCALPHA)
        for k, col in enumerate(DOTS):
            head = int(now * (60 + 14 * k)) % n + k * (n // len(DOTS))
            for j in range(70):                                       # a fading tail
                a = self._loop[(head - j) % n]
                b = self._loop[(head - j - 1) % n]
                pygame.draw.line(ov, (*col, int(170 * (1 - j / 70))), a, b, 3)
            pygame.draw.circle(ov, (*col, 70), self._loop[head % n], 12)
            pygame.draw.circle(ov, (*col, 255), self._loop[head % n], 5)
        screen.blit(ov, (0, 0))

    def handle(self, ev: pygame.event.Event) -> None:
        if ev.type == pygame.QUIT:
            self.quit = True
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                self.quit = True
            elif ev.key in (pygame.K_DOWN, pygame.K_s):
                self.index = (self.index + 1) % len(self.modes)
            elif ev.key in (pygame.K_UP, pygame.K_w):
                self.index = (self.index - 1) % len(self.modes)
            elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.done = self.modes[self.index].key
            elif pygame.K_1 <= ev.key <= pygame.K_9 and ev.key - pygame.K_1 < len(self.modes):
                self.done = self.modes[ev.key - pygame.K_1].key
        elif ev.type == pygame.MOUSEMOTION:
            for i, r in enumerate(self.rects):
                if r.collidepoint(ev.pos):
                    self.index = i
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for i, r in enumerate(self.rects):
                if r.collidepoint(ev.pos):
                    self.done = self.modes[i].key

    def draw(self, screen: pygame.Surface) -> None:
        c = self.colours
        w, h = self.size
        self._background(screen, pygame.time.get_ticks() / 1000.0)
        title = self.big.get("RL RACING LINE", c["hud_text"])
        screen.blit(title, title.get_rect(midtop=(w // 2, int(h * 0.10))))
        sub = self.text.get("a car that learns its racing line", c["hud_dim"])
        screen.blit(sub, sub.get_rect(midtop=(w // 2, int(h * 0.12) + title.get_height() + 6)))
        bw, bh, gap = min(640, w - 80), 76, 14
        x, y = (w - bw) // 2, int(h * 0.36)
        self.rects = []
        for i, m in enumerate(self.modes):
            r = pygame.Rect(x, y + i * (bh + gap), bw, bh)
            self.rects.append(r)
            panel(screen, r, (255, 255, 255, 60) if i == self.index else (10, 12, 16, 200))
            pygame.draw.rect(screen, DOTS[i % len(DOTS)] if i == self.index else (60, 64, 74), (r.x, r.y, 5, r.h))
            screen.blit(self.text.get(f"{i + 1}  {m.title}", c["hud_text"]), (r.x + 18, r.y + 12))
            screen.blit(self.text.get(m.blurb, c["hud_dim"]), (r.x + 18, r.y + 12 + 28))
        lines = textwrap.wrap(self.credits, 150) if self.credits else []
        for k, line in enumerate(reversed(lines)):                   # attribution, wrapped, at the very bottom
            small = self.small.get(line, c["hud_dim"])
            screen.blit(small, small.get_rect(midbottom=(w // 2, h - 6 - k * (self.fonts[2].get_linesize() + 1))))
        hint = self.text.get("Up/Down or click · Enter opens · Esc quits", c["hud_dim"])
        screen.blit(hint, hint.get_rect(midbottom=(w // 2, h - 8 - (len(lines) + 1) * (self.fonts[2].get_linesize() + 1))))
