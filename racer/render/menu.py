"""Mode menu (T124): the first screen of the demo. Up/Down or click to choose, Enter to open, Esc quits."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pygame

from racer.render.hud import TextCache, panel


@dataclass
class Menu:
    modes: Sequence[Any]                              # objects with .key, .title, .blurb
    size: tuple[int, int]
    fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]
    colours: Mapping[str, Any]
    index: int = 0
    done: str | None = None                           # the chosen mode's key
    quit: bool = False
    rects: list[pygame.Rect] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.text = TextCache(self.fonts[0])
        self.big = TextCache(self.fonts[1])

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
        screen.fill(c["outside"])
        title = self.big.get("RL Racing Line", c["hud_text"])
        screen.blit(title, title.get_rect(midtop=(w // 2, int(h * 0.12))))
        sub = self.text.get("a car that learns its racing line", c["hud_dim"])
        screen.blit(sub, sub.get_rect(midtop=(w // 2, int(h * 0.12) + title.get_height() + 6)))
        bw, bh, gap = min(640, w - 80), 76, 14
        x, y = (w - bw) // 2, int(h * 0.36)
        self.rects = []
        for i, m in enumerate(self.modes):
            r = pygame.Rect(x, y + i * (bh + gap), bw, bh)
            self.rects.append(r)
            panel(screen, r, (255, 255, 255, 55) if i == self.index else c["hud_bg"])
            screen.blit(self.text.get(f"{i + 1}  {m.title}", c["hud_text"]), (r.x + 18, r.y + 12))
            screen.blit(self.text.get(m.blurb, c["hud_dim"]), (r.x + 18, r.y + 12 + 28))
        hint = self.text.get("Up/Down or click · Enter opens · Esc quits", c["hud_dim"])
        screen.blit(hint, hint.get_rect(midbottom=(w // 2, h - 24)))
