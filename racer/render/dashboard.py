"""Dashboard for the followed car (T131): speed and cosmetic rpm gauge, gear, pedals, steering, lap times.

Bottom left of the race window. The rpm and gear come from the cosmetic gearbox
(:mod:`racer.render.gearbox`), the same one the engine sound uses, so gauge and sound agree.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pygame

from racer.env.race import Race, RaceCar
from racer.render.hud import TextCache, fmt_time, panel


def arc_points(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 40) -> list[tuple[float, float]]:
    """Points along an arc from ``a0`` to ``a1`` degrees (0° = right, counter-clockwise on screen)."""
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy - r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


class Dashboard:
    SWEEP = (225.0, -45.0)                           # gauge from bottom-left over the top to bottom-right

    def __init__(self, fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font],
                 colours: Mapping[str, Any], rpm_range: tuple[float, float] = (1.0, 2.9), size=(470, 150)) -> None:
        self.font, self.big, self.small = fonts
        self.c = colours
        self.size = size
        self.swarm = False                       # a swarm has no laps: show distance instead
        self.rpm0, self.rpm1 = rpm_range
        self.t = TextCache(self.font)
        self.tb = TextCache(self.big)
        self.ts = TextCache(self.small)

    def draw(self, screen: pygame.Surface, race: Race, car: RaceCar, colour: Sequence[int], rpm: float, gear: int) -> pygame.Rect:
        c = self.c
        w, h = self.size
        rect = pygame.Rect(12, screen.get_height() - h - 12, w, h)
        panel(screen, rect, c["hud_bg"])
        phys = car.env.car
        # gauge
        cx, cy, r = rect.x + 78, rect.y + 82, 56
        a0, a1 = self.SWEEP
        frac = max(min((rpm - self.rpm0) / max(self.rpm1 - self.rpm0, 1e-6), 1.0), 0.0)
        pygame.draw.lines(screen, (52, 56, 66), False, arc_points(cx, cy, r, a0, a1), 7)
        red = 0.86
        pygame.draw.lines(screen, (150, 55, 55), False, arc_points(cx, cy, r, a0 + (a1 - a0) * red, a1, 8), 7)
        if frac > 0.01:
            col = c["hud_bad"] if frac > red else (255, 170, 50)
            pygame.draw.lines(screen, col, False, arc_points(cx, cy, r, a0, a0 + (a1 - a0) * frac, max(int(40 * frac), 2)), 7)
        kmh = self.tb.get(f"{phys.speed * 3.6:.0f}", c["hud_text"])
        screen.blit(kmh, kmh.get_rect(center=(cx, cy - 6)))
        screen.blit(self.ts.get("km/h", c["hud_dim"]), self.ts.get("km/h", c["hud_dim"]).get_rect(center=(cx, cy + 20)))
        g = self.tb.get("N" if phys.speed < 0.5 and gear <= 1 and rpm < 1.15 else str(gear), c["hud_good"])
        screen.blit(g, g.get_rect(center=(cx, cy + 42)))
        # text
        x = rect.x + 160
        last = car.lap_times[-1] if car.lap_times else None
        best = min(car.lap_times) if car.lap_times else None
        state = {"respawn": f"CRASHED ({(car.last_crash or '').replace('_', ' ')})", "finished": "FINISHED"}.get(car.state, "")
        screen.blit(self.t.get(car.label, colour), (x, rect.y + 6))
        lh = self.font.get_linesize()
        rows = [f"lap {race.lap_of(car)}/{race.rules.laps}   {fmt_time(race.current_lap_time(car))}",
                f"last {fmt_time(last)}   best {fmt_time(best)}",
                f"lateral {phys.a_lat / 9.81:+.1f} g   crashes {car.crashes}"]
        if self.swarm:
            rows = [f"driven {car.dist / 1000:.2f} km", f"time {fmt_time(max(race.t, 0.0))}", rows[2]]
        if state:
            rows[2] = state
        for k, s in enumerate(rows):
            col = c["hud_text"] if k < 2 else (c["hud_bad"] if car.state == "respawn" else c["hud_good"]) if state else c["hud_dim"]
            screen.blit(self.t.get(s, col), (x, rect.y + 6 + lh * (k + 1)))
        # pedals and steering
        by = rect.bottom - 18
        self._bar(screen, pygame.Rect(x, by, 70, 8), phys.throttle, c["gas_bar"], "gas")
        self._bar(screen, pygame.Rect(x + 90, by, 70, 8), phys.brake, c["brake_bar"], "brake")
        steer = max(min(phys.steer / math.radians(30.0), 1.0), -1.0)      # front-wheel angle, 30° = full lock at low speed
        self._bar(screen, pygame.Rect(x + 180, by, 100, 8), steer, c["steer_bar"], "steer", centred=True)
        return rect

    def _bar(self, screen: pygame.Surface, r: pygame.Rect, v: float, col, label: str, centred: bool = False) -> None:
        pygame.draw.rect(screen, (40, 44, 52), r)
        if centred:
            w = int(abs(v) * r.w / 2)
            pygame.draw.rect(screen, col, (r.centerx if v >= 0 else r.centerx - w, r.y, w, r.h))
            pygame.draw.line(screen, (130, 135, 145), (r.centerx, r.y - 2), (r.centerx, r.bottom + 2))
        else:
            pygame.draw.rect(screen, col, (r.x, r.y, int(max(min(v, 1.0), 0.0) * r.w), r.h))
        screen.blit(self.ts.get(label, self.c["hud_dim"]), (r.x, r.y - self.small.get_linesize()))
