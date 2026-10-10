"""Race HUD for the Model Arena: leaderboard, countdown, results and the followed car's readout."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import pygame

from racer.env.race import Race, RaceCar


def fmt_time(t: float | None) -> str:
    if t is None:
        return "-"
    m, s = divmod(t, 60.0)
    return f"{int(m)}:{s:06.3f}" if m >= 1 else f"{s:.3f}"


def short_label(label: str) -> str:
    """Compact car label: ``R002_curric_wide@1500k`` → ``R002@1500k`` (the run id identifies the run)."""
    m = re.match(r"^([RS]\d{3})_.*@(\d+k)$", label)
    if m:
        return f"{m.group(1)}@{m.group(2)}"
    return label if len(label) <= 16 else label[:15] + "…"


class TextCache:
    """Rendered text surfaces, reused while the same string is drawn every frame."""

    def __init__(self, font: pygame.font.Font, limit: int = 2000) -> None:
        self.font = font
        self.limit = limit
        self._cache: dict[tuple[str, tuple[int, ...]], pygame.Surface] = {}

    def get(self, s: str, colour: Sequence[int]) -> pygame.Surface:
        key = (s, tuple(colour))
        img = self._cache.get(key)
        if img is None:
            if len(self._cache) > self.limit:
                self._cache.clear()
            img = self._cache[key] = self.font.render(s, True, colour)
        return img


def panel(screen: pygame.Surface, rect: pygame.Rect, colour: Sequence[int]) -> None:
    bg = pygame.Surface(rect.size, pygame.SRCALPHA)
    bg.fill(colour)
    screen.blit(bg, rect.topleft)


class RaceHud:
    def __init__(self, font: pygame.font.Font, font_big: pygame.font.Font, colours: Mapping[str, Any],
                 width: int) -> None:
        self.text = TextCache(font)
        self.big = TextCache(font_big)
        self.font = font
        self.c = colours
        self.width = width

    def leaderboard(self, screen: pygame.Surface, race: Race, colour_of, selected: int,
                    title: str) -> list[tuple[pygame.Rect, int]]:
        """Draw the standings on the right; returns (row rect, car index) for mouse picking."""
        lh = self.font.get_linesize() + 2
        cars = race.standings()
        x0 = screen.get_width() - self.width - 12
        rect = pygame.Rect(x0, 12, self.width, lh * (len(cars) + 2) + 12)
        panel(screen, rect, self.c["hud_bg"])
        t = self.text
        laps = race.rules.laps
        clock = fmt_time(max(race.t, 0.0)) if race.started else f"start in {-race.t:.0f}"
        screen.blit(t.get(f"{title}   {clock}", self.c["hud_text"]), (x0 + 10, 18))
        y = 18 + lh
        cols = (x0 + 10, x0 + 36, x0 + 54, x0 + 190, x0 + 245, x0 + 330, x0 + 410)
        for s, cx in zip(("P", "", "car", "lap", "gap", "best", "crash"), cols):
            screen.blit(t.get(s, self.c["hud_dim"]), (cx, y))
        y += lh
        rows = []
        for pos, car in enumerate(cars, 1):
            row = pygame.Rect(x0 + 4, y - 1, self.width - 8, lh)
            if car.index == selected:
                panel(screen, row, (255, 255, 255, 40))
            rows.append((row, car.index))
            pygame.draw.rect(screen, colour_of(car.index), (cols[1], y + 3, 12, lh - 8))
            done = car.finished_t is not None
            gap = race.gap(car)
            gap_s = "FIN" if done and gap == 0 else ("-" if gap is None else ("+" + fmt_time(gap) if gap > 0.0005 else "lead"))
            best = min(car.lap_times) if car.lap_times else None
            dim = self.c["hud_dim"] if car.state == "respawn" else self.c["hud_text"]
            label = short_label(car.label)
            for s, cx in zip((str(pos), label, f"{race.lap_of(car)}/{laps}" if not done else "done", gap_s,
                              fmt_time(best), str(car.crashes) if car.crashes else ""),
                             (cols[0], cols[2], cols[3], cols[4], cols[5], cols[6])):
                screen.blit(t.get(s, dim), (cx, y))
            y += lh
        return rows

    def car_readout(self, screen: pygame.Surface, race: Race, car: RaceCar, colour) -> None:
        """Followed car: speed, lap, lap times (bottom left)."""
        lh = self.font.get_linesize()
        env = car.env
        last = car.lap_times[-1] if car.lap_times else None
        best = min(car.lap_times) if car.lap_times else None
        state = {"respawn": f"CRASHED ({(car.last_crash or '').replace('_', ' ')})", "finished": "FINISHED"}.get(car.state, "")
        lines = [
            (car.label, colour),
            (f"{env.car.speed * 3.6:5.0f} km/h   lat {env.car.a_lat / 9.81:+4.1f} g   lap {race.lap_of(car)}/{race.rules.laps}",
             self.c["hud_text"]),
            (f"lap {fmt_time(race.current_lap_time(car))}   last {fmt_time(last)}   best {fmt_time(best)}",
             self.c["hud_text"]),
            (state or f"crashes {car.crashes}", self.c["hud_bad"] if car.state == "respawn" else self.c["hud_dim"]),
        ]
        h = lh * len(lines) + 12
        top = screen.get_height() - h - 12
        panel(screen, pygame.Rect(12, top, 460, h), self.c["hud_bg"])
        for k, (s, col) in enumerate(lines):
            screen.blit(self.text.get(s, col), (20, top + 6 + k * lh))

    def countdown(self, screen: pygame.Surface, race: Race) -> None:
        if race.started:
            if race.t < 1.0:
                self._centre(screen, "GO!", self.c["hud_good"])
            return
        self._centre(screen, str(int(-race.t) + 1), self.c["hud_text"])

    def results(self, screen: pygame.Surface, race: Race, colour_of) -> None:
        cars = race.standings()
        lh = self.font.get_linesize() + 4
        w = 620
        h = lh * (len(cars) + 3)
        x0, y0 = (screen.get_width() - w) // 2, max((screen.get_height() - h) // 2, 10)
        panel(screen, pygame.Rect(x0, y0, w, h), (10, 12, 16, 220))
        t = self.text
        screen.blit(t.get(f"RESULTS — {race.rules.laps} laps   (Esc: back)", self.c["hud_text"]), (x0 + 16, y0 + 8))
        y = y0 + 8 + lh * 2
        for pos, car in enumerate(cars, 1):
            pygame.draw.rect(screen, colour_of(car.index), (x0 + 46, y + 3, 12, lh - 10))
            total = fmt_time(car.finished_t) if car.finished_t is not None else "DNF"
            best = fmt_time(min(car.lap_times)) if car.lap_times else "-"
            for s, cx in ((str(pos), x0 + 16), (car.label, x0 + 66), (total, x0 + 350), (f"best {best}", x0 + 440),
                          (f"{car.crashes} crash" if car.crashes else "", x0 + 548)):
                screen.blit(t.get(s, self.c["hud_text"]), (cx, y))
            y += lh

    def _centre(self, screen: pygame.Surface, s: str, colour) -> None:
        img = self.big.get(s, colour)
        rect = img.get_rect(center=(screen.get_width() // 2, screen.get_height() // 3))
        panel(screen, rect.inflate(60, 24), self.c["hud_bg"])
        screen.blit(img, rect)
