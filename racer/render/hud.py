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

    def swarm_board(self, screen: pygame.Surface, race: Race, colour_of, selected: int, title: str,
                    max_rows: int = 10) -> list[tuple[pygame.Rect, int]]:
        """Standings of a swarm: distance driven, speed and crashes (laps and gaps mean nothing for scattered cars)."""
        lh = self.font.get_linesize() + 2
        cars = sorted(race.cars, key=lambda c: -c.dist)
        alive = sum(1 for c in race.cars if c.state == "racing")
        x0 = screen.get_width() - self.width - 12
        shown = cars[:max_rows]
        rect = pygame.Rect(x0, 12, self.width, lh * (len(shown) + 2) + 12)
        panel(screen, rect, self.c["hud_bg"])
        t = self.text
        screen.blit(t.get(f"{title}   {alive}/{len(race.cars)} driving   {fmt_time(max(race.t, 0.0))}", self.c["hud_text"]),
                    (x0 + 10, 18))
        y = 18 + lh
        cols = (x0 + 10, x0 + 36, x0 + 54, x0 + 250, x0 + 330, x0 + 410)
        for s, cx in zip(("P", "", "car", "km", "km/h", "crash"), cols):
            screen.blit(t.get(s, self.c["hud_dim"]), (cx, y))
        y += lh
        rows = []
        for pos, car in enumerate(shown, 1):
            row = pygame.Rect(x0 + 4, y - 1, self.width - 8, lh)
            if car.index == selected:
                panel(screen, row, (255, 255, 255, 40))
            rows.append((row, car.index))
            pygame.draw.rect(screen, colour_of(car.index), (cols[1], y + 3, 12, lh - 8))
            dim = self.c["hud_dim"] if car.state == "respawn" else self.c["hud_text"]
            for s, cx in ((str(pos), cols[0]), (car.label.rsplit("#", 1)[-1].join(["#", ""]), cols[2]),
                          (f"{car.dist / 1000:.2f}", cols[3]), (f"{car.env.car.speed * 3.6:.0f}", cols[4]),
                          (str(car.crashes) if car.crashes else "", cols[5])):
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
        """Start lights: one more red light each second, all green at the start."""
        if race.started and race.t >= 1.0:
            return
        n = 3
        left = int(-race.t) + 1 if not race.started else 0           # seconds to go (3, 2, 1) or 0
        lit = n - left + 1 if left else n
        w, r, gap = 3 * 54 + 28, 20, 54
        bar = pygame.Rect(0, 0, w, 64)
        bar.midtop = (screen.get_width() // 2, 14)
        pygame.draw.rect(screen, (14, 16, 20), bar, border_radius=12)
        pygame.draw.rect(screen, (70, 74, 84), bar, 2, border_radius=12)
        for i in range(n):
            on = i < lit if left else True
            col = (235, 50, 45) if left else (70, 225, 110)
            colour = col if on else (48, 28, 28)
            centre = (bar.x + 14 + r + i * gap + 6, bar.centery)
            if on:
                glow = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
                pygame.draw.circle(glow, (*col, 60), (r * 2, r * 2), r * 2)
                screen.blit(glow, glow.get_rect(center=centre))
            pygame.draw.circle(screen, colour, centre, r)
        if not left:
            self._centre(screen, "GO!", self.c["hud_good"])

    def results(self, screen: pygame.Surface, race: Race, colour_of) -> None:
        cars = race.standings()
        lh = self.font.get_linesize() + 4
        w = 720
        title_h = self.big.font.get_linesize()
        h = lh * (len(cars) + 2) + title_h + 24
        x0, y0 = (screen.get_width() - w) // 2, max((screen.get_height() - h) // 2, 10)
        panel(screen, screen.get_rect(), (0, 0, 0, 140))                       # dim everything behind
        panel(screen, pygame.Rect(x0, y0, w, h), (10, 12, 16, 245))
        t = self.text
        screen.blit(self.big.get("RESULTS", self.c["hud_text"]), (x0 + 20, y0 + 10))
        lapw = "lap" if race.rules.laps == 1 else "laps"
        screen.blit(t.get(f"{race.rules.laps} {lapw}   ·   Enter: race again   ·   Esc: back", self.c["hud_dim"]),
                    (x0 + 20, y0 + 10 + title_h + 2))
        y = y0 + 10 + title_h + 2 + lh * 2
        fastest = min((min(c.lap_times) for c in cars if c.lap_times), default=None)
        medals = ((255, 205, 60), (200, 205, 215), (205, 135, 70))
        for pos, car in enumerate(cars, 1):
            if pos <= 3 and car.finished_t is not None:
                pygame.draw.circle(screen, medals[pos - 1], (x0 + 26, y + lh // 2 - 2), 9)
            pygame.draw.rect(screen, colour_of(car.index), (x0 + 46, y + 3, 12, lh - 10))
            total = fmt_time(car.finished_t) if car.finished_t is not None else "DNF"
            best = fmt_time(min(car.lap_times)) if car.lap_times else "-"
            purple = fastest is not None and car.lap_times and min(car.lap_times) == fastest
            for s, cx, col in ((str(pos), x0 + 16, self.c["hud_text"]), (car.label, x0 + 66, self.c["hud_text"]),
                               (total, x0 + 380, self.c["hud_text"]),
                               (f"best {best}", x0 + 480, (200, 130, 255) if purple else self.c["hud_text"]),
                               (f"{car.crashes} crash" if car.crashes else "", x0 + 620, self.c["hud_dim"])):
                screen.blit(t.get(s, col), (cx, y))
            y += lh

    def keys(self, screen: pygame.Surface, items: Sequence[tuple[str, str]], shown: bool) -> None:
        """Key list (bottom right): ``(key, action)`` rows, or just a hint when hidden (T82)."""
        lh = self.font.get_linesize()
        t = self.text
        if not shown:
            hint = t.get("H  keys", self.c["hud_dim"])
            r = hint.get_rect(bottomright=(screen.get_width() - 16, screen.get_height() - 12))
            panel(screen, r.inflate(16, 8), self.c["hud_bg"])
            screen.blit(hint, r)
            return
        key_w = max(t.get(k, self.c["hud_text"]).get_width() for k, _ in items) + 16
        act_w = max(t.get(a, self.c["hud_text"]).get_width() for _, a in items)
        w, h = key_w + act_w + 20, lh * (len(items) + 1) + 12
        x0, y0 = screen.get_width() - w - 12, screen.get_height() - h - 12
        panel(screen, pygame.Rect(x0, y0, w, h), self.c["hud_bg"])
        screen.blit(t.get("KEYS   (H hides)", self.c["hud_dim"]), (x0 + 10, y0 + 6))
        for k, (key, action) in enumerate(items, 1):
            y = y0 + 6 + k * lh
            screen.blit(t.get(key, self.c["hud_good"]), (x0 + 10, y))
            screen.blit(t.get(action, self.c["hud_text"]), (x0 + 10 + key_w, y))

    def _centre(self, screen: pygame.Surface, s: str, colour) -> None:
        img = self.big.get(s, colour)
        rect = img.get_rect(center=(screen.get_width() // 2, screen.get_height() // 3))
        panel(screen, rect.inflate(60, 24), self.c["hud_bg"])
        screen.blit(img, rect)
