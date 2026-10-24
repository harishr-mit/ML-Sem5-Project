"""Evolution Race (T126, PROJECT_CONTEXT §10.1): the checkpoints of one run race together.

The picker lists the runs that have at least two raceable checkpoints; the strategy (every N-th,
evenly spaced, top by score) preselects checkpoints and a click on a checkpoint adds or removes
it. The cars are coloured from cool (early) to warm (late) so the race reads as training time.

Keys: Up/Down run · Left/Right strategy · +/− how many · click a checkpoint to (de)select ·
click a track · [ ] laps · Enter race · Esc back.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pygame

from racer.agents.checkpoints import STRATEGIES, choose, run_checkpoints
from racer.render.hud import TextCache, panel
from racer.render.ui import Selection


def gradient_palette(n: int) -> list[tuple[int, int, int]]:
    """``n`` colours from blue (first) to red (last), for cars ordered by training time."""
    n = max(n, 1)
    out = []
    for i in range(n):
        hue = (240.0 - 240.0 * (i / (n - 1 if n > 1 else 1))) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def raceable_runs(rows: Sequence[Mapping[str, Any]], blocked: Mapping[str, str]) -> dict[str, list[dict[str, Any]]]:
    """run id → its raceable checkpoints, for runs with at least two of them (newest run first)."""
    ids = sorted({r["run_id"] for r in rows if r.get("path")}, reverse=True)
    out = {}
    for rid in ids:
        cps = [r for r in run_checkpoints(rows, rid) if r["model_id"] not in blocked]
        if len(cps) >= 2:
            out[rid] = cps
    return out


@dataclass
class EvoPicker:
    rows: list[dict[str, Any]]
    tracks: list[tuple[str, str]]
    size: tuple[int, int]
    font: pygame.font.Font
    colours: Mapping[str, Any]
    max_cars: int = 20
    laps: int = 3
    track: str = "training_ground"
    blocked: Mapping[str, str] = field(default_factory=dict)
    strategy: str = "spaced"
    count: int = 6
    run_i: int = 0
    selected: list[str] = field(default_factory=list)
    done: Selection | None = None
    quit: bool = False
    scroll: int = 0

    def __post_init__(self) -> None:
        self.text = TextCache(self.font)
        self.lh = self.font.get_linesize() + 4
        self.runs = raceable_runs(self.rows, self.blocked)
        self.run_ids = list(self.runs)
        self._hit: list[tuple[pygame.Rect, str, Any]] = []
        self.apply_strategy()

    @property
    def run_id(self) -> str | None:
        return self.run_ids[self.run_i] if self.run_ids else None

    def checkpoints(self) -> list[dict[str, Any]]:
        return self.runs[self.run_id] if self.run_id else []

    def apply_strategy(self) -> None:
        picked = choose(self.checkpoints(), self.strategy, self.count) if self.run_id else []
        self.selected = [r["model_id"] for r in picked][: self.max_cars]
        self.scroll = 0

    def toggle(self, model_id: str) -> None:
        if model_id in self.selected:
            self.selected.remove(model_id)
        elif len(self.selected) < self.max_cars:
            order = {r["model_id"]: i for i, r in enumerate(self.checkpoints())}
            self.selected = sorted(self.selected + [model_id], key=lambda m: order[m])

    def start(self) -> None:
        if len(self.selected) >= 2:
            self.done = Selection(self.track, list(self.selected), False, self.laps, False, gradient=True)

    def handle(self, ev: pygame.event.Event) -> None:
        if ev.type == pygame.QUIT:
            self.quit = True
        elif ev.type == pygame.MOUSEWHEEL:
            self.scroll = max(self.scroll - ev.y * 3, 0)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for rect, kind, value in self._hit:
                if rect.collidepoint(ev.pos):
                    if kind == "track":
                        self.track = value
                    elif kind == "run":
                        self.run_i = value
                        self.apply_strategy()
                    elif kind == "cp":
                        self.toggle(value)
                    elif kind == "start":
                        self.start()
                    break
        elif ev.type == pygame.KEYDOWN:
            k = ev.key
            names = list(STRATEGIES)
            if k == pygame.K_ESCAPE:
                self.quit = True
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.start()
            elif k == pygame.K_DOWN and self.run_ids:
                self.run_i = (self.run_i + 1) % len(self.run_ids)
                self.apply_strategy()
            elif k == pygame.K_UP and self.run_ids:
                self.run_i = (self.run_i - 1) % len(self.run_ids)
                self.apply_strategy()
            elif k == pygame.K_RIGHT:
                self.strategy = names[(names.index(self.strategy) + 1) % len(names)]
                self.apply_strategy()
            elif k == pygame.K_LEFT:
                self.strategy = names[(names.index(self.strategy) - 1) % len(names)]
                self.apply_strategy()
            elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.count = min(self.count + 1, self.max_cars)
                self.apply_strategy()
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.count = max(self.count - 1, 1)
                self.apply_strategy()
            elif k == pygame.K_RIGHTBRACKET:
                self.laps = min(self.laps + 1, 20)
            elif k == pygame.K_LEFTBRACKET:
                self.laps = max(self.laps - 1, 1)

    def draw(self, screen: pygame.Surface) -> None:
        c, t, lh = self.colours, self.text, self.lh
        w, h = self.size
        screen.fill(c["outside"])
        self._hit = []
        screen.blit(t.get("EVOLUTION RACE - the checkpoints of one run race together", c["hud_text"]), (20, 14))
        screen.blit(t.get("Up/Down: run · Left/Right: strategy · +/−: how many · click: add/remove a checkpoint · "
                          "[ ]: laps · Enter: race · Esc: back", c["hud_dim"]), (20, 14 + lh))
        top = 20 + lh * 3
        # runs
        x = 20
        panel(screen, pygame.Rect(x - 8, top - 8, 330, h - top - 60), c["hud_bg"])
        screen.blit(t.get("RUNS (2+ checkpoints)", c["hud_dim"]), (x, top))
        y = top + lh
        n_fit = max((h - top - 100) // lh, 1)
        first = max(min(self.run_i - n_fit // 2, len(self.run_ids) - n_fit), 0)
        for i in range(first, min(first + n_fit, len(self.run_ids))):
            rid = self.run_ids[i]
            rect = pygame.Rect(x, y - 2, 312, lh)
            if i == self.run_i:
                panel(screen, rect, (255, 255, 255, 50))
            tag = self.runs[rid][0]["model_id"].split("@")[0]
            screen.blit(t.get(f"{tag[:30]}  ({len(self.runs[rid])})", c["hud_text"]), (x + 6, y))
            self._hit.append((rect, "run", i))
            y += lh
        # checkpoints
        x1 = 360
        panel(screen, pygame.Rect(x1 - 8, top - 8, 520, h - top - 60), c["hud_bg"])
        screen.blit(t.get(f"CHECKPOINTS  strategy: {self.strategy} ({STRATEGIES[self.strategy][:38]}), N = {self.count}",
                          c["hud_dim"]), (x1, top))
        cps = self.checkpoints()
        n_fit = max((h - top - 100) // lh, 1)
        self.scroll = min(self.scroll, max(len(cps) - n_fit, 0))
        palette = gradient_palette(max(len(self.selected), 1))
        y = top + lh
        for r in cps[self.scroll:self.scroll + n_fit]:
            mid = r["model_id"]
            rect = pygame.Rect(x1, y - 2, 500, lh)
            on = mid in self.selected
            if on:
                panel(screen, rect, (255, 255, 255, 35))
                pygame.draw.rect(screen, palette[self.selected.index(mid)], (x1 + 2, y + 2, 12, lh - 8))
            steps = f"{int(r.get('steps') or 0) // 1000}k"
            seen = r.get("completion_rate")
            unseen = r.get("unseen_completion")
            info = f"{steps:>8}   seen {seen:.0%}" if seen is not None else f"{steps:>8}   -"
            if unseen is not None:
                info += f"   unseen {unseen:.0%}"
            screen.blit(t.get(info, c["hud_text"] if on else c["hud_dim"]), (x1 + 22, y))
            self._hit.append((rect, "cp", mid))
            y += lh
        # tracks
        x2 = 900
        panel(screen, pygame.Rect(x2 - 8, top - 8, w - x2 - 4, h - top - 60), c["hud_bg"])
        y = top
        last = None
        for folder, name in self.tracks:
            if y > h - 90:
                break
            if folder != last:
                screen.blit(t.get(folder.upper(), c["hud_dim"]), (x2, y))
                y += lh
                last = folder
            rect = pygame.Rect(x2, y - 2, w - x2 - 20, lh)
            if name == self.track:
                panel(screen, rect, (255, 255, 255, 50))
            screen.blit(t.get(name, c["hud_text"]), (x2 + 8, y))
            self._hit.append((rect, "track", name))
            y += lh
        screen.blit(t.get(f"{self.run_id or '-'} · track {self.track} · {len(self.selected)} car(s) · {self.laps} laps",
                          c["hud_text"]), (20, h - 40))
        btn = pygame.Rect(w - 170, h - 48, 150, 34)
        ready = len(self.selected) >= 2
        pygame.draw.rect(screen, c["hud_good"] if ready else c["hud_dim"], btn, border_radius=6)
        img = t.get("RACE  >", c["outside"])
        screen.blit(img, img.get_rect(center=btn.center))
        self._hit.append((btn, "start", None))


def run_evolution(ctx, preset: Selection | None) -> str:
    """The Evolution Race mode: picker → race → back to the picker."""
    from racer.render.arena import blocked_models, list_tracks, run_race

    blocked = None
    while True:
        selection = preset
        if selection is None:
            if blocked is None:
                blocked = blocked_models(ctx.reg_rows)
            picker = EvoPicker(ctx.reg_rows, list_tracks(), ctx.screen.get_size(), ctx.fonts[0], ctx.colours,
                               max_cars=int(ctx.acfg["max_cars"]), laps=ctx.laps, track=ctx.track, blocked=blocked)
            while picker.done is None and not picker.quit:
                for ev in pygame.event.get():
                    picker.handle(ev)
                picker.draw(ctx.screen)
                pygame.display.flip()
                ctx.clock.tick(60)
            if picker.quit:
                return "back"
            selection = picker.done
            ctx.track, ctx.laps = selection.track, selection.laps
        result = run_race(ctx, selection)
        if result == "quit" or preset is not None:
            return result
