"""Model picker for the Model Arena (D16): pick a track and up to 20 models from the registry.

Keys / mouse: click a track; click model rows to select (up to ``max_cars``); click a column
header to sort (again to reverse); type to filter by id; Backspace edits the filter;
``Tab`` best checkpoints only; ``Shift+B`` add the PID baseline; ``Shift+P`` fair physics;
``+``/``−`` laps; wheel scrolls; ``Enter`` starts; ``Esc`` quits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pygame

from racer.render.hud import TextCache, panel

COLUMNS = [  # (header, registry key, width px, format)
    ("model", "model_id", 300, "{}"),
    ("steps", "steps", 70, "{}"),
    ("best", "is_best", 46, "{}"),
    ("show", "completion_rate", 60, "{:.0%}"),
    ("held", "heldout_completion", 60, "{:.0%}"),
    ("TG", "best_lap_training_ground", 70, "{:.2f}"),
    ("Tech", "best_lap_technical", 70, "{:.2f}"),
    ("Flow", "best_lap_fast_flow", 70, "{:.2f}"),
]
DESC_FIRST = {"steps", "is_best", "completion_rate", "heldout_completion"}   # "more is better" columns


@dataclass
class Selection:
    track: str
    models: list[str]
    baseline: bool = False
    laps: int = 3
    fair: bool = False


@dataclass
class ModelPicker:
    rows: list[dict[str, Any]]            # typed registry rows (models.csv)
    tracks: list[tuple[str, str]]         # (folder, track name)
    size: tuple[int, int]
    font: pygame.font.Font
    colours: Mapping[str, Any]
    palette: Sequence[Sequence[int]]
    max_cars: int = 20
    laps: int = 3
    track: str = "training_ground"
    selected: list[str] = field(default_factory=list)
    filter: str = ""
    sort_key: str = "model_id"
    sort_desc: bool = False
    best_only: bool = False
    baseline: bool = False
    fair: bool = False
    scroll: int = 0
    done: Selection | None = None
    quit: bool = False

    def __post_init__(self) -> None:
        self.text = TextCache(self.font)
        self.lh = self.font.get_linesize() + 4
        self.rows = [r for r in self.rows if r.get("path")]     # only models with a folder (not the baseline row)
        self._hit: list[tuple[pygame.Rect, str, Any]] = []

    # ------------------------------------------------------------ state
    def visible(self) -> list[dict[str, Any]]:
        rows = [r for r in self.rows if self.filter.lower() in str(r["model_id"]).lower()
                and (not self.best_only or r.get("is_best"))]
        key = self.sort_key
        if key == "model_id":    # natural order: run, then steps
            return sorted(rows, key=lambda r: (str(r["run_id"]), r.get("steps") or 0), reverse=self.sort_desc)
        filled = sorted((r for r in rows if r.get(key) is not None), key=lambda r: r[key], reverse=self.sort_desc)
        return filled + [r for r in rows if r.get(key) is None]       # empty cells always last

    def toggle(self, model_id: str) -> None:
        if model_id in self.selected:
            self.selected.remove(model_id)
        elif len(self.selected) + int(self.baseline) < self.max_cars:
            self.selected.append(model_id)

    def start(self) -> None:
        if self.selected or self.baseline:
            self.done = Selection(self.track, list(self.selected), self.baseline, self.laps, self.fair)

    # ------------------------------------------------------------ events
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
                    elif kind == "row":
                        self.toggle(value)
                    elif kind == "sort":
                        self.sort_desc = (not self.sort_desc) if self.sort_key == value else value in DESC_FIRST
                        self.sort_key = value
                    elif kind == "start":
                        self.start()
                    break
        elif ev.type == pygame.KEYDOWN:
            k = ev.key
            if k == pygame.K_ESCAPE:
                self.quit = True
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.start()
            elif k == pygame.K_BACKSPACE:
                self.filter = self.filter[:-1]
                self.scroll = 0
            elif k == pygame.K_TAB:
                self.best_only = not self.best_only
                self.scroll = 0
            elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.laps = min(self.laps + 1, 20)
            elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.laps = max(self.laps - 1, 1)
            elif ev.unicode and (ev.unicode.isalnum() or ev.unicode in "_@") and not self._shortcut(ev.unicode):
                self.filter += ev.unicode
                self.scroll = 0

    def _shortcut(self, ch: str) -> bool:
        """Upper-case B / P are shortcuts (lower case types into the filter)."""
        if ch == "B":
            self.baseline = not self.baseline
            return True
        if ch == "P":
            self.fair = not self.fair
            return True
        return False

    # ------------------------------------------------------------ drawing
    def draw(self, screen: pygame.Surface) -> None:
        c = self.colours
        t = self.text
        w, h = self.size
        screen.fill(c["outside"])
        self._hit = []
        lh = self.lh
        screen.blit(t.get("MODEL ARENA — pick a track and up to 20 models", c["hud_text"]), (20, 14))
        help_s = ("type: filter · Tab: best only · Shift+B: baseline car · Shift+P: fair physics · +/−: laps · "
                  "click header: sort · Enter: race · Esc: quit")
        screen.blit(t.get(help_s, c["hud_dim"]), (20, 14 + lh))

        # tracks
        x, y = 20, 20 + lh * 3
        panel(screen, pygame.Rect(x - 8, y - 8, 230, h - y - 60), c["hud_bg"])
        last_folder = None
        for folder, name in self.tracks:
            if folder != last_folder:
                screen.blit(t.get(folder.upper(), c["hud_dim"]), (x, y))
                y += lh
                last_folder = folder
            rect = pygame.Rect(x, y - 2, 210, lh)
            if name == self.track:
                panel(screen, rect, (255, 255, 255, 50))
            screen.blit(t.get(name, c["hud_text"]), (x + 8, y))
            self._hit.append((rect, "track", name))
            y += lh

        # models table
        x0, y0 = 270, 20 + lh * 3
        panel(screen, pygame.Rect(x0 - 8, y0 - 8, w - x0 - 12, h - y0 - 60), c["hud_bg"])
        cx = x0 + 22
        for head, key, cw, _ in COLUMNS:
            mark = (" v" if self.sort_desc else " ^") if key == self.sort_key else ""   # the UI font has no arrows
            screen.blit(t.get(head + mark, c["hud_dim"]), (cx, y0))
            self._hit.append((pygame.Rect(cx, y0, cw, lh), "sort", key))
            cx += cw
        rows = self.visible()
        n_fit = max((h - y0 - 80) // lh - 1, 1)
        self.scroll = min(self.scroll, max(len(rows) - n_fit, 0))
        y = y0 + lh
        for r in rows[self.scroll:self.scroll + n_fit]:
            mid = r["model_id"]
            rect = pygame.Rect(x0, y - 2, w - x0 - 30, lh)
            if mid in self.selected:
                k = self.selected.index(mid)
                panel(screen, rect, (255, 255, 255, 35))
                pygame.draw.rect(screen, self.palette[k % len(self.palette)], (x0 + 2, y + 2, 12, lh - 8))
            self._hit.append((rect, "row", mid))
            cx = x0 + 22
            for _, key, cw, f in COLUMNS:
                v = r.get(key)
                if key == "is_best":
                    s = "*" if v else ""
                elif key == "steps" and v is not None:
                    s = f"{int(v) // 1000}k"
                else:
                    s = f.format(v) if v is not None else "-"
                screen.blit(t.get(s, c["hud_text"]), (cx, y))
                cx += cw
            y += lh

        # footer
        extras = (["+ baseline"] if self.baseline else []) + (["fair physics"] if self.fair else []) \
            + (["best only"] if self.best_only else [])
        foot = (f"track {self.track} · {len(self.selected)} model(s) · {self.laps} laps"
                + ("".join(f" · {e}" for e in extras)) + (f" · filter '{self.filter}'" if self.filter else ""))
        screen.blit(t.get(foot, c["hud_text"]), (20, h - 40))
        btn = pygame.Rect(w - 170, h - 48, 150, 34)
        ready = bool(self.selected or self.baseline)
        pygame.draw.rect(screen, c["hud_good"] if ready else c["hud_dim"], btn, border_radius=6)
        img = t.get("RACE  >", c["outside"])
        screen.blit(img, img.get_rect(center=btn.center))
        self._hit.append((btn, "start", None))
