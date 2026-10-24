"""Swarm view (T132, PROJECT_CONTEXT §10.1): one policy, many cars scattered over the track.

Pick one model (or the baseline) and how many copies drive at once. Cars start at random places and
speeds, crashed cars respawn, and the view shows the whole track: the early-training chaos shot, or a
trained model's variety of lines. Cars are ghosts, so nothing interacts.
"""
from __future__ import annotations

from dataclasses import dataclass

import pygame

from racer.render.ui import ModelPicker, Selection


@dataclass
class SwarmPicker(ModelPicker):
    """The arena picker restricted to one model; ``+``/``-`` set how many cars (the ``laps`` field counts them)."""

    max_swarm: int = 60

    def __post_init__(self) -> None:
        super().__post_init__()
        self.title = "SWARM - pick one model (or Shift+B for the baseline) and how many cars"
        self.extra_help = " · +/−: number of cars"
        self.count_label = "cars"
        self.max_cars = 1

    def ready(self) -> bool:
        return len(self.selected) == 1 or (self.baseline and not self.selected)

    def start(self) -> None:
        if self.ready():
            self.done = Selection(self.track, list(self.selected), self.baseline and not self.selected, 3, self.fair,
                                  gradient=True, swarm=self.laps)

    def handle(self, ev: pygame.event.Event) -> None:
        if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.laps = min(self.laps + 4, self.max_swarm)
        elif ev.type == pygame.KEYDOWN and ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.laps = max(self.laps - 4, 2)
        else:
            super().handle(ev)

    def toggle(self, model_id: str) -> None:
        if model_id in self.selected:
            self.selected.remove(model_id)
        elif model_id not in self.blocked:
            self.selected = [model_id]                       # exactly one model


def run_swarm(ctx, preset: Selection | None) -> str:
    """The swarm mode: picker → swarm → back to the picker."""
    from racer.render.arena import blocked_models, list_tracks, run_race

    blocked = None
    while True:
        selection = preset
        if selection is None:
            if blocked is None:
                blocked = blocked_models(ctx.reg_rows)
            sw = ctx.acfg["swarm"]
            picker = SwarmPicker(ctx.reg_rows, list_tracks(), ctx.screen.get_size(), ctx.fonts[0], ctx.colours,
                                 ctx.acfg["palette"], max_cars=1, laps=int(sw["default_cars"]), track=ctx.track,
                                 blocked=blocked, max_swarm=int(sw["max_cars"]))
            while picker.done is None and not picker.quit:
                for ev in pygame.event.get():
                    picker.handle(ev)
                picker.draw(ctx.screen)
                pygame.display.flip()
                ctx.clock.tick(60)
            if picker.quit:
                return "back"
            selection = picker.done
            ctx.track = selection.track
        result = run_race(ctx, selection)
        if result == "quit" or preset is not None:
            return result
