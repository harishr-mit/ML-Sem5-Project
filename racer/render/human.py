"""Human vs AI (T127, PROJECT_CONTEXT §10.1): you drive a car in the race against the chosen models.

Input: arrows or WASD (all-or-nothing, so an ABS assist keeps hard braking inside the friction
circle, as in ``drive.py``) or a gamepad (left stick steers, triggers or A/B for gas and brake).
Your car is a ghost like the others (no collisions) and uses the physics preset you pick
(Shift+C in the picker); a crash respawns it on the centreline like any other car.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pygame

from racer.config import PROJECT_ROOT
from racer.env.race import Entry
from racer.env.racing_env import make_env_config
from racer.render.ui import ModelPicker, Selection

LEFT, RIGHT = (pygame.K_LEFT, pygame.K_a), (pygame.K_RIGHT, pygame.K_d)
UP, DOWN = (pygame.K_UP, pygame.K_w), (pygame.K_DOWN, pygame.K_s)


def _any(keys, group) -> bool:
    return any(keys[k] for k in group)


def read_input(keys, pad: Mapping[str, float] | None, cfg: Mapping[str, Any]) -> tuple[float, float]:
    """``(steer, gas)`` from the keyboard, else the gamepad. Steer +1 is left (counter-clockwise)."""
    if any(_any(keys, g) for g in (LEFT, RIGHT, UP, DOWN)):
        return (float(_any(keys, LEFT)) - float(_any(keys, RIGHT)),
                float(_any(keys, UP)) - float(_any(keys, DOWN)))
    if not pad:
        return 0.0, 0.0
    dz = float(cfg["deadzone"])
    dead = lambda v: 0.0 if abs(v) < dz else float(v)                  # noqa: E731
    return -dead(pad["steer"]), float(np.clip(dead(pad["gas"]) - dead(pad["brake"]), -1.0, 1.0))


def poll_pad(joystick, cfg: Mapping[str, Any]) -> dict[str, float] | None:
    """The first gamepad's stick and triggers as ``{steer, gas, brake}``; ``None`` without one."""
    if joystick is None:
        return None
    try:
        ax, bt = cfg["pad_axes"], cfg["pad_buttons"]
        n = joystick.get_numaxes()
        trig = lambda i: (joystick.get_axis(i) + 1.0) / 2.0 if i < n else 0.0     # noqa: E731  triggers rest at -1
        gas = max(trig(ax["gas"]), float(joystick.get_button(bt["gas"])))
        brake = max(trig(ax["brake"]), float(joystick.get_button(bt["brake"])))
        return {"steer": joystick.get_axis(ax["steer"]), "gas": gas, "brake": brake}
    except pygame.error:
        return None


class HumanDriver:
    """The human car's controller: holds the last input and applies the ABS assist when it acts."""

    def __init__(self, cfg: Mapping[str, Any], abs_cfg: Mapping[str, Any] | None = None) -> None:
        a = dict(abs_cfg if abs_cfg is not None else cfg["abs"])
        self.abs_on = bool(a["enabled"])
        self.margin = float(a["margin"])
        self.min_brake = float(a["min_brake"])
        self.abs_active = False
        self.steer = 0.0
        self.gas = 0.0

    def set_input(self, steer: float, gas: float) -> None:
        self.steer, self.gas = float(steer), float(gas)

    def toggle_abs(self) -> None:
        self.abs_on = not self.abs_on

    def act(self, obs: np.ndarray, envs: Sequence[Any]) -> np.ndarray:
        """``Entry.act_batch``: one car, so ``envs[0]`` is the human's env."""
        gas = self.gas
        if gas < 0.0 and self.abs_on:
            env = envs[0]
            limit = env.car.abs_brake_limit(env._wheels.tolist(), self.margin)
            gas = -max(limit, self.min_brake)               # cap the all-or-nothing brake inside the grip limit
            self.abs_active = limit < 1.0
        else:
            self.abs_active = False
        return np.array([[self.steer, gas]], dtype=np.float32)


def human_entry(track: str, physics: str, driver: HumanDriver) -> Entry:
    return Entry("You", make_env_config([track], physics=physics), driver.act, key="human")


def physics_presets() -> list[str]:
    return sorted(p.stem for p in (PROJECT_ROOT / "configs" / "physics").glob("*.yaml"))


@dataclass
class HumanPicker(ModelPicker):
    """The arena picker plus your car's physics (Shift+C); starts with no model at all."""

    physics_options: list[str] = field(default_factory=physics_presets)
    physics: str = "default"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.title = "HUMAN VS AI - pick a track, your car's physics and the opponents (0 to 19)"
        self.extra_help = " · Shift+C: your car's physics"

    def ready(self) -> bool:
        return True

    def start(self) -> None:
        self.done = Selection(self.track, list(self.selected), self.baseline, self.laps, self.fair, human=self.physics)

    def _shortcut(self, ch: str) -> bool:
        if ch == "C":
            i = (self.physics_options.index(self.physics) + 1) % len(self.physics_options)
            self.physics = self.physics_options[i]
            return True
        return super()._shortcut(ch)

    def draw(self, screen: pygame.Surface) -> None:
        self.extra_foot = f" · your car: {self.physics}"
        super().draw(screen)


def run_human(ctx, preset: Selection | None) -> str:
    """The Human vs AI mode: picker → race → back to the picker."""
    from racer.render.arena import blocked_models, list_tracks, run_race

    blocked = None
    while True:
        selection = preset
        if selection is None:
            if blocked is None:
                blocked = blocked_models(ctx.reg_rows)
            picker = HumanPicker(ctx.reg_rows, list_tracks(), ctx.screen.get_size(), ctx.fonts[0], ctx.colours,
                                 ctx.acfg["palette"], max_cars=int(ctx.acfg["max_cars"]) - 1, laps=ctx.laps,
                                 track=ctx.track, blocked=blocked, physics=str(ctx.acfg["human"]["physics"]))
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
