"""Demo application shell (T124): one window, a shared context, a mode menu.

``scripts/demo.py`` parses arguments and calls :func:`run`. Modes are registered in :data:`MODES`;
a mode is a function ``run(ctx, preset) -> "back" | "quit"``. ``preset`` is a ready
:class:`~racer.render.ui.Selection` when the command line already chose the cars (no picker).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pygame

from racer.agents.checkpoints import choose, run_checkpoints
from racer.config import load_preset
from racer.registry.registry import Registry, typed
from racer.render.debug_view import load_font
from racer.render.ui import Selection


@dataclass
class Context:
    """Everything a mode needs: window, fonts, configs, the loaded-model cache and the CLI choices."""

    screen: pygame.Surface
    clock: pygame.time.Clock
    fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]    # normal, big, small
    colours: dict[str, tuple[int, ...]]
    acfg: dict[str, Any]                  # configs/demo/arena.yaml
    dcfg: dict[str, Any]                  # configs/demo/render.yaml → debug
    reg_rows: list[dict[str, Any]]
    track: str = "training_ground"
    laps: int = 3
    sim_speed: float = 1.0
    screenshot: str | None = None         # headless run: save a PNG after ``frames`` frames
    frames: int = 600
    direct: bool = False                  # the command line chose the cars: Esc quits instead of going back
    models: dict[str, Any] = field(default_factory=dict)    # model id → LoadedModel, loaded once
    probes: dict[str, Any] = field(default_factory=dict)    # model id → ActivationProbe (brain panel)
    audio: Any = None                     # racer.render.audio.AudioController, or None (no sound at all)


@dataclass(frozen=True)
class Mode:
    key: str
    title: str
    blurb: str
    run: Callable[[Context, Selection | None], str]


def _arena(ctx: Context, preset: Selection | None) -> str:
    from racer.render.arena import run_arena
    return run_arena(ctx, preset)


def _evolution(ctx: Context, preset: Selection | None) -> str:
    from racer.render.evorace import run_evolution
    return run_evolution(ctx, preset)


def _swarm(ctx: Context, preset: Selection | None) -> str:
    from racer.render.swarm import run_swarm
    return run_swarm(ctx, preset)


def _human(ctx: Context, preset: Selection | None) -> str:
    from racer.render.human import run_human
    return run_human(ctx, preset)


MODES: list[Mode] = [
    Mode("arena", "Model Arena", "Pick up to 20 models and race them on a track.", _arena),
    Mode("evolution", "Evolution Race", "The checkpoints of one run race: watch it learn.", _evolution),
    Mode("human", "Human vs AI", "Drive a car yourself against the models (keyboard or gamepad).", _human),
    Mode("swarm", "Swarm", "One model, dozens of cars at once: chaos early on, variety later.", _swarm),
]


def build_context(*, track: str, laps: int | None, sim_speed: float, screenshot: str | None, frames: int,
                  title: str | None, direct: bool, sound: bool | None = None) -> Context:
    """Open the window and load the configs (call after SDL_VIDEODRIVER is set for headless runs)."""
    acfg = load_preset("demo", "arena")
    dcfg = load_preset("demo", "render")["debug"]
    colours = {k: tuple(int(v) for v in c) for k, c in dcfg["colours"].items()}
    pygame.init()
    screen = pygame.display.set_mode(tuple(dcfg["window"]))
    pygame.display.set_caption(title or "RL Racing Line")
    fc = dcfg["fonts"]
    fonts = (load_font(fc["body"]["file"], int(fc["body"]["size"])), load_font(fc["display"]["file"], int(fc["display"]["size"])),
             load_font(fc["small"]["file"], int(fc["small"]["size"])))
    rows = [{k: typed(v) for k, v in r.items()} for r in Registry().models()]
    ctx = Context(screen, pygame.time.Clock(), fonts, colours, acfg, dcfg, rows, track=track,
                  laps=laps or int(acfg["race"]["laps"]), sim_speed=sim_speed, screenshot=screenshot,
                  frames=frames, direct=direct)
    # D28: sound is on when watching a race; headless captures stay silent unless asked (--sound).
    acfg_audio = load_preset("demo", "audio")
    want = sound if sound is not None else (bool(acfg_audio["default_on"]) and not screenshot)
    if want or sound is False or not screenshot:
        from racer.render.audio import AudioController
        ctx.audio = AudioController(acfg_audio, enabled=want)     # M toggles it in every race window
    return ctx


def preset_selection(ctx: Context, *, models: list[str] | None, run: str | None, best: bool, every: int,
                     baseline: bool, spaced: int | None = None, top: int | None = None,
                     human: str | None = None, swarm: int | None = None) -> Selection | None:
    """The cars the command line asked for, or ``None`` to open the menu. Raises ValueError if empty."""
    if not (models or run or best or baseline or human or swarm):
        return None
    rows = ctx.reg_rows
    if models:
        ids = list(models)
    elif run:
        strategy, n = ("spaced", spaced) if spaced else ("top", top) if top else ("every", every)
        ids = [r["model_id"] for r in choose(run_checkpoints(rows, run), strategy, n)]
    elif best:
        ids = [r["model_id"] for r in rows if r.get("is_best") and r.get("path")]
    else:
        ids = []                                       # --baseline alone
    ids = ids[: int(ctx.acfg["max_cars"]) - int(baseline) - int(bool(human))]
    if swarm:
        if not ids and not baseline:
            raise ValueError("a swarm needs one model (--models) or --baseline")
        return Selection(ctx.track, ids[:1], baseline and not ids, ctx.laps, False, gradient=True, swarm=swarm)
    if not ids and not baseline and not human:
        raise ValueError("no models selected")
    return Selection(ctx.track, ids, baseline, ctx.laps, False, gradient=bool(run), human=human)


def run(ctx: Context, preset: Selection | None = None, mode: str | None = None) -> int:
    """Run the app until the window closes. A preset (or ``mode``) skips the menu."""
    from racer.render.menu import Menu

    by_key = {m.key: m for m in MODES}
    pick = by_key["arena"] if preset is not None else by_key.get(mode or "")
    while True:
        if pick is None:
            menu = Menu(MODES, ctx.screen.get_size(), ctx.fonts, ctx.colours, credits=str(ctx.acfg.get("credits", "")))
            while menu.done is None and not menu.quit:
                for ev in pygame.event.get():
                    menu.handle(ev)
                menu.draw(ctx.screen)
                pygame.display.flip()
                ctx.clock.tick(60)
            if menu.quit:
                break
            pick = by_key[menu.done]
        result = pick.run(ctx, preset)
        if result == "quit" or ctx.direct or mode is not None or preset is not None:
            break
        pick = None
    if ctx.audio is not None:
        ctx.audio.close()
    pygame.quit()
    return 0
