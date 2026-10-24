"""The simulation never imports pygame, and the debug window runs headless."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_simulation_is_pygame_free():
    code = ("import sys, racer.physics.car, racer.track.track, racer.track.generator, "
            "racer.track.progress, racer.env.racing_env; sys.exit(1 if 'pygame' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_drive_headless_screenshot(tmp_path):
    out = tmp_path / "shot.png"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "drive.py"), "--mode", "baseline",
                        "--screenshot", str(out), "--frames", "90"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.is_file() and out.stat().st_size > 10_000


def test_drive_human_mode_runs_past_terminations(tmp_path):
    """Human mode with no input: the car would be 'stuck' after 3 s; free drive must keep running."""
    out = tmp_path / "human.png"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "drive.py"), "--mode", "human",
                        "--screenshot", str(out), "--frames", "600"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_arena_headless_baseline_race(tmp_path):
    out = tmp_path / "arena.png"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "demo.py"), "--baseline", "--track", "technical",
                        "--screenshot", str(out), "--frames", "240"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.is_file() and out.stat().st_size > 10_000


def test_model_picker_selects_and_starts():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from racer.config import load_preset
    from racer.render.ui import ModelPicker

    pygame.init()
    screen = pygame.display.set_mode((1280, 800))
    rows = [{"model_id": f"R00{r}_x@{s}k", "run_id": f"R00{r}", "steps": s * 1000, "path": "models/x",
             "is_best": s == 200, "completion_rate": s / 400, "best_lap_technical": 90 - s / 100}
            for r in (1, 2) for s in (100, 200, 300)]
    rows.append({"model_id": "BASE_pid", "run_id": "BASE", "steps": 0, "path": None})
    colours = {k: tuple(v) for k, v in load_preset("demo", "render")["debug"]["colours"].items()}
    pk = ModelPicker(rows, [("showcase", "technical"), ("showcase", "fast_flow")], (1280, 800),
                     pygame.font.Font(None, 20), colours, load_preset("demo", "arena")["palette"], max_cars=3)
    assert "BASE_pid" not in [r["model_id"] for r in pk.visible()]          # no folder → not pickable
    for ch in "R002":
        pk.handle(pygame.event.Event(pygame.KEYDOWN, key=0, unicode=ch, mod=0))
    assert [r["model_id"] for r in pk.visible()] == ["R002_x@100k", "R002_x@200k", "R002_x@300k"]
    for r in pk.visible() + [{"model_id": "R001_x@100k"}]:
        pk.toggle(r["model_id"])
    assert len(pk.selected) == 3                                            # max_cars
    pk.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, unicode="\t", mod=0))
    assert [r["model_id"] for r in pk.visible()] == ["R002_x@200k"]         # best only
    pk.draw(screen)
    pk.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r", mod=0))
    assert pk.done is not None and pk.done.models == ["R002_x@100k", "R002_x@200k", "R002_x@300k"]
    pygame.quit()


def test_arena_key_list_matches_the_key_handling():
    """Every key run_race handles is in the on-screen key list, and the panel draws (T82)."""
    import importlib
    import os
    import re
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from racer.config import load_preset
    from racer.render.hud import RaceHud

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        demo = importlib.import_module("demo")
    finally:
        sys.path.pop(0)
    src = (ROOT / "racer" / "render" / "arena.py").read_text(encoding="utf-8")
    body = src[src.index("def run_race("):]
    handled = set(re.findall(r"pygame\.K_([A-Za-z0-9_]+)", body))
    listed = " ".join(k for k, _ in demo.arena_keys(False, human=True)).lower()
    names = {"ESCAPE": "esc", "TAB": "tab", "SPACE": "space", "PLUS": "+", "EQUALS": "+", "KP_PLUS": "+",
             "MINUS": "−", "KP_MINUS": "−"}
    missing = [k for k in handled if names.get(k, k.lower()) not in listed]
    assert not missing, f"keys handled but not listed: {missing}"
    assert demo.arena_keys(True)[-1] == ("Esc", "quit")

    pygame.init()
    screen = pygame.display.set_mode((1280, 800))
    colours = {k: tuple(v) for k, v in load_preset("demo", "render")["debug"]["colours"].items()}
    hud = RaceHud(pygame.font.Font(None, 20), pygame.font.Font(None, 40), colours, 460)
    hud.keys(screen, demo.arena_keys(False), True)
    hud.keys(screen, demo.arena_keys(False), False)


def test_mode_menu_navigation():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from racer.config import load_preset
    from racer.render.app import MODES
    from racer.render.menu import Menu

    pygame.init()
    screen = pygame.display.set_mode((1280, 800))
    fonts = (pygame.font.Font(None, 22), pygame.font.Font(None, 60), pygame.font.Font(None, 16))
    colours = {k: tuple(v) for k, v in load_preset("demo", "render")["debug"]["colours"].items()}
    menu = Menu(MODES, (1280, 800), fonts, colours)
    menu.draw(screen)
    assert len(menu.rects) == len(MODES)
    key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k, unicode="", mod=0)     # noqa: E731
    menu.handle(key(pygame.K_DOWN))
    assert menu.index == (1 % len(MODES))
    menu.handle(key(pygame.K_UP))
    assert menu.index == 0
    menu.handle(key(pygame.K_UP))
    assert menu.index == len(MODES) - 1                                     # wraps
    menu.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=menu.rects[0].center))
    assert menu.done == MODES[0].key
    menu2 = Menu(MODES, (1280, 800), fonts, colours)
    menu2.handle(key(pygame.K_ESCAPE))
    assert menu2.quit and menu2.done is None
    pygame.quit()


def test_preset_selection_from_the_command_line():
    from types import SimpleNamespace
    from racer.render.app import preset_selection

    rows = [{"model_id": f"R001_x@{s}k", "run_id": "R001", "steps": s * 1000, "is_best": s == 200, "path": "m"}
            for s in (100, 200, 300, 400)]
    ctx = SimpleNamespace(reg_rows=rows, acfg={"max_cars": 3}, track="technical", laps=2)
    assert preset_selection(ctx, models=None, run=None, best=False, every=1, baseline=False) is None
    s = preset_selection(ctx, models=None, run="R001", best=False, every=2, baseline=True)
    assert s.models == ["R001_x@100k", "R001_x@300k"] and s.baseline and s.track == "technical" and s.laps == 2
    assert preset_selection(ctx, models=None, run="R001", best=False, every=1, baseline=True).models ==         ["R001_x@100k", "R001_x@200k"]                                      # max_cars leaves room for the baseline
    assert preset_selection(ctx, models=None, run=None, best=True, every=1, baseline=False).models == ["R001_x@200k"]
    try:
        preset_selection(ctx, models=None, run="R999", best=False, every=1, baseline=False)
    except ValueError:
        pass
    else:
        raise AssertionError("an empty selection must raise")


def test_gradient_palette_runs_from_blue_to_red():
    from racer.render.evorace import gradient_palette
    pal = gradient_palette(5)
    assert len(pal) == 5 and len(set(pal)) == 5
    first, last = pal[0], pal[-1]
    assert first[2] > first[0] and last[0] > last[2]            # blue first, red last
    assert len(gradient_palette(1)) == 1


def test_evolution_picker_strategies_and_toggle():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from racer.config import load_preset
    from racer.render.evorace import EvoPicker, raceable_runs

    pygame.init()
    screen = pygame.display.set_mode((1280, 800))
    rows = [{"model_id": f"R00{r}_x@{s}k", "run_id": f"R00{r}", "steps": s * 1000, "path": "models/x",
             "completion_rate": s / 1000, "score": float(s)} for r in (1, 2) for s in (100, 200, 300, 400, 500, 600)]
    rows.append({"model_id": "R003_y@100k", "run_id": "R003", "steps": 100000, "path": "models/y"})   # one only
    runs = raceable_runs(rows, blocked={"R002_x@600k": "needs a newer engine"})
    assert list(runs) == ["R002", "R001"] and len(runs["R002"]) == 5                                  # newest first
    colours = {k: tuple(v) for k, v in load_preset("demo", "render")["debug"]["colours"].items()}
    pk = EvoPicker(rows, [("showcase", "technical")], (1280, 800), pygame.font.Font(None, 20), colours,
                   blocked={"R002_x@600k": "x"}, count=3)
    key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k, unicode="", mod=0)     # noqa: E731
    assert pk.run_id == "R002"
    assert pk.selected == ["R002_x@100k", "R002_x@300k", "R002_x@500k"]            # spaced 3 of 5
    pk.handle(key(pygame.K_RIGHT))                                                  # next strategy: top
    assert pk.strategy == "top" and len(pk.selected) == 3
    pk.handle(key(pygame.K_LEFT))
    pk.toggle("R002_x@200k")                                                        # hand-pick one more
    assert pk.selected == ["R002_x@100k", "R002_x@200k", "R002_x@300k", "R002_x@500k"]
    pk.handle(key(pygame.K_DOWN))                                                   # next run resets the picks
    assert pk.run_id == "R001" and len(pk.selected) == 3
    pk.draw(screen)
    pk.handle(key(pygame.K_RETURN))
    assert pk.done is not None and pk.done.gradient and pk.done.track == "training_ground" and len(pk.done.models) == 3
    pygame.quit()


def test_evolution_race_headless_from_the_command_line(tmp_path):
    """--run with --spaced races a run's checkpoints (skips folders that are missing, as in a fresh checkout)."""
    out = tmp_path / "evo.png"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "demo.py"), "--run", "R001", "--spaced", "3", "--baseline",
                        "--track", "technical", "--screenshot", str(out), "--frames", "200"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.is_file() and out.stat().st_size > 10_000


def test_title_screen_carries_the_attribution_line():
    """The CC BY 4.0 engine sound must be credited on the demo's title screen (assets/LICENSES.md)."""
    from racer.config import load_preset
    credits = load_preset("demo", "arena")["credits"]
    assert "skyernaklea" in credits and "CC BY 4.0" in credits
