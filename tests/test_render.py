"""The simulation never imports pygame, and the debug window runs headless."""
import pytest
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
             "MINUS": "−", "KP_MINUS": "−", "RETURN": "enter", "KP_ENTER": "enter"}
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


def test_camera_zooms_out_with_speed_and_back_in():
    from racer.config import load_preset
    from racer.render.camera import Camera
    cfg = load_preset("demo", "render")["debug"]
    cam = Camera((1280, 800), cfg)
    base = cam.view_zoom
    for _ in range(300):
        cam.update(0.0, 0.0, 70.0, 0.0)                                  # flat out
    assert cam.view_zoom == pytest.approx(base * (1 - cfg["camera_speed_zoom"]), rel=0.02)
    for _ in range(300):
        cam.update(0.0, 0.0, 0.0, 0.0)
    assert cam.view_zoom == pytest.approx(base, rel=0.02)
    cam.overview = True
    cam.zoom_fit = 3.0
    assert cam.view_zoom == 3.0                                          # the overview ignores the speed zoom


def test_minimap_maps_the_track_into_its_box_and_marks_cars():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from types import SimpleNamespace
    import numpy as np
    from racer.config import load_preset
    from racer.render.minimap import Minimap

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    colours = {k: tuple(v) for k, v in load_preset("demo", "render")["debug"]["colours"].items()}
    th = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    pts = np.stack([300 * np.cos(th), 120 * np.sin(th)], axis=1)                     # a 600 x 240 m ellipse
    track = SimpleNamespace(centreline=SimpleNamespace(points=pts), key="ellipse")
    mm = Minimap(colours, size=(250, 170), margin=12)
    mm.set_track(track)
    xs, ys = zip(*(mm.to_map(x, y) for x, y in pts))
    assert 12 - 1 <= min(xs) and max(xs) <= 250 - 12 + 1 and 12 - 1 <= min(ys) and max(ys) <= 170 - 12 + 1
    assert mm.to_map(0.0, 100.0)[1] < mm.to_map(0.0, -100.0)[1]                      # north is up on screen
    cars = [SimpleNamespace(index=0, pose=(300.0, 0.0, 0.0)), SimpleNamespace(index=1, pose=(-300.0, 0.0, 0.0))]
    before = pygame.image.tobytes(screen, "RGB")
    mm.draw(screen, (10, 10), cars, 0, lambda i: (255, 0, 0) if i == 0 else (0, 0, 255))
    assert pygame.image.tobytes(screen, "RGB") != before
    pygame.quit()


def test_dashboard_draws_gauge_text_and_bars():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from types import SimpleNamespace
    from racer.config import load_preset
    from racer.render.dashboard import Dashboard, arc_points

    pts = arc_points(0.0, 0.0, 10.0, 225.0, -45.0, 4)
    assert pts[0][0] < 0 < pts[-1][0] and pts[0][1] > 0 > 0 - 1                      # starts bottom-left, ends bottom-right
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    fonts = (pygame.font.Font(None, 22), pygame.font.Font(None, 40), pygame.font.Font(None, 16))
    colours = {k: tuple(v) for k, v in load_preset("demo", "render")["debug"]["colours"].items()}
    dash = Dashboard(fonts, colours)
    phys = SimpleNamespace(speed=30.0, a_lat=9.81, throttle=0.7, brake=0.0, steer=0.2)
    car = SimpleNamespace(label="R000_x@1k", env=SimpleNamespace(car=phys), lap_times=[61.2], state="racing",
                          crashes=1, last_crash=None)
    race = SimpleNamespace(lap_of=lambda c: 2, rules=SimpleNamespace(laps=3), current_lap_time=lambda c: 12.3)
    for rpm in (1.0, 2.0, 3.5):                                                       # beyond the redline is clamped
        rect = dash.draw(screen, race, car, (255, 0, 0), rpm, 3)
    assert rect.w == dash.size[0] and rect.bottom <= 600
    car.state = "respawn"
    car.last_crash = "off_road"
    dash.draw(screen, race, car, (255, 0, 0), 1.0, 1)
    pygame.quit()
