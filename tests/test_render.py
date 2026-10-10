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
