"""Swarm view (T132): scattered starts in Race, the picker, and a headless swarm."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from racer.config import load_preset

ROOT = Path(__file__).resolve().parents[1]


def baseline_entries(n, track="training_ground"):
    from racer.agents.baseline import BaselineController
    from racer.env.race import Entry
    from racer.env.racing_env import make_env_config
    ctrl = BaselineController(load_preset("train", "baseline"))
    return [Entry(f"PID #{i + 1}", make_env_config([track]), lambda obs, envs: np.stack([ctrl.act(e) for e in envs]),
                  key="baseline") for i in range(n)]


def swarm_rules(**kw):
    sw = load_preset("demo", "arena")["swarm"]
    return dict(load_preset("demo", "arena")["race"], laps=int(sw["laps"]), countdown_s=0.0, start="scatter",
                scatter_speed_max_ms=float(sw["speed_max_ms"]), scatter_lateral_frac=float(sw["lateral_frac"]), **kw)


def test_scattered_start_spreads_the_cars_and_starts_the_clock_at_once():
    from racer.env.race import Race
    race = Race("training_ground", baseline_entries(8), swarm_rules())
    s = np.array([c.env.tracker.s for c in race.cars])
    assert len(set(np.round(s))) == 8 and s.max() - s.min() > 0.3 * race.length            # spread over the track
    speeds = np.array([c.env.car.speed for c in race.cars])
    assert speeds.min() >= 0 and speeds.max() <= swarm_rules()["scatter_speed_max_ms"] + 1e-6 and speeds.max() > 1.0
    for _ in range(60):
        race.tick()
    assert race.started and all(c.state == "racing" for c in race.cars)
    again = Race("training_ground", baseline_entries(8), swarm_rules())
    assert [round(c.env.tracker.s, 3) for c in again.cars] == [round(float(v), 3) for v in s]    # deterministic


def test_the_grid_start_is_unchanged_by_the_new_rule_keys():
    from racer.env.race import Race
    race = Race("training_ground", baseline_entries(4), dict(load_preset("demo", "arena")["race"], laps=1))
    assert race.rules.start == "grid" and all(c.state == "grid" for c in race.cars)
    assert race.cars[0].env.car.speed == 0.0


def test_swarm_picker_takes_exactly_one_model_and_counts_cars():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from racer.render.swarm import SwarmPicker

    pygame.init()
    screen = pygame.display.set_mode((1280, 800))
    rows = [{"model_id": f"R001_x@{s}k", "run_id": "R001", "steps": s * 1000, "path": "models/x"} for s in (100, 200)]
    colours = {k: tuple(v) for k, v in load_preset("demo", "render")["debug"]["colours"].items()}
    pk = SwarmPicker(rows, [("showcase", "technical")], (1280, 800), pygame.font.Font(None, 20), colours,
                     load_preset("demo", "arena")["palette"], max_cars=1, laps=24, max_swarm=30)
    pk.draw(screen)
    key = lambda k: pygame.event.Event(pygame.KEYDOWN, key=k, unicode="", mod=0)     # noqa: E731
    pk.handle(key(pygame.K_RETURN))
    assert pk.done is None                                               # nothing picked yet
    pk.toggle("R001_x@100k")
    pk.toggle("R001_x@200k")
    assert pk.selected == ["R001_x@200k"]
    for _ in range(5):
        pk.handle(key(pygame.K_PLUS))
    assert pk.laps == 30                                                 # capped
    pk.handle(key(pygame.K_RETURN))
    assert pk.done.swarm == 30 and pk.done.models == ["R001_x@200k"] and pk.done.gradient
    pygame.quit()


def test_swarm_headless_with_the_baseline(tmp_path):
    out = tmp_path / "swarm.png"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "demo.py"), "--swarm", "16", "--baseline", "--track",
                        "technical", "--screenshot", str(out), "--frames", "240"], capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.is_file() and out.stat().st_size > 10_000


def test_swarm_without_a_policy_is_refused():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "demo.py"), "--swarm", "8"], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode != 0 and "swarm needs" in r.stderr


def test_twenty_cars_with_effects_hold_the_frame_rate():
    """PROJECT_CONTEXT §10.2 targets 60 FPS with 20 cars; the test uses a lenient floor so a busy machine doesn't flake."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import bench_demo
    finally:
        sys.path.pop(0)
    mean, low = bench_demo.measure(20, 400, "technical")
    assert mean >= 45, f"20 cars ran at {mean:.0f} fps (1% low {low:.0f})"
