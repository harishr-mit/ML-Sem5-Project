"""Multi-car race (T47, D14): grid, countdown, laps, gaps, crashes and respawns."""
import numpy as np
import pytest

from racer.agents.baseline import BaselineController
from racer.config import PROJECT_ROOT, load_preset
from racer.env.race import Entry, Race
from racer.env.racing_env import make_env_config

RULES = dict(load_preset("demo", "arena")["race"], laps=1, countdown_s=1.0)
MODEL = PROJECT_ROOT / "models" / "R001_ppo_rays5_tg@1200k"


def baseline_entry(label="BASE"):
    ctrl = BaselineController(load_preset("train", "baseline"))
    return Entry(label, make_env_config(["training_ground"]),
                 lambda obs, envs: np.stack([ctrl.act(e) for e in envs]))


def flat_out_entry(label):
    return Entry(label, make_env_config(["training_ground"]),
                 lambda obs, envs: np.tile(np.array([0.0, 1.0], np.float32), (len(obs), 1)))


def test_grid_and_countdown():
    race = Race("training_ground", [flat_out_entry(f"c{i}") for i in range(5)], RULES)
    d = [c.dist for c in race.cars]
    assert d == pytest.approx([-8, -8, -16, -16, -24], abs=0.6)          # 2-wide rows behind the line
    poses = [c.pose for c in race.cars]
    for _ in range(30):                                                   # still counting down
        race.tick()
    assert [c.pose for c in race.cars] == poses and not race.started
    for _ in range(60):
        race.tick()
    assert race.started and all(c.dist > d0 for c, d0 in zip(race.cars, d))


def test_crash_respawns_without_losing_distance():
    race = Race("training_ground", [flat_out_entry("flat")], RULES)
    car = race.cars[0]
    seen_respawn = False
    for _ in range(60 * 90):
        race.tick()
        if car.state == "respawn":
            seen_respawn = True
            d_crash = car.dist
        elif seen_respawn:
            assert car.dist == pytest.approx(d_crash, abs=0.5)           # same race distance (it may already roll)
            assert abs(car.env.tracker.lateral) < 0.5 and car.env.car.speed < 1.0
            break
    assert car.crashes >= 1 and seen_respawn


def test_baseline_race_finishes_with_times_and_gaps():
    race = Race("training_ground", [baseline_entry("A"), baseline_entry("B")], RULES)
    while not race.over:
        race.tick()
        assert race.t < 200
    a, b = race.standings()
    assert a.finished_t is not None and b.finished_t is not None
    assert 40 < a.lap_times[0] < 60                       # 46.6 s flying; standing start from the grid
    assert abs(a.lap_times[0] - b.lap_times[0]) < 2.0    # same driver, either side of the front row
    assert race.gap(a) == 0.0 and race.gap(b) == pytest.approx(b.finished_t - a.finished_t)


@pytest.mark.skipif(not (MODEL / "model.zip").is_file(), reason="R001 not on this machine")
def test_models_race_batched():
    from racer.agents.loader import LoadedModel
    m = LoadedModel(MODEL)
    entries = [Entry(f"m{i}", m.env_config, lambda obs, envs: m.act_batch(obs), key=m) for i in range(4)]
    race = Race("training_ground", entries + [baseline_entry()], RULES)
    while not race.over:
        race.tick()
        assert race.t < 200
    assert sum(c.finished_t is not None for c in race.cars) == 5
    assert min(c.lap_times[0] for c in race.cars[:4]) < race.cars[4].lap_times[0]   # RL beats the baseline
