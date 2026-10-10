"""RacingEnv: API checks, observation layout, terminations, and the baseline lapping tracks."""
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env

from racer.agents.baseline import BaselineController
from racer.config import load_preset
from racer.env.racing_env import RacingEnv, make_env_config

SHOWCASE = ["training_ground", "technical", "fast_flow"]


@pytest.fixture(scope="module")
def env():
    return RacingEnv(make_env_config(["training_ground"]))


def test_gymnasium_check_env(env):
    gym_check_env(env, skip_render_check=True)


def test_sb3_check_env():
    from stable_baselines3.common.env_checker import check_env
    check_env(RacingEnv(make_env_config(["training_ground"])), warn=True)


def test_obs_layout(env):
    obs, _ = env.reset(seed=0, options={"start_s": 10.0, "lateral": 0.0, "heading_offset": 0.0, "speed": 0.0})
    assert obs.shape == (13,) and obs.dtype == np.float32
    assert env.observer.labels[:5] == ["ray-90", "ray-45", "ray+0", "ray+45", "ray+90"]
    hw = env.track.half_width
    assert obs[0] * 60 == pytest.approx(hw, abs=0.2)    # right ray reaches the right edge
    assert obs[4] * 60 == pytest.approx(hw, abs=0.2)
    assert obs[5] == pytest.approx(0.0, abs=0.05) and obs[6] == pytest.approx(1.0, abs=0.01)  # sin, cos
    assert obs[10] == 1.0     # all wheels on road
    assert env.observation_space.contains(obs)


def test_wrong_way_terminates(env):
    env.reset(seed=1, options={"start_s": 200.0, "lateral": 0.0, "heading_offset": math.pi, "speed": 10.0})
    reason = None
    for _ in range(200):
        _, r, term, trunc, info = env.step(np.array([0.0, 0.3]))
        if term or trunc:
            reason = info["termination"]
            break
    assert reason in ("wrong_way", "stuck")
    assert r < -10


def test_stuck_terminates(env):
    env.reset(seed=2, options={"start_s": 200.0, "speed": 0.0, "lateral": 0.0, "heading_offset": 0.0})
    for _ in range(100):
        _, _, term, trunc, info = env.step(np.array([0.0, -1.0]))
        if term:
            break
    assert info["termination"] == "stuck"


def test_steering_into_wall_crashes(env):
    env.reset(seed=3, options={"start_s": 100.0, "speed": 20.0, "lateral": 0.0, "heading_offset": 0.0})
    for _ in range(400):
        _, _, term, trunc, info = env.step(np.array([1.0, 1.0]))
        if term:
            break
    assert info["termination"] in ("barrier", "offroad")


@pytest.mark.parametrize("track", SHOWCASE)
def test_baseline_laps_showcase(track):
    env = RacingEnv(make_env_config([track]))
    ctrl = BaselineController(load_preset("train", "baseline"))
    env.reset(seed=0, options={"start_s": 0.0, "lateral": 0.0, "heading_offset": 0.0, "speed": 0.0})
    info = {}
    for _ in range(env.ep.max_steps):
        _, _, term, trunc, info = env.step(ctrl.act(env))
        if term or trunc:
            break
    assert info["termination"] is None, info
    assert info["laps"] >= 2
    assert len(env.lap_times) >= 1


def test_evaluation_covers_each_track():
    from racer.agents.evaluation import evaluate
    ctrl = BaselineController(load_preset("train", "baseline"))
    s = evaluate(make_env_config(["training_ground"]), lambda obs, env: ctrl.act(env),
                 ["training_ground", "technical"], n_starts=1)
    assert [e["track"] for e in s["episodes"]] == ["Training Ground", "Technical"]


def test_free_drive_walls_are_solid():
    env = RacingEnv(make_env_config(["training_ground"]))
    env.free_drive = True
    env.reset(seed=3, options={"start_s": 100.0, "speed": 30.0, "lateral": 0.0, "heading_offset": 0.0})
    warnings = set()
    for _ in range(600):   # 30 s of full lock + throttle: hits the wall, goes off-road, gets stuck
        _, _, term, trunc, info = env.step(np.array([1.0, 1.0]))
        assert not term and not trunc
        warnings.add(info["warning"])
        f = env.track.field
        d, i = env.track.sample(env.car.x, env.car.y)
        assert d <= env.track.half_width + env.track.runoff   # never through the barrier
    assert "wall" in warnings


def test_free_drive_off_by_default():
    assert RacingEnv(make_env_config(["training_ground"])).free_drive is False
