"""RacingEnv: API checks, observation layout, terminations, and the baseline lapping tracks."""
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env

from racer.agents.baseline import BaselineController
from racer.config import CONFIG_DIR, load_preset
from racer.env.racing_env import RacingEnv, make_env_config
from racer.physics.surfaces import KERB_ID

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
    assert [e["track"] for e in s["episodes"]] == ["training_ground", "technical"]   # track keys (file stems)


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


def limits_env(on: bool, track: str = "training_ground") -> RacingEnv:
    cfg = make_env_config([track])
    cfg["env"]["episode"]["track_limits"] = on
    env = RacingEnv(cfg)
    env.set_track(track)
    return env


def first_step(env, lateral, start_s=100.0, speed=0.0):
    env.reset(seed=0, options={"start_s": start_s, "lateral": lateral, "heading_offset": 0.0, "speed": speed})
    return env.step(np.array([0.0, 0.0]))


def test_track_limits_off_for_every_existing_preset():
    """Old presets and snapshots don't have the key, so they keep their exact episode rules."""
    for p in (CONFIG_DIR / "env").glob("*.yaml"):
        if "track_limits" not in load_preset("env", p.stem)["episode"]:
            assert RacingEnv(make_env_config(["training_ground"], env=p.stem)).ep.track_limits is False, p.stem


def test_centre_over_the_edge_fails_only_with_track_limits():
    hw = limits_env(True).track.half_width
    _, r, term, _, info = first_step(limits_env(True), lateral=hw + 0.3)
    assert term and info["termination"] == "track_limits" and r < -10
    _, _, term, _, info = first_step(limits_env(False), lateral=hw + 0.3)
    assert not term and info["termination"] is None


def test_wheels_over_the_edge_with_centre_on_road_is_allowed():
    env = limits_env(True)
    hw = env.track.half_width
    _, _, term, _, _ = first_step(env, lateral=hw - 0.4)     # outer wheels (0.8 m off-centre) on the grass
    assert not term and env._onroad_frac() < 1.0


def test_kerb_counts_as_road_for_track_limits():
    """D1: a car whose centre is on a kerb is still on track."""
    env = limits_env(True, "technical")
    tr = env.track
    i = int(np.flatnonzero(tr.kerb_mask)[0])
    lateral = tr.half_width - tr.kerb_m / 2
    env.reset(seed=0, options={"start_s": i * tr.ds, "lateral": lateral, "heading_offset": 0.0, "speed": 0.0})
    assert tr.surface_at(env.car.x, env.car.y) == KERB_ID
    _, _, term, _, info = env.step(np.array([0.0, 0.0]))
    assert not term, info


def test_driving_off_the_road_ends_at_the_edge_not_the_barrier():
    env = limits_env(True)
    env.reset(seed=0, options={"start_s": 100.0, "lateral": 0.0, "heading_offset": 0.0, "speed": 20.0})
    for _ in range(200):
        _, _, term, _, info = env.step(np.array([1.0, 0.5]))
        if term:
            break
    assert info["termination"] == "track_limits"
    d, _ = env.track.sample(env.car.x, env.car.y)
    assert env.track.half_width < d < env.track.half_width + env.car.speed * env.params.dt   # within one substep


def test_free_drive_reports_track_limits_as_a_warning():
    env = limits_env(True)
    env.free_drive = True
    _, _, term, _, info = first_step(env, lateral=env.track.half_width + 0.3)
    assert not term and info["warning"] == "track_limits"


def test_rays7_lean_inputs():
    env = RacingEnv(make_env_config(["training_ground"], env="rays7_lean"))
    assert env.observation_space.shape == (10,)
    assert env.observer.labels == ["ray-55", "ray-17", "ray-5.5", "ray+0", "ray+5.5", "ray+17", "ray+55",
                                   "sin_head", "cos_head", "v_long"]
    assert env.ep.track_limits is True
    gym_check_env(env, skip_render_check=True)
