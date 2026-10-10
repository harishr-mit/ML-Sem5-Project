"""Experiment presets (T39): every env / reward / physics preset builds a valid env."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from racer.config import CONFIG_DIR
from racer.env.racing_env import RacingEnv, make_env_config


def names(kind: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / kind).glob("*.yaml"))


@pytest.mark.parametrize("env_name", names("env"))
@pytest.mark.parametrize("reward", names("reward"))
def test_env_reward_combinations_pass_check_env(env_name, reward):
    env = RacingEnv(make_env_config(["training_ground"], env=env_name, reward=reward))
    check_env(env, skip_render_check=True)
    spec = env.config["env"]["obs_spec"]
    if spec["version"] == 1:
        assert env.observation_space.shape == (len(spec["rays"]) + 8,)
    assert env.observation_space.shape == (len(env.observer.labels),)


@pytest.mark.parametrize("physics", names("physics"))
def test_physics_presets_drive(physics):
    env = RacingEnv(make_env_config(["training_ground"], physics=physics))
    env.reset(seed=0, options={"start_s": 0.0, "lateral": 0.0, "heading_offset": 0.0, "speed": 20.0})
    for _ in range(40):
        obs, r, term, trunc, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
        assert np.isfinite(obs).all() and np.isfinite(r)
    assert info["speed"] > 20.0


def test_no_grip_limit_never_slides_on_road():
    """E2: full lock at speed on the road stays inside the friction circle."""
    env = RacingEnv(make_env_config(["training_ground"], physics="no_grip_limit"))
    env.reset(seed=0, options={"start_s": 0.0, "lateral": 0.0, "heading_offset": 0.0, "speed": 30.0})
    env.free_drive = True
    for _ in range(10):
        env.step(np.array([1.0, -1.0], dtype=np.float32))
        assert not env.car.sliding


def test_speed_reward_ignores_direction():
    env = RacingEnv(make_env_config(["training_ground"], reward="speed_naive"))
    assert env.reward_fn(-5.0, 0.05, 0.0, 0.0, 0, False, 20.0) == pytest.approx(1.0)
    assert env.reward_fn(+5.0, 0.05, 0.0, 0.0, 0, False, 20.0) == pytest.approx(1.0)
