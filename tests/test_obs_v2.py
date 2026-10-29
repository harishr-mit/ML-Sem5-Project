"""obs_spec v2 features (D21) and action layouts (D22)."""
import copy

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from racer.env.features import FEATURES, describe_features
from racer.env.racing_env import RacingEnv, make_env_config

V1_AS_V2 = [
    {"feature": "rays", "rays": [{"angle_deg": a, "range_m": r}
                                 for a, r in ((-90, 60), (-45, 60), (0, 200), (45, 60), (90, 60))]},
    {"feature": "heading_sincos"}, {"feature": "v_long", "scale": 70.0}, {"feature": "v_lat", "scale": 10.0},
    {"feature": "yaw_rate", "scale": 3.0}, {"feature": "onroad_frac"}, {"feature": "prev_action"},
]


def v2_config(features, layout=None, tracks=("technical",)):
    cfg = make_env_config(list(tracks))
    cfg["env"]["obs_spec"] = {"version": 2, "name": "test_v2", "features": copy.deepcopy(features)}
    if layout:
        cfg["env"]["action_spec"] = {"layout": layout}
    return cfg


def rollout(env, actions, seed=4):
    obs = [env.reset(seed=seed)[0]]
    rewards = []
    for a in actions:
        o, r, term, trunc, _ = env.step(a)
        obs.append(o)
        rewards.append(r)
        if term or trunc:
            break
    return np.array(obs), np.array(rewards)


def test_v2_copy_of_rays5_matches_v1_exactly():
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(200, 2)).astype(np.float32)
    v1 = RacingEnv(make_env_config(["technical"]))
    v2 = RacingEnv(v2_config(V1_AS_V2))
    o1, r1 = rollout(v1, actions)
    o2, r2 = rollout(v2, actions)
    assert v2.observer.labels == v1.observer.labels
    assert np.array_equal(o1, o2) and np.array_equal(r1, r2)
    assert np.array_equal(v1.observation_space.low, v2.observation_space.low)


def test_adding_and_removing_features_changes_only_the_size():
    fewer = [f for f in V1_AS_V2 if f["feature"] not in ("v_lat", "yaw_rate")]
    more = V1_AS_V2 + [{"feature": "steer_angle"}, {"feature": "curvature_ahead", "distances_m": [20, 60]}]
    assert RacingEnv(v2_config(fewer)).observation_space.shape == (11,)
    env = RacingEnv(v2_config(more))
    assert env.observation_space.shape == (16,)
    assert env.observer.labels[-3:] == ["steer_angle", "curv+20m", "curv+60m"]
    obs, _ = rollout(env, np.tile([0.3, 0.5], (50, 1)).astype(np.float32))
    assert np.all(np.abs(obs) <= 1.0 + 1e-6)
    assert np.any(obs[:, -2:] != 0.0)               # the technical track has corners within 60 m somewhere


def test_bad_specs_are_rejected():
    with pytest.raises(ValueError, match="unknown observation feature"):
        RacingEnv(v2_config([{"feature": "radar"}]))
    with pytest.raises(ValueError, match="unknown parameters"):
        RacingEnv(v2_config([{"feature": "v_long", "scale": 70, "colour": "red"}]))
    with pytest.raises(ValueError, match="missing parameter"):
        RacingEnv(v2_config([{"feature": "v_long"}]))
    with pytest.raises(ValueError, match="only supports the steer_gas"):
        cfg = make_env_config(["technical"])
        cfg["env"]["action_spec"] = {"layout": "steer_throttle_brake"}
        RacingEnv(cfg)


def test_throttle_brake_layout():
    env = RacingEnv(v2_config(V1_AS_V2, layout="steer_throttle_brake"))
    assert env.action_space.shape == (3,) and env.observation_space.shape == (14,)
    assert env.observer.labels[-3:] == ["prev_steer", "prev_throttle", "prev_brake"]
    check_env(env, skip_render_check=True)
    env.reset(seed=1, options={"speed": 20.0, "start_s": 0.0, "lateral": 0.0, "heading_offset": 0.0})
    env.step(np.array([0.0, -1.0, 1.0], dtype=np.float32))      # no throttle, full brake
    assert env.controls == (0.0, -1.0) and env.car.brake > 0.9
    obs, *_ = env.step(np.array([0.2, 1.0, -1.0], dtype=np.float32))
    assert np.allclose(env.controls, (0.2, 1.0)) and np.allclose(obs[-3:], [0.2, 1.0, -1.0])


def test_every_feature_is_described():
    names = {d["feature"] for d in describe_features()}
    assert names == set(FEATURES)
    assert all(d["description"] for d in describe_features())


def test_v2_preset_loads():
    env = RacingEnv(make_env_config(["technical"], env="rays7_narrow"))
    assert env.observer.version == 2 and env.observer.n_rays == 7
    check_env(env, skip_render_check=True)


def test_rays2_has_two_values_per_ray_and_its_first_crossing_is_the_plain_ray():
    """D33: rays2 = rays that go on through the road edge; its first value per ray is what `rays` gives."""
    rays = [{"angle_deg": a, "range_m": 200.0} for a in (-55.0, -17.0, 0.0, 17.0, 55.0)]
    e1 = RacingEnv(v2_config([{"feature": "rays", "rays": rays}]))
    e2 = RacingEnv(v2_config([{"feature": "rays2", "rays": rays}]))
    opts = {"track": "technical", "start_s": 200.0, "lateral": 0.0, "speed": 10.0}
    o1, _ = e1.reset(seed=3, options=opts)
    o2, _ = e2.reset(seed=3, options=opts)
    assert len(o2) == 10 and len(o1) == 5
    assert np.allclose(o2[0::2], o1)                               # first crossings agree
    assert np.all(o2[1::2] >= o2[0::2] - 1e-9) and np.all(o2 <= 1.0)  # the second is never before the first
    assert e2.observer.labels[:4] == ["ray-55", "ray-55#2", "ray-17", "ray-17#2"]
    assert (e2.observer.n_rays, e2.observer.ray_dist.shape) == (5, (5,))
    assert np.any(o2[1::2] < 1.0)                                  # on the Technical track some ray sees the road again
