"""Engine compatibility and model manifests (D23)."""
import copy

import pytest

from racer.agents.compat import IncompatibleModel, build_manifest, check, count_params, ensure_compatible, manifest_of
from racer.config import load_preset
from racer.env.racing_env import make_env_config


def snapshot(**env_changes):
    env = make_env_config(["technical"])
    for k, v in env_changes.items():
        env["env"][k] = v
    return {"model_id": "X@1k", "algorithm": "ppo", "env_config": env, "train": load_preset("train", "ppo_default")}


def test_current_models_are_supported():
    assert check(snapshot()) == []
    assert check(snapshot(obs_spec=load_preset("env", "rays7_narrow")["obs_spec"])) == []


def test_missing_pieces_are_reported_not_crashed():
    s = snapshot(obs_spec={"version": 2, "name": "future", "features": [{"feature": "lidar_3d"}]})
    s["env_config"]["physics"] = dict(s["env_config"]["physics"], version=7)
    s["algorithm"] = "dreamer"
    s["env_config"]["env"]["action_spec"] = {"layout": "joystick"}
    problems = " ".join(check(s))
    for word in ("lidar_3d", "physics 7", "dreamer", "joystick"):
        assert word in problems
    with pytest.raises(IncompatibleModel, match="can't run"):
        ensure_compatible(s)


def test_manifest_of_old_snapshot_is_derived():
    m = manifest_of(snapshot())
    assert m["obs_dim"] == 13 and m["action_dim"] == 2 and m["action_layout"] == "steer_gas"
    assert m["obs_labels"][:2] == ["ray-90", "ray-45"] and m["inputs"][0]["feature"] == "rays"
    # PPO MlpPolicy [256, 256]: 13·256+256 + 256·256+256 + 256·2+2 + 2 (log_std)
    assert m["n_params"] == 13 * 256 + 256 + 256 * 256 + 256 + 256 * 2 + 2 + 2


def test_parameter_count_matches_sb3():
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO, SAC

    obs = gym.spaces.Box(-1, 1, (15,), np.float32)
    act = gym.spaces.Box(-1, 1, (3,), np.float32)

    class Dummy(gym.Env):
        observation_space, action_space = obs, act

    ppo = PPO("MlpPolicy", Dummy(), policy_kwargs={"net_arch": {"pi": [64, 32], "vf": [64, 32]}}, device="cpu")
    n = count_params("ppo", 15, 3, [64, 32])
    assert n["total"] == sum(p.numel() for p in ppo.policy.parameters())
    sac = SAC("MlpPolicy", Dummy(), policy_kwargs={"net_arch": {"pi": [64, 32], "qf": [64, 32]}}, device="cpu",
              buffer_size=10)
    n = count_params("sac", 15, 3, [64, 32])
    assert n["policy"] == sum(p.numel() for p in sac.actor.parameters())
    assert n["total"] == n["policy"] + sum(p.numel() for p in sac.critic.parameters())


def test_manifest_with_new_layout():
    env = make_env_config(["technical"], env="rays7_narrow")
    env["env"]["action_spec"] = {"layout": "steer_throttle_brake"}
    m = build_manifest(env, copy.deepcopy(load_preset("train", "ppo_default")))
    assert m["obs_dim"] == 16 and m["action_dim"] == 3 and m["obs_spec_version"] == 2
