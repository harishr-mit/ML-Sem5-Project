"""Neuroevolution runs (D26, docs/NEUROEVOLUTION_PLAN.md): exporting a car as a model folder."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from racer.agents.neuroevolution import layer_sizes, unflatten


class _Spaces(gym.Env):
    """Only the spaces: SB3 needs an env to build a policy, and nothing here steps it."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        self.observation_space, self.action_space = observation_space, action_space


def export_model(folder: Path, flat: np.ndarray, train_cfg: Mapping[str, Any],
                 observation_space: gym.Space, action_space: gym.Space) -> None:
    """Write ``model.zip`` (the weights in an SB3 PPO actor) and an identity ``vecnormalize.pkl``.

    ``config.yaml`` is the caller's: it holds the run's snapshot, as for every model (§9).
    """
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from racer.agents.algorithms import get_algorithm

    sizes = layer_sizes(observation_space.shape[0], train_cfg["policy"]["net_arch"], action_space.shape[0])
    layers = unflatten(np.asarray(flat, dtype=np.float32), sizes)
    folder.mkdir(parents=True, exist_ok=False)
    venv = DummyVecEnv([lambda: _Spaces(observation_space, action_space)])
    policy = get_algorithm("ga").build(train_cfg, venv, None, None)
    linears = [m for m in policy.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lin, (w, b) in zip([*linears, policy.policy.action_net], layers):
            lin.weight.copy_(torch.as_tensor(w))
            lin.bias.copy_(torch.as_tensor(b))
    policy.save(folder / "model.zip")
    VecNormalize(venv, training=False, norm_obs=False, norm_reward=False).save(str(folder / "vecnormalize.pkl"))
