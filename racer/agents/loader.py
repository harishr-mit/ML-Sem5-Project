"""Load a self-contained model folder (PROJECT_CONTEXT §9).

    models/<model_id>/  model.zip  vecnormalize.pkl  config.yaml

``config.yaml`` holds the full snapshot, including ``env_config`` (physics, obs spec,
reward, track settings), so the model always runs with what it was trained with.
Week 2 adds lookup by model id through the registry.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from racer.config import load_yaml


class LoadedModel:
    """A policy plus its observation normaliser and config snapshot."""

    def __init__(self, folder: Path | str) -> None:
        from stable_baselines3 import PPO   # imported lazily: torch is slow to import

        self.folder = Path(folder)
        self.config: dict[str, Any] = load_yaml(self.folder / "config.yaml")
        self.model_id = self.config.get("model_id", self.folder.name)
        self.env_config: dict[str, Any] = self.config["env_config"]
        self.policy = PPO.load(self.folder / "model.zip", device="cpu")
        vn_path = self.folder / "vecnormalize.pkl"
        self.vecnorm = None
        if vn_path.is_file():
            with vn_path.open("rb") as f:
                self.vecnorm = pickle.load(f)
            self.vecnorm.training = False

    def normalise(self, obs: np.ndarray) -> np.ndarray:
        if self.vecnorm is None or not self.vecnorm.norm_obs:
            return obs
        return self.vecnorm.normalize_obs(obs)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _ = self.policy.predict(self.normalise(obs), deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)
