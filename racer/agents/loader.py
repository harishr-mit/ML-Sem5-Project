"""Load a self-contained model folder (PROJECT_CONTEXT §9).

    models/<model_id>/  model.zip  vecnormalize.pkl  config.yaml  [eval.json  thumb_*.png]

``config.yaml`` holds the full snapshot, including ``env_config`` (physics, obs spec,
reward, track settings), so the model always runs with what it was trained with.
A model can be referred to by folder path or by model id (registry, then ``models/``,
then ``outputs/scratch/``). Inference runs the torch policy directly, batched.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from racer.config import PROJECT_ROOT, load_yaml

MODELS_DIR = PROJECT_ROOT / "models"
SCRATCH_DIR = PROJECT_ROOT / "outputs" / "scratch"


def resolve_model(ref: str | Path) -> Path:
    """Folder of a model given its path or id."""
    p = Path(ref)
    if (p / "config.yaml").is_file():
        return p
    ref = str(ref)
    try:
        from racer.registry.registry import Registry
        row = Registry().model(ref)
        if row and row.get("path") and (PROJECT_ROOT / row["path"] / "config.yaml").is_file():
            return PROJECT_ROOT / row["path"]
    except (ImportError, OSError):
        pass
    for base in (MODELS_DIR, SCRATCH_DIR):
        if (base / ref / "config.yaml").is_file():
            return base / ref
    raise FileNotFoundError(f"model not found: {ref}")


def _algorithm_class(name: str):
    from stable_baselines3 import PPO, SAC   # imported lazily: torch is slow to import
    classes = {"ppo": PPO, "sac": SAC}
    if name not in classes:
        raise ValueError(f"unknown algorithm {name!r} (known: {sorted(classes)})")
    return classes[name]


class LoadedModel:
    """A policy plus its observation normaliser and config snapshot."""

    def __init__(self, folder: Path | str) -> None:
        import torch

        self._torch = torch
        self.folder = resolve_model(folder)
        self.config: dict[str, Any] = load_yaml(self.folder / "config.yaml")
        self.model_id = self.config.get("model_id", self.folder.name)
        self.env_config: dict[str, Any] = self.config["env_config"]
        self.algorithm = self.config.get("algorithm", "ppo")
        self.policy = _algorithm_class(self.algorithm).load(self.folder / "model.zip", device="cpu")
        self.vecnorm = None
        vn_path = self.folder / "vecnormalize.pkl"
        if vn_path.is_file():
            with vn_path.open("rb") as f:
                self.vecnorm = pickle.load(f)
            self.vecnorm.training = False
        space = self.policy.action_space
        self._low, self._high = space.low, space.high

    def normalise(self, obs: np.ndarray) -> np.ndarray:
        if self.vecnorm is None or not self.vecnorm.norm_obs:
            return obs
        return self.vecnorm.normalize_obs(obs)

    def act_batch(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Actions for a batch of raw observations, shape (n, obs_dim) → (n, 2).

        Same result as SB3 ``predict`` (normalise, policy, clip to the action space),
        without its per-call overhead.
        """
        torch = self._torch
        x = torch.as_tensor(np.asarray(self.normalise(obs)), dtype=torch.float32)
        with torch.no_grad():
            a = self.policy.policy._predict(x, deterministic=deterministic).cpu().numpy()
        return np.clip(a, self._low, self._high).astype(np.float32)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.act_batch(obs[None, :], deterministic)[0]
