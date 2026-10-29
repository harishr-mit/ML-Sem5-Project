"""Input normalisation for neuroevolution (D33, PROJECT_CONTEXT §8).

PPO standardises its inputs with VecNormalize's running statistics. A GA car can't: the statistics
would move under the population, and a car could no longer be rebuilt from the lineage. So the
statistics are measured once, before generation 0, from the hand-coded baseline driving the
training maps (clean, and with action noise so slides and corrections are in the data), frozen in
the run's snapshot, and applied to every car with VecNormalize's own formula.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

EPSILON = 1e-8          # VecNormalize's


def calibrate_obs(env_config: Mapping[str, Any], maps: Sequence[str], seed: int, min_std: float, clip: float,
                  steps_per_map: int = 4000, noise: float = 0.3) -> dict[str, Any]:
    """Mean and variance of every input over baseline driving: half the steps clean, half noisy.

    Deterministic in ``seed``. The variance has a floor of ``min_std``², so an input that hardly
    moves under the baseline (a second ray crossing that is almost always 'none') isn't blown up
    into huge values when a car later makes it move.
    """
    from racer.agents.baseline import BaselineController
    from racer.config import load_preset
    from racer.env.racing_env import RacingEnv

    rng = np.random.default_rng([int(seed), 77])
    env = RacingEnv(dict(env_config))
    ctrl = BaselineController(load_preset("train", "baseline"))
    total, s1, s2 = 0, 0.0, 0.0
    for name in maps:
        for sigma in (0.0, noise):
            done = 0
            while done < steps_per_map // 2:
                obs, _ = env.reset(seed=int(rng.integers(2 ** 31)), options={"track": name})
                for _ in range(steps_per_map // 2 - done):
                    act = np.clip(ctrl.act(env) + sigma * rng.standard_normal(2), -1.0, 1.0)
                    obs, _, term, trunc, _ = env.step(act)
                    x = obs.astype(np.float64)
                    s1, s2, total, done = s1 + x, s2 + x * x, total + 1, done + 1
                    if term or trunc:
                        break
    mean = s1 / total
    var = np.maximum(s2 / total - mean * mean, min_std ** 2)
    return {"mean": [float(v) for v in mean], "var": [float(v) for v in var], "clip": float(clip), "count": total}


def standardise(obs: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    """VecNormalize.normalize_obs with frozen statistics."""
    mean, var = np.asarray(stats["mean"]), np.asarray(stats["var"])
    return np.clip((obs - mean) / np.sqrt(var + EPSILON), -stats["clip"], stats["clip"]).astype(np.float32)
