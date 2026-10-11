"""Algorithm registry (PROJECT_CONTEXT §9, D23): how to build, load and describe each RL algorithm.

The train preset's ``algorithm`` names an entry here; its hyperparameters live in the
preset section of the same name (``ppo:``, ``sac:``). The loader uses the same registry,
so a model of any registered algorithm runs in the demo. Adding an algorithm = one entry
(and a train preset); existing models are never affected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class Algorithm:
    name: str
    sb3_class: str              # attribute of stable_baselines3
    description: str
    build: Callable[..., Any]   # (train_cfg, venv, tb_dir, seed, device="cpu") -> model
    update_steps: Callable[[Mapping[str, Any]], int]   # env steps (all envs) between policy updates


def _activation(name: str):
    import torch
    return {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[name]


def _lr(value: float, schedule: str):
    v = float(value)
    return (lambda frac: v * frac) if schedule == "linear" else v


def _build_ppo(train_cfg, venv, tb_dir, seed, device="cpu"):
    from stable_baselines3 import PPO

    p = train_cfg["ppo"]
    pol = train_cfg["policy"]
    return PPO(
        "MlpPolicy", venv, learning_rate=_lr(p["learning_rate"], p["lr_schedule"]), n_steps=p["n_steps"],
        batch_size=p["batch_size"], n_epochs=p["n_epochs"], gamma=p["gamma"], gae_lambda=p["gae_lambda"],
        clip_range=p["clip_range"], ent_coef=p["ent_coef"], vf_coef=p["vf_coef"], max_grad_norm=p["max_grad_norm"],
        policy_kwargs={"net_arch": {"pi": list(pol["net_arch"]), "vf": list(pol["net_arch"])},
                       "activation_fn": _activation(pol["activation"])},
        tensorboard_log=str(tb_dir) if tb_dir else None, seed=seed, device=device, verbose=0,
    )


def _build_sac(train_cfg, venv, tb_dir, seed, device="cpu"):
    from stable_baselines3 import SAC

    s = train_cfg["sac"]
    pol = train_cfg["policy"]
    return SAC(
        "MlpPolicy", venv, learning_rate=_lr(s["learning_rate"], s["lr_schedule"]), buffer_size=s["buffer_size"],
        learning_starts=s["learning_starts"], batch_size=s["batch_size"], tau=s["tau"], gamma=s["gamma"],
        train_freq=s["train_freq"], gradient_steps=s["gradient_steps"], ent_coef=s["ent_coef"],
        policy_kwargs={"net_arch": {"pi": list(pol["net_arch"]), "qf": list(pol["net_arch"])},
                       "activation_fn": _activation(pol["activation"])},
        tensorboard_log=str(tb_dir) if tb_dir else None, seed=seed, device=device, verbose=0,
    )


ALGORITHMS: dict[str, Algorithm] = {
    "ppo": Algorithm(
        "ppo", "PPO",
        "Proximal Policy Optimisation: on-policy; collects a batch of fresh driving, then improves the policy "
        "in small, clipped steps. Stable and works well with many parallel envs.",
        _build_ppo, lambda cfg: int(cfg["ppo"]["n_steps"]) * int(cfg["n_envs"])),
    "sac": Algorithm(
        "sac", "SAC",
        "Soft Actor-Critic: off-policy; learns from a replay buffer of all past driving and rewards keeping "
        "the policy random (entropy). Usually more sample-efficient, slower per step (E7).",
        _build_sac, lambda cfg: int(cfg["sac"]["train_freq"]) * int(cfg["n_envs"])),
}


def get_algorithm(name: str) -> Algorithm:
    if name not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {name!r} (known: {sorted(ALGORITHMS)})")
    return ALGORITHMS[name]


def algorithm_class(name: str):
    """The SB3 class used to load a saved model of this algorithm."""
    import stable_baselines3   # imported lazily: torch is slow to import

    return getattr(stable_baselines3, get_algorithm(name).sb3_class)
