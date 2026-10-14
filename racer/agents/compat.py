"""Model manifest and engine compatibility check (PROJECT_CONTEXT §9, D23).

Every part of a model that the engine must understand is versioned and registered:
observation spec versions and features, action layouts, physics versions, reward types and
algorithms. :func:`check` compares what a model's snapshot needs with what this engine
supports, so a model the engine can't run yet is reported (and greyed out in the picker)
instead of crashing the demo. To support it, register the missing piece.

The **manifest** summarises a model for comparisons and the AI Sandbox: input and output
sizes and labels, network shape and parameter count. New snapshots store it
(``manifest:`` in config.yaml); for older ones it is derived from the snapshot.
"""
from __future__ import annotations

from typing import Any, Mapping


class IncompatibleModel(RuntimeError):
    """The engine lacks something this model needs."""


def engine_support() -> dict[str, list]:
    """Everything this engine can run."""
    from racer.agents.algorithms import ALGORITHMS
    from racer.env.actions import LAYOUTS
    from racer.env.features import FEATURES
    from racer.env.obs_spec import OBSERVERS
    from racer.env.reward import REWARDS
    from racer.physics.versions import PHYSICS

    return {"obs_spec": sorted(OBSERVERS), "features": sorted(FEATURES), "action_layout": sorted(LAYOUTS),
            "physics": sorted(PHYSICS), "reward": sorted(REWARDS), "algorithm": sorted(ALGORITHMS)}


def requirements(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """What a model's snapshot needs from the engine."""
    from racer.env.actions import action_spec_of
    from racer.physics.versions import physics_version

    env = snapshot["env_config"]
    spec = env["env"]["obs_spec"]
    return {
        "obs_spec": spec.get("version"),
        "features": [f.get("feature") for f in spec.get("features", [])],
        "action_layout": action_spec_of(env["env"]).get("layout"),
        "physics": physics_version(env["physics"]),
        "reward": env["reward"].get("type"),
        "algorithm": snapshot.get("algorithm", "ppo"),
    }


def check(snapshot: Mapping[str, Any]) -> list[str]:
    """Reasons this engine can't run the model (empty list = it can)."""
    sup = engine_support()
    req = requirements(snapshot)
    problems = []
    for key in ("obs_spec", "action_layout", "physics", "reward", "algorithm"):
        if req[key] not in sup[key]:
            problems.append(f"needs {key} {req[key]!r} (engine has {sup[key]})")
    missing = [f for f in req["features"] if f not in sup["features"]]
    if missing:
        problems.append(f"needs observation features {missing}")
    return problems


def ensure_compatible(snapshot: Mapping[str, Any]) -> None:
    problems = check(snapshot)
    if problems:
        mid = snapshot.get("model_id", "model")
        raise IncompatibleModel(f"{mid} can't run on this engine: " + "; ".join(problems))


# ------------------------------------------------------------------ manifest
def _mlp(n_in: int, arch: list[int], n_out: int | None) -> int:
    """Weights + biases of a fully connected stack."""
    sizes = [n_in, *arch] + ([n_out] if n_out is not None else [])
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def count_params(algorithm: str, obs_dim: int, act_dim: int, net_arch: list[int]) -> dict[str, int]:
    """Parameter counts of SB3's MlpPolicy: ``policy`` = the network that drives, ``total`` = all trained."""
    arch = [int(n) for n in net_arch]
    if algorithm == "ppo":
        policy = _mlp(obs_dim, arch, act_dim) + act_dim               # + log_std
        return {"policy": policy, "total": policy + _mlp(obs_dim, arch, 1)}
    if algorithm == "sac":
        h = arch[-1] if arch else obs_dim
        policy = _mlp(obs_dim, arch, None) + 2 * (h * act_dim + act_dim)    # latent + mean + log_std heads
        return {"policy": policy, "total": policy + 2 * _mlp(obs_dim + act_dim, arch, 1)}   # twin critics
    if algorithm == "ga":
        policy = _mlp(obs_dim, arch, act_dim)                          # the evolved weights, nothing else trains
        return {"policy": policy, "total": policy}
    raise ValueError(f"no parameter count for algorithm {algorithm!r}")


def build_manifest(env_config: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Manifest from an env config and train preset (needs the engine to support them)."""
    from racer.env.actions import action_spec_of, build_action_layout
    from racer.env.obs_spec import build_observer
    from racer.physics.versions import physics_version

    env = env_config["env"]
    layout = build_action_layout(action_spec_of(env))
    obs = build_observer(env["obs_spec"], env["raycast"], layout)
    algorithm = train_cfg.get("algorithm", "ppo")
    pol = train_cfg["policy"]
    params = count_params(algorithm, obs.dim, layout.dim, list(pol["net_arch"]))
    return {
        "obs_spec": env["obs_spec"]["name"], "obs_spec_version": int(env["obs_spec"]["version"]),
        "obs_dim": int(obs.dim), "obs_labels": list(obs.labels), "inputs": obs.describe(),
        "action_layout": layout.name, "action_dim": layout.dim, "action_labels": list(layout.labels),
        "physics_version": physics_version(env_config["physics"]), "algorithm": algorithm,
        "net_arch": [int(n) for n in pol["net_arch"]], "activation": pol["activation"],
        "n_params": params["policy"], "n_params_total": params["total"],
    }


def manifest_of(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """The stored manifest, or one derived from the snapshot (models from before D23)."""
    if snapshot.get("manifest"):
        return dict(snapshot["manifest"])
    return build_manifest(snapshot["env_config"], dict(snapshot["train"], algorithm=snapshot.get("algorithm", "ppo")))
