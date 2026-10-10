"""Train PPO on the racing env (PROJECT_CONTEXT §8). Minimal Week 1 version.

    .venv\\Scripts\\python scripts\\train.py --tag ppo_rays5_tg --tracks training_ground
                                           [--steps 2000000] [--train ppo_default] [--physics default]
                                           [--env rays5] [--reward progress] [--seed 0] [--n-envs 6]
                                           [--scratch]

Every checkpoint (every ``checkpoint_every`` steps) and the final model are written as
self-contained folders (§9): models/R{run:03d}_{tag}@{steps}k/ with model.zip,
vecnormalize.pkl and config.yaml (full snapshot: env config, train config, seed, git hash).
--scratch writes everything to outputs/scratch/ instead (git-ignored) and uses S-ids.
TensorBoard logs go to runs/<run id>_<tag>/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import platform
import re
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from racer.config import PROJECT_ROOT, deep_merge, load_preset

MODELS_DIR = PROJECT_ROOT / "models"
SCRATCH_DIR = PROJECT_ROOT / "outputs" / "scratch"
RUNS_DIR = PROJECT_ROOT / "runs"


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def next_run_id(prefix: str, *dirs: Path) -> str:
    """Next free R### (or S### for scratch) across model folders and TensorBoard runs."""
    used = [0]
    pat = re.compile(rf"^{prefix}(\d{{3}})_")
    for d in dirs:
        if d.is_dir():
            used += [int(m.group(1)) for p in d.iterdir() if (m := pat.match(p.name))]
    return f"{prefix}{max(used) + 1:03d}"


def make_env(config: dict[str, Any], rank: int, seed: int):
    """Top-level factory (picklable for SubprocVecEnv on Windows)."""
    import torch
    torch.set_num_threads(1)
    from stable_baselines3.common.monitor import Monitor
    from racer.env.racing_env import RacingEnv
    env = RacingEnv(config)
    env.reset(seed=seed + rank)
    return Monitor(env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="short tag for the model id, e.g. ppo_rays5_tg")
    ap.add_argument("--tracks", nargs="+", default=["training_ground"])
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--train", default="ppo_default")
    ap.add_argument("--physics", default="default")
    ap.add_argument("--env", default="rays5")
    ap.add_argument("--reward", default="progress")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--scratch", action="store_true", help="write to outputs/scratch/ (git-ignored)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="strict override of the train preset, e.g. --set ppo.ent_coef=0.01 (repeatable)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not re.fullmatch(r"[a-z0-9_]+", args.tag):
        ap.error("--tag must be lowercase letters, digits and underscores")

    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from racer.agents.evaluation import evaluate, format_summary
    from racer.env.racing_env import make_env_config

    train_cfg = load_preset("train", args.train)
    if args.steps is not None:
        train_cfg["total_steps"] = args.steps
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.n_envs is not None:
        train_cfg["n_envs"] = args.n_envs
    for item in args.set:
        key, _, value = item.partition("=")
        override: dict[str, Any] = {}
        node = override
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
        train_cfg = deep_merge(train_cfg, override, strict=True)
    if train_cfg["algorithm"] != "ppo":
        ap.error(f"train preset {args.train!r} is not PPO")
    env_config = make_env_config(args.tracks, physics=args.physics, env=args.env, reward=args.reward)

    out_dir = SCRATCH_DIR if args.scratch else MODELS_DIR
    run_id = next_run_id("S" if args.scratch else "R", out_dir, RUNS_DIR if not args.scratch else SCRATCH_DIR)
    run_name = f"{run_id}_{args.tag}"
    tb_dir = (SCRATCH_DIR / "tb") if args.scratch else RUNS_DIR
    seed = int(train_cfg["seed"])
    torch.set_num_threads(int(train_cfg["torch_threads"]))

    base_snapshot = {
        "run_id": run_id, "tag": args.tag, "algorithm": "ppo", "seed": seed,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
        "obs_spec_version": env_config["env"]["obs_spec"]["version"], "notes": args.notes,
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "stable_baselines3": __import__("stable_baselines3").__version__,
                     "gymnasium": __import__("gymnasium").__version__, "numpy": np.__version__},
        "train": train_cfg, "env_config": env_config,
    }

    def save_folder(model, vecnorm, steps: int) -> Path:
        model_id = f"{run_name}@{steps // 1000}k"
        folder = out_dir / model_id
        if folder.exists():
            raise FileExistsError(f"{folder} already exists (model folders are immutable)")
        folder.mkdir(parents=True)
        model.save(folder / "model.zip")
        vecnorm.save(str(folder / "vecnormalize.pkl"))
        snap = dict(base_snapshot, model_id=model_id, steps=steps,
                    saved_at=dt.datetime.now().isoformat(timespec="seconds"))
        with (folder / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(snap, f, sort_keys=False, allow_unicode=True)
        return folder

    class RacingCallback(BaseCallback):
        """Checkpoints as model folders + episode stats (laps, lap time, terminations) to TensorBoard."""

        def __init__(self, every: int) -> None:
            super().__init__()
            self.every = every
            self.next_ckpt = every
            self.saved: list[Path] = []
            self.saved_steps = 0

        def _on_step(self) -> bool:
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                if "lap_time" in info:
                    self.logger.record_mean("race/lap_time", info["lap_time"])
                if done:
                    reason = info.get("termination") or "truncated"
                    for r in ("barrier", "offroad", "stuck", "wrong_way", "truncated"):
                        self.logger.record_mean(f"end/{r}", float(reason == r))
                    self.logger.record_mean("race/laps", info.get("laps", 0))
                    self.logger.record_mean("race/progress_m", info.get("progress", 0.0))
            if self.num_timesteps >= self.next_ckpt:
                self.saved.append(save_folder(self.model, self.training_env, self.num_timesteps))
                self.saved_steps = self.num_timesteps
                print(f"[{run_name}] checkpoint {self.saved[-1].name}", flush=True)
                self.next_ckpt += self.every
            return True

    n_envs = int(train_cfg["n_envs"])
    venv = SubprocVecEnv([partial(make_env, env_config, i, seed) for i in range(n_envs)])
    vn = train_cfg["vecnormalize"]
    venv = VecNormalize(venv, norm_obs=vn["norm_obs"], norm_reward=vn["norm_reward"],
                        clip_obs=vn["clip_obs"], gamma=vn["gamma"])
    venv.seed(seed)
    p = train_cfg["ppo"]
    lr0 = float(p["learning_rate"])
    lr = (lambda frac: lr0 * frac) if p["lr_schedule"] == "linear" else lr0
    pol = train_cfg["policy"]
    act_fn = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[pol["activation"]]
    model = PPO(
        "MlpPolicy", venv, learning_rate=lr, n_steps=p["n_steps"], batch_size=p["batch_size"],
        n_epochs=p["n_epochs"], gamma=p["gamma"], gae_lambda=p["gae_lambda"], clip_range=p["clip_range"],
        ent_coef=p["ent_coef"], vf_coef=p["vf_coef"], max_grad_norm=p["max_grad_norm"],
        policy_kwargs={"net_arch": {"pi": list(pol["net_arch"]), "vf": list(pol["net_arch"])}, "activation_fn": act_fn},
        tensorboard_log=str(tb_dir), seed=seed, device="cpu", verbose=0,
    )
    total = int(train_cfg["total_steps"])
    print(f"[{run_name}] PPO on {args.tracks} for {total:,} steps, {n_envs} envs, seed {seed}", flush=True)
    cb = RacingCallback(int(train_cfg["checkpoint_every"]))
    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=total, callback=cb, tb_log_name=run_name, progress_bar=False)
    finally:
        minutes = (time.perf_counter() - t0) / 60
        print(f"[{run_name}] trained {model.num_timesteps:,} steps in {minutes:.1f} min "
              f"({model.num_timesteps / max(minutes * 60, 1e-9):,.0f} steps/s)", flush=True)
    # The last checkpoint doubles as the final model if it is less than one rollout old.
    if not cb.saved or model.num_timesteps - cb.saved_steps >= p["n_steps"] * n_envs:
        cb.saved.append(save_folder(model, venv, model.num_timesteps))
    venv.close()
    final = cb.saved[-1]
    print(f"[{run_name}] final model: {final.relative_to(PROJECT_ROOT)}", flush=True)

    # Quick deterministic check on the training tracks (the full protocol is Week 2's evaluate.py).
    from racer.agents.loader import LoadedModel
    loaded = LoadedModel(final)
    summary = evaluate(env_config, lambda obs, env: loaded.act(obs), args.tracks, n_starts=3)
    print(format_summary(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
