"""Train a model on the racing env (PROJECT_CONTEXT §8; PPO by default, any registered algorithm).

    .venv\\Scripts\\python scripts\\train.py --tag ppo_rays5_tg --tracks training_ground
    .venv\\Scripts\\python scripts\\train.py --tag curric_wide --trackset gen_wide --parent R001_ppo_rays5_tg@1200k
                                           --train ppo_finetune
    .venv\\Scripts\\python scripts\\train.py --tag narrow --trackset maps7 --env rays7_narrow
        [--steps N] [--train ppo_default|sac_default] [--physics default] [--env rays5] [--reward progress]
        [--action-layout steer_throttle_brake] [--seed 0] [--n-envs 6] [--curriculum TEXT] [--notes TEXT]
        [--set ppo.ent_coef=0.01] [--scratch] [--allow-dirty] [--no-eval] [--eval-workers 6] [--protocol v2]

The snapshot also stores a manifest (inputs, outputs, network size; D23), so models can be
compared and the engine can check it supports a model before loading it.

Every checkpoint (every ``checkpoint_every`` steps) and the final model are written as
self-contained folders (§9): models/R{run:03d}_{tag}@{steps}k/ with model.zip,
vecnormalize.pkl and config.yaml (full snapshot: env config, train config, seed, git hash,
parent). The run and every checkpoint go into the registry as they are saved; after
training every checkpoint is evaluated (protocol v2, ``--protocol``), the best is marked ``is_best`` and
registry.xlsx is regenerated.

--parent fine-tunes a saved model (curriculum, D8): its weights and VecNormalize statistics
are the starting point; its obs spec and network must match. --scratch writes everything
to outputs/scratch/ (git-ignored, S-ids, scratch registry). TensorBoard logs go to runs/.
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

from racer.config import PROJECT_ROOT, deep_merge, load_preset, load_yaml

MODELS_DIR = PROJECT_ROOT / "models"
SCRATCH_DIR = PROJECT_ROOT / "outputs" / "scratch"
RUNS_DIR = PROJECT_ROOT / "runs"


# Paths whose uncommitted changes make a snapshot "-dirty" (code and configs; not the
# registry, eval copies or docs, which every run updates).
CODE_PATHS = ["racer", "scripts", "configs", "tracks", "pyproject.toml", "requirements.txt"]


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS],
                               cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
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


def reserve_run_id(prefix: str, out_dir: Path, *dirs: Path) -> str:
    """Next free run id, reserved atomically (a marker folder in ``out_dir/.run_ids``).

    A run's first model folder only appears at its first checkpoint, so two runs started
    together (parallel queue, T64) would otherwise get the same id. ``mkdir`` either creates
    the marker or fails, so exactly one process gets each id. Markers stay (ids are never reused).
    """
    markers = out_dir / ".run_ids"
    markers.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = next_run_id(prefix, *dirs, markers)
        try:
            (markers / f"{run_id}_reserved").mkdir()
            return run_id
        except FileExistsError:
            continue


def make_env(config: dict[str, Any], rank: int, seed: int):
    """Top-level factory (picklable for SubprocVecEnv on Windows)."""
    import torch
    torch.set_num_threads(1)
    from stable_baselines3.common.monitor import Monitor
    from racer.env.racing_env import RacingEnv
    env = RacingEnv(config)
    env.reset(seed=seed + rank)
    return Monitor(env)


def apply_sets(train_cfg: dict[str, Any], items: list[str]) -> dict[str, Any]:
    """Strict ``key.path=value`` overrides of the train preset."""
    for item in items:
        key, _, value = item.partition("=")
        override: dict[str, Any] = {}
        node = override
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
        train_cfg = deep_merge(train_cfg, override, strict=True)
    return train_cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True, help="short tag for the model id, e.g. ppo_rays5_tg")
    tg = ap.add_mutually_exclusive_group()
    tg.add_argument("--tracks", nargs="+", help="explicit track names (default: training_ground)")
    tg.add_argument("--trackset", help="track set preset in configs/trackset/ (D7)")
    ap.add_argument("--parent", default=None, help="model id or folder to fine-tune from (D8)")
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--train", default="ppo_default")
    ap.add_argument("--physics", default="default")
    ap.add_argument("--env", default="rays5")
    ap.add_argument("--reward", default="progress")
    ap.add_argument("--action-layout", default=None,
                    help="override the env preset's action layout, e.g. steer_throttle_brake (needs obs_spec v2)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--scratch", action="store_true", help="write to outputs/scratch/ (git-ignored)")
    ap.add_argument("--curriculum", default="", help="free text for the registry, e.g. 'stage 2 of 3'")
    ap.add_argument("--notes", default="")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="allow a real (R) run with uncommitted changes (snapshots then record '-dirty')")
    ap.add_argument("--no-eval", action="store_true", help="skip evaluating the checkpoints after training")
    ap.add_argument("--eval-workers", type=int, default=None, help="processes for evaluation (default: n_envs)")
    ap.add_argument("--protocol", default="v2", help="evaluation protocol after training (default v2, D20)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="strict override of the train preset, e.g. --set ppo.ent_coef=0.01 (repeatable)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not re.fullmatch(r"[a-z0-9_]+", args.tag):
        ap.error("--tag must be lowercase letters, digits and underscores")

    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from racer.agents.algorithms import algorithm_class, get_algorithm
    from racer.agents.compat import build_manifest
    from racer.agents.loader import resolve_model
    from racer.env.actions import action_spec_of
    from racer.env.racing_env import make_env_config
    from racer.registry.record import evaluate_and_record, registry_for
    from racer.registry.registry import register_folder, run_row

    train_cfg = load_preset("train", args.train)
    if args.steps is not None:
        train_cfg["total_steps"] = args.steps
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.n_envs is not None:
        train_cfg["n_envs"] = args.n_envs
    train_cfg = apply_sets(train_cfg, args.set)
    if not args.scratch and not args.allow_dirty and git_commit().endswith("-dirty"):
        ap.error("uncommitted changes: commit first so the snapshot's git_commit points at real code "
                 "(or pass --allow-dirty / --scratch)")
    try:
        algo = get_algorithm(train_cfg["algorithm"])
    except ValueError as e:
        ap.error(f"train preset {args.train!r}: {e}")
    if args.trackset:
        env_config = make_env_config(trackset=args.trackset, physics=args.physics, env=args.env, reward=args.reward)
        track_label = args.trackset
    else:
        tracks = args.tracks or ["training_ground"]
        env_config = make_env_config(tracks, physics=args.physics, env=args.env, reward=args.reward)
        track_label = "+".join(tracks)
    if args.action_layout:
        env_config["env"]["action_spec"] = {"layout": args.action_layout}
    try:
        manifest = build_manifest(env_config, train_cfg)     # also validates obs spec + action layout
    except ValueError as e:
        ap.error(str(e))

    parent_dir, parent_id = None, None
    if args.parent:
        parent_dir = resolve_model(args.parent)
        parent_snap = load_yaml(parent_dir / "config.yaml")
        parent_id = parent_snap.get("model_id", parent_dir.name)
        if parent_snap.get("algorithm", "ppo") != algo.name:
            ap.error(f"parent {parent_id} is not a {algo.name.upper()} model")
        if parent_snap["env_config"]["env"]["obs_spec"] != env_config["env"]["obs_spec"]:
            ap.error(f"parent {parent_id} was trained with a different obs_spec")
        if action_spec_of(parent_snap["env_config"]["env"]) != action_spec_of(env_config["env"]):
            ap.error(f"parent {parent_id} was trained with a different action layout")
        if parent_snap["train"]["policy"] != train_cfg["policy"]:
            ap.error(f"parent {parent_id} has a different policy network ({parent_snap['train']['policy']})")

    out_dir = SCRATCH_DIR if args.scratch else MODELS_DIR
    run_id = reserve_run_id("S" if args.scratch else "R", out_dir, out_dir, RUNS_DIR if not args.scratch else SCRATCH_DIR)
    run_name = f"{run_id}_{args.tag}"
    tb_dir = (SCRATCH_DIR / "tb") if args.scratch else RUNS_DIR
    seed = int(train_cfg["seed"])
    torch.set_num_threads(int(train_cfg["torch_threads"]))
    reg = registry_for(args.scratch)

    base_snapshot = {
        "run_id": run_id, "tag": args.tag, "algorithm": algo.name, "seed": seed,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
        "obs_spec_version": env_config["env"]["obs_spec"]["version"], "notes": args.notes,
        "parent_model": parent_id, "curriculum": args.curriculum,
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "stable_baselines3": __import__("stable_baselines3").__version__,
                     "gymnasium": __import__("gymnasium").__version__, "numpy": np.__version__},
        "train": train_cfg, "env_config": env_config, "manifest": manifest,
    }
    reg.upsert_run(run_row(base_snapshot))

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
        register_folder(reg, folder)
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
                    for r in ("barrier", "track_limits", "offroad", "stuck", "wrong_way", "truncated"):
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
    raw = SubprocVecEnv([partial(make_env, env_config, i, seed) for i in range(n_envs)])
    vn = train_cfg["vecnormalize"]
    if parent_dir is not None:
        # Continue from the parent's normaliser statistics (they keep updating).
        venv = VecNormalize.load(str(parent_dir / "vecnormalize.pkl"), raw)
        venv.training = True
        venv.norm_obs, venv.norm_reward = vn["norm_obs"], vn["norm_reward"]
        venv.clip_obs, venv.gamma = vn["clip_obs"], vn["gamma"]
    else:
        venv = VecNormalize(raw, norm_obs=vn["norm_obs"], norm_reward=vn["norm_reward"],
                            clip_obs=vn["clip_obs"], gamma=vn["gamma"])
    venv.seed(seed)
    model = algo.build(train_cfg, venv, tb_dir, seed)
    if parent_dir is not None:
        parent_model = algorithm_class(algo.name).load(parent_dir / "model.zip", device="cpu")
        model.policy.load_state_dict(parent_model.policy.state_dict())
        del parent_model
    total = int(train_cfg["total_steps"])
    origin = f", fine-tuning {parent_id}" if parent_id else ""
    print(f"[{run_name}] {algo.sb3_class} on {track_label} for {total:,} steps, {n_envs} envs, seed {seed}{origin}; "
          f"{manifest['obs_dim']} inputs, {manifest['action_dim']} outputs ({manifest['action_layout']}), "
          f"{manifest['n_params']:,} policy parameters", flush=True)
    cb = RacingCallback(int(train_cfg["checkpoint_every"]))
    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=total, callback=cb, tb_log_name=run_name, progress_bar=False)
    finally:
        minutes = (time.perf_counter() - t0) / 60
        print(f"[{run_name}] trained {model.num_timesteps:,} steps in {minutes:.1f} min "
              f"({model.num_timesteps / max(minutes * 60, 1e-9):,.0f} steps/s)", flush=True)
        reg.upsert_run({"run_id": run_id, "total_steps": model.num_timesteps, "wall_clock_min": round(minutes, 1)})
    # The last checkpoint doubles as the final model if it is less than one rollout old.
    if not cb.saved or model.num_timesteps - cb.saved_steps >= algo.update_steps(train_cfg):
        cb.saved.append(save_folder(model, venv, model.num_timesteps))
    venv.close()
    print(f"[{run_name}] final model: {cb.saved[-1].relative_to(PROJECT_ROOT)}", flush=True)

    if not args.no_eval:
        print(f"[{run_name}] evaluating {len(cb.saved)} checkpoints (protocol {args.protocol})", flush=True)
        evaluate_and_record(cb.saved, args.protocol, args.eval_workers or n_envs)
        best = next((r for r in reg.models() if r["run_id"] == run_id and r.get("is_best") == "true"), None)
        if best:
            print(f"[{run_name}] best: {best['model_id']}  seen {best['completion_rate']}  "
                  f"held-out {best['heldout_completion']}  unseen {best.get('unseen_completion') or '-'}  "
                  f"overall {best['overall_completion']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
