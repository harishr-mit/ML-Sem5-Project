"""Training throughput benchmark: CPU vs GPU, minibatch size, runs in parallel (log.md T62).

    .venv\\Scripts\\python scripts\\bench_train.py --steps 200000 [--n-envs 6] [--batch-size 256]
    .venv-gpu\\Scripts\\python scripts\\bench_train.py --device cuda --steps 200000 --batch-size 1024
    .venv\\Scripts\\python scripts\\bench_train.py --runs 2 --n-envs 3 --steps 200000   2 trainings at once

Trains PPO (``ppo_default`` + overrides) without checkpoints, logs or evaluation, and reports
env steps per second, split into collection (env stepping) and update (gradient steps)
time. With ``--runs K`` it starts K copies at the same time (separate processes, different
seeds) and reports their total throughput. Nothing is saved; use an idle machine.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def make_env(config: dict[str, Any], rank: int, seed: int):
    """Top-level factory (picklable for SubprocVecEnv on Windows)."""
    import torch
    torch.set_num_threads(1)
    from stable_baselines3.common.monitor import Monitor
    from racer.env.racing_env import RacingEnv
    env = RacingEnv(config)
    env.reset(seed=seed + rank)
    return Monitor(env)


def bench_one(args) -> dict[str, Any]:
    from functools import partial

    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from racer.agents.algorithms import get_algorithm
    from racer.config import load_preset
    from racer.env.racing_env import make_env_config

    torch.set_num_threads(1)
    cfg = load_preset("train", "ppo_default")
    cfg["n_envs"] = args.n_envs
    cfg["ppo"]["batch_size"] = args.batch_size
    env_config = make_env_config(trackset=args.trackset, env=args.env)
    venv = VecNormalize(SubprocVecEnv([partial(make_env, env_config, i, args.seed) for i in range(args.n_envs)]),
                        gamma=cfg["vecnormalize"]["gamma"])
    model = get_algorithm("ppo").build(cfg, venv, None, args.seed, device=args.device)

    class Phases(BaseCallback):
        def __init__(self):
            super().__init__()
            self.collect = 0.0
            self.t = None

        def _on_rollout_start(self):
            self.t = time.perf_counter()

        def _on_rollout_end(self):
            self.collect += time.perf_counter() - self.t

        def _on_step(self):
            return True

    cb = Phases()
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=cb)
    wall = time.perf_counter() - t0
    venv.close()
    return {"device": args.device, "n_envs": args.n_envs, "batch_size": args.batch_size, "steps": model.num_timesteps,
            "wall_s": round(wall, 1), "steps_per_s": round(model.num_timesteps / wall),
            "collect_s": round(cb.collect, 1), "update_s": round(wall - cb.collect, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu", help="cpu | cuda (needs the CUDA torch build, .venv-gpu)")
    ap.add_argument("--runs", type=int, default=1, help="start this many trainings at the same time")
    ap.add_argument("--trackset", default="maps7")
    ap.add_argument("--env", default="rays5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print one JSON line (used for --runs)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.runs <= 1:
        res = bench_one(args)
        print(json.dumps(res) if args.json else
              f"{res['device']} · {res['n_envs']} envs · batch {res['batch_size']}: {res['steps_per_s']} steps/s "
              f"(collect {res['collect_s']} s, update {res['update_s']} s, {res['steps']:,} steps)")
        return 0

    base = [sys.executable, str(Path(__file__).resolve()), "--json", "--steps", str(args.steps),
            "--n-envs", str(args.n_envs), "--batch-size", str(args.batch_size), "--device", args.device,
            "--trackset", args.trackset, "--env", args.env]
    t0 = time.perf_counter()
    procs = [subprocess.Popen(base + ["--seed", str(args.seed + 100 * k)], stdout=subprocess.PIPE, text=True)
             for k in range(args.runs)]
    results = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
    wall = time.perf_counter() - t0
    total = sum(r["steps_per_s"] for r in results)     # each run's own training time (start-up excluded)
    print(f"{args.runs} runs × {args.n_envs} envs · {args.device} · batch {args.batch_size}: "
          f"total {total} steps/s (each {', '.join(str(r['steps_per_s']) for r in results)} steps/s; "
          f"{wall:.0f} s wall incl. start-up)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
