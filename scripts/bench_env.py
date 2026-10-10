"""Benchmark env throughput and run the baseline driver on tracks.

    .venv\\Scripts\\python scripts\\bench_env.py [--steps 20000] [--tracks showcase heldout]
                                               [--trackset final_mix]

Reports single-core env steps/s with random actions (target ≥ 3k, §7.1; with
--trackset this includes track switching and generation, which short random episodes stress) and the
baseline's lap times / completion per track (a sanity check of env + tracks).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from racer.agents.baseline import BaselineController
from racer.config import load_preset
from racer.env.racing_env import RacingEnv, make_env_config
from racer.track.io import TRACKS_DIR


def bench_random(steps: int, trackset: str | None = None) -> float:
    env = RacingEnv(make_env_config(trackset=trackset) if trackset else make_env_config(["training_ground"]))
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    env.step(np.zeros(2))   # JIT warm-up
    actions = rng.uniform(-1, 1, (steps, 2)).astype(np.float32)
    actions[:, 1] = np.abs(actions[:, 1])   # mostly forward, so episodes last
    t0 = time.perf_counter()
    for k in range(steps):
        _, _, term, trunc, _ = env.step(actions[k])
        if term or trunc:
            env.reset()
    return steps / (time.perf_counter() - t0)


def run_baseline(track: str) -> dict:
    env = RacingEnv(make_env_config([track]))
    ctrl = BaselineController(load_preset("train", "baseline"))
    env.reset(seed=0, options={"start_s": 0.0, "lateral": 0.0, "heading_offset": 0.0, "speed": 0.0})
    info, speeds = {}, []
    for _ in range(env.ep.max_steps):
        _, _, term, trunc, info = env.step(ctrl.act(env))
        speeds.append(info["speed"])
        if term or trunc:
            break
    return {"laps": info["laps"], "termination": info["termination"], "lap_times": env.lap_times,
            "mean_speed": float(np.mean(speeds)), "length": env.track.length}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--tracks", nargs="*", default=["showcase", "heldout"])
    ap.add_argument("--trackset", default=None, help="benchmark random actions on this track set")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    sps = bench_random(args.steps, args.trackset)
    print(f"random actions{' on ' + args.trackset if args.trackset else ''}: {sps:,.0f} env steps/s on one core ({'OK' if sps >= 3000 else 'BELOW'} target 3k)")
    ok = True
    for group in args.tracks:
        for path in sorted((TRACKS_DIR / group).glob("*.json")):
            r = run_baseline(path.stem)
            lap = f"{r['lap_times'][0]:.1f} s" if r["lap_times"] else "-"
            good = r["termination"] is None and r["laps"] >= 1
            ok &= good
            print(f"  {'OK  ' if good else 'FAIL'} baseline {path.stem:<18} {r['length']:5.0f} m  lap {lap:>8}  "
                  f"mean {r['mean_speed'] * 3.6:5.1f} km/h  laps {r['laps']}  end={r['termination']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
