"""Deterministic evaluation episodes (minimal Week 1 version of PROJECT_CONTEXT §8's protocol).

Each start is a fixed point on the track (evenly spaced along it, from standstill, on the
centreline), so results are reproducible. Week 2 adds the full protocol (showcase ×3 +
held-out ×1, registry metrics, thumbnails).
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from racer.env.racing_env import RacingEnv


def run_episode(env: RacingEnv, act: Callable[[np.ndarray, RacingEnv], np.ndarray], track: str,
                start_s: float) -> dict[str, Any]:
    obs, _ = env.reset(seed=0, options={"track": track, "start_s": start_s, "lateral": 0.0,
                                        "heading_offset": 0.0, "speed": 0.0})
    info: dict[str, Any] = {}
    total_r, speeds, offroad = 0.0, [], 0.0
    while True:
        obs, r, term, trunc, info = env.step(act(obs, env))
        total_r += r
        speeds.append(info["speed"])
        offroad += info["offroad_frac"]
        if term or trunc:
            break
    return {
        "track": env.track.name, "start_s": start_s, "return": total_r, "laps": info["laps"],
        "termination": info["termination"], "lap_times": list(env.lap_times),
        "mean_speed": float(np.mean(speeds)), "offroad_pct": 100.0 * offroad / len(speeds),
        "steps": env.steps,
    }


def evaluate(env_config: Mapping[str, Any], act: Callable[[np.ndarray, RacingEnv], np.ndarray],
             tracks: Sequence[str], n_starts: int = 3) -> dict[str, Any]:
    """Run ``n_starts`` evenly spaced starts on each track; summarise completion and lap times."""
    config = dict(env_config)
    config["tracks"] = list(tracks)
    env = RacingEnv(config)
    episodes = []
    for name in tracks:
        env.set_track(name)
        for k in range(n_starts):
            episodes.append(run_episode(env, act, name, start_s=k * env.track.length / n_starts))
    laps = [lt for e in episodes for lt in e["lap_times"]]
    summary = {
        "episodes": episodes,
        "completion_rate": float(np.mean([e["termination"] is None and e["laps"] >= 1 for e in episodes])),
        "crash_rate": float(np.mean([e["termination"] is not None for e in episodes])),
        "best_lap": min(laps) if laps else None,
        "mean_lap": float(np.mean(laps)) if laps else None,
        "mean_speed": float(np.mean([e["mean_speed"] for e in episodes])),
    }
    return summary


def format_summary(summary: Mapping[str, Any]) -> str:
    lines = []
    for e in summary["episodes"]:
        lap = ", ".join(f"{t:.1f}s" for t in e["lap_times"]) or "-"
        lines.append(f"  {e['track']:<16} start {e['start_s']:6.0f} m  laps {e['laps']}  end={e['termination']}  "
                     f"laps [{lap}]  mean {e['mean_speed'] * 3.6:5.1f} km/h  off-road {e['offroad_pct']:4.1f}%")
    best = f"{summary['best_lap']:.2f}s" if summary["best_lap"] else "-"
    lines.append(f"  completion {summary['completion_rate']:.0%}  crashes {summary['crash_rate']:.0%}  best lap {best}")
    return "\n".join(lines)
