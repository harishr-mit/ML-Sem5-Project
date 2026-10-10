"""Deterministic evaluation (PROJECT_CONTEXT §8, D9, D10).

Every start is a fixed point on the track (from standstill, on the centreline), so results
are reproducible. :func:`evaluate` is the quick check used by ``train.py``; the versioned
protocol (``configs/eval/v1.yaml``) runs through :func:`run_protocol`, and
:func:`evaluate_model` writes its files:

* ``models/<id>/eval.json`` + ``thumb_<track>.png``: added once, never overwritten (D10);
  a later protocol version writes ``eval_v2.json`` / ``thumb_v2_<track>.png``.
* a copy of the same files in ``outputs/eval/<id>/`` (scratch models: ``outputs/scratch/eval/``).

The hand-coded baseline is evaluated with the same protocol as pseudo-model ``BASE_pid``;
its best laps are the reference for ``lap_ratio``.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from racer.config import PROJECT_ROOT, load_preset, load_yaml
from racer.env.racing_env import RacingEnv, make_env_config

EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
SCRATCH_EVAL_DIR = PROJECT_ROOT / "outputs" / "scratch" / "eval"
SCRATCH_DIR = PROJECT_ROOT / "outputs" / "scratch"

Act = Callable[[np.ndarray, RacingEnv], np.ndarray]


# ----------------------------------------------------------------- episodes
def run_episode(env: RacingEnv, act: Act, track: str, start_s: float,
                start: Mapping[str, float] | None = None, trajectory: bool = False) -> dict[str, Any]:
    start = start or {"lateral": 0.0, "heading_offset_deg": 0.0, "speed": 0.0}
    obs, _ = env.reset(seed=0, options={"track": track, "start_s": start_s, "lateral": start["lateral"],
                                        "heading_offset": math.radians(start["heading_offset_deg"]),
                                        "speed": start["speed"]})
    info: dict[str, Any] = {}
    g = env.params.gravity
    total_r, speeds, offroad, lat_g = 0.0, [], 0.0, 0.0
    traj: list[tuple[float, float, float]] = []
    while True:
        obs, r, term, trunc, info = env.step(act(obs, env))
        total_r += r
        speeds.append(info["speed"])
        offroad += info["offroad_frac"]
        lat_g += abs(env.car.a_lat) / g
        if trajectory:
            traj.append((env.car.x, env.car.y, info["speed"]))
        if term or trunc:
            break
    out = {
        "track": env.track.key, "start_s": start_s, "return": total_r, "laps": info["laps"],
        "termination": info["termination"], "lap_times": list(env.lap_times),
        "mean_speed": float(np.mean(speeds)), "offroad_pct": 100.0 * offroad / len(speeds),
        "mean_lat_g": lat_g / len(speeds), "steps": env.steps,
        "completed": info["termination"] is None and info["laps"] >= 1,
    }
    if trajectory:
        out["trajectory"] = traj
    return out


def evaluate(env_config: Mapping[str, Any], act: Act, tracks: Sequence[str], n_starts: int = 3) -> dict[str, Any]:
    """Quick check: ``n_starts`` evenly spaced starts on each track; completion and lap times."""
    config = dict(env_config)
    config["tracks"] = list(tracks)
    env = RacingEnv(config)
    episodes = []
    for name in tracks:
        env.set_track(name)
        for k in range(n_starts):
            episodes.append(run_episode(env, act, name, start_s=k * env.track.length / n_starts))
    laps = [lt for e in episodes for lt in e["lap_times"]]
    return {
        "episodes": episodes,
        "completion_rate": float(np.mean([e["completed"] for e in episodes])),
        "crash_rate": float(np.mean([e["termination"] is not None for e in episodes])),
        "best_lap": min(laps) if laps else None,
        "mean_lap": float(np.mean(laps)) if laps else None,
        "mean_speed": float(np.mean([e["mean_speed"] for e in episodes])),
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    lines = []
    for e in summary["episodes"]:
        lap = ", ".join(f"{t:.1f}s" for t in e["lap_times"]) or "-"
        lines.append(f"  {e['track']:<16} start {e['start_s']:6.0f} m  laps {e['laps']}  end={e['termination']}  "
                     f"laps [{lap}]  mean {e['mean_speed'] * 3.6:5.1f} km/h  off-road {e['offroad_pct']:4.1f}%")
    best = f"{summary['best_lap']:.2f}s" if summary["best_lap"] else "-"
    lines.append(f"  completion {summary['completion_rate']:.0%}  crashes {summary['crash_rate']:.0%}  best lap {best}")
    return "\n".join(lines)


# ----------------------------------------------------------------- protocol
def load_protocol(name: str = "v1") -> dict[str, Any]:
    p = load_preset("eval", name)
    p["name"] = name
    return p


def run_protocol(env_config: Mapping[str, Any], act: Act, protocol: Mapping[str, Any],
                 baseline_laps: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Run every episode of the protocol and compute the registry metrics."""
    groups = protocol["groups"]
    all_tracks = [t for g in groups.values() for t in g["tracks"]]
    config = dict(env_config)
    config["tracks"] = list(all_tracks)
    env = RacingEnv(config)
    episodes: list[dict[str, Any]] = []
    for gname, g in groups.items():
        for name in g["tracks"]:
            env.set_track(name)
            for k in range(int(g["starts"])):
                e = run_episode(env, act, name, k * env.track.length / int(g["starts"]), protocol["start"],
                                trajectory=(gname == "showcase" and k == 0))
                e["group"] = gname
                episodes.append(e)
    return {"metrics": compute_metrics(episodes, groups, baseline_laps), "episodes": episodes}


def compute_metrics(episodes: Sequence[Mapping[str, Any]], groups: Mapping[str, Any],
                    baseline_laps: Mapping[str, float] | None) -> dict[str, Any]:
    def mean(xs):
        xs = list(xs)
        return float(np.mean(xs)) if xs else None

    show = [e for e in episodes if e["group"] == "showcase"]
    held = [e for e in episodes if e["group"] == "heldout"]
    m: dict[str, Any] = {
        "completion_rate": mean(e["completed"] for e in show),
        "heldout_completion": mean(e["completed"] for e in held),
        "overall_completion": mean(e["completed"] for e in episodes),
        "crash_rate": mean(e["termination"] is not None for e in episodes),
        "mean_speed": mean(e["mean_speed"] for e in episodes),
        "offroad_pct": mean(e["offroad_pct"] for e in episodes),
        "mean_lat_g": mean(e["mean_lat_g"] for e in episodes),
    }
    ratios = []
    for name in groups.get("showcase", {}).get("tracks", []):
        laps = [t for e in show if e["track"] == name for t in e["lap_times"]]
        best = min(laps) if laps else None
        m[f"best_lap_{name}"] = best
        m[f"mean_lap_{name}"] = mean(laps)
        if best is not None and baseline_laps and baseline_laps.get(name):
            ratios.append(best / baseline_laps[name])
    m["lap_ratio"] = mean(ratios)
    m["score"] = m["overall_completion"]
    # Lap times are multiples of the 0.05 s agent step; strip float noise.
    return {k: (round(v, 3 if "lap_" in k else 4) if isinstance(v, float) else v) for k, v in m.items()}


def rank_key(metrics: Mapping[str, Any]) -> tuple[float, float]:
    """Sort key (ascending = better): completion first, then lap time relative to the baseline."""
    comp = metrics.get("overall_completion")
    ratio = metrics.get("lap_ratio")
    return (-(comp if comp is not None else -1.0), ratio if ratio is not None else math.inf)


# ------------------------------------------------------------------- files
def eval_filenames(version: int) -> tuple[str, str]:
    """(eval json name, thumbnail prefix) for a protocol version; v1 keeps §9's plain names."""
    return ("eval.json", "thumb_") if version == 1 else (f"eval_v{version}.json", f"thumb_v{version}_")


def eval_root_for(folder: Path) -> Path:
    try:
        folder.resolve().relative_to(SCRATCH_DIR.resolve())
        return SCRATCH_EVAL_DIR
    except ValueError:
        return EVAL_DIR


def save_thumbnail(track, trajectory: Sequence[tuple[float, float, float]], path: Path,
                   px: int, speed_max: float, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 0.9))
    for off in (-track.half_width, track.half_width):
        e = track.edges(off)
        ax.plot(np.r_[e[:, 0], e[:1, 0]], np.r_[e[:, 1], e[:1, 1]], color="#555555", lw=0.6)
    if len(trajectory) > 1:
        t = np.asarray(trajectory)
        seg = np.stack([t[:-1, :2], t[1:, :2]], axis=1)
        lc = LineCollection(seg, cmap="turbo", norm=plt.Normalize(0, speed_max), linewidths=1.4)
        lc.set_array(t[1:, 2])
        ax.add_collection(lc)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.5, 0.95, title, ha="center", va="center", fontsize=7)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_eval_files(folder: Path | None, model_id: str, result: Mapping[str, Any], protocol: Mapping[str, Any],
                     env_config: Mapping[str, Any], eval_root: Path) -> dict[str, Any]:
    """Write eval json + thumbnails to the model folder (only if absent) and copy them to ``eval_root``."""
    from racer.env.trackset import get_track

    json_name, thumb_prefix = eval_filenames(int(protocol["version"]))
    out_dir = eval_root / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [d for d in (folder, out_dir) if d is not None]
    doc = {
        "model_id": model_id, "protocol": protocol["name"], "protocol_version": int(protocol["version"]),
        "evaluated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "metrics": result["metrics"], "baseline_laps": result.get("baseline_laps"),
        "episodes": [{k: v for k, v in e.items() if k != "trajectory"} for e in result["episodes"]],
    }
    text = json.dumps(doc, indent=1)
    th = protocol["thumbnail"]
    thumbs = []
    for e in result["episodes"]:
        if "trajectory" not in e:
            continue
        track = get_track(e["track"], env_config["track"])
        best = result["metrics"].get(f"best_lap_{e['track']}")
        title = f"{model_id}  {track.name}  " + (f"{best:.2f} s" if best else "no lap")
        name = f"{thumb_prefix}{e['track']}.png"
        tmp = out_dir / f".{name}.tmp.png"
        save_thumbnail(track, e["trajectory"], tmp, int(th["px"]), float(th["speed_max"]), title)
        thumbs.append((name, tmp))
    for d in targets:
        if not (d / json_name).exists():      # add once, never overwrite (D10)
            (d / json_name).write_text(text, encoding="utf-8")
        for name, tmp in thumbs:
            if not (d / name).exists():
                shutil.copyfile(tmp, d / name)
    for _, tmp in thumbs:
        tmp.unlink()
    return doc


# ------------------------------------------------------------------ models
def baseline_laps(protocol: Mapping[str, Any], eval_root: Path = EVAL_DIR) -> dict[str, float]:
    """Best showcase laps of the baseline under this protocol (evaluated once, then read from disk)."""
    json_name, _ = eval_filenames(int(protocol["version"]))
    path = eval_root / protocol["baseline_id"] / json_name
    if not path.is_file():
        path = EVAL_DIR / protocol["baseline_id"] / json_name
    if not path.is_file():
        evaluate_baseline(protocol)
    metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    return {t: metrics[f"best_lap_{t}"] for t in protocol["groups"]["showcase"]["tracks"]
            if metrics.get(f"best_lap_{t}")}


def evaluate_baseline(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the hand-coded baseline (default presets) as pseudo-model ``baseline_id``."""
    from racer.agents.baseline import BaselineController

    env_config = make_env_config(["training_ground"])
    ctrl = BaselineController(load_preset("train", "baseline"))
    result = run_protocol(env_config, lambda obs, env: ctrl.act(env), protocol)
    laps = {t: result["metrics"][f"best_lap_{t}"] for t in protocol["groups"]["showcase"]["tracks"]
            if result["metrics"].get(f"best_lap_{t}")}
    result["metrics"] = compute_metrics(result["episodes"], protocol["groups"], laps)
    result["baseline_laps"] = laps
    return write_eval_files(None, protocol["baseline_id"], result, protocol, env_config, EVAL_DIR)


def evaluate_model(folder: Path | str, protocol_name: str = "v1") -> dict[str, Any]:
    """Evaluate one model folder (or return its existing results for this protocol)."""
    folder = Path(folder)
    protocol = load_protocol(protocol_name)
    json_name, _ = eval_filenames(int(protocol["version"]))
    eval_root = eval_root_for(folder)
    model_id = load_yaml(folder / "config.yaml").get("model_id", folder.name)
    if (folder / json_name).is_file():
        doc = json.loads((folder / json_name).read_text(encoding="utf-8"))
        copy = eval_root / model_id
        copy.mkdir(parents=True, exist_ok=True)
        for f in folder.glob("*"):
            if (f.name == json_name or f.name.startswith(eval_filenames(int(protocol["version"]))[1])) \
                    and not (copy / f.name).exists():
                shutil.copyfile(f, copy / f.name)
        return doc

    import torch
    torch.set_num_threads(1)
    from racer.agents.loader import LoadedModel

    m = LoadedModel(folder)
    laps = baseline_laps(protocol, eval_root)
    result = run_protocol(m.env_config, lambda obs, env: m.act(obs), protocol, laps)
    result["baseline_laps"] = laps
    return write_eval_files(folder, model_id, result, protocol, m.env_config, eval_root)


def _evaluate_model_worker(args: tuple[str, str]) -> dict[str, Any]:
    return evaluate_model(args[0], args[1])


def evaluate_many(folders: Sequence[Path | str], protocol_name: str = "v1", workers: int = 6) -> list[dict[str, Any]]:
    """Evaluate several model folders in a process pool (callers need the __main__ guard on Windows)."""
    protocol = load_protocol(protocol_name)
    roots = {eval_root_for(Path(f)) for f in folders}
    for root in roots:
        baseline_laps(protocol, root)          # make sure the reference exists before forking
    if workers <= 1 or len(folders) <= 1:
        return [evaluate_model(f, protocol_name) for f in folders]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(workers, len(folders))) as pool:
        return list(pool.map(_evaluate_model_worker, [(str(f), protocol_name) for f in folders]))
