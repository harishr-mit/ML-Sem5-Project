"""Telemetry of a driven lap (T135, PROJECT_CONTEXT §11): what a model does on a track, step by step.

:func:`run_telemetry` drives one policy on one track under the model's own rules (physics, sensors,
episode rules) and records, per agent step, where the car is, how fast, what it commands, how hard it
corners and brakes. The Workbench draws racing lines, speed traces, g-g diagrams and braking zones
from it. Nothing here writes to a model folder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from racer.env.racing_env import RacingEnv, make_env_config

Act = Callable[[np.ndarray, RacingEnv], np.ndarray]
CHANNELS = ("t", "s", "x", "y", "theta", "speed", "steer", "gas", "throttle", "brake", "lat_g", "long_g", "offroad",
            "sliding")


@dataclass
class Telemetry:
    label: str
    track: str
    length: float                                # track length in metres
    dt: float                                    # seconds per row
    data: dict[str, np.ndarray]
    lap_starts: list[int] = field(default_factory=list)      # row index where each completed lap started (first = 0)
    lap_times: list[float] = field(default_factory=list)
    termination: str | None = None
    obs: np.ndarray | None = None                # (rows, obs_dim) when recorded

    def __len__(self) -> int:
        return len(self.data["t"])

    @property
    def best_lap(self) -> float | None:
        return min(self.lap_times) if self.lap_times else None

    def frame(self) -> pd.DataFrame:
        df = pd.DataFrame({k: v for k, v in self.data.items()})
        df["kmh"] = df["speed"] * 3.6
        return df

    def lap(self, k: int = -1) -> "Telemetry":
        """The ``k``-th completed lap (default: the last one), else the whole drive when no lap was completed."""
        if not self.lap_times:
            return self
        k = k % len(self.lap_times)
        a = self.lap_starts[k]
        b = self.lap_starts[k + 1] if k + 1 < len(self.lap_starts) else len(self)
        sl = slice(a, b)
        sub = {name: v[sl] for name, v in self.data.items()}
        sub["t"] = sub["t"] - sub["t"][0]
        ds = np.diff(sub["s"])
        ds[ds < -self.length / 2] += self.length             # the row after the line wrapped to ~0: unwrap it
        sub["s"] = np.concatenate([sub["s"][:1], sub["s"][0] + np.cumsum(ds)])
        return Telemetry(self.label, self.track, self.length, self.dt, sub, [0], [self.lap_times[k]], self.termination,
                         None if self.obs is None else self.obs[sl])

    def braking_zones(self, threshold: float = 0.1, min_len_m: float = 5.0) -> list[tuple[float, float]]:
        """``(s_start, s_end)`` of the stretches where the brake is on, in lap distance (metres)."""
        return braking_zones(self.data["s"], self.data["brake"], threshold, min_len_m)


def braking_zones(s: np.ndarray, brake: np.ndarray, threshold: float = 0.1, min_len_m: float = 5.0
                  ) -> list[tuple[float, float]]:
    on = np.asarray(brake) > threshold
    zones: list[tuple[float, float]] = []
    start = None
    for i, flag in enumerate(on):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            zones.append((float(s[start]), float(s[i - 1])))
            start = None
    if start is not None:
        zones.append((float(s[start]), float(s[-1])))
    return [(a, b) for a, b in zones if b - a >= min_len_m]


def run_telemetry(act: Act, env_config: Mapping[str, Any], track: str, label: str = "", laps: int = 1,
                  start_s: float = 0.0, start_speed: float = 0.0, max_steps: int = 6000,
                  record_obs: bool = False) -> Telemetry:
    """Drive ``laps`` laps (or until the model's own episode rules end the drive) and record every step.

    The drive starts at ``start_s`` on the centreline. Truncation by the episode's lap and step limits is
    ignored, so a model can be watched for several laps; a crash or a training termination ends the drive.
    """
    cfg = dict(env_config)
    cfg["tracks"] = [track]
    env = RacingEnv(cfg)
    env.set_track(track)
    obs, _ = env.reset(seed=0, options={"track": track, "start_s": start_s, "lateral": 0.0, "heading_offset": 0.0,
                                        "speed": start_speed})
    g = env.params.gravity
    rows: dict[str, list[float]] = {k: [] for k in CHANNELS}
    obs_rows: list[np.ndarray] = []
    lap_starts, lap_times = [0], []
    prev_v_long = env.car.v_long
    termination = None
    for step in range(max_steps):
        a = np.asarray(act(obs, env), dtype=np.float32)
        if record_obs:
            obs_rows.append(np.asarray(obs, dtype=np.float32))
        obs, _, term, _, info = env.step(a)
        c = env.car
        rows["t"].append((step + 1) * env.dt_agent)
        rows["s"].append(info["s"])
        rows["x"].append(c.x)
        rows["y"].append(c.y)
        rows["theta"].append(c.theta)
        rows["speed"].append(c.speed)
        rows["steer"].append(float(a[0]))
        rows["gas"].append(float(a[1]) if len(a) > 1 else 0.0)
        rows["throttle"].append(c.throttle)
        rows["brake"].append(c.brake)
        rows["lat_g"].append(c.a_lat / g)
        rows["long_g"].append((c.v_long - prev_v_long) / env.dt_agent / g)
        rows["offroad"].append(info["offroad_frac"])
        rows["sliding"].append(float(c.sliding))
        prev_v_long = c.v_long
        if len(env.lap_times) > len(lap_times):
            lap_times.append(env.lap_times[-1])
            lap_starts.append(step + 1)
            if len(lap_times) >= laps:
                break
        if term:
            termination = info["termination"]
            break
    data = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
    if len(lap_starts) > len(lap_times):                 # a lap that was still running when the drive ended
        lap_starts = lap_starts[:len(lap_times) + 1] if lap_times else [0]
    return Telemetry(label or track, track, float(env.track.length), float(env.dt_agent), data,
                     lap_starts[:max(len(lap_times), 1)] if lap_times else [0], lap_times, termination,
                     np.stack(obs_rows) if obs_rows else None)


def baseline_policy() -> tuple[Act, dict[str, Any], str]:
    """The hand-coded PID driver as ``(act, env_config, label)`` (E8 reference)."""
    from racer.agents.baseline import BaselineController
    from racer.config import load_preset
    ctrl = BaselineController(load_preset("train", "baseline"))
    return (lambda obs, env: ctrl.act(env)), make_env_config(["training_ground"]), "PID baseline"


def model_policy(model_id: str) -> tuple[Act, dict[str, Any], str]:
    """A registered model as ``(act, env_config, label)``; raises ``IncompatibleModel`` if this engine can't run it."""
    from racer.agents.loader import LoadedModel
    m = LoadedModel(model_id)
    return (lambda obs, env: m.act(obs)), m.env_config, m.model_id


def drive(who: str, track: str, **kw: Any) -> Telemetry:
    """Telemetry of a model id (or ``"baseline"``) on ``track``."""
    act, cfg, label = baseline_policy() if who == "baseline" else model_policy(who)
    return run_telemetry(act, cfg, track, label, **kw)
