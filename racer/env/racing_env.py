"""Gymnasium racing environment (PROJECT_CONTEXT §7).

The env is built from one plain-dict config (see :func:`make_env_config`), the same dict
that is snapshotted into every model folder, so a saved model always gets back the exact
physics, sensors, reward and episode rules it was trained with.

Each ``step`` holds the action for ``physics_hz / agent_hz`` physics substeps. After every
substep the field is probed for the wheel surfaces and the barrier; progress, terminations
and the reward are evaluated once per agent step.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np

from racer.config import from_dict, load_preset
from racer.env.obs_spec import build_observer
from racer.env.reward import build_reward
from racer.physics.car import Car, PhysicsParams
from racer.physics.surfaces import ROAD_ID, SurfaceSet
from racer.track import field as F
from racer.track.progress import ProgressTracker
from racer.track.track import Track


@dataclass(frozen=True)
class EpisodeParams:
    agent_hz: int
    start_lateral_max_m: float
    start_heading_max_deg: float
    start_speed_min: float
    start_speed_max: float
    offroad_limit_s: float
    noprogress_window_s: float
    noprogress_min_m: float
    wrongway_deg: float
    wrongway_limit_s: float
    max_laps: int
    max_steps: int


def make_env_config(tracks: Sequence[str], physics: str = "default", env: str = "rays5",
                    reward: str = "progress", track: str = "default",
                    overrides: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Assemble the full env config from presets. ``overrides`` maps section → strict overrides."""
    overrides = overrides or {}
    unknown = set(overrides) - {"physics", "env", "reward", "track"}
    if unknown:
        raise ValueError(f"unknown override sections {sorted(unknown)}")
    return {
        "presets": {"physics": physics, "env": env, "reward": reward, "track": track},
        "tracks": list(tracks),
        "physics": load_preset("physics", physics, overrides.get("physics")),
        "env": load_preset("env", env, overrides.get("env")),
        "reward": load_preset("reward", reward, overrides.get("reward")),
        "track": load_preset("track", track, overrides.get("track")),
    }


# Process-wide cache: building a field takes ~50 ms and a few MB, and tracks are immutable.
_TRACK_CACHE: dict[tuple[str, int], Track] = {}


def get_track(name: str, track_cfg: Mapping[str, Any]) -> Track:
    key = (name, id(track_cfg))
    if key not in _TRACK_CACHE:
        _TRACK_CACHE[key] = Track.load(name, track_cfg)
    return _TRACK_CACHE[key]


class RacingEnv(gym.Env):
    """One car on one track per episode. Action = [steer, gas] in [−1, 1]²."""

    metadata = {"render_modes": []}

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.params = PhysicsParams.from_config(config["physics"])
        self.surfaces = SurfaceSet(self.params.surfaces)
        env_cfg = config["env"]
        unknown = set(env_cfg) - {"obs_spec", "raycast", "episode"}
        if unknown:
            raise ValueError(f"env config: unknown keys {sorted(unknown)}")
        self.ep = from_dict(EpisodeParams, env_cfg["episode"], where="env.episode")
        if self.params.physics_hz % self.ep.agent_hz:
            raise ValueError("physics_hz must be a multiple of agent_hz")
        self.substeps = self.params.physics_hz // self.ep.agent_hz
        self.dt_agent = 1.0 / self.ep.agent_hz
        self.observer = build_observer(env_cfg["obs_spec"], env_cfg["raycast"])
        self.reward_fn = build_reward(config["reward"])
        self.track_names = list(config["tracks"])
        if not self.track_names:
            raise ValueError("config['tracks'] is empty")

        low, high = self.observer.bounds()
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.car = Car(self.params, self.surfaces)
        self._wheel_off = np.array(self.car.wheel_offsets, dtype=np.float64)
        self._corner_off = np.array(self.car.corner_offsets, dtype=np.float64)
        self._wheels = np.zeros(4, dtype=np.int64)
        self._obs = np.zeros(self.observer.dim, dtype=np.float32)
        self.track: Track | None = None
        self.tracker: ProgressTracker | None = None
        # Renderers set this to get the car pose after every physics substep (training leaves it off).
        self.record_substeps = False
        self.substep_poses: list[tuple[float, float, float]] = []
        # Free drive (human play in the debug window; never in training): walls are solid,
        # nothing terminates or truncates, and the would-be termination is reported in
        # info["warning"] instead.
        self.free_drive = False
        n_window = int(round(self.ep.noprogress_window_s * self.ep.agent_hz))
        self._progress_hist: deque[float] = deque(maxlen=n_window + 1)

    # ------------------------------------------------------------------ helpers
    def set_track(self, name: str) -> None:
        if self.track is None or self.track.name != get_track(name, self.config["track"]).name:
            self.track = get_track(name, self.config["track"])
            self.tracker = ProgressTracker(self.track)

    def _probe(self) -> bool:
        f = self.track.field
        c = self.car
        return bool(F.probe_car(f.dist, f.idx, f.geom, self.track.kerb_mask, c.x, c.y, c.theta,
                                self._wheel_off, self._corner_off, self.tracker.idx, self._wheels))

    def _onroad_frac(self) -> float:
        return float(np.count_nonzero(self._wheels >= ROAD_ID)) * 0.25

    def _observe(self) -> np.ndarray:
        return self.observer.observe(self.track, self.car, self.tracker.heading_error(self.car.theta),
                                     self._onroad_frac(), self.prev_action, self._obs).copy()

    # --------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Options (all optional): track, start_s, lateral, heading_offset (rad), speed."""
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        name = options.get("track") or self.track_names[int(rng.integers(len(self.track_names)))]
        self.set_track(name)
        tr = self.track
        ep = self.ep
        s0 = options.get("start_s", rng.uniform(0.0, tr.length))
        lateral = options.get("lateral", rng.uniform(-ep.start_lateral_max_m, ep.start_lateral_max_m))
        dh = options.get("heading_offset", math.radians(rng.uniform(-ep.start_heading_max_deg, ep.start_heading_max_deg)))
        speed = options.get("speed", rng.uniform(ep.start_speed_min, ep.start_speed_max))
        x, y, heading = tr.pose_at(s0, lateral)
        self.car.reset(x, y, heading + dh, speed)
        self.tracker.reset(x, y)
        self._probe()

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.time = 0.0
        self.laps = 0
        self.offroad_time = 0.0
        self.wrongway_time = 0.0
        self._progress_hist.clear()
        self._progress_hist.append(0.0)
        self._last_cross_t: float | None = None
        self._last_cross_total = 0.0
        self.lap_times: list[float] = []
        info = {"track": tr.name, "start_s": float(s0)}
        return self._observe(), info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        steer, gas = float(a[0]), float(a[1])
        car, ep, dt = self.car, self.ep, self.params.dt

        crashed = False
        offroad_acc = 0.0
        all_off = False
        record = self.record_substeps
        if record:
            self.substep_poses.clear()
        free = self.free_drive
        wall_hit = False
        for _ in range(self.substeps):
            prev_pose = (car.x, car.y, car.theta)
            car.step(steer, gas, self._wheels.tolist())
            crashed = self._probe()
            if crashed and free:
                # Solid wall: undo this substep's move and stop the car against it.
                car.x, car.y, car.theta = prev_pose
                car.vx = car.vy = 0.0
                crashed = False
                wall_hit = True
                self._probe()
            if record:
                self.substep_poses.append((car.x, car.y, car.theta))
            on = self._onroad_frac()
            offroad_acc += 1.0 - on
            all_off = on == 0.0
            self.offroad_time = self.offroad_time + dt if all_off else 0.0
            if crashed:
                break
        offroad_frac = offroad_acc / self.substeps

        ds, crossing, left = self.tracker.update(car.x, car.y)
        crashed = crashed or (left and not free)
        self.steps += 1
        self.time += self.dt_agent
        total = self.tracker.total
        self._progress_hist.append(total)

        # Lap timing at the start/finish line (only full laps of net progress count).
        lap_time = None
        if crossing > 0:
            if self._last_cross_t is not None and total - self._last_cross_total >= 0.99 * self.track.length:
                lap_time = self.time - self._last_cross_t
                self.lap_times.append(lap_time)
            self._last_cross_t, self._last_cross_total = self.time, total
        elif crossing < 0:
            self._last_cross_t = None

        laps_now = int(total // self.track.length) if total > 0 else 0
        new_laps = max(laps_now - self.laps, 0)
        self.laps = max(self.laps, laps_now)

        if abs(self.tracker.heading_error(car.theta)) > math.radians(ep.wrongway_deg):
            self.wrongway_time += self.dt_agent
        else:
            self.wrongway_time = 0.0

        reason = None
        if crashed:
            reason = "barrier"
        elif self.offroad_time > ep.offroad_limit_s:
            reason = "offroad"
        elif (len(self._progress_hist) == self._progress_hist.maxlen
              and total - self._progress_hist[0] < ep.noprogress_min_m):
            reason = "stuck"
        elif self.wrongway_time > ep.wrongway_limit_s:
            reason = "wrong_way"
        warning = None
        if free:
            warning, reason = reason, None
            if wall_hit:
                warning = "wall"
        terminated = reason is not None
        truncated = not free and not terminated and (self.laps >= ep.max_laps or self.steps >= ep.max_steps)

        reward = self.reward_fn(ds, self.dt_agent, steer - float(self.prev_action[0]), offroad_frac,
                                new_laps, terminated, car.speed)
        self.prev_action[0], self.prev_action[1] = steer, gas
        info = {
            "track": self.track.name, "s": self.tracker.s, "progress": total, "laps": self.laps,
            "speed": car.speed, "offroad_frac": offroad_frac, "sliding": car.sliding,
            "termination": reason, "time": self.time,
        }
        if free:
            info["warning"] = warning
        if lap_time is not None:
            info["lap_time"] = lap_time
        return self._observe(), float(reward), terminated, truncated, info
