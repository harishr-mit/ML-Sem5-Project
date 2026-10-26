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
from racer.env.actions import action_spec_of, build_action_layout
from racer.env.obs_spec import build_observer
from racer.env.reward import build_reward
from racer.physics.surfaces import ROAD_ID, SurfaceSet
from racer.physics.versions import build_physics
from racer.track import field as F
from racer.env.trackset import TrackSampler, get_track, resolve_trackset
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
    # D26: end the episode as soon as the car's centre is beyond the road edge (kerbs are road).
    # Added after models were saved, so a snapshot without the key keeps the old rules.
    track_limits: bool = False


def make_env_config(tracks: Sequence[str] | None = None, physics: str = "default", env: str = "rays5",
                    reward: str = "progress", track: str = "default",
                    overrides: Mapping[str, Mapping[str, Any]] | None = None,
                    trackset: str | None = None) -> dict[str, Any]:
    """Assemble the full env config from presets. ``overrides`` maps section → strict overrides.

    Give either ``tracks`` (a list of track names) or ``trackset`` (a preset in
    configs/trackset/, stored expanded so the snapshot never changes meaning).
    """
    overrides = overrides or {}
    unknown = set(overrides) - {"physics", "env", "reward", "track"}
    if unknown:
        raise ValueError(f"unknown override sections {sorted(unknown)}")
    if (tracks is None) == (trackset is None):
        raise ValueError("give exactly one of tracks / trackset")
    presets = {"physics": physics, "env": env, "reward": reward, "track": track}
    if trackset is not None:
        presets["trackset"] = trackset
    return {
        "presets": presets,
        "tracks": resolve_trackset(trackset) if trackset is not None else list(tracks),
        "physics": load_preset("physics", physics, overrides.get("physics")),
        "env": load_preset("env", env, overrides.get("env")),
        "reward": load_preset("reward", reward, overrides.get("reward")),
        "track": load_preset("track", track, overrides.get("track")),
    }


class RacingEnv(gym.Env):
    """One car on one track per episode. Action = [steer, gas] in [−1, 1]²."""

    metadata = {"render_modes": []}

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.params, car_cls = build_physics(config["physics"])
        self.surfaces = SurfaceSet(self.params.surfaces)
        env_cfg = config["env"]
        unknown = set(env_cfg) - {"obs_spec", "raycast", "episode", "action_spec"}
        if unknown:
            raise ValueError(f"env config: unknown keys {sorted(unknown)}")
        self.ep = from_dict(EpisodeParams, env_cfg["episode"], where="env.episode")
        if self.params.physics_hz % self.ep.agent_hz:
            raise ValueError("physics_hz must be a multiple of agent_hz")
        self.substeps = self.params.physics_hz // self.ep.agent_hz
        self.dt_agent = 1.0 / self.ep.agent_hz
        self.action_layout = build_action_layout(action_spec_of(env_cfg))
        self.observer = build_observer(env_cfg["obs_spec"], env_cfg["raycast"], self.action_layout)
        self.reward_fn = build_reward(config["reward"])
        self.sampler = TrackSampler(config["tracks"], config["track"])
        self.track_names = self.sampler.names

        low, high = self.observer.bounds()
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.action_layout.dim,), dtype=np.float32)
        self.controls = (0.0, 0.0)     # (steer, gas) the car received on the last step

        self.car = car_cls(self.params, self.surfaces)
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
    def set_track(self, track: str | Track) -> None:
        """Switch to a track (a name from tracks/, or a Track object, e.g. generated or drawn)."""
        if isinstance(track, str):
            track = get_track(track, self.config["track"])
        if track is not self.track:
            self.track = track
            self.tracker = ProgressTracker(track)

    def _probe(self) -> bool:
        f = self.track.field
        c = self.car
        return bool(F.probe_car(f.dist, f.idx, f.geom, self.track.kerb_mask, c.x, c.y, c.theta,
                                self._wheel_off, self._corner_off, self.tracker.idx, self._wheels))

    def _onroad_frac(self) -> float:
        return float(np.count_nonzero(self._wheels >= ROAD_ID)) * 0.25

    def _observe(self) -> np.ndarray:
        return self.observer.observe(self.track, self.car, self.tracker.heading_error(self.car.theta),
                                     self._onroad_frac(), self.prev_action, self._obs, self.tracker,
                                     self._wheels).copy()

    # --------------------------------------------------------------------- API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Options (all optional): track, start_s, lateral, heading_offset (rad), speed."""
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        self.set_track(options.get("track") or self.sampler.sample(rng))
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

        self.prev_action = self.action_layout.neutral()
        getattr(self.observer, "reset", lambda: None)()
        self.prev_steer = 0.0
        self.controls = (0.0, 0.0)
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
        steer, gas = self.action_layout.to_controls(a)
        if gas < 0.0 and getattr(self.action_layout, "abs_margin", None):     # ABS-assisted layouts (D30)
            gas = -min(-gas, self.car.abs_brake_limit(self._wheels.tolist(), self.action_layout.abs_margin))
        self.controls = (steer, gas)
        car, ep, dt = self.car, self.ep, self.params.dt

        crashed = False
        offroad_acc = 0.0
        all_off = False
        record = self.record_substeps
        if record:
            self.substep_poses.clear()
        free = self.free_drive
        wall_hit = False
        over_limits = False
        limits = ep.track_limits
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
            if limits and not over_limits:
                over_limits = self.track.sample(car.x, car.y)[0] > self.track.half_width
            if crashed or (over_limits and not free):
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
        elif over_limits:
            reason = "track_limits"
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

        reward = self.reward_fn(ds, self.dt_agent, steer - self.prev_steer, offroad_frac,
                                new_laps, terminated, car.speed)
        self.prev_action[:] = a
        self.prev_steer = float(np.float32(steer))   # float32 like the stored action (keeps v1 rewards bit-identical)
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
