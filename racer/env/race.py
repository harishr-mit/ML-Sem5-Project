"""Multi-car race for the demo (PROJECT_CONTEXT §10.1, D14, D15). Simulation only: no pygame.

Every car is its own :class:`RacingEnv` built from its model's snapshot (own physics, obs
spec and episode rules, or a common "fair" physics), all on the same track. Cars are
ghosts: they never interact, and rays only see the road.

Rules (``configs/demo/arena.yaml``): a staggered 2-wide grid behind the start line, a
countdown, then an N-lap race. A car that crashes (any training termination) waits
``respawn_s`` and restarts from standstill on the centreline where it was, and the crash
is counted. Env truncation (lap / step limits) is ignored, so cars keep racing.

Time advances one physics substep per :meth:`Race.tick`. A car needs a new action every
``physics_hz / agent_hz`` substeps; cars that need one in the same tick are batched per
model. ``race distance`` is measured from the start line (grid slots are negative).
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from racer.config import from_dict
from racer.env.racing_env import RacingEnv
from racer.track.track import Track


@dataclass(frozen=True)
class RaceRules:
    laps: int
    grid_gap_m: float           # distance between grid rows
    grid_lateral_frac: float    # lateral grid offset as a fraction of the half width
    countdown_s: float
    respawn_s: float
    finish_timeout_s: float     # race ends this long after the winner (or when all finish)
    start: str = "grid"          # "grid" (staggered 2-wide) or "scatter" (random places and speeds: the swarm view)
    scatter_speed_max_ms: float = 15.0
    scatter_lateral_frac: float = 0.6


@dataclass
class Entry:
    """One car: a label, a policy and its env config.

    ``act_batch(obs[n, d], envs) -> actions[n, a]`` gets the cars' observations and envs (the
    hand-coded baseline reads the env; trained models only use the observations). ``a`` is
    the size of the car's action layout (``env.action_layout``, D22); each car has its own
    observation spec and layout, so models with different inputs and outputs race together.
    """

    label: str
    env_config: Mapping[str, Any]
    act_batch: Callable[[np.ndarray, list], np.ndarray]
    key: Any = None             # cars with the same key are batched together (e.g. the model object)


@dataclass
class RaceCar:
    entry: Entry
    index: int
    env: RacingEnv
    track: Track
    obs: np.ndarray
    offset: float                               # race distance at the env's last reset
    poses: deque = field(default_factory=deque)
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    t_state: float = 0.0                        # race time of the env's current state
    state: str = "grid"                         # grid | racing | respawn | finished
    respawn_until: float = 0.0
    respawn_s: float = 0.0                      # track position to respawn at
    lap_times: list[float] = field(default_factory=list)
    last_cross_t: float = 0.0
    finished_t: float | None = None
    crashes: int = 0
    last_crash: str | None = None
    action: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    @property
    def dist(self) -> float:
        return self.offset + self.env.tracker.total

    @property
    def label(self) -> str:
        return self.entry.label


class Race:
    def __init__(self, track: str | Track, entries: Sequence[Entry], rules: Mapping[str, Any] | RaceRules,
                 physics_override: Mapping[str, Any] | None = None, spread_decisions: bool = True) -> None:
        if not entries:
            raise ValueError("a race needs at least one car")
        self.spread_decisions = spread_decisions
        self.rules = rules if isinstance(rules, RaceRules) else from_dict(RaceRules, rules, where="arena race rules")
        self.cars: list[RaceCar] = []
        hz = set()
        for i, e in enumerate(entries):
            cfg = copy.deepcopy(dict(e.env_config))
            if physics_override is not None:
                cfg["physics"] = copy.deepcopy(dict(physics_override))
            cfg["tracks"] = [track if isinstance(track, str) else "training_ground"]   # sampler is never used
            env = RacingEnv(cfg)
            env.record_substeps = True
            env.set_track(track)
            hz.add(env.params.physics_hz)
            self.cars.append(RaceCar(entry=e, index=i, env=env, track=env.track, obs=np.zeros(1), offset=0.0))
        if len(hz) != 1:
            raise ValueError(f"cars use different physics rates {sorted(hz)}; use a common physics preset")
        self.dt = 1.0 / hz.pop()
        self.track = self.cars[0].track
        self.length = self.track.length
        self.t = -self.rules.countdown_s       # race time; the start is at t = 0
        self.over = False
        self.first_finish: float | None = None
        n = int(self.rules.laps * self.length) + 2
        self._first_t = np.full(n, np.nan)     # earliest race time any car reached each metre
        self._grid()
        if self.started:                       # no countdown (the swarm): everyone is racing from the first tick
            for car in self.cars:
                car.state, car.t_state = "racing", self.t

    # -------------------------------------------------------------- setup
    def _grid(self) -> None:
        r = self.rules
        if r.start == "scatter":
            rng = np.random.default_rng(1234)
            for car in self.cars:
                s = float(rng.uniform(0.0, self.length))
                lat = float(rng.uniform(-1, 1)) * r.scatter_lateral_frac * car.track.half_width
                self._place(car, s, lat, s, speed=float(rng.uniform(0.0, r.scatter_speed_max_ms)))
            return
        for car in self.cars:
            row, side = divmod(car.index, 2)
            d0 = -(row + 1) * r.grid_gap_m
            lateral = (1 if side == 0 else -1) * r.grid_lateral_frac * car.track.half_width
            self._place(car, d0 % self.length, lateral, d0)

    def _place(self, car: RaceCar, s: float, lateral: float, dist: float, speed: float = 0.0) -> None:
        obs, _ = car.env.reset(seed=car.index, options={"track": car.track, "start_s": s, "lateral": lateral,
                                                         "heading_offset": 0.0, "speed": speed})
        car.obs = obs
        car.offset = dist
        car.poses.clear()
        car.pose = (car.env.car.x, car.env.car.y, car.env.car.theta)
        car.action = car.env.action_layout.neutral()

    # ------------------------------------------------------------ stepping
    @property
    def started(self) -> bool:
        return self.t >= 0.0

    def tick(self) -> None:
        """Advance the race by one physics substep."""
        if self.over:
            return
        if not self.started:
            self.t += self.dt
            if self.started:
                for car in self.cars:
                    car.state, car.t_state = "racing", self.t
            return
        # Respawns that are due.
        for car in self.cars:
            if car.state == "respawn" and self.t >= car.respawn_until:
                self._place(car, car.respawn_s, 0.0, car.dist)
                car.state = "finished" if car.finished_t is not None else "racing"
                car.t_state = self.t
        # Cars that need a new action, batched per policy. A car with nothing left to show must
        # step now; with ``spread_decisions`` other cars may step up to one agent step early
        # (their poses are played back later), so each tick handles ~1/substeps of the cars
        # instead of all of them landing on the same frame (T51). Cars don't interact and
        # timing uses each car's own simulated time, so results are the same either way.
        active = [c for c in self.cars if c.state in ("racing", "finished")]
        need = [c for c in active if not c.poses]
        if self.spread_decisions:
            budget = math.ceil(sum(1.0 / c.env.substeps for c in active))
            early = sorted((c for c in active if c.poses and len(c.poses) <= c.env.substeps),
                           key=lambda c: len(c.poses))
            need += early[:max(budget - len(need), 0)]
        groups: dict[Any, list[RaceCar]] = {}
        for c in need:
            groups.setdefault(c.entry.key if c.entry.key is not None else id(c.entry), []).append(c)
        for cars in groups.values():
            actions = cars[0].entry.act_batch(np.stack([c.obs for c in cars]), [c.env for c in cars])
            for c, a in zip(cars, actions):
                self._step_car(c, a)
        for car in self.cars:
            if car.poses:
                car.pose = car.poses.popleft()
        self.t += self.dt
        if self.first_finish is not None and (self.t >= self.first_finish + self.rules.finish_timeout_s
                                              or all(c.finished_t is not None for c in self.cars)):
            self.over = True

    def _step_car(self, car: RaceCar, action: np.ndarray) -> None:
        d_prev, t_prev = car.dist, car.t_state
        s_prev = car.env.tracker.s
        obs, _, terminated, _, info = car.env.step(action)     # truncation is ignored in a race
        car.obs, car.action = obs, np.asarray(action, dtype=np.float32)
        car.t_state = t_prev + car.env.dt_agent
        car.poses.extend(car.env.substep_poses)
        if terminated:
            # Crash: hold the car where it is, then respawn on the centreline at the last good point.
            car.crashes += 1
            car.last_crash = info["termination"]
            car.state = "respawn"
            car.respawn_until = car.t_state + self.rules.respawn_s
            car.respawn_s = s_prev
            car.offset = d_prev - car.env.tracker.total       # race distance stays at d_prev
            return
        self._timing(car, d_prev, t_prev)

    def _timing(self, car: RaceCar, d_prev: float, t_prev: float) -> None:
        d_new, t_new = car.dist, car.t_state
        if d_new <= d_prev:
            return
        frac = lambda d: t_prev + (d - d_prev) / (d_new - d_prev) * (t_new - t_prev)
        # Earliest time each metre was reached (for gaps).
        lo, hi = max(int(math.floor(d_prev)) + 1, 0), min(int(math.floor(d_new)), len(self._first_t) - 1)
        if hi >= lo:
            # Keep the minimum: with spread decisions a later car may be stepped before an earlier one.
            seg = self._first_t[lo:hi + 1]
            np.fmin(seg, [frac(d) for d in np.arange(lo, hi + 1)], out=seg)
        # Lap completions (line crossings at multiples of the track length).
        if car.finished_t is None:
            k = len(car.lap_times) + 1
            while d_prev < k * self.length <= d_new:
                tc = frac(k * self.length)
                car.lap_times.append(tc - car.last_cross_t)
                car.last_cross_t = tc
                if k >= self.rules.laps:
                    car.finished_t = tc
                    car.state = "finished"
                    # Cars stepped in the same tick finish at interpolated times: keep the earliest.
                    self.first_finish = tc if self.first_finish is None else min(self.first_finish, tc)
                    break
                k += 1

    # ------------------------------------------------------------ standings
    def gap(self, car: RaceCar) -> float | None:
        """Seconds behind the first car to reach this car's race distance."""
        if not self.started:
            return None
        if car.finished_t is not None:
            return car.finished_t - (self.first_finish or car.finished_t)
        i = min(max(int(math.floor(car.dist)), 0), len(self._first_t) - 1)
        t0 = self._first_t[i]
        return None if np.isnan(t0) else max(car.t_state - t0, 0.0)

    def standings(self) -> list[RaceCar]:
        """Finished cars by finish time, then everyone else by race distance."""
        done = sorted((c for c in self.cars if c.finished_t is not None), key=lambda c: c.finished_t)
        rest = sorted((c for c in self.cars if c.finished_t is None), key=lambda c: -c.dist)
        return done + rest

    def lap_of(self, car: RaceCar) -> int:
        """Current lap number (1-based, capped at the race length)."""
        return min(len(car.lap_times) + 1, self.rules.laps)

    def current_lap_time(self, car: RaceCar) -> float | None:
        if not self.started or car.finished_t is not None:
            return None
        return car.t_state - car.last_cross_t
