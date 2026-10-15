"""Neuroevolution runs (D26, docs/NEUROEVOLUTION_PLAN.md): the evolution loop, its worker pool,
the lineage database, pause/resume, and exporting a car as a model folder.

A run lives in ``outputs/evolve/<run_name>/``: ``state.npz`` (the population to drive next;
deleted when the run finishes), ``lineage.sqlite`` (every car of every generation),
``snapshot.yaml`` (the run's snapshot, reused on resume) and ``stop.request`` (pause).
Generation ``g`` draws its map and start from ``default_rng([seed, g, 0])`` and is bred from
``default_rng([seed, g, 1])``, and every car drives alone from a fully specified start, so a
run is identical however often it was paused and however many workers drove it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import sqlite3
import time
from dataclasses import dataclass
from multiprocessing import TimeoutError as PoolTimeout
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Mapping

import gymnasium as gym
import numpy as np

from racer.agents import neuroevolution as ne
from racer.config import PROJECT_ROOT, from_dict

EVOLVE_DIR = PROJECT_ROOT / "outputs" / "evolve"
PAUSED_EXIT = 3          # train.py's exit code for a paused run; the queue marks it "interrupted"
FINISHED = "finished.json"
_DRAW, _BREED = 0, 1


@dataclass(frozen=True)
class GAParams:
    population: int
    elite_frac: float
    sigma: float
    speed_scale: float
    workers: int
    checkpoint_every: int
    state_every: int
    validation: dict

    @property
    def n_elite(self) -> int:
        return max(1, int(self.population * self.elite_frac))


def training_maps(env_config: Mapping[str, Any]) -> list[str]:
    """The named maps a run evolves on (its ``tracks``); a generated pool isn't allowed."""
    tracks = env_config["tracks"]
    if isinstance(tracks, list):
        return list(tracks)
    if tracks.get("generated"):
        raise ValueError(f"track set {tracks.get('trackset')!r} has a generated pool; neuroevolution drives "
                         "named maps (use --tracks or a track set such as train7)")
    return [f["track"] for f in tracks["fixed"]]


def settings_key(env_config: Mapping[str, Any], train_cfg: Mapping[str, Any]) -> str:
    """What must match to resume a run: everything except the worker count and the step budget."""
    train = json.loads(json.dumps(train_cfg))
    train["ga"].pop("workers", None)
    train.pop("total_steps", None)
    blob = json.dumps({"env": env_config, "train": train}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def find_run(prefix: str, tag: str, root: Path = EVOLVE_DIR) -> Path | None:
    """The newest run folder with this tag (paused, crashed or finished), if any."""
    found = sorted(root.glob(f"{prefix}[0-9][0-9][0-9]_GA_{tag}")) if root.is_dir() else []
    return found[-1] if found else None


def is_finished(run_dir: Path) -> bool:
    return (run_dir / FINISHED).is_file()


def unfinished_run_ids(root: Path = EVOLVE_DIR) -> set[str]:
    if not root.is_dir():
        return set()
    return {p.name.split("_", 1)[0] for p in root.iterdir() if p.is_dir() and not is_finished(p)}


# ----------------------------------------------------------------- lineage
class Lineage:
    """One row per car per generation (``cars``) and per generation (``generations``)."""

    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(str(path))
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS cars (
                generation INTEGER, car INTEGER, parent INTEGER, seed INTEGER, fitness REAL,
                progress_m REAL, time_s REAL, steps INTEGER, ending TEXT, PRIMARY KEY (generation, car));
            CREATE TABLE IF NOT EXISTS generations (
                generation INTEGER PRIMARY KEY, map TEXT, start_s REAL, lateral REAL, heading_offset REAL,
                speed REAL, best REAL, mean REAL, median REAL, finished INTEGER, env_steps INTEGER,
                validation REAL);
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)

    def set_meta(self, **kv: Any) -> None:
        self.db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in kv.items()])
        self.db.commit()

    def meta(self) -> dict[str, Any]:
        return {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM meta")}

    def write(self, generation: int, cars: list[tuple], gen_row: Mapping[str, Any]) -> None:
        # INSERT OR REPLACE: after a crash, generations since the last state save are driven
        # again and rewritten with identical rows.
        self.db.executemany("INSERT OR REPLACE INTO cars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", cars)
        self.db.execute("INSERT OR REPLACE INTO generations VALUES "
                        "(:generation, :map, :start_s, :lateral, :heading_offset, :speed, :best, :mean, :median, "
                        ":finished, :env_steps, :validation)", {"generation": generation, **gen_row})
        self.db.commit()

    def origin(self, generation: int, car: int) -> tuple[int, int]:
        row = self.db.execute("SELECT parent, seed FROM cars WHERE generation = ? AND car = ?",
                              (generation, car)).fetchone()
        if row is None:
            raise KeyError(f"no car {car} in generation {generation}")
        return int(row[0]), int(row[1])

    def close(self) -> None:
        self.db.close()


def rebuild_car(run_dir: Path, generation: int, car: int) -> np.ndarray:
    """Weights of any car of a run, replayed from its lineage (needs the same numpy version)."""
    lin = Lineage(run_dir / "lineage.sqlite")
    try:
        m = lin.meta()
        return ne.rebuild(lin.origin, generation, car, m["sizes"], m["sigma"])
    finally:
        lin.close()


# ----------------------------------------------------------------- workers
_W: dict[str, Any] = {}


def _below_normal_priority() -> None:
    import psutil
    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)


def _init_worker(env_config: Mapping[str, Any], val_cfg: Mapping[str, Any]) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)      # Ctrl+C pauses the run in the main process only
    _below_normal_priority()
    from racer.env.racing_env import RacingEnv
    _W["env"] = RacingEnv(env_config)
    _W["val_cfg"] = (int(val_cfg["stream"]), env_config["track"])
    _W["val"] = {}


def _validation_track(k: int):
    if k not in _W["val"]:
        from racer.track.generator import generate
        stream, track_cfg = _W["val_cfg"]
        _W["val"][k] = generate(k, stream, name=f"val_s{stream}_{k}", track_cfg=track_cfg)[0]
    return _W["val"][k]


def drive(env, layers, start: Mapping[str, Any], speed_scale: float) -> tuple:
    """One try from a fully specified start: until one lap, a failure, or the env's step limit.

    Returns (fitness, progress_m, time_s, steps, ending), ending = "lap", "time" or the
    env's termination reason.
    """
    obs, _ = env.reset(options=dict(start))
    length = env.track.length
    while True:
        obs, _, term, trunc, info = env.step(ne.forward(layers, obs))
        finished = info["progress"] >= length and info["termination"] is None
        if finished or term or trunc:
            break
    ending = "lap" if finished else (info["termination"] or "time")
    fit = ne.fitness(info["progress"], length_m=length, finished=finished, time_s=env.time,
                     speed_scale=speed_scale)
    return fit, float(info["progress"]), float(env.time), int(env.steps), ending


def _drive_task(task: tuple) -> list[tuple]:
    weights, sizes, start, speed_scale, val_index = task
    if val_index is not None:
        start = dict(start, track=_validation_track(val_index))
    return [drive(_W["env"], ne.unflatten(w, sizes), start, speed_scale) for w in weights]


# --------------------------------------------------------------------- run
@dataclass
class State:
    generation: int
    weights: np.ndarray
    parents: np.ndarray
    seeds: np.ndarray
    env_steps: int = 0
    best_validation: float | None = None
    last_gain: int = 0              # generation of the last validation gain
    wall_s: float = 0.0
    settings: str = ""

    def save(self, path: Path) -> None:
        meta = {k: getattr(self, k) for k in ("generation", "env_steps", "best_validation", "last_gain",
                                               "wall_s", "settings")}
        tmp = path.with_name(path.name + ".partial")
        with tmp.open("wb") as f:
            np.savez(f, weights=self.weights, parents=self.parents, seeds=self.seeds, meta=json.dumps(meta))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "State":
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            return cls(weights=z["weights"], parents=z["parents"], seeds=z["seeds"], **meta)


Checkpoint = Callable[[np.ndarray, int, int, float], Any]     # (weights, generation, env_steps, fitness)


class Evolution:
    """The loop of one run. ``run`` returns "finished" (plateau or budget) or "paused"."""

    def __init__(self, run_dir: Path, env_config: Mapping[str, Any], train_cfg: Mapping[str, Any],
                 obs_dim: int, act_dim: int, checkpoint: Checkpoint | None = None, writer=None,
                 log: Callable[[str], None] = print) -> None:
        self.run_dir = Path(run_dir)
        self.env_config, self.train_cfg = env_config, train_cfg
        self.p = from_dict(GAParams, train_cfg["ga"], where="train.ga")
        self.seed = int(train_cfg["seed"])
        self.budget = train_cfg.get("total_steps")
        self.sizes = ne.layer_sizes(obs_dim, train_cfg["policy"]["net_arch"], act_dim)
        self.maps = training_maps(env_config)
        self.ep = env_config["env"]["episode"]
        self.checkpoint, self.writer, self.log = checkpoint, writer, log
        self.state_path = self.run_dir / "state.npz"
        self.stop_path = self.run_dir / "stop.request"
        self.pause_requested = False
        self.final: State | None = None

    # ------------------------------------------------------------ setup
    def _rng(self, generation: int, purpose: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, generation, purpose])

    def _initial_state(self) -> State:
        key = settings_key(self.env_config, self.train_cfg)
        if self.state_path.is_file():
            st = State.load(self.state_path)
            if st.settings != key:
                raise ValueError(f"{self.run_dir.name} was started with different settings; only ga.workers "
                                 "and total_steps may change when resuming")
            self.log(f"[{self.run_dir.name}] resuming at generation {st.generation} "
                     f"({st.env_steps:,} env steps so far)")
            return st
        g0 = ne.first_generation(self.sizes, self.p.population, self._rng(0, _BREED))
        st = State(0, g0.weights, g0.parents, g0.seeds, settings=key)
        st.save(self.state_path)          # from now on a rerun finds and resumes this run, even after a crash
        return st

    def _start(self, generation: int) -> dict[str, Any]:
        """Map and start of a generation, the same for every car."""
        from racer.env.trackset import get_track
        rng = self._rng(generation, _DRAW)
        name = self.maps[int(rng.integers(len(self.maps)))]
        length = get_track(name, self.env_config["track"]).length
        ep = self.ep
        return {"track": name, "start_s": float(rng.uniform(0.0, length)),
                "lateral": float(rng.uniform(-ep["start_lateral_max_m"], ep["start_lateral_max_m"])),
                "heading_offset": math.radians(float(rng.uniform(-ep["start_heading_max_deg"],
                                                                 ep["start_heading_max_deg"]))),
                "speed": float(rng.uniform(ep["start_speed_min"], ep["start_speed_max"]))}

    def _on_sigint(self, signum, frame) -> None:
        if self.pause_requested:
            raise KeyboardInterrupt
        self.pause_requested = True
        self.log(f"[{self.run_dir.name}] pausing after this generation (Ctrl+C again to abort without saving)")

    # ------------------------------------------------------------ driving
    @staticmethod
    def _map(pool, tasks: list) -> list:
        # pool.map blocks in a lock wait that Ctrl+C can't interrupt on Windows; polling keeps the
        # main thread running Python, so a second Ctrl+C aborts at once.
        res = pool.map_async(_drive_task, tasks)
        while True:
            try:
                return res.get(0.2)
            except PoolTimeout:
                continue

    def _drive_all(self, pool, weights: np.ndarray, start: Mapping[str, Any]) -> list[tuple]:
        n_chunks = min(len(weights), self.p.workers * 4)
        chunks = np.array_split(np.arange(len(weights)), n_chunks)
        tasks = [(weights[c], self.sizes, start, self.p.speed_scale, None) for c in chunks]
        return [r for part in self._map(pool, tasks) for r in part]

    def _validate(self, pool, flat: np.ndarray) -> float:
        start = {"start_s": 0.0, "lateral": 0.0, "heading_offset": 0.0, "speed": 0.0}
        n = int(self.p.validation["tracks"])
        tasks = [(flat[None], self.sizes, start, self.p.speed_scale, k) for k in range(n)]
        return float(np.mean([part[0][0] for part in self._map(pool, tasks)]))

    # ------------------------------------------------------------ the loop
    def run(self, pause_after: int | None = None) -> str:
        """Evolve until the validation plateau, the step budget, or a pause.

        ``pause_after``: pause once this many generations are done (tests).
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        st = self._initial_state()
        lineage = Lineage(self.run_dir / "lineage.sqlite")
        lineage.set_meta(sizes=self.sizes, sigma=self.p.sigma, seed=self.seed, population=self.p.population)
        old_handler = signal.signal(signal.SIGINT, self._on_sigint)
        v = self.p.validation
        ctx = get_context("spawn")
        t_start = time.perf_counter()
        outcome = "paused"
        try:
            with ctx.Pool(self.p.workers, initializer=_init_worker, initargs=(self.env_config, v)) as pool:
                while True:
                    g = st.generation
                    start = self._start(g)
                    res = self._drive_all(pool, st.weights, start)
                    scores = np.array([r[0] for r in res])
                    st.env_steps += sum(r[3] for r in res)
                    best = int(ne.rank(scores)[0])

                    validation = None
                    if (g + 1) % int(v["every"]) == 0:
                        validation = self._validate(pool, st.weights[best])
                        if st.best_validation is None or validation > st.best_validation * (1 + float(v["min_gain"])):
                            st.best_validation, st.last_gain = validation, g
                    lineage.write(g, [(g, k, int(st.parents[k]), int(st.seeds[k]), *res[k]) for k in range(len(res))],
                                  {"map": start["track"], **{k: start[k] for k in ("start_s", "lateral",
                                                                                   "heading_offset", "speed")},
                                   "best": float(scores.max()), "mean": float(scores.mean()),
                                   "median": float(np.median(scores)),
                                   "finished": sum(r[4] == "lap" for r in res), "env_steps": st.env_steps,
                                   "validation": validation})
                    self._log_generation(g, start["track"], res, scores, validation, st)

                    plateau = st.best_validation is not None and g - st.last_gain >= int(v["patience"])
                    over_budget = self.budget is not None and st.env_steps >= int(self.budget)
                    done = plateau or over_budget
                    if self.checkpoint and ((g + 1) % self.p.checkpoint_every == 0 or done):
                        self.checkpoint(st.weights[best], g, st.env_steps, float(scores[best]))
                    if done:
                        outcome = "finished"
                        reason = "validation plateau" if plateau else "step budget"
                        self.log(f"[{self.run_dir.name}] finished after generation {g} ({reason}); "
                                 f"best validation {st.best_validation}")
                        break

                    off = ne.next_generation(st.weights, scores, self.p.n_elite, self.p.sigma, self._rng(g + 1, _BREED))
                    st.generation, st.weights, st.parents, st.seeds = g + 1, off.weights, off.parents, off.seeds
                    stop_file = self.stop_path.is_file()
                    pause = (self.pause_requested or stop_file
                             or (pause_after is not None and st.generation >= pause_after))
                    if pause or st.generation % self.p.state_every == 0:
                        st.wall_s += time.perf_counter() - t_start
                        t_start = time.perf_counter()
                        st.save(self.state_path)
                    if pause:
                        self.stop_path.unlink(missing_ok=True)
                        self.log(f"[{self.run_dir.name}] paused before generation {st.generation} "
                                 f"({st.env_steps:,} env steps); rerun the same command to continue")
                        break
        finally:
            signal.signal(signal.SIGINT, old_handler)
            lineage.close()
        if outcome == "finished":
            st.wall_s += time.perf_counter() - t_start
            (self.run_dir / FINISHED).write_text(json.dumps(
                {"generations": st.generation + 1, "env_steps": st.env_steps, "wall_s": round(st.wall_s, 1),
                 "best_validation": st.best_validation}), encoding="utf-8")
            self.state_path.unlink(missing_ok=True)
        self.final = st
        return outcome

    def _log_generation(self, g, track, res, scores, validation, st) -> None:
        endings = [r[4] for r in res]
        laps = [r[2] for r in res if r[4] == "lap"]
        if self.writer is not None:
            w, step = self.writer, st.env_steps
            w.add_scalar("ga/best_fitness", float(scores.max()), step)
            w.add_scalar("ga/mean_fitness", float(scores.mean()), step)
            w.add_scalar("ga/median_fitness", float(np.median(scores)), step)
            w.add_scalar("ga/generation", g, step)
            w.add_scalar("ga/map", self.maps.index(track), step)
            w.add_scalar("race/laps", len(laps) / len(res), step)
            w.add_scalar("race/progress_m", float(np.mean([r[1] for r in res])), step)
            if laps:
                w.add_scalar("race/lap_time", min(laps), step)
            for reason in ("lap", "time", "barrier", "track_limits", "offroad", "stuck", "wrong_way"):
                w.add_scalar(f"end/{reason}", endings.count(reason) / len(res), step)
            if validation is not None:
                w.add_scalar("ga/validation", validation, step)
            w.flush()
        val = f"  validation {validation:.3f}" if validation is not None else ""
        self.log(f"[{self.run_dir.name}] gen {g:5d}  {track:<16} best {scores.max():.3f}  "
                 f"mean {scores.mean():.3f}  laps {len(laps)}/{len(res)}  {st.env_steps:,} steps{val}")


# ------------------------------------------------------------------ export
class _Spaces(gym.Env):
    """Only the spaces: SB3 needs an env to build a policy, and nothing here steps it."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        self.observation_space, self.action_space = observation_space, action_space


def export_model(folder: Path, flat: np.ndarray, train_cfg: Mapping[str, Any],
                 observation_space: gym.Space, action_space: gym.Space) -> None:
    """Write ``model.zip`` (the weights in an SB3 PPO actor) and an identity ``vecnormalize.pkl``.

    ``config.yaml`` is the caller's: it holds the run's snapshot, as for every model (§9).
    """
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from racer.agents.algorithms import get_algorithm

    sizes = ne.layer_sizes(observation_space.shape[0], train_cfg["policy"]["net_arch"], action_space.shape[0])
    layers = ne.unflatten(np.asarray(flat, dtype=np.float32), sizes)
    folder.mkdir(parents=True, exist_ok=False)
    venv = DummyVecEnv([lambda: _Spaces(observation_space, action_space)])
    policy = get_algorithm("ga").build(train_cfg, venv, None, None)
    linears = [m for m in policy.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    with torch.no_grad():
        for lin, (w, b) in zip([*linears, policy.policy.action_net], layers):
            lin.weight.copy_(torch.as_tensor(w))
            lin.bias.copy_(torch.as_tensor(b))
    policy.save(folder / "model.zip")
    VecNormalize(venv, training=False, norm_obs=False, norm_reward=False).save(str(folder / "vecnormalize.pkl"))
