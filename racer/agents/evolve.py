"""Neuroevolution runs (D26, docs/NEUROEVOLUTION_PLAN.md): the evolution loop, its worker pool,
the lineage database, pause/resume, and exporting a car as a model folder.

A run lives in ``outputs/evolve/<run_name>/``: ``state.npz`` (the population to drive next; kept
when the run finishes, so it can be extended, D29), ``lineage.sqlite`` (every car of every
generation), ``snapshot.yaml`` (the run's snapshot, reused on resume), ``finished.json`` (the
run has stopped) and ``stop.request`` (pause).
Generation ``g`` draws its map (and, with ``ga.start: random``, its start) from ``default_rng([seed, g, 0])`` and is bred from
``default_rng([seed, g, 1])``, and every car drives alone from a fully specified start, so a
run is identical however often it was paused and however many workers drove it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
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
import yaml

from racer.agents import neuroevolution as ne
from racer.config import PROJECT_ROOT, from_dict

EVOLVE_DIR = PROJECT_ROOT / "outputs" / "evolve"
PAUSED_EXIT = 3          # train.py's exit code for a paused run; the queue marks it "interrupted"
FINISHED = "finished.json"
_DRAW, _BREED = 0, 1
RESUME_FREE = ("workers", "max_generations", "evaluate")     # ga settings a resumed run may change


@dataclass(frozen=True)
class GAParams:
    population: int
    elite_frac: float
    sigma: float
    speed_scale: float
    workers: int
    max_generations: int | None
    state_every: int
    validation: dict
    evaluate: dict
    start: str = "random"         # "line": every try on the start line, standing; older snapshots lack it

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


def settings_key(env_config: Mapping[str, Any], train_cfg: Mapping[str, Any], legacy: bool = False) -> str:
    """What must match to resume a run: everything except the worker count, the generation budget,
    the patience (they only decide when to stop; D29) and which checkpoints are evaluated afterwards.
    ``legacy``: the key of states saved before D29, which included the patience."""
    train = json.loads(json.dumps(train_cfg))
    for k in RESUME_FREE:
        train["ga"].pop(k, None)
    if not legacy:
        train["ga"].get("validation", {}).pop("patience", None)
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
    """One row per car per generation (``cars``), per generation (``generations``) and per
    validated elite (``validations``: its mean score over the validation tracks)."""

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
            CREATE TABLE IF NOT EXISTS validations (
                generation INTEGER, car INTEGER, score REAL, PRIMARY KEY (generation, car));
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        """)

    def set_meta(self, **kv: Any) -> None:
        self.db.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in kv.items()])
        self.db.commit()

    def meta(self) -> dict[str, Any]:
        return {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM meta")}

    def write(self, generation: int, cars: list[tuple], gen_row: Mapping[str, Any],
              validations: Mapping[int, float] | None = None) -> None:
        # INSERT OR REPLACE: after a crash, generations since the last state save are driven
        # again and rewritten with identical rows.
        self.db.executemany("INSERT OR REPLACE INTO cars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", cars)
        self.db.executemany("INSERT OR REPLACE INTO validations VALUES (?, ?, ?)",
                            [(generation, int(c), float(s)) for c, s in (validations or {}).items()])
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


# (weights, generation, car, env_steps, fitness, validation score)
Checkpoint = Callable[[np.ndarray, int, int, int, float, float], Any]


class Evolution:
    """The loop of one run. ``run`` returns "finished" (plateau or budget) or "paused"."""

    def __init__(self, run_dir: Path, env_config: Mapping[str, Any], train_cfg: Mapping[str, Any],
                 obs_dim: int, act_dim: int, checkpoint: Checkpoint | None = None, writer=None,
                 log: Callable[[str], None] = print, extend: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.extend = extend          # continue a finished run (D29); otherwise it is refused
        self.env_config, self.train_cfg = env_config, train_cfg
        self.p = from_dict(GAParams, train_cfg["ga"], where="train.ga")
        if self.p.start not in ("line", "random"):
            raise ValueError(f"train.ga.start must be 'line' or 'random', not {self.p.start!r}")
        self.seed = int(train_cfg["seed"])
        self.budget = self.p.max_generations
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
            if st.settings not in (key, settings_key(self.env_config, self.train_cfg, legacy=True)):
                raise ValueError(f"{self.run_dir.name} was started with different settings; only ga."
                                 f"{', ga.'.join(RESUME_FREE)} and ga.validation.patience may change when resuming")
            st.settings = key
            self.log(f"[{self.run_dir.name}] resuming at generation {st.generation} "
                     f"({st.env_steps:,} env steps so far)")
            return st
        if self.extend and is_finished(self.run_dir):       # finished before D29: no state was kept
            st = self.restore_state()
            self.log(f"[{self.run_dir.name}] state restored from the lineage: resuming at generation "
                     f"{st.generation} ({st.env_steps:,} env steps so far)")
            st.save(self.state_path)
            return st
        g0 = ne.first_generation(self.sizes, self.p.population, self._rng(0, _BREED))
        st = State(0, g0.weights, g0.parents, g0.seeds, settings=key)
        st.save(self.state_path)          # from now on a rerun finds and resumes this run, even after a crash
        return st

    def restore_state(self) -> State:
        """The state a run would have kept when it finished, rebuilt from its lineage (runs that
        finished before D29 kept none): the last driven generation replayed car by car, bred into
        the next one, and the validation counters replayed with ``min_gain``."""
        lin = Lineage(self.run_dir / "lineage.sqlite")
        try:
            last = lin.db.execute("SELECT MAX(generation) FROM cars").fetchone()[0]
            if last is None:
                raise ValueError(f"{self.run_dir.name} has no driven generation to restore")
            pop = None
            for g in range(last + 1):
                rows = lin.db.execute("SELECT parent, seed FROM cars WHERE generation = ? ORDER BY car", (g,)).fetchall()
                if g == 0:
                    pop = np.stack([ne.init_weights(self.sizes, int(s)) for _, s in rows])
                else:
                    pop = np.stack([pop[p] if s == ne.NO_NOISE else ne.mutate(pop[p], self.p.sigma, int(s))
                                    for p, s in rows])
            scores = np.array([r[0] for r in lin.db.execute(
                "SELECT fitness FROM cars WHERE generation = ? ORDER BY car", (last,))])
            env_steps = lin.db.execute("SELECT env_steps FROM generations WHERE generation = ?", (last,)).fetchone()[0]
            best, gain = None, 0
            for g, val in lin.db.execute("SELECT generation, validation FROM generations "
                                         "WHERE validation IS NOT NULL ORDER BY generation"):
                if best is None or val > best * (1 + float(self.p.validation["min_gain"])):
                    best, gain = val, g
        finally:
            lin.close()
        done = self.run_dir / FINISHED
        wall_s = float(json.loads(done.read_text(encoding="utf-8")).get("wall_s", 0.0)) if done.is_file() else 0.0
        off = ne.next_generation(pop, scores, self.p.n_elite, self.p.sigma, self._rng(last + 1, _BREED))
        return State(last + 1, off.weights, off.parents, off.seeds, int(env_steps), best, gain, wall_s,
                     settings_key(self.env_config, self.train_cfg))

    def _start(self, generation: int) -> dict[str, Any]:
        """Map and start of a generation, the same for every car."""
        from racer.env.trackset import get_track
        rng = self._rng(generation, _DRAW)
        name = self.maps[int(rng.integers(len(self.maps)))]
        if self.p.start == "line":         # the map is drawn first, so both modes evolve on the same maps
            return {"track": name, "start_s": 0.0, "lateral": 0.0, "heading_offset": 0.0, "speed": 0.0}
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

    def _validate(self, pool, cars: np.ndarray) -> np.ndarray:
        """Each car's mean fitness over the validation tracks (standing start on the start line)."""
        start = {"start_s": 0.0, "lateral": 0.0, "heading_offset": 0.0, "speed": 0.0}
        n = int(self.p.validation["tracks"])
        tasks = [(cars, self.sizes, start, self.p.speed_scale, k) for k in range(n)]
        per_track = [[r[0] for r in part] for part in self._map(pool, tasks)]
        return np.mean(np.array(per_track), axis=0)

    # ------------------------------------------------------------ the loop
    def run(self, pause_after: int | None = None) -> str:
        """Evolve until the validation plateau, the generation budget, or a pause.

        ``pause_after``: pause once this many generations are done (tests).
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        finished = is_finished(self.run_dir)
        if finished and not self.extend:
            raise RuntimeError(f"{self.run_dir.name} has finished; extend it to continue it (train.py --extend)")
        st = self._initial_state()
        if finished:
            (self.run_dir / FINISHED).unlink()      # running again: a crash from now on resumes it as an unfinished run
        kept: State | None = None
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
                    over_budget = self.budget is not None and g + 1 >= int(self.budget)

                    # Validate the elites (the cars that go on unchanged), not just this map's winner:
                    # the winner on one map is often not the car that drives other tracks best.
                    validation, val_scores, val_car = None, {}, None
                    if (g + 1) % int(v["every"]) == 0 or over_budget:
                        elites = ne.rank(scores)[:self.p.n_elite]
                        means = self._validate(pool, st.weights[elites])
                        val_scores = {int(c): float(m) for c, m in zip(elites, means)}
                        val_car = int(elites[int(np.argmax(means))])     # ties: the better-ranked elite
                        validation = val_scores[val_car]
                        if st.best_validation is None or validation > st.best_validation * (1 + float(v["min_gain"])):
                            st.best_validation, st.last_gain = validation, g
                    lineage.write(g, [(g, k, int(st.parents[k]), int(st.seeds[k]), *res[k]) for k in range(len(res))],
                                  {"map": start["track"], **{k: start[k] for k in ("start_s", "lateral",
                                                                                   "heading_offset", "speed")},
                                   "best": float(scores.max()), "mean": float(scores.mean()),
                                   "median": float(np.median(scores)),
                                   "finished": sum(r[4] == "lap" for r in res), "env_steps": st.env_steps,
                                   "validation": validation}, val_scores)
                    self._log_generation(g, start["track"], res, scores, validation, st)

                    plateau = val_car is not None and g - st.last_gain >= int(v["patience"])
                    done = plateau or over_budget
                    if self.checkpoint and val_car is not None:
                        self.checkpoint(st.weights[val_car], g, val_car, st.env_steps, float(scores[val_car]),
                                        validation)
                    if done:
                        outcome = "finished"
                        off = ne.next_generation(st.weights, scores, self.p.n_elite, self.p.sigma,
                                                 self._rng(g + 1, _BREED))
                        kept = State(g + 1, off.weights, off.parents, off.seeds, st.env_steps, st.best_validation,
                                     st.last_gain, st.wall_s, st.settings)       # where an extension continues (D29)
                        reason = "validation plateau" if plateau else "generation budget"
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
            kept.wall_s = st.wall_s
            kept.save(self.state_path)          # before finished.json: a crash in between leaves a resumable run
            (self.run_dir / FINISHED).write_text(json.dumps(
                {"generations": st.generation + 1, "env_steps": st.env_steps, "wall_s": round(st.wall_s, 1),
                 "best_validation": st.best_validation}), encoding="utf-8")
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


def write_model_folder(folder: Path, flat: np.ndarray, snapshot: Mapping[str, Any],
                       observation_space: gym.Space, action_space: gym.Space) -> None:
    """A complete model folder, or nothing: written under a dot name (skipped by the registry,
    backup and integrity scans) and renamed once ``config.yaml`` is in it."""
    # The PID keeps a sandbox export and the trainer's checkpoint of the same car apart.
    partial = folder.with_name(f".{folder.name}.{os.getpid()}.partial")
    shutil.rmtree(partial, ignore_errors=True)
    if partial.exists():
        raise OSError(f"can't remove the unfinished export {partial}; delete it and try again")
    export_model(partial, flat, snapshot["train"], observation_space, action_space)
    with (partial / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(snapshot), f, sort_keys=False, allow_unicode=True)
    for attempt in range(20):      # a virus scanner or the indexer may hold the new model.zip briefly
        try:
            partial.rename(folder)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


# ------------------------------------------------------- export any car later
def run_dir_of(run_id: str, project: Path = PROJECT_ROOT) -> Path:
    """The evolve folder of a GA run id (R### in outputs/evolve/, S### in outputs/scratch/evolve/)."""
    root = project / "outputs" / ("scratch/evolve" if run_id.startswith("S") else "evolve")
    found = sorted(root.glob(f"{run_id}_GA_*")) if root.is_dir() else []
    if not found:
        raise FileNotFoundError(f"no neuroevolution run {run_id} in {root}")
    return found[0]


def generation_table(run_dir: Path) -> list[dict[str, Any]]:
    """Per driven generation: map, best/mean fitness, finishers, env steps, validation (if validated)."""
    lin = Lineage(run_dir / "lineage.sqlite")
    try:
        cur = lin.db.execute("SELECT generation, map, best, mean, finished, env_steps, validation "
                             "FROM generations ORDER BY generation")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        lin.close()


def pick_car(run_dir: Path, generation: int) -> int:
    """The generation's best-validated elite if it was validated, else its best car on its map."""
    lin = Lineage(run_dir / "lineage.sqlite")
    try:
        # Ties as the checkpoint breaks them: the elite ranked higher (fitness, then index) wins.
        row = lin.db.execute("SELECT v.car FROM validations v JOIN cars c ON c.generation = v.generation "
                             "AND c.car = v.car WHERE v.generation = ? ORDER BY v.score DESC, c.fitness DESC, v.car "
                             "LIMIT 1", (generation,)).fetchone()
        if row is None:
            row = lin.db.execute("SELECT car FROM cars WHERE generation = ? ORDER BY fitness DESC, car LIMIT 1",
                                 (generation,)).fetchone()
        if row is None:
            raise KeyError(f"generation {generation} of {run_dir.name} hasn't been driven")
        return int(row[0])
    finally:
        lin.close()


def export_car(run_dir: Path, generation: int, car: int | None = None, out_dir: Path | None = None,
               registry=None) -> tuple[Path, bool]:
    """Rebuild a car from the lineage and save it as a registered model folder.

    ``car`` defaults to :func:`pick_car`. ``out_dir``/``registry`` default to the run's own
    (models/ for R runs, outputs/scratch/ for S runs). Returns (folder, created); created is
    False if that car is already saved (e.g. as a checkpoint).
    """
    from racer.config import load_yaml
    from racer.env.racing_env import RacingEnv
    from racer.registry.record import registry_for
    from racer.registry.registry import register_folder

    base = load_yaml(run_dir / "snapshot.yaml")
    if base["versions"]["numpy"] != np.__version__:
        raise RuntimeError(f"{run_dir.name} ran with numpy {base['versions']['numpy']}, this is {np.__version__}: "
                           "rebuilt cars would differ (random streams are only fixed within a numpy version)")
    car = pick_car(run_dir, generation) if car is None else int(car)
    lin = Lineage(run_dir / "lineage.sqlite")
    try:
        g_row = lin.db.execute("SELECT env_steps FROM generations WHERE generation = ?", (generation,)).fetchone()
        c_row = lin.db.execute("SELECT fitness FROM cars WHERE generation = ? AND car = ?", (generation, car)).fetchone()
        v_row = lin.db.execute("SELECT score FROM validations WHERE generation = ? AND car = ?",
                               (generation, car)).fetchone()
        if g_row is None or c_row is None:
            raise KeyError(f"no car {car} in generation {generation} of {run_dir.name}")
        meta = lin.meta()
        flat = ne.rebuild(lin.origin, generation, car, meta["sizes"], meta["sigma"])
    finally:
        lin.close()
    scratch = run_dir.name.startswith("S")
    out_dir = out_dir if out_dir is not None else PROJECT_ROOT / ("outputs/scratch" if scratch else "models")
    env_steps = int(g_row[0])
    model_id = f"{run_dir.name}@{env_steps // 1000}k"
    folder = out_dir / model_id
    if folder.exists():
        saved = load_yaml(folder / "config.yaml")
        same_car = saved.get("car", car) == car       # checkpoints from before T104 have no car
        if saved.get("generation") == generation and same_car:
            return folder, False
        model_id = f"{run_dir.name}_g{generation}c{car}@{env_steps // 1000}k"
        folder = out_dir / model_id
        if folder.exists():
            return folder, False
    snap = dict(base, model_id=model_id, steps=env_steps, generation=generation, car=car,
                fitness=round(float(c_row[0]), 6), validation=None if v_row is None else round(float(v_row[0]), 6),
                saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"), exported_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    env = RacingEnv(base["env_config"])
    write_model_folder(folder, flat, snap, env.observation_space, env.action_space)
    register_folder(registry if registry is not None else registry_for(scratch), folder)
    return folder, True

