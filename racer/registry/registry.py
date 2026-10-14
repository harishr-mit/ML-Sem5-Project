"""Model registry: ``registry/runs.csv`` and ``registry/models.csv`` (PROJECT_CONTEXT §9, D11).

The CSVs are the source of truth (``registry.xlsx`` is generated from them). Rows are
upserted by id with an atomic rewrite, because evaluation updates rows that training
created; every write holds a lock file, so parallel runs never lose each other's rows. Scratch runs (``train.py --scratch``) use their own registry in
``outputs/scratch/registry/`` so smoke tests never touch the real one.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from racer.config import PROJECT_ROOT, load_yaml

REGISTRY_DIR = PROJECT_ROOT / "registry"
SCRATCH_REGISTRY_DIR = PROJECT_ROOT / "outputs" / "scratch" / "registry"
SHOWCASE = ("training_ground", "technical", "fast_flow")
# Maps added with the map library (D19) and evaluated by protocol v2. Columns are only ever
# appended, so older CSVs and readers keep working.
LIBRARY_MAPS = ("hairpin_valley", "esses", "oval_infield", "city_blocks", "alpine", "coastline")

RUN_COLUMNS = [
    "run_id", "created_at", "tag", "algorithm", "parent_model", "obs_spec", "physics_preset", "reward_preset",
    "track_set", "curriculum", "total_steps", "wall_clock_min", "seed", "hparams_json", "notes",
    # model shape (D23): number of inputs, output layout, parameters of the driving network
    "obs_dim", "action_layout", "n_params",
]
MODEL_COLUMNS = [
    "model_id", "run_id", "steps", "path", "is_best", "completion_rate", "crash_rate", "heldout_completion",
    *[f"best_lap_{t}" for t in SHOWCASE], "mean_speed", "offroad_pct", "mean_lat_g",
    "overall_completion", "lap_ratio", "score", "eval_protocol", "eval_date", "tags", "notes",
    # protocol v2 (D20)
    "test_completion", "unseen_completion", *[f"best_lap_{t}" for t in LIBRARY_MAPS],
]


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def typed(v: str) -> Any:
    """Parse a CSV cell back into int / float / bool / None / str."""
    if v == "":
        return None
    if v in ("true", "false"):
        return v == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


@contextmanager
def file_lock(path: Path, timeout_s: float = 300.0, stale_s: float = 600.0) -> Iterator[None]:
    """Cross-process lock: an exclusively created lock file (works on Windows; T64).

    Parallel training runs read-modify-write the same CSVs, so every write holds this lock.
    A lock older than ``stale_s`` (a crashed writer) is broken.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except (FileExistsError, PermissionError):
            # Windows raises PermissionError (not FileExistsError) while the previous holder's
            # lock file is still being deleted (T76): it is held, so wait like for a live lock.
            try:
                if time.time() - path.stat().st_mtime > stale_s:
                    path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            except PermissionError:
                pass
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError(f"{path} is held by another process (delete it if no run is active)")
            time.sleep(0.05)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


class Registry:
    def __init__(self, root: Path | str = REGISTRY_DIR) -> None:
        self.root = Path(root)
        self.runs_path = self.root / "runs.csv"
        self.models_path = self.root / "models.csv"

    def lock(self):
        """Hold while reading-modifying-writing this registry (parallel runs, T64)."""
        return file_lock(self.root / "registry.lock")

    # ----------------------------------------------------------- reading
    @staticmethod
    def _read(path: Path) -> list[dict[str, str]]:
        if not path.is_file():
            return []
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def runs(self) -> list[dict[str, str]]:
        return self._read(self.runs_path)

    def models(self) -> list[dict[str, str]]:
        return self._read(self.models_path)

    def run(self, run_id: str) -> dict[str, str] | None:
        return next((r for r in self.runs() if r["run_id"] == run_id), None)

    def model(self, model_id: str) -> dict[str, str] | None:
        return next((r for r in self.models() if r["model_id"] == model_id), None)

    # ----------------------------------------------------------- writing
    @staticmethod
    def _write(path: Path, columns: list[str], rows: Iterable[Mapping[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in columns})
        for attempt in range(10):     # Excel locks an open CSV on Windows
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 9:
                    raise PermissionError(f"{path} is locked (open in Excel?); new rows are in {tmp}")
                time.sleep(0.5)

    def _upsert(self, path: Path, columns: list[str], key: str, row: Mapping[str, Any]) -> None:
        unknown = set(row) - set(columns)
        if unknown:
            raise KeyError(f"unknown registry columns {sorted(unknown)}")
        rows = self._read(path)
        new = {k: _cell(v) for k, v in row.items()}
        for r in rows:
            if r[key] == new[key]:
                r.update(new)
                break
        else:
            rows.append(new)
        self._write(path, columns, rows)

    def upsert_run(self, row: Mapping[str, Any]) -> None:
        with self.lock():
            self._upsert(self.runs_path, RUN_COLUMNS, "run_id", row)

    def upsert_model(self, row: Mapping[str, Any]) -> None:
        with self.lock():
            self._upsert(self.models_path, MODEL_COLUMNS, "model_id", row)

    def set_best(self, run_id: str) -> str | None:
        """Mark the best evaluated checkpoint of a run (D9 ranking); returns its id."""
        with self.lock():
            return self._set_best(run_id)

    def _set_best(self, run_id: str) -> str | None:
        from racer.agents.evaluation import protocol_version, rank_key

        rows = self.models()
        cands = [r for r in rows if r["run_id"] == run_id and r.get("eval_date")]
        if not cands:
            return None
        # Rank only within the newest protocol used in this run (metrics differ between protocols).
        newest = max(protocol_version(r.get("eval_protocol")) for r in cands)
        cands = [r for r in cands if protocol_version(r.get("eval_protocol")) == newest]
        best = min(cands, key=lambda r: (rank_key({k: typed(v) for k, v in r.items()}), int(r["steps"] or 0)))
        for r in rows:
            if r["run_id"] == run_id:
                r["is_best"] = _cell(r is best)
        self._write(self.models_path, MODEL_COLUMNS, rows)
        return best["model_id"]


# ------------------------------------------------------------- row builders
def rel_path(folder: Path) -> str:
    try:
        return folder.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return folder.as_posix()


def track_set_label(env_config: Mapping[str, Any]) -> str:
    presets = env_config.get("presets", {})
    if presets.get("trackset"):
        return presets["trackset"]
    tracks = env_config.get("tracks")
    return "+".join(tracks) if isinstance(tracks, list) else str(tracks)


def run_row(snapshot: Mapping[str, Any], total_steps: int | None = None,
            wall_clock_min: float | None = None) -> dict[str, Any]:
    """runs.csv row from a model folder's config.yaml snapshot."""
    from racer.agents.compat import manifest_of

    env = snapshot["env_config"]
    train = snapshot["train"]
    hp = {k: train[k] for k in ("ppo", "sac", "ga", "policy", "vecnormalize", "n_envs") if k in train}
    try:
        man = manifest_of(snapshot)
    except (KeyError, ValueError):      # a model this engine can't describe (e.g. an unknown feature)
        man = {}
    return {
        "run_id": snapshot["run_id"], "created_at": snapshot.get("created_at"), "tag": snapshot.get("tag"),
        "algorithm": snapshot.get("algorithm", "ppo"), "parent_model": snapshot.get("parent_model"),
        "obs_spec": env["env"]["obs_spec"]["name"], "physics_preset": env["presets"]["physics"],
        "reward_preset": env["presets"]["reward"], "track_set": track_set_label(env),
        "curriculum": snapshot.get("curriculum"), "total_steps": total_steps, "wall_clock_min": wall_clock_min,
        "seed": snapshot.get("seed"), "hparams_json": json.dumps(hp, sort_keys=True), "notes": snapshot.get("notes"),
        "obs_dim": man.get("obs_dim"), "action_layout": man.get("action_layout"), "n_params": man.get("n_params"),
    }


def model_row(folder: Path, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": snapshot.get("model_id", folder.name), "run_id": snapshot["run_id"],
        "steps": snapshot.get("steps"), "path": rel_path(folder), "tags": snapshot.get("tag"),
    }


def metrics_row(doc: Mapping[str, Any]) -> dict[str, Any]:
    """models.csv metric columns from an eval.json document.

    Every metric column is set (missing ones blank), so a row never mixes two protocols.
    """
    m = doc["metrics"]
    first = MODEL_COLUMNS.index("completion_rate")
    metric_cols = [c for c in MODEL_COLUMNS[first:] if c not in ("eval_protocol", "eval_date", "tags", "notes")]
    row = {k: m.get(k) for k in metric_cols}
    row.update({"model_id": doc["model_id"], "eval_protocol": doc["protocol"], "eval_date": doc["evaluated_at"]})
    return row


def newer_or_same_protocol(existing: Mapping[str, Any] | None, doc: Mapping[str, Any]) -> bool:
    """True if ``doc`` may replace the metrics of ``existing`` (the newest protocol wins, §9)."""
    from racer.agents.evaluation import protocol_version

    if not existing or not existing.get("eval_protocol"):
        return True
    return protocol_version(doc["protocol"]) >= protocol_version(existing["eval_protocol"])


def register_folder(reg: Registry, folder: Path) -> dict[str, Any]:
    """Register (or refresh) one model folder's checkpoint row; returns its snapshot."""
    snap = load_yaml(folder / "config.yaml")
    reg.upsert_model(model_row(folder, snap))
    return snap


def backfill(reg: Registry, models_dir: Path) -> list[str]:
    """Register every run and checkpoint found in ``models_dir`` (e.g. runs from before the registry)."""
    by_run: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for cfg in sorted(models_dir.glob("*/config.yaml")):
        snap = register_folder(reg, cfg.parent)
        by_run.setdefault(snap["run_id"], []).append((cfg.parent, snap))
    for run_id, items in by_run.items():
        items.sort(key=lambda it: it[1].get("steps", 0))
        first, last = items[0][1], items[-1][1]
        minutes = None
        if first.get("created_at") and last.get("saved_at"):
            t0 = dt.datetime.fromisoformat(first["created_at"])
            t1 = dt.datetime.fromisoformat(last["saved_at"])
            minutes = round((t1 - t0).total_seconds() / 60, 1)
        existing = reg.run(run_id)
        if existing and existing.get("total_steps"):
            continue      # train.py already wrote the run row
        reg.upsert_run(run_row(last, last.get("steps"), minutes))
    return sorted(by_run)


def ablation_group(tag: str) -> str:
    """Experiment group of a run tag: the tag without a trailing seed suffix (``_s1``)."""
    return re.sub(r"_s\d+$", "", tag or "")
