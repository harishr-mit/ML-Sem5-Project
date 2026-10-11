"""Read access to everything the sandbox shows: runs, checkpoints, snapshots, manifests,
evaluation files, thumbnails and run notes (T67).

Real runs live in ``registry/`` + ``models/`` + ``outputs/eval/`` + ``runs/``; trial runs
(``train.py --scratch``) in ``outputs/scratch/``. Files are re-read only when they change
(cached by modification time), so the pages stay fast with hundreds of checkpoints.
The only file written here is ``registry/run_notes.yaml`` (owner, T65); models and the
registry CSVs are never touched.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from racer.config import PROJECT_ROOT

FOLDER_TTL_S = 30.0       # how long a model folder's existence is remembered

NOTES_HEADER = """\
# Why each run was made and what it showed (AI Sandbox, T67; owner decision T65).
# Model snapshots can never change, so explanations written after a run live here, keyed by
# run id. Edited by hand or in the sandbox's Model Explorer. Fields: why, outcome.
"""


@dataclass(frozen=True)
class Source:
    """Where one kind of run keeps its files."""

    kind: str            # "real" or "trial"
    registry: Path
    models: Path
    evals: Path
    tensorboard: Path


_cache: dict[tuple[str, str], tuple[tuple[int, int], Any]] = {}


def _cached(path: Path, tag: str, load: Callable[[Path], Any]) -> Any:
    """``load(path)``, re-run only when the file changes (mtime + size)."""
    st = path.stat()
    key = (str(path), tag)
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(key)
    if hit and hit[0] == stamp:
        return hit[1]
    value = load(path)
    _cache[key] = (stamp, value)
    return value


def _load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    from racer.registry.registry import Registry, typed
    return [{k: typed(v) for k, v in r.items()} for r in Registry._read(path)]


class Store:
    """All sandbox reads for one project folder (tests point it at a temporary one)."""

    def __init__(self, root: Path | str = PROJECT_ROOT) -> None:
        self.root = Path(root)
        scratch = self.root / "outputs" / "scratch"
        self.real = Source("real", self.root / "registry", self.root / "models", self.root / "outputs" / "eval",
                           self.root / "runs")
        self.trial = Source("trial", scratch / "registry", scratch, scratch / "eval", scratch / "tb")
        self.notes_path = self.root / "registry" / "run_notes.yaml"
        self.track_previews = self.root / "outputs" / "plots" / "tracks"
        self._folders: dict[str, tuple[float, Path | None]] = {}

    # ------------------------------------------------------------ registry rows
    def sources(self, trials: bool = False) -> list[Source]:
        return [self.real, self.trial] if trials else [self.real]

    def source_of(self, run_or_model_id: str) -> Source:
        return self.trial if str(run_or_model_id).startswith("S") else self.real

    def _rows(self, src: Source, name: str) -> list[dict[str, Any]]:
        path = src.registry / name
        if not path.is_file():
            return []
        rows = _cached(path, "csv", _read_csv)
        return [dict(r, kind=src.kind) for r in rows]

    def runs(self, trials: bool = False) -> list[dict[str, Any]]:
        """runs.csv rows (typed), with the run notes' ``why`` / ``outcome`` added."""
        notes = self.notes()
        out = []
        for src in self.sources(trials):
            for r in self._rows(src, "runs.csv"):
                n = notes.get(str(r["run_id"]), {})
                out.append(dict(r, why=n.get("why"), outcome=n.get("outcome")))
        return out

    def models(self, trials: bool = False) -> list[dict[str, Any]]:
        """models.csv rows (typed); each row holds the metrics of its newest evaluation protocol."""
        return [r for src in self.sources(trials) for r in self._rows(src, "models.csv")]

    def run(self, run_id: str) -> dict[str, Any] | None:
        return next((r for r in self.runs(trials=True) if r["run_id"] == run_id), None)

    def model_row(self, model_id: str) -> dict[str, Any] | None:
        src = self.source_of(model_id)
        row = self._model_index(src).get(model_id)
        return dict(row, kind=src.kind) if row else None

    def checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        """Checkpoint rows of a run that still have a model folder, by steps."""
        rows = [r for r in self.models(trials=True) if r["run_id"] == run_id and self.folder(r["model_id"])]
        return sorted(rows, key=lambda r: int(r.get("steps") or 0))

    def best(self, run_id: str) -> dict[str, Any] | None:
        """The run's ``is_best`` checkpoint, else its last one."""
        cps = self.checkpoints(run_id)
        return next((r for r in cps if r.get("is_best") is True), cps[-1] if cps else None)

    def explorable_runs(self, trials: bool = False) -> list[dict[str, Any]]:
        """Runs with at least one model folder (the baseline pseudo-run has none): real runs
        first, newest first."""
        runs = [r for r in self.runs(trials) if self.checkpoints(str(r["run_id"]))]
        return sorted(runs, key=lambda r: (r["kind"] == "real", str(r["run_id"])), reverse=True)

    # ------------------------------------------------------------ model folders
    def _model_index(self, src: Source) -> dict[str, dict[str, Any]]:
        path = src.registry / "models.csv"
        if not path.is_file():
            return {}
        return _cached(path, "index", lambda p: {str(r["model_id"]): r for r in _read_csv(p)})

    def folder(self, model_id: str) -> Path | None:
        """The model's folder (registry path first, then by id), or None if it is gone.
        Remembered for ``FOLDER_TTL_S`` (hundreds of checkpoints are checked on every page)."""
        now = time.monotonic()
        hit = self._folders.get(model_id)
        if hit and now - hit[0] < FOLDER_TTL_S:
            return hit[1]
        src = self.source_of(model_id)
        row = self._model_index(src).get(model_id)
        cands = [self.root / str(row["path"])] if row and row.get("path") else []
        cands.append(src.models / model_id)
        found = next((c for c in cands if (c / "config.yaml").is_file()), None)
        self._folders[model_id] = (now, found)
        return found

    def snapshot(self, model_id: str) -> dict[str, Any]:
        folder = self.folder(model_id)
        if folder is None:
            raise FileNotFoundError(f"no model folder for {model_id}")
        return _cached(folder / "config.yaml", "yaml", _load_yaml)

    def run_snapshot(self, run_id: str) -> dict[str, Any] | None:
        """Snapshot of the run's best checkpoint (all checkpoints of a run share the config)."""
        best = self.best(run_id)
        return self.snapshot(best["model_id"]) if best else None

    def manifest(self, model_id: str) -> dict[str, Any]:
        """The model's manifest (stored, or derived for models from before D23); {} if the
        engine can't describe it (see :meth:`problems`)."""
        from racer.agents.compat import manifest_of
        snap = self.snapshot(model_id)
        return _cached(self.folder(model_id) / "config.yaml", "manifest", lambda _p: _safe(manifest_of, snap, {}))

    def problems(self, model_id: str) -> list[str]:
        """Why the engine can't run this model (empty = it can)."""
        from racer.agents.compat import check
        try:
            snap = self.snapshot(model_id)
        except FileNotFoundError as e:
            return [str(e)]
        return _cached(self.folder(model_id) / "config.yaml", "check", lambda _p: _safe(check, snap, ["unreadable"]))

    def parent_chain(self, model_id: str) -> list[str]:
        """The model, its parent, the parent's parent, … (stops at a missing folder or a loop)."""
        chain = [model_id]
        while True:
            try:
                parent = self.snapshot(chain[-1]).get("parent_model")
            except FileNotFoundError:
                break
            if not parent:
                break
            parent = Path(str(parent)).name          # older snapshots may store a folder path
            if parent in chain:
                break
            chain.append(parent)
        return chain

    # ------------------------------------------------------------ evaluation files
    def _eval_dirs(self, model_id: str) -> list[Path]:
        dirs = [self.source_of(model_id).evals / model_id]
        folder = self.folder(model_id)
        return ([folder] if folder else []) + dirs

    def eval_protocols(self, model_id: str) -> list[str]:
        """Protocols this model was evaluated with, oldest first (e.g. ['v1', 'v2'])."""
        found = set()
        for d in self._eval_dirs(model_id):
            if (d / "eval.json").is_file():
                found.add(1)
            for p in d.glob("eval_v*.json"):
                v = p.stem.removeprefix("eval_v")
                if v.isdigit():
                    found.add(int(v))
        return [f"v{v}" for v in sorted(found)]

    def eval_doc(self, model_id: str, protocol: str | None = None) -> dict[str, Any] | None:
        """The evaluation document for a protocol (default: the newest one), or None."""
        from racer.agents.evaluation import eval_filenames, protocol_version
        protos = self.eval_protocols(model_id)
        if not protos:
            return None
        protocol = protocol or protos[-1]
        name, _ = eval_filenames(protocol_version(protocol))
        for d in self._eval_dirs(model_id):
            if (d / name).is_file():
                return _cached(d / name, "json", _load_json)
        return None

    def thumbnails(self, model_id: str, protocol: str) -> dict[str, Path]:
        """Trajectory thumbnails of one evaluation, by track name."""
        from racer.agents.evaluation import eval_filenames, protocol_version
        _, prefix = eval_filenames(protocol_version(protocol))
        out: dict[str, Path] = {}
        for d in reversed(self._eval_dirs(model_id)):            # the model folder wins
            for p in d.glob(f"{prefix}*.png"):
                track = p.stem[len(prefix):]
                if protocol_version(protocol) == 1 and track.startswith("v") and "_" in track \
                        and track.split("_", 1)[0][1:].isdigit():
                    continue                                       # thumb_v2_* is not a v1 thumbnail
                out[track] = p
        return dict(sorted(out.items()))

    def checkpoint_metrics(self, run_id: str, protocol: str) -> list[dict[str, Any]]:
        """Metrics of every checkpoint of a run evaluated with ``protocol`` (from the eval files,
        so older protocols stay available after a re-evaluation), by steps."""
        rows = []
        for cp in self.checkpoints(run_id):
            doc = self.eval_doc(cp["model_id"], protocol) if protocol in self.eval_protocols(cp["model_id"]) else None
            if doc:
                rows.append({"model_id": cp["model_id"], "steps": int(cp.get("steps") or 0),
                             "is_best": cp.get("is_best") is True, **doc["metrics"]})
        return rows

    def baseline_laps(self, protocol: str) -> dict[str, float]:
        from racer.agents.evaluation import baseline_laps, load_protocol
        try:
            return baseline_laps(load_protocol(protocol), self.real.evals)
        except (FileNotFoundError, KeyError, ValueError):
            return {}

    # ------------------------------------------------------------ tracks
    def track_preview(self, name: str) -> Path | None:
        p = self.track_previews / f"{name}.png"
        return p if p.is_file() else None

    # ------------------------------------------------------------ learning curves
    def tensorboard_dirs(self, run_id: str) -> list[Path]:
        """TensorBoard folders of a run (``<run_id>_<tag>_<n>``)."""
        tb = self.source_of(run_id).tensorboard
        return sorted(p for p in tb.glob(f"{run_id}_*") if p.is_dir()) if tb.is_dir() else []

    # ------------------------------------------------------------ run notes
    def notes(self) -> dict[str, dict[str, str]]:
        if not self.notes_path.is_file():
            return {}
        data = _cached(self.notes_path, "yaml", _load_yaml) or {}
        return {str(k): dict(v or {}) for k, v in data.items()}

    def save_note(self, run_id: str, field: str, text: str) -> None:
        """Set (or clear, with empty text) one field of a run's notes; atomic rewrite."""
        if field not in ("why", "outcome"):
            raise ValueError(f"unknown note field {field!r}")
        notes = self.notes()
        entry = notes.setdefault(str(run_id), {})
        if text.strip():
            entry[field] = text.strip()
        else:
            entry.pop(field, None)
        if not entry:
            notes.pop(str(run_id))
        body = yaml.safe_dump({k: {f: v[f] for f in ("why", "outcome") if f in v} for k, v in sorted(notes.items())},
                              allow_unicode=True, sort_keys=False, width=100)
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.notes_path.with_suffix(".yaml.tmp")
        tmp.write_text(NOTES_HEADER + body, encoding="utf-8")
        os.replace(tmp, self.notes_path)


def _safe(fn: Callable, arg: Any, default: Any) -> Any:
    try:
        return fn(arg)
    except (KeyError, ValueError, TypeError) as e:
        return default if not isinstance(default, list) else [*default, str(e)]


def experiment_group(tag: str | None) -> str:
    """Experiment group of a run tag (the tag without its seed suffix, e.g. ``e4_rays9``)."""
    from racer.registry.registry import ablation_group
    return ablation_group(str(tag or ""))


STORE = Store()
