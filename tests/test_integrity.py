"""Cross-component integrity (T79): registry, model folders, eval copies, run notes, docs, pins.

The other tests check each component on its own; these check that they still agree with each
other, e.g. after an interrupted run, a hand edit or a doc change. Model folders are local
(git-ignored), so those checks skip on a machine without them.
"""
from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
import yaml

from racer.config import PROJECT_ROOT
from racer.registry.registry import Registry

MODELS_DIR = PROJECT_ROOT / "models"
EVAL_COPIES = PROJECT_ROOT / "outputs" / "eval"
MODEL_FILES = ("model.zip", "vecnormalize.pkl", "config.yaml")


def eval_file(protocol: str) -> str:
    return "eval.json" if protocol == "v1" else f"eval_{protocol}.json"


def is_true(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1")


@pytest.fixture(scope="module")
def registry() -> tuple[list[dict], list[dict]]:
    reg = Registry()
    return reg.runs(), reg.models()


def local_models() -> set[str]:
    if not MODELS_DIR.is_dir():
        return set()
    return {p.name for p in MODELS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")}


# ---------------------------------------------------------------- registry on its own

def test_registry_ids_are_unique_and_linked(registry):
    runs, models = registry
    run_ids = [r["run_id"] for r in runs]
    model_ids = [m["model_id"] for m in models]
    assert len(set(run_ids)) == len(run_ids), "duplicate run_id in runs.csv"
    assert len(set(model_ids)) == len(model_ids), "duplicate model_id in models.csv"
    orphans = [m["model_id"] for m in models if m["run_id"] not in run_ids]
    assert not orphans, f"models of unknown runs: {orphans[:5]}"


def test_every_run_has_exactly_one_best_checkpoint(registry):
    runs, models = registry
    bad = {}
    for r in runs:
        best = [m for m in models if m["run_id"] == r["run_id"] and is_true(m["is_best"])]
        if len(best) != 1:
            bad[r["run_id"]] = len(best)
    assert not bad, f"runs without exactly one best checkpoint: {bad}"


def test_every_evaluated_model_has_its_eval_copy(registry):
    """outputs/eval/ is tracked, so this holds on every machine (D10)."""
    _, models = registry
    missing = [m["model_id"] for m in models if m.get("eval_protocol")
               and not (EVAL_COPIES / m["model_id"] / eval_file(m["eval_protocol"])).is_file()]
    assert not missing, f"eval copies missing in outputs/eval: {missing[:5]}"


def test_run_notes_match_the_runs(registry):
    runs, _ = registry
    notes = yaml.safe_load((PROJECT_ROOT / "registry" / "run_notes.yaml").read_text(encoding="utf-8")) or {}
    run_ids = {r["run_id"] for r in runs} - {"BASE"}
    assert not run_ids - set(notes), f"runs without notes: {sorted(run_ids - set(notes))}"
    assert not set(notes) - run_ids, f"notes for unknown runs: {sorted(set(notes) - run_ids)}"


# ---------------------------------------------------------------- registry vs local model folders

def test_model_folders_and_registry_agree(registry):
    folders = local_models()
    if not folders:
        pytest.skip("no local models/ (git-ignored)")
    _, models = registry
    registered = {m["model_id"] for m in models if not m["run_id"] == "BASE"}
    assert not folders - registered, f"model folders not in the registry: {sorted(folders - registered)[:5]}"
    assert not registered - folders, f"registered models without a folder: {sorted(registered - folders)[:5]}"


def test_model_folders_are_complete(registry):
    folders = local_models()
    if not folders:
        pytest.skip("no local models/ (git-ignored)")
    _, models = registry
    bad = []
    for m in models:
        if m["model_id"] not in folders:
            continue
        folder = MODELS_DIR / m["model_id"]
        bad += [f"{m['model_id']}/{f}" for f in MODEL_FILES if not (folder / f).is_file()]
        if m.get("eval_protocol") and not (folder / eval_file(m["eval_protocol"])).is_file():
            bad.append(f"{m['model_id']}/{eval_file(m['eval_protocol'])}")
    assert not bad, f"missing model files: {bad[:5]}"


# ---------------------------------------------------------------- docs, tracks, pins

def test_models_named_in_the_docs_exist(registry):
    _, models = registry
    ids = {m["model_id"] for m in models}
    text = "".join((PROJECT_ROOT / f).read_text(encoding="utf-8") for f in ("README.md", "PROJECT_CONTEXT.md"))
    named = set(re.findall(r"R\d{3}_[a-z0-9_]+@\d+k", text))
    assert not named - ids, f"docs name models that aren't registered: {sorted(named - ids)}"


def test_presets_used_in_readme_commands_exist():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    kinds = {"env": "env", "trackset": "trackset", "physics": "physics", "reward": "reward", "train": "train"}
    missing = [f"--{flag} {name}" for flag, name in re.findall(r"--(env|trackset|physics|reward|train) ([a-z0-9_]+)", text)
               if not (PROJECT_ROOT / "configs" / kinds[flag] / f"{name}.yaml").is_file()]
    assert not missing, f"README uses presets that don't exist: {missing}"


def test_track_names_are_unique_across_folders():
    seen: dict[str, str] = {}
    dupes = []
    for f in sorted((PROJECT_ROOT / "tracks").glob("*/*.json")):
        if f.stem in seen:
            dupes.append(f"{f.stem} ({seen[f.stem]}, {f.parent.name})")
        seen[f.stem] = f.parent.name
    assert not dupes, f"track names are looked up by name, so they must be unique: {dupes}"


def test_installed_packages_match_the_pins():
    """Pins keep saved models loadable (§14.1); a drifted package is caught here."""
    drift = []
    for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s#]+)", line.strip())
        if not m:
            continue
        try:
            got = version(m.group(1))
        except PackageNotFoundError:
            got = "not installed"
        if got != m.group(2):
            drift.append(f"{m.group(1)}: pinned {m.group(2)}, installed {got}")
    assert not drift, drift
