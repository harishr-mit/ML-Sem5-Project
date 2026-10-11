"""AI Sandbox (T66–T72): data, explanations, diagrams, builder, queue, launcher, and the
Streamlit pages rendered headless with ``streamlit.testing`` (no browser)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE_KEYS = ["overview", "explorer", "compare", "build", "launch", "glossary"]


def _run_page(key: str) -> None:            # executed by AppTest as a script
    import importlib
    importlib.import_module(f"racer.sandbox.pages.{key}").render()


# ---------------------------------------------------------------- launcher (T66)

def test_server_settings_are_local_and_private():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        sandbox = importlib.import_module("sandbox")
    finally:
        sys.path.pop(0)
    s = sandbox.server_settings(8501)
    opts = dict(zip(s[::2], s[1::2]))
    assert opts["--server.address"] == "localhost"
    assert opts["--browser.gatherUsageStats"] == "false"
    assert opts["--server.headless"] == "true"          # no first-run e-mail prompt, no ~/.streamlit


def test_every_page_is_routed():
    from racer.sandbox import pages
    assert [k for k, *_ in pages.PAGES] == PAGE_KEYS


def _run_app() -> None:                     # executed by AppTest as a script
    from racer.sandbox.pages import run_app
    run_app()


def test_app_with_navigation_renders_home():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_run_app, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title and at.title[0].value == "Overview"


# ---------------------------------------------------------------- fixtures: a tiny project

def make_snapshot(run_id: str, model_id: str, steps: int, env: str = "rays5", parent: str | None = None,
                  tag: str = "t", features: list | None = None) -> dict:
    """A config.yaml snapshot like train.py writes (manifest included)."""
    from racer.agents.compat import build_manifest
    from racer.config import load_preset
    env_cfg = load_preset("env", env)
    if features is not None:
        env_cfg["obs_spec"] = {"version": 2, "name": "custom", "features": features}
    train = load_preset("train", "ppo_default")
    env_config = {"presets": {"physics": "default", "env": env, "reward": "progress", "track": "default",
                              "trackset": "tg"},
                  "tracks": {"trackset": "tg", "fixed": [{"track": "training_ground", "weight": 1.0}]},
                  "physics": load_preset("physics", "default"), "env": env_cfg,
                  "reward": load_preset("reward", "progress"), "track": load_preset("track", "default")}
    snap = {"run_id": run_id, "tag": tag, "algorithm": "ppo", "seed": 1, "created_at": "2026-10-11T00:00:00",
            "git_commit": "abc", "notes": "why it was made", "parent_model": parent, "curriculum": "",
            "train": train, "env_config": env_config, "model_id": model_id, "steps": steps}
    try:
        snap["manifest"] = build_manifest(env_config, train)
    except (KeyError, ValueError):
        pass                                     # e.g. an unknown feature: the engine can't describe it
    return snap


def add_model(root: Path, snap: dict, metrics: dict | None = None, protocols=("v2",), is_best=False,
              trial: bool = False) -> None:
    import json

    import yaml

    from racer.registry.registry import Registry, model_row, run_row
    base = root / "outputs" / "scratch" if trial else root / "models"
    folder = base / snap["model_id"]
    folder.mkdir(parents=True)
    (folder / "config.yaml").write_text(yaml.safe_dump(snap, allow_unicode=True), encoding="utf-8")
    reg = Registry(root / ("outputs/scratch/registry" if trial else "registry"))
    reg.upsert_run(run_row(snap))
    row = model_row(folder, snap)
    row["path"] = folder.relative_to(root).as_posix()
    if metrics:
        row.update({k: v for k, v in metrics.items() if k in ("overall_completion", "lap_ratio")},
                   eval_protocol=protocols[-1], eval_date="2026-10-11", is_best=is_best)
    reg.upsert_model(row)
    eval_dir = root / "outputs" / ("scratch/eval" if trial else "eval") / snap["model_id"]
    for p in protocols if metrics else ():
        eval_dir.mkdir(parents=True, exist_ok=True)
        name, thumb = ("eval.json", "thumb_") if p == "v1" else (f"eval_{p}.json", f"thumb_{p}_")
        doc = {"model_id": snap["model_id"], "protocol": p, "metrics": dict(metrics, protocol_tag=p), "episodes": []}
        (eval_dir / name).write_text(json.dumps(doc), encoding="utf-8")
        (eval_dir / f"{thumb}training_ground.png").write_bytes(b"png")


@pytest.fixture()
def project(tmp_path):
    """R001 (3 checkpoints, @200k best), R002 fine-tuned from R001@200k, BASE without folders."""
    from racer.registry.registry import Registry
    add_model(tmp_path, make_snapshot("R001", "R001_t@100k", 100_000), {"overall_completion": 0.2}, ("v1", "v2"))
    add_model(tmp_path, make_snapshot("R001", "R001_t@200k", 200_000), {"overall_completion": 0.9}, ("v2",),
              is_best=True)
    add_model(tmp_path, make_snapshot("R001", "R001_t@300k", 300_000), {"overall_completion": 0.5}, ("v2",))
    add_model(tmp_path, make_snapshot("R002", "R002_ft@100k", 100_000, env="rays9", parent="R001_t@200k"))
    Registry(tmp_path / "registry").upsert_run({"run_id": "BASE", "tag": "baseline"})
    from racer.sandbox.data import Store
    return Store(tmp_path)


# ---------------------------------------------------------------- data (T67)

def test_store_runs_checkpoints_and_best(project):
    runs = [r["run_id"] for r in project.explorable_runs()]
    assert runs == ["R002", "R001"]                         # BASE has no model folder
    cps = project.checkpoints("R001")
    assert [c["steps"] for c in cps] == [100_000, 200_000, 300_000]
    assert project.best("R001")["model_id"] == "R001_t@200k"
    assert project.best("R002")["model_id"] == "R002_ft@100k"   # no is_best → the last checkpoint
    assert project.snapshot("R001_t@300k")["steps"] == 300_000
    assert project.manifest("R002_ft@100k")["obs_dim"] == 17
    assert project.problems("R001_t@100k") == []


def test_store_eval_files_and_thumbnails(project):
    assert project.eval_protocols("R001_t@100k") == ["v1", "v2"]
    assert project.eval_doc("R001_t@100k")["protocol"] == "v2"                  # newest by default
    assert project.eval_doc("R001_t@100k", "v1")["metrics"]["protocol_tag"] == "v1"
    assert list(project.thumbnails("R001_t@100k", "v1")) == ["training_ground"]
    assert project.thumbnails("R001_t@100k", "v1")["training_ground"].name == "thumb_training_ground.png"
    assert project.thumbnails("R001_t@100k", "v2")["training_ground"].name == "thumb_v2_training_ground.png"
    rows = project.checkpoint_metrics("R001", "v2")
    assert [r["steps"] for r in rows] == [100_000, 200_000, 300_000]
    assert [r["is_best"] for r in rows] == [False, True, False]
    assert [r["steps"] for r in project.checkpoint_metrics("R001", "v1")] == [100_000]
    assert project.eval_doc("R002_ft@100k") is None


def test_store_parent_chain_and_trials(project, tmp_path):
    assert project.parent_chain("R002_ft@100k") == ["R002_ft@100k", "R001_t@200k"]
    add_model(tmp_path, make_snapshot("S001", "S001_try@20k", 20_000), trial=True)
    assert [r["run_id"] for r in project.explorable_runs(trials=True)] == ["R002", "R001", "S001"]
    assert project.folder("S001_try@20k") == tmp_path / "outputs" / "scratch" / "S001_try@20k"
    assert project.best("S001")["kind"] == "trial"


def test_run_notes_round_trip(project):
    project.save_note("R001", "outcome", "laps the training ground")
    project.save_note("R001", "why", "first run")
    assert project.notes()["R001"] == {"why": "first run", "outcome": "laps the training ground"}
    assert next(r for r in project.runs() if r["run_id"] == "R001")["outcome"] == "laps the training ground"
    text = project.notes_path.read_text(encoding="utf-8")
    assert text.startswith("# Why each run")                # the header survives a rewrite
    project.save_note("R001", "why", "")
    project.save_note("R001", "outcome", " ")
    assert "R001" not in project.notes()
    with pytest.raises(ValueError):
        project.save_note("R001", "other", "x")


def test_model_the_engine_cant_run_is_reported(tmp_path):
    from racer.sandbox.data import Store
    snap = make_snapshot("R009", "R009_x@100k", 100_000, features=[{"feature": "lidar_3d"}])
    add_model(tmp_path, snap)
    problems = Store(tmp_path).problems("R009_x@100k")
    assert problems and "lidar_3d" in " ".join(problems)
    assert Store(tmp_path).manifest("R009_x@100k") == {}


# ---------------------------------------------------------------- explanations (T67)

def test_glossary_entries_are_complete():
    from racer.sandbox.explain import glossary
    g = glossary()
    assert len(g) > 150
    bad = [k for k, e in g.items() if not e.get("short") or not e.get("affects")]
    assert not bad, bad


def test_every_preset_key_is_explained():
    import yaml

    from racer.sandbox.explain import preset_rows, unexplained
    keys = set()
    for kind in ("physics", "reward", "train", "env", "trackset", "track"):
        for f in (PROJECT_ROOT / "configs" / kind).glob("*.yaml"):
            keys |= set(preset_rows(kind, yaml.safe_load(f.read_text(encoding="utf-8")) or {}))
    assert keys and unexplained(keys) == []


def test_every_metric_column_is_explained():
    from racer.registry.registry import MODEL_COLUMNS
    from racer.sandbox.explain import unexplained
    first = MODEL_COLUMNS.index("completion_rate")
    metrics = [c for c in MODEL_COLUMNS[first:] if c not in ("eval_protocol", "eval_date", "tags", "notes")]
    assert unexplained(f"metric.{c}" for c in metrics) == []


def test_every_saved_snapshot_key_is_explained():
    """One snapshot per run (every checkpoint of a run shares its config)."""
    import yaml

    from racer.agents.compat import manifest_of
    from racer.sandbox.explain import snapshot_rows, unexplained
    seen, keys = set(), set()
    for base in (PROJECT_ROOT / "models", PROJECT_ROOT / "outputs" / "scratch"):
        for cfg in sorted(base.glob("*/config.yaml")):
            run = cfg.parent.name.split("_", 1)[0]
            if run in seen:
                continue
            seen.add(run)
            snap = yaml.safe_load(cfg.read_text(encoding="utf-8"))
            keys |= set(snapshot_rows(snap, manifest_of(snap)))
    if not seen:
        pytest.skip("no saved models (models/ is local only)")
    assert unexplained(keys) == []


def test_snapshot_rows_and_diff():
    from racer.sandbox.explain import diff, flatten, fmt, row_kind, snapshot_rows
    assert flatten({"a": {"b": 1, "c": [1, 2]}, "d": {}}) == {"a.b": 1, "a.c": [1, 2], "d": {}}
    a = make_snapshot("R001", "R001_t@100k", 100_000)
    b = make_snapshot("R002", "R002_t@100k", 100_000, env="rays9")
    ra, rb = snapshot_rows(a), snapshot_rows(b)
    assert ra["physics.mass_kg"] == 1000.0 and ra["train.ppo.gamma"] == 0.995
    assert row_kind("run.git_commit") == "identity" and row_kind("manifest.obs_dim") == "derived"
    assert row_kind("train.seed") == "identity" and row_kind("physics.mass_kg") == "config"
    rows = {r["key"]: r for r in diff([ra, rb])}
    changed = {k for k, r in rows.items() if r["differs"] and r["kind"] == "config"}
    assert changed == {"presets.env", "env.obs_spec.name", "env.obs_spec.rays"}
    assert rows["physics.mass_kg"]["note"].startswith("Car mass")
    assert fmt([{"angle_deg": -90.0, "range_m": 60.0}, {"angle_deg": 0, "range_m": 200}]) == "-90°/60 m, +0°/200 m"
    assert fmt(1.0) == "1" and fmt(None) == "—" and fmt(True) == "yes"


def test_real_diff_r004_vs_r008_is_only_the_rays():
    """The E4 ablation changed only the ray list (T67 'done when')."""
    from racer.sandbox.data import STORE
    from racer.sandbox.explain import diff, snapshot_rows
    a, b = STORE.best("R004"), STORE.best("R008")
    if not a or not b:
        pytest.skip("R004 / R008 not available locally")
    rows = [snapshot_rows(STORE.snapshot(m["model_id"]), STORE.manifest(m["model_id"])) for m in (a, b)]
    changed = {r["key"] for r in diff(rows) if r["differs"] and r["kind"] == "config"}
    assert changed == {"presets.env", "env.obs_spec.name", "env.obs_spec.rays"}


# ---------------------------------------------------------------- pages (headless)

@pytest.mark.parametrize("key", PAGE_KEYS)
def test_page_renders(key):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_run_page, args=(key,), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_glossary_search():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_run_page, args=("glossary",), default_timeout=60)
    at.run()
    at.text_input(key="glossary_q").input("grip").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "mu_static" in text and "wheelbase_m" not in text
