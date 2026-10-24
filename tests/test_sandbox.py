"""AI Sandbox (T66–T72): data, explanations, diagrams, builder, queue, launcher, and the
Streamlit pages rendered headless with ``streamlit.testing`` (no browser)."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAGE_KEYS = ["overview", "explorer", "compare", "workbench", "build", "presets", "draw", "launch", "glossary"]


@pytest.fixture(autouse=True)
def _restore_main_module():
    """AppTest runs pages as ``__main__`` and leaves it replaced; Windows' spawn-based process
    pools in later tests (test_queue.py) would then re-run that script in their workers."""
    main = sys.modules.get("__main__")
    yield
    if main is not None:
        sys.modules["__main__"] = main


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


def _sandbox_script():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        return importlib.import_module("sandbox")
    finally:
        sys.path.pop(0)


def test_window_settings_keep_webview_data_in_the_project():      # D25, T75
    sandbox = _sandbox_script()
    cfg = sandbox.window_settings()
    storage = (PROJECT_ROOT / cfg["storage_dir"]).resolve()
    assert storage.is_relative_to((PROJECT_ROOT / "outputs" / "sandbox").resolve())   # git-ignored
    assert len(cfg["size_px"]) == 2 and len(cfg["min_size_px"]) == 2


def test_window_falls_back_to_the_browser(monkeypatch):
    sandbox = _sandbox_script()
    opened = []
    monkeypatch.setattr(sandbox.webbrowser, "open", opened.append)
    monkeypatch.setattr(sandbox, "open_window", lambda url, cfg: False)     # e.g. no WebView2
    assert sandbox.show("http://localhost:1", "window", {}) is False
    assert sandbox.show("http://localhost:1", "browser", {}) is False
    assert sandbox.show("http://localhost:1", "none", {}) is False
    assert opened == ["http://localhost:1", "http://localhost:1"]


def test_window_blocks_until_closed(monkeypatch):
    sandbox = _sandbox_script()
    monkeypatch.setattr(sandbox, "open_window", lambda url, cfg: True)
    monkeypatch.setattr(sandbox.webbrowser, "open", lambda url: pytest.fail("browser opened"))
    assert sandbox.show("http://localhost:1", "window", {}) is True


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


def test_goto_switches_page_on_the_next_run():
    """Buttons hand a selection to another page (e.g. Explorer → Launch)."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_run_app, default_timeout=120)
    at.run()
    at.session_state["_goto"] = "glossary"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Glossary"


def test_explorer_launch_button_preselects_the_model():
    from streamlit.testing.v1 import AppTest

    from racer.sandbox.data import STORE
    best = STORE.best("R015")
    if not best:
        pytest.skip("R015 not available locally")
    at = AppTest.from_function(_run_app, default_timeout=180)
    at.session_state["_goto"] = "explorer"
    at.session_state["explorer_model"] = best["model_id"]
    at.run()
    assert at.title[0].value == "Model Explorer"
    at.button(key="ex_launch").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value == "Launch"
    assert at.multiselect(key="ln_models").value == [best["model_id"]]


# ---------------------------------------------------------------- fixtures: a tiny project

def make_snapshot(run_id: str, model_id: str, steps: int, env: str = "rays5", parent: str | None = None,
                  tag: str = "t", features: list | None = None, layout: str | None = None) -> dict:
    """A config.yaml snapshot like train.py writes (manifest included)."""
    from racer.agents.compat import build_manifest
    from racer.config import load_preset
    env_cfg = load_preset("env", env)
    if features is not None:
        env_cfg["obs_spec"] = {"version": 2, "name": "custom", "features": features}
    if layout:
        env_cfg["action_spec"] = {"layout": layout}
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


# ---------------------------------------------------------------- diagrams (T68)

RAYS7 = [{"angle_deg": a, "range_m": r} for a, r in
         ((-60, 60), (-20, 120), (-7, 150), (0, 150), (7, 150), (20, 120), (60, 60))]


@pytest.mark.parametrize("offset", [0.0, 2.0, -3.5])
def test_straight_hits_match_the_sensor(offset):
    """The straight-road diagram agrees with the real ray sensor within 0.1 m."""
    import numpy as np

    from racer.config import load_preset
    from racer.env.sensors import cast_rays, pack_raycast_params
    from racer.sandbox.diagrams import straight_hits
    from racer.track.io import spec_from_dict
    from racer.track.track import Track
    # a stadium: long straights at y = 0 and y = 200 (x 0..600), 12 m wide
    pts = [[x, 0] for x in range(0, 601, 100)] + [[700, 100]] + [[x, 200] for x in range(600, -1, -100)] + [[-100, 100]]
    tr = Track(spec_from_dict({"name": "stadium", "control_points": pts, "width_m": 12.0, "runoff_m": 15.0,
                               "kerb_m": 1.2, "direction": "ccw"}))
    raycast = load_preset("env", "rays5")["raycast"]
    out = np.empty(len(RAYS7))
    f = tr.field
    cast_rays(f.dist, f.idx, f.geom, 300.0, offset, 0.0, np.radians([r["angle_deg"] for r in RAYS7]),
              np.array([float(r["range_m"]) for r in RAYS7]), pack_raycast_params(raycast), out)
    expected = straight_hits(RAYS7, 12.0, offset)
    assert np.allclose(out, expected, atol=0.1), (out, expected)


def test_straight_hits_geometry():
    from racer.sandbox.diagrams import straight_hits
    d = straight_hits([{"angle_deg": 90, "range_m": 60}, {"angle_deg": -30, "range_m": 60},
                       {"angle_deg": 0, "range_m": 200}, {"angle_deg": 7, "range_m": 20}], 12.0)
    assert d[0] == pytest.approx(6.0) and d[1] == pytest.approx(12.0)
    assert d[2] == 200 and d[3] == 20                            # capped at the range


def test_layer_table_adds_up_to_the_manifest_count():
    from racer.agents.compat import count_params
    from racer.sandbox.diagrams import layer_table
    for algo in ("ppo", "sac", "ga"):
        for obs, act, arch in ((13, 2, [256, 256]), (15, 3, [64, 64, 64]), (9, 2, [32])):
            rows = layer_table(algo, obs, act, arch)
            want = count_params(algo, obs, act, arch)
            assert sum(r["params"] for r in rows if r["network"] == "actor") == want["policy"]
            assert sum(r["params"] for r in rows) == want["total"]


def test_every_algorithm_on_the_build_page_validates():
    from racer.sandbox import builder as B
    for algo in B.BUILDABLE:
        plan = B.plan_from_presets(algorithm=algo)
        plan.tag, plan.why = "check", "every listed algorithm must build"
        assert B.validate(plan) == [], algo


def test_diagrams_render():
    from racer.sandbox import diagrams as D
    for fig in (D.ray_diagram_straight([("a", RAYS7), ("b", RAYS7[:3])], offset_m=1.0),
                D.ray_diagram_map("training_ground", 100.0, 0.0, [("a", RAYS7, _raycast())]),
                D.network_diagram([f"in{i}" for i in range(30)], [256, 256], ["steer", "gas"])):
        assert D.png(fig)[:4] == b"\x89PNG"
    assert D.rays_of([{"feature": "v_long", "params": {}}]) == []


def _raycast() -> dict:
    from racer.config import load_preset
    return load_preset("env", "rays5")["raycast"]


# ---------------------------------------------------------------- launcher (T70)

FAKE_DEMO = "import sys, time\nprint('fake demo', sys.argv[1:], flush=True)\ntime.sleep(120)\n"


@pytest.fixture()
def fake_launcher(tmp_path):
    from racer.sandbox.launcher import Launcher
    (tmp_path / "fake_demo.py").write_text(FAKE_DEMO, encoding="utf-8")
    launcher = Launcher(root=tmp_path, script="fake_demo.py")
    yield launcher
    launcher.stop_all()


def test_demo_argv():
    from racer.sandbox.launcher import demo_argv
    assert demo_argv("technical", ["A", "B"], laps=2, baseline=True, fair=True, title="T") == [
        "--track", "technical", "--models", "A", "B", "--laps", "2", "--baseline", "--fair", "--title", "T"]
    assert demo_argv("x", run="R001", every=3) == ["--track", "x", "--run", "R001", "--every", "3"]
    assert demo_argv("x", baseline=True) == ["--track", "x", "--baseline"]
    with pytest.raises(ValueError):
        demo_argv("x")
    with pytest.raises(ValueError):
        demo_argv("x", ["A"], run="R001")


def test_launch_list_and_stop(fake_launcher):
    from racer.sandbox import procs
    from racer.sandbox.launcher import Launcher
    a = fake_launcher.launch("technical", ["R001_a@100k"], laps=1)
    b = fake_launcher.launch("alpine", ["R002_b@100k", "R003_c@100k"], baseline=True)
    assert a["id"] == 1 and b["id"] == 2 and "#2 · alpine" in b["title"]
    assert [i["id"] for i in fake_launcher.running()] == [1, 2]
    again = Launcher(root=fake_launcher.root, script="fake_demo.py")       # e.g. the sandbox restarted
    assert [i["id"] for i in again.running()] == [1, 2]
    assert again.stop(1) in ("closed", "killed")
    assert not procs.alive(a["pid"], a["created"])
    assert procs.alive(b["pid"], b["created"])                             # the other window keeps running
    assert [(i["id"], i["running"]) for i in fake_launcher.instances()] == [(1, False), (2, True)]
    import time
    for _ in range(50):                                                    # the game's output goes to its log
        if "fake demo" in Launcher.log_tail(b):
            break
        time.sleep(0.1)
    assert "fake demo" in Launcher.log_tail(b) and "--title" in Launcher.log_tail(b)
    assert again.stop(1) == "not running"


def test_recycled_pid_is_not_ours():
    import os

    from racer.sandbox import procs
    me = os.getpid()
    assert procs.alive(me, procs.created(me))
    assert not procs.alive(me, procs.created(me) - 100.0)                 # same PID, other creation time
    assert procs.stop(me, procs.created(me) - 100.0) == "not running"


def test_launch_page_launches_and_stops(fake_launcher, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from racer.sandbox.data import STORE
    from racer.sandbox.pages import launch
    best = STORE.best("R015")
    if not best:
        pytest.skip("R015 not available locally")
    monkeypatch.setattr(launch, "_launcher", lambda: fake_launcher)
    at = AppTest.from_function(_run_page, args=("launch",), default_timeout=120)
    at.session_state["launch_models"] = [best["model_id"]]
    at.run()
    assert at.multiselect(key="ln_models").value == [best["model_id"]]
    at.button(key="ln_go").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Opened game window #1" in m.value for m in at.success)
    item = fake_launcher.running()[0]
    assert item["models"] == [best["model_id"]] and "--title" in item["argv"]
    at.button(key="stop_1").click().run()
    assert not fake_launcher.running()


# ---------------------------------------------------------------- builder (T71)

def test_v1_to_v2_gives_the_identical_observation():
    import numpy as np

    from racer.env.racing_env import RacingEnv, make_env_config
    from racer.sandbox.builder import v1_to_v2
    v1 = make_env_config(["technical"], env="rays9")
    v2 = make_env_config(["technical"], env="rays9")
    v2["env"]["obs_spec"] = {"version": 2, "name": "x", "features": v1_to_v2(v1["env"]["obs_spec"])}
    actions = np.random.default_rng(0).uniform(-1, 1, size=(150, 2)).astype(np.float32)
    outs = []
    for cfg in (v1, v2):
        env = RacingEnv(cfg)
        obs = [env.reset(seed=3)[0]]
        for a in actions:
            o, _, term, trunc, _ = env.step(a)
            obs.append(o)
            if term or trunc:
                break
        outs.append(np.array(obs))
    assert np.array_equal(outs[0], outs[1])


def test_plan_from_presets_reuses_the_preset():
    from racer.sandbox import builder as B
    p = B.plan_from_presets("rays7_narrow")
    p.tag, p.why = "narrow_again", "repeat R015 with a second seed to see how much is luck"
    assert B.matching_env_preset(p) == "rays7_narrow"
    assert B.train_sets(p) == []
    man = B.preview(p)
    assert man["obs_dim"] == 15 and man["n_params"] == 70404
    v1 = B.plan_from_presets("rays5")
    assert B.matching_env_preset(v1) == "rays5"           # a converted v1 spec still matches its preset


def test_plan_changes_become_sets_and_a_new_preset(tmp_path):
    import importlib.util

    import yaml

    from racer.env.obs_spec import build_observer
    from racer.sandbox import builder as B
    p = B.plan_from_presets("rays7_narrow")
    p.tag, p.why, p.seeds = "narrow9", "two more shallow rays: does seeing further help on Fast Flow?", [1, 2]
    p.features[0]["rays"] += [{"angle_deg": 3.0, "range_m": 200.0}, {"angle_deg": -3.0, "range_m": 200.0}]
    p.features.append({"feature": "steer_angle", "scale_deg": 30.0})
    p.hparams["ent_coef"] = 3e-05
    p.net_arch = [128, 128]
    assert B.matching_env_preset(p) is None
    sets = B.train_sets(p)
    assert sets == ["ppo.ent_coef=3.0e-05", "policy.net_arch=[128,128]"]
    spec = importlib.util.spec_from_file_location("train_script", PROJECT_ROOT / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    cfg = train.apply_sets(B.load_preset("train", "ppo_default"), sets)        # exactly as train.py reads them
    assert cfg["ppo"]["ent_coef"] == 3e-05 and cfg["policy"]["net_arch"] == [128, 128]
    (tmp_path / "proj" / "configs" / "env").mkdir(parents=True)
    path = B.write_env_preset(p, tmp_path / "proj")
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    obs = build_observer(body["obs_spec"], body["raycast"])
    assert path.name == "sb_narrow9.yaml" and obs.dim == 18                    # 9 rays + 8 + steer angle
    assert [f["feature"] for f in body["obs_spec"]["features"]][-2:] == ["onroad_frac", "prev_action"]
    assert B.write_env_preset(p, tmp_path / "proj") == path                     # same content: fine
    p2 = p.copy()
    p2.features = p2.features[:1]
    with pytest.raises(FileExistsError):
        B.write_env_preset(p2, tmp_path / "proj")
    entries = B.queue_entries(p, "sb_narrow9")
    assert [e["name"] for e in entries] == ["narrow9_s1", "narrow9_s2"]
    a = entries[1]["args"]
    assert a[a.index("--seed") + 1] == "2" and a[a.index("--env") + 1] == "sb_narrow9"
    assert a[a.index("--notes") + 1].startswith("two more") and "--scratch" not in a
    assert B.preview(p)["obs_dim"] == 18


def test_plan_validation():
    from racer.sandbox import builder as B
    p = B.plan_from_presets("rays7_narrow")
    errs = " ".join(B.validate(p))
    assert "Tag" in errs and "Why" in errs
    p.tag, p.why = "Bad Tag", "a reason long enough"
    assert any("Tag" in e for e in B.validate(p))
    p.tag, p.steps = "fine", 50_000
    assert any("Steps" in e for e in B.validate(p))
    p.steps = 1_000_000
    assert B.validate(p) == []
    assert any("already used" in e for e in B.validate(p, taken_tags={"fine_s1"}))
    p.tag = "fine_s3"
    p.seeds = [3]
    assert B.run_tags(p) == [("fine_s3", 3)]
    p.features = [f for f in p.features if f["feature"] != "rays"]
    assert any("No rays" in w for w in B.warnings(p))


def test_ga_plan_queues_a_valid_train_command():
    import importlib.util

    from racer.sandbox import builder as B
    p = B.plan_from_presets("rays7_lean", "ga")
    p.tag, p.why = "ga_main", "neuroevolution on the 7 training maps (E10)"
    assert p.trackset == "train7" and p.steps is None
    assert B.validate(p) == [] and B.matching_env_preset(p) == "rays7_lean" and B.train_sets(p) == []
    assert B.preview(p)["n_params"] == 69_122
    assert B.estimate_minutes(p.steps, 1, 1) is None
    assert B.estimate_minutes(50_000_000, 1, 1, algorithm="ga") is None     # PPO's rates don't apply
    p.hparams["sigma"] = 0.05
    p.hparams["validation"] = dict(p.hparams["validation"], patience=300)
    sets = B.train_sets(p)
    spec = importlib.util.spec_from_file_location("train_script", PROJECT_ROOT / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    cfg = train.apply_sets(B.load_preset("train", "ga_default"), sets)
    assert cfg["ga"]["sigma"] == 0.05 and cfg["ga"]["validation"]["patience"] == 300
    assert cfg["ga"]["validation"]["every"] == 20
    args = B.queue_entries(p, "rays7_lean")[0]["args"]
    assert "--steps" not in args and args[args.index("--train") + 1] == "ga_default"
    p.steps, p.generations = 50_000_000, 500                     # GA runs are budgeted in generations
    args = B.queue_entries(p, "rays7_lean")[0]["args"]
    assert args[args.index("--generations") + 1] == "500" and "--steps" not in args
    assert B.train_config(p)["ga"]["max_generations"] == 500 and "max_generations=" not in " ".join(B.train_sets(p))
    assert B.estimate_minutes(None, 1, 1, algorithm="ga", generations=500)["run"] > 0
    p.trackset = "maps7"
    assert any("generated pool" in e for e in B.validate(p))


def test_plan_from_a_real_snapshot():
    from racer.sandbox import builder as B
    from racer.sandbox.data import STORE
    for run, preset in (("R015", "rays7_narrow"), ("R004", "rays5")):
        best = STORE.best(run)
        if not best:
            pytest.skip(f"{run} not available locally")
        p = B.plan_from_snapshot(STORE.snapshot(best["model_id"]))
        assert B.matching_env_preset(p) == preset and B.train_sets(p) == []
        assert p.started_from == best["model_id"]


def test_time_estimate():
    from racer.sandbox.builder import estimate_minutes
    cfg = {"queue": {"minutes_per_million_alone": 10.0, "minutes_per_million_parallel": 13.0,
                     "eval_minutes_per_checkpoint": 0.0}}
    assert estimate_minutes(2_000_000, 1, 1, cfg) == {"run": 20.0, "queue": 20.0}
    assert estimate_minutes(2_000_000, 4, 2, cfg) == {"run": 26.0, "queue": 52.0}


# ---------------------------------------------------------------- queue (T71)

def test_queue_file_round_trip(tmp_path):
    import importlib.util

    from racer.sandbox.queue import QueueControl, steps_of
    qc = QueueControl(root=tmp_path, file="configs/queue/sandbox.yaml")
    assert qc.load() == {"parallel": 2, "runs": []}
    notes = 'Goal: test "quotes" and: colons'
    qc.add([{"name": "a_s1", "args": ["--tag", "a_s1", "--steps", "20000", "--notes", notes]},
            {"name": "b_s1", "args": ["--tag", "b_s1"]}])
    with pytest.raises(ValueError):
        qc.add([{"name": "a_s1", "args": []}])
    spec = importlib.util.spec_from_file_location("qo", PROJECT_ROOT / "scripts" / "queue_overnight.py")
    qo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qo)
    runs, parallel = qo.load_queue(qc.path)                 # the runner reads exactly what was written
    assert parallel == 2 and runs[0]["argv"][-1] == notes
    assert [e["status"] for e in qc.entries()] == ["waiting", "waiting"]
    assert qc.entries()[0]["steps"] == 20000 and steps_of(["--steps", "x"]) is None
    qc.remove("b_s1")
    assert qc.names() == {"a_s1"}
    qc.log_dir.mkdir(parents=True)
    (qc.log_dir / "state.json").write_text('{"a_s1": {"status": "running", "pid": 1}}', encoding="utf-8")
    assert qc.entries()[0]["status"] == "interrupted"       # no live runner
    (qc.log_dir / "state.json").write_text('{"a_s1": {"status": "done"}}', encoding="utf-8")
    with pytest.raises(ValueError):
        qc.remove("a_s1")                                      # finished entries stay
    assert not qc.request_stop()


SLOW = "import sys, time\nprint('slow', sys.argv[1:], flush=True)\ntime.sleep(120)\n"


def test_stop_request_interrupts_a_running_queue(tmp_path):
    """The sandbox's Stop queue: the runner stops its runs (whole process tree) and marks them interrupted."""
    import shutil
    import subprocess
    import sys
    import time
    import uuid

    from racer.sandbox import procs
    from racer.sandbox.queue import QueueControl
    stem = f"pytest_stop_{uuid.uuid4().hex[:8]}"
    (tmp_path / "slow.py").write_text(SLOW, encoding="utf-8")
    queue = tmp_path / f"{stem}.yaml"
    queue.write_text("parallel: 2\nruns:\n  - {name: one, args: [x]}\n  - {name: two, args: [y]}\n", encoding="utf-8")
    qc = QueueControl(file=str(queue))
    runner = subprocess.Popen([sys.executable, str(PROJECT_ROOT / "scripts" / "queue_overnight.py"), str(queue),
                               "--script", str(tmp_path / "slow.py"), "--poll", "0.2"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):                                   # up to 60 s (slow while trainings run)
            if [e["status"] for e in qc.entries()] == ["running", "running"] and qc.runner() is not None:
                break
            time.sleep(0.2)
        assert qc.runner() is not None
        assert [e["status"] for e in qc.entries()] == ["running", "running"]
        pids = [qc.state()[n]["pid"] for n in ("one", "two")]
        assert any("a queue is running" in r for r in qc.busy())
        assert qc.request_stop()
        assert runner.wait(timeout=30) == 130
        assert [e["status"] for e in qc.entries()] == ["interrupted", "interrupted"]
        assert qc.runner() is None and not (qc.log_dir / "runner.json").exists()
        assert not any(procs.alive(p, None) for p in pids)
    finally:
        if runner.poll() is None:                              # kill the venv launcher's child and its runs too
            import psutil
            for p in [*psutil.Process(runner.pid).children(recursive=True), psutil.Process(runner.pid)]:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            runner.wait(timeout=10)
        shutil.rmtree(qc.log_dir, ignore_errors=True)


# ---------------------------------------------------------------- pages (headless)

@pytest.mark.parametrize("key", PAGE_KEYS)
def test_page_renders(key):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_run_page, args=(key,), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("run", ["R001", "R015"])
def test_explorer_shows_real_models(run):
    """R001: v1 inputs, old snapshot (no manifest, track list); R015: v2 features."""
    from streamlit.testing.v1 import AppTest

    from racer.sandbox.data import STORE
    best = STORE.best(run)
    if not best:
        pytest.skip(f"{run} not available locally")
    at = AppTest.from_function(_run_page, args=("explorer",), default_timeout=180)
    at.session_state["explorer_model"] = best["model_id"]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.subheader[0].value == best["model_id"]
    text = " ".join(m.value for m in at.markdown)
    assert "mass_kg" in text and "gamma" in text and "progress" in text


def test_explorer_three_output_trial_model(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from racer.sandbox.data import Store
    from racer.sandbox.pages import _ui
    feats = [{"feature": "rays", "rays": RAYS7}, {"feature": "v_long", "scale": 70.0}, {"feature": "prev_action"}]
    add_model(tmp_path, make_snapshot("S001", "S001_tb@20k", 20_000, features=feats, layout="steer_throttle_brake"),
              trial=True)
    monkeypatch.setattr(_ui, "STORE", Store(tmp_path))
    at = AppTest.from_function(_run_page, args=("explorer",), default_timeout=180)
    at.session_state["explorer_model"] = "S001_tb@20k"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.subheader[0].value == "S001_tb@20k"
    text = " ".join(m.value for m in at.markdown)
    assert "`throttle`" in text and "`brake`" in text and "prev_throttle" in text


def test_compare_real_models():
    from streamlit.testing.v1 import AppTest

    from racer.sandbox.data import STORE
    best = [STORE.best(r) for r in ("R004", "R008", "R015")]
    if not all(best):
        pytest.skip("R004 / R008 / R015 not available locally")
    at = AppTest.from_function(_run_page, args=("compare",), default_timeout=180)
    at.session_state["compare_models"] = [b["model_id"] for b in best]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "`env.obs_spec.rays`" in text and "`presets.trackset`" in text
    assert "`physics.mass_kg`" not in text                 # same in all three: hidden by default
    assert "`env.action_spec.layout`" not in text          # old snapshots mean steer_gas (D22)


def test_compare_without_a_common_protocol_warns(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from racer.sandbox.data import Store
    from racer.sandbox.pages import _ui
    add_model(tmp_path, make_snapshot("R001", "R001_a@100k", 100_000), {"overall_completion": 0.5}, ("v1",))
    add_model(tmp_path, make_snapshot("R002", "R002_b@100k", 100_000, env="rays9"), {"overall_completion": 0.5},
              ("v2",))
    monkeypatch.setattr(_ui, "STORE", Store(tmp_path))
    at = AppTest.from_function(_run_page, args=("compare",), default_timeout=120)
    at.session_state["compare_models"] = ["R001_a@100k", "R002_b@100k"]
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("no evaluation protocol in common" in w.value for w in at.warning)


def test_build_page_adds_a_run_to_the_queue(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from racer.sandbox.pages import build
    from racer.sandbox.queue import QueueControl
    qc = QueueControl(root=tmp_path, file="configs/queue/sandbox.yaml")
    monkeypatch.setattr(build, "_qc", lambda: qc)
    at = AppTest.from_function(_run_page, args=("build",), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.button(key="bq_add").disabled                     # no tag / reason yet
    at.text_input(key="bq_tag_0").input("sbtest")
    at.text_area(key="bq_why_0").input("check that the form queues a run with the right arguments")
    at.text_input(key="bq_seeds_0").input("1, 2")
    at.run()
    assert not at.button(key="bq_add").disabled, [e.value for e in at.error]
    at.button(key="bq_add").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert qc.names() == {"sbtest_s1", "sbtest_s2"}
    args = qc.load()["runs"][1]["args"]
    assert args[args.index("--env") + 1] == "rays7_narrow" and args[args.index("--seed") + 1] == "2"
    assert "--scratch" not in args                              # real run by default (owner)
    assert any("Added sbtest_s1, sbtest_s2" in m.value for m in at.success)
    at.button(key="bq_rm_sbtest_s2").click().run()
    assert qc.names() == {"sbtest_s1"}


def test_build_page_queues_a_neuroevolution_run(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from racer.sandbox.pages import build
    from racer.sandbox.queue import QueueControl
    qc = QueueControl(root=tmp_path, file="configs/queue/sandbox.yaml")
    monkeypatch.setattr(build, "_qc", lambda: qc)
    at = AppTest.from_function(_run_page, args=("build",), default_timeout=180)
    at.run()
    at.selectbox(key="bq_from_env").select("rays7_lean")
    at.selectbox(key="bq_from_algo").select("ga").run()
    at.button(key="bq_load_presets").click().run()
    assert not at.exception, [e.value for e in at.exception]
    at.text_input(key="bq_tag_1").input("gatest")
    at.text_area(key="bq_why_1").input("check that the form queues a neuroevolution run")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.button(key="bq_add").disabled, [e.value for e in at.error]
    at.button(key="bq_add").click().run()
    args = qc.load()["runs"][0]["args"]
    assert args[args.index("--train") + 1] == "ga_default" and args[args.index("--trackset") + 1] == "train7"
    assert "--steps" not in args and "--set" not in args


def test_overview_leaderboard_order(tmp_path, monkeypatch):
    """Sorted by unseen completion then lap ratio; another car or reward goes last (not comparable)."""
    from streamlit.testing.v1 import AppTest

    from racer.sandbox.data import Store
    from racer.sandbox.pages import _ui
    from racer.sandbox.pages.overview import leaderboard
    for run, unseen, lap, physics in (("R001", 0.5, 1.1, "default"), ("R002", 0.9, 0.97, "default"),
                                      ("R003", 1.0, 0.4, "no_grip_limit"), ("R004", 0.5, 0.95, "default")):
        snap = make_snapshot(run, f"{run}_x@100k", 100_000)
        snap["env_config"]["presets"]["physics"] = physics
        add_model(tmp_path, snap, {"overall_completion": unseen, "unseen_completion": unseen, "lap_ratio": lap},
                  ("v2",), is_best=True)
    monkeypatch.setattr(_ui, "STORE", Store(tmp_path))
    df = leaderboard("v2")
    assert list(df["run"]) == ["R002", "R004", "R001", "R003"]
    assert df.iloc[-1]["not comparable"] == "physics: no_grip_limit"
    at = AppTest.from_function(_run_page, args=("overview",), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [m.value for m in at.metric][:2] == ["4", "4"]          # runs, checkpoints


def test_glossary_search():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_run_page, args=("glossary",), default_timeout=60)
    at.run()
    at.text_input(key="glossary_q").input("grip").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "mu_static" in text and "wheelbase_m" not in text


def test_explorer_evolution_tab_saves_any_generation(tmp_path, monkeypatch):
    import numpy as np
    import yaml
    from streamlit.testing.v1 import AppTest

    from racer.agents.compat import build_manifest
    from racer.agents.evolve import Evolution, export_car
    from racer.config import load_preset
    from racer.registry.registry import Registry, run_row
    from racer.sandbox.data import Store
    from racer.sandbox.pages import _ui
    snap = make_snapshot("S900", "S900_GA_x@0k", 0, env="rays7_lean", tag="x")
    train = load_preset("train", "ga_default")
    train["ga"].update(population=6, elite_frac=0.34, workers=1, max_generations=3,
                       validation={"every": 2, "tracks": 1, "stream": 2, "patience": 1000, "min_gain": 0.01})
    snap.update(algorithm="ga", train=train, manifest=build_manifest(snap["env_config"], train),
                versions={"numpy": np.__version__})
    run_dir = tmp_path / "outputs" / "scratch" / "evolve" / "S900_GA_x"
    Evolution(run_dir, snap["env_config"], train, 10, 2, log=lambda s: None).run()
    (run_dir / "snapshot.yaml").write_text(yaml.safe_dump(snap), encoding="utf-8")
    reg = Registry(tmp_path / "outputs" / "scratch" / "registry")
    reg.upsert_run(run_row(snap))
    first, _ = export_car(run_dir, 1, out_dir=tmp_path / "outputs" / "scratch", registry=reg)
    monkeypatch.setattr(_ui, "STORE", Store(tmp_path))
    at = AppTest.from_function(_run_page, args=("explorer",), default_timeout=180)
    at.session_state["explorer_model"] = first.name
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.number_input(key="ex_ga_gen_S900").set_value(0).run()
    at.button(key="ex_ga_save_S900").click().run()
    assert not at.exception, [e.value for e in at.exception]
    ids = {m["model_id"] for m in reg.models() if m["run_id"] == "S900"}
    assert len(ids) == 2
    assert at.subheader[0].value == (ids - {first.name}).pop()        # the new model is selected


@pytest.mark.parametrize("kind", ["env", "physics", "reward", "train", "trackset"])
def test_presets_page_shows_every_preset(kind):
    from streamlit.testing.v1 import AppTest

    from racer.sandbox import builder as B
    from racer.sandbox.explain import PRESET_KINDS
    for name in B.preset_names(kind):
        at = AppTest.from_function(_run_page, args=("presets",), default_timeout=120)
        at.session_state["pr_kind"] = kind
        at.session_state[f"pr_name_{kind}"] = name
        at.run()
        assert not at.exception, (kind, name, [e.value for e in at.exception])
        if name != "baseline":                         # the PID driver's settings, not a train preset
            assert any(f"{PRESET_KINDS[kind][1]} {name}" in c.value for c in at.code), (kind, name)


def test_runs_are_matched_to_their_presets():
    from racer.sandbox.explain import preset_header, runs_using
    runs = [{"run_id": "R1", "obs_spec": "rays7_lean", "track_set": "train7", "algorithm": "ga"},
            {"run_id": "R2", "obs_spec": "rays5", "track_set": "maps7", "algorithm": "ppo"}]
    assert [r["run_id"] for r in runs_using("env", "rays7_lean", runs)] == ["R1"]
    assert [r["run_id"] for r in runs_using("trackset", "maps7", runs)] == ["R2"]
    assert [r["run_id"] for r in runs_using("train", "ga_default", runs)] == ["R1"]
    assert preset_header("env", "rays7_lean").startswith("Environment preset")


def test_queue_list_shows_a_generation_budget(tmp_path):
    from racer.sandbox.queue import QueueControl
    qc = QueueControl(root=tmp_path, file="configs/queue/sandbox.yaml")
    qc.add([{"name": "ga_a", "args": ["--tag", "ga_a", "--train", "ga_default", "--generations", "300"]},
            {"name": "ppo_a", "args": ["--tag", "ppo_a", "--steps", "2000000"]}])
    e = {x["name"]: x for x in qc.entries()}
    assert e["ga_a"]["generations"] == 300 and e["ga_a"]["steps"] is None
    assert e["ppo_a"]["steps"] == 2_000_000 and e["ppo_a"]["generations"] is None
    from racer.sandbox.builder import budget_label
    assert budget_label(None, 300) == "300 generations" and budget_label(2_000_000, None) == "2M steps"
    assert budget_label(None, None) == "until it stops improving"



def test_demo_argv_extra_arguments_come_last():
    from racer.sandbox.launcher import demo_argv
    argv = demo_argv("x", ["A", "B"], title="T", extra=["--gradient"])
    assert argv[-1] == "--gradient" and argv[:4] == ["--track", "x", "--models", "A"]


def test_launch_sound_checkbox_adds_no_sound_only_when_unchecked():
    from racer.sandbox.pages.launch import _sound_args
    assert _sound_args(True) == [] and _sound_args(False) == ["--no-sound"]


def test_demo_argv_drive_yourself():
    from racer.sandbox.launcher import demo_argv
    assert demo_argv("x", human="downforce", title="T") == ["--track", "x", "--human", "downforce", "--title", "T"]
    assert demo_argv("x", ["A"], baseline=True, human="default")[-2:] == ["--human", "default"]
    with pytest.raises(ValueError):
        demo_argv("x")


def test_drop_marked_removes_only_the_ticked_rows():
    from racer.sandbox.builder import drop_marked
    rows = [{"angle_deg": a, "range_m": 100.0} for a in (-30.0, 0.0, 30.0)]
    assert drop_marked(rows, [False, True, False]) == [rows[0], rows[2]]
    assert drop_marked(rows, [False, False, False]) == rows
    assert drop_marked(rows, [True, True, True]) == []
    assert drop_marked([], []) == []


def test_workbench_drives_the_baseline_and_draws_every_plot():
    """The Workbench drives a lap right here and renders table, racing line, traces, g-g and braking zones."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_run_page, args=("workbench",), default_timeout=240)
    at.session_state["wb_run"] = {"who": ["baseline"], "track": "training_ground", "laps": 1, "speed": 0.0}
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert {"Registry", "Compare results", "Learning curves", "Drive & analyse", "Brain", "Racing line", "Traces"} <= {t.label for t in at.tabs}
    assert any("training_ground" in s.value for s in at.subheader)
    charts = at.get("vega_lite_chart")
    assert len(charts) >= 4                                      # registry scatter + racing line + traces + g-g (+ more)
    assert at.get("dataframe")                                   # the lap summary table


def test_workbench_brain_tab_needs_a_driven_model_first():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_run_page, args=("workbench",), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Drive a model" in c.value for c in at.caption)
