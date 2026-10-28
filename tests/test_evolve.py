"""Neuroevolution runs (D26): pause/resume determinism, stopping, lineage, and model folders."""
import copy
import json
import sqlite3

import numpy as np
import pytest
import yaml

from racer.agents import neuroevolution as ne
from racer.agents.compat import build_manifest, check
from racer.agents.evolve import FINISHED, Evolution, State, export_model, find_run, is_finished, rebuild_car, \
    settings_key, training_maps, unfinished_run_ids
from racer.agents.loader import LoadedModel
from racer.config import load_preset
from racer.env.racing_env import RacingEnv, make_env_config


def small_run(workers=1, **ga):
    """A tiny GA on Training Ground: 6 cars, 2 elites, validation on 1 generated track."""
    env_config = make_env_config(["training_ground"], env="rays7_lean")
    train = load_preset("train", "ga_default")
    train["ga"].update(population=6, elite_frac=0.34, workers=workers, state_every=3,
                       validation={"every": 3, "tracks": 1, "stream": 2, "patience": 1000, "min_gain": 0.01})
    train["ga"].update(ga)
    return env_config, train


def evolution(run_dir, env_config, train, checkpoint=None, **kw):
    return Evolution(run_dir, env_config, train, obs_dim=10, act_dim=2, checkpoint=checkpoint, log=lambda s: None, **kw)


def lineage_rows(run_dir):
    with sqlite3.connect(str(run_dir / "lineage.sqlite")) as db:
        return (db.execute("SELECT * FROM cars ORDER BY generation, car").fetchall(),
                db.execute("SELECT * FROM generations ORDER BY generation").fetchall(),
                db.execute("SELECT * FROM validations ORDER BY generation, car").fetchall())


def test_paused_and_resumed_with_other_workers_equals_an_uninterrupted_run(tmp_path):
    env_config, train = small_run(workers=2)
    assert evolution(tmp_path / "a", env_config, train).run(pause_after=8) == "paused"

    env1, train1 = small_run(workers=1)
    assert evolution(tmp_path / "b", env1, train1).run(pause_after=4) == "paused"
    assert State.load(tmp_path / "b" / "state.npz").generation == 4
    assert evolution(tmp_path / "b", env_config, train).run(pause_after=8) == "paused"

    a, b = State.load(tmp_path / "a" / "state.npz"), State.load(tmp_path / "b" / "state.npz")
    assert a.generation == b.generation == 8
    assert np.array_equal(a.weights, b.weights) and np.array_equal(a.seeds, b.seeds)
    assert a.env_steps == b.env_steps > 0 and a.best_validation == b.best_validation is not None
    assert lineage_rows(tmp_path / "a") == lineage_rows(tmp_path / "b")
    cars, gens, _ = lineage_rows(tmp_path / "a")
    assert len(cars) == 8 * 6 and len(gens) == 8
    assert len({g[1] for g in gens}) == 1 and {g[2] for g in gens} == {0.0}   # one map; always the start line (ga.start: line)


def test_any_car_is_rebuilt_from_the_lineage(tmp_path):
    env_config, train = small_run()
    evolution(tmp_path, env_config, train).run(pause_after=5)
    st = State.load(tmp_path / "state.npz")
    sizes = ne.layer_sizes(10, [256, 256], 2)
    from racer.agents.evolve import Lineage
    lin = Lineage(tmp_path / "lineage.sqlite")

    def origin(g, c):            # generation 5 is bred but not driven yet: its lineage is in the state
        return (int(st.parents[c]), int(st.seeds[c])) if g == 5 else lin.origin(g, c)

    for car in range(6):
        assert np.array_equal(ne.rebuild(origin, 5, car, sizes, train["ga"]["sigma"]), st.weights[car])
    lin.close()
    w4 = rebuild_car(tmp_path, 4, 0)
    assert w4.shape == (69_122,)


def test_resume_refuses_changed_settings(tmp_path):
    env_config, train = small_run()
    evolution(tmp_path, env_config, train).run(pause_after=1)
    changed = copy.deepcopy(train)
    changed["ga"]["sigma"] = 0.05
    with pytest.raises(ValueError, match="different settings"):
        evolution(tmp_path, env_config, changed).run(pause_after=2)


def test_stop_request_pauses_and_is_consumed(tmp_path):
    env_config, train = small_run()
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "stop.request").touch()
    assert evolution(tmp_path, env_config, train).run() == "paused"
    assert State.load(tmp_path / "state.npz").generation == 1
    assert not (tmp_path / "stop.request").exists()


def test_validation_plateau_finishes_with_a_final_checkpoint(tmp_path):
    env_config, train = small_run(validation={"every": 2, "tracks": 1, "stream": 2, "patience": 4,
                                              "min_gain": 1e9})     # nothing after the first validation counts
    saved = []
    ev = evolution(tmp_path, env_config, train, checkpoint=lambda w, g, car, steps, fit, val: saved.append(g))
    assert ev.run() == "finished"
    assert ev.final.generation == 5                  # first validation after gen 1, plateau 4 generations later
    assert saved == [1, 3, 5]                        # a checkpoint at every validation
    assert is_finished(tmp_path) and State.load(tmp_path / "state.npz").generation == 6   # kept: the next generation (D29)


def test_generation_budget_finishes_with_a_validated_checkpoint(tmp_path):
    env_config, train = small_run(max_generations=2)       # validation every 3: the budget ends before one
    saved = []
    ev = evolution(tmp_path, env_config, train, checkpoint=lambda w, g, car, steps, fit, val: saved.append(g))
    assert ev.run() == "finished" and ev.final.generation == 1
    assert saved == [1]                              # the last generation is validated and saved


def test_all_elites_are_validated_and_the_best_one_is_saved(tmp_path):
    env_config, train = small_run(population=8, elite_frac=0.5)       # 4 elites
    saved = []
    evolution(tmp_path, env_config, train,
              checkpoint=lambda w, g, car, steps, fit, val: saved.append((g, car, val))).run(pause_after=3)
    cars, gens, vals = lineage_rows(tmp_path)
    assert {v[0] for v in vals} == {2} and len(vals) == 4             # generation 2: its 4 elites
    fitness = {c[1]: c[4] for c in cars if c[0] == 2}
    elites = sorted(fitness, key=lambda c: (-fitness[c], c))[:4]
    assert sorted(v[1] for v in vals) == sorted(elites)
    best = max(vals, key=lambda v: (v[2], -elites.index(v[1])))
    assert saved == [(2, best[1], pytest.approx(best[2]))]
    assert gens[2][-1] == pytest.approx(best[2])                      # the generation's validation = best elite's


def test_runs_are_found_by_tag(tmp_path):
    for name, finished in (("R003_GA_main_s1", True), ("R007_GA_main_s1", False), ("R008_GA_other", False)):
        (tmp_path / name).mkdir()
        if finished:
            (tmp_path / name / FINISHED).touch()
    assert find_run("R", "main_s1", root=tmp_path).name == "R007_GA_main_s1"
    assert find_run("S", "main_s1", root=tmp_path) is None
    assert unfinished_run_ids(tmp_path) == {"R007", "R008"}


def test_a_crash_after_the_last_save_replays_to_the_same_run(tmp_path):
    """After a crash the run resumes from its last periodic save and drives some generations again;
    their lineage rows are rewritten and their checkpoints offered again, with identical results."""
    import shutil
    env_config, train = small_run(validation={"every": 2, "tracks": 1, "stream": 2, "patience": 1000,
                                              "min_gain": 0.01})
    calls_a, calls_b = [], []

    def rec(calls):
        return lambda w, g, car, s, f, v: calls.append((g, car, s, w.tobytes()))
    evolution(tmp_path / "a", env_config, train, rec(calls_a)).run(pause_after=6)
    b = tmp_path / "b"
    evolution(b, env_config, train, rec(calls_b)).run(pause_after=3)
    shutil.copy(b / "state.npz", tmp_path / "state3.npz")
    evolution(b, env_config, train, rec(calls_b)).run(pause_after=5)
    shutil.copy(tmp_path / "state3.npz", b / "state.npz")          # the crash: generations 3–4 are lost
    evolution(b, env_config, train, rec(calls_b)).run(pause_after=6)
    assert lineage_rows(tmp_path / "a") == lineage_rows(b)
    assert np.array_equal(State.load(tmp_path / "a" / "state.npz").weights, State.load(b / "state.npz").weights)
    assert set(calls_b) == set(calls_a) and len(calls_b) == len(calls_a) + 1      # gen 3's checkpoint offered twice


def test_a_new_run_can_be_resumed_from_its_first_generation(tmp_path):
    env_config, train = small_run()
    evolution(tmp_path, env_config, train)._initial_state()
    assert State.load(tmp_path / "state.npz").generation == 0


def plateau_run(patience):
    """Validation every 2 generations, and no gain ever counts after the first: stops `patience` later."""
    return small_run(validation={"every": 2, "tracks": 1, "stream": 2, "patience": patience, "min_gain": 1e9})


def same_state(a, b):
    assert (a.generation, a.env_steps, a.best_validation, a.last_gain) == (b.generation, b.env_steps,
                                                                           b.best_validation, b.last_gain)
    assert np.array_equal(a.weights, b.weights) and np.array_equal(a.parents, b.parents)
    assert np.array_equal(a.seeds, b.seeds)


def test_the_patience_is_not_part_of_the_settings_key():
    (env_config, short), (_, long), (_, other) = plateau_run(4), plateau_run(1000), plateau_run(4)
    other["ga"]["validation"]["min_gain"] = 0.05           # last_gain was computed with min_gain: it stays fixed
    assert settings_key(env_config, short) == settings_key(env_config, long)
    assert settings_key(env_config, short) != settings_key(env_config, other)


def test_an_extension_is_stored_in_the_snapshot_so_a_plain_resume_keeps_it(tmp_path):
    from racer.agents.evolve import extended_snapshot, write_snapshot
    env_config, run_train = plateau_run(500)
    snapshot = {"run_id": "R030", "train": run_train}
    _, command = plateau_run(1500)
    command["ga"]["validation"]["min_gain"] = 0.5               # the stopping rules only: min_gain stays the run's
    command["ga"]["max_generations"] = 3000
    new = extended_snapshot(snapshot, command, "2026-10-24T20:00:00")
    assert new["train"]["ga"]["validation"] == dict(run_train["ga"]["validation"], patience=1500)
    assert new["extensions"] == [{"at": "2026-10-24T20:00:00", "patience": 1500, "max_generations": 3000}]
    assert snapshot["train"]["ga"]["validation"]["patience"] == 500 and "extensions" not in snapshot   # a copy
    again = extended_snapshot(new, plateau_run(2500)[1], "2026-10-25T08:00:00")
    assert [e["patience"] for e in again["extensions"]] == [1500, 2500]
    write_snapshot(tmp_path, again)
    assert yaml.safe_load((tmp_path / "snapshot.yaml").read_text(encoding="utf-8")) == again
    assert not list(tmp_path.glob("*.partial"))


def test_a_state_saved_before_d29_still_resumes(tmp_path):
    env_config, train = small_run()
    evolution(tmp_path, env_config, train).run(pause_after=2)
    st = State.load(tmp_path / "state.npz")
    st.settings = settings_key(env_config, train, legacy=True)          # as saved before the patience was left out
    st.save(tmp_path / "state.npz")
    assert evolution(tmp_path, env_config, train).run(pause_after=3) == "paused"
    assert State.load(tmp_path / "state.npz").settings == settings_key(env_config, train)    # now the new key
    changed = copy.deepcopy(train)
    changed["ga"]["sigma"] = 0.05
    with pytest.raises(ValueError, match="different settings"):
        evolution(tmp_path, env_config, changed).run(pause_after=4)


def test_an_extended_run_only_evaluates_checkpoints_without_this_protocols_file(tmp_path):
    folders = [tmp_path / "a", tmp_path / "b"]
    for f in folders:
        f.mkdir()
    (folders[0] / "eval_v2.json").write_text("{}", encoding="utf-8")
    (folders[1] / "eval.json").write_text("{}", encoding="utf-8")          # v1's file doesn't count for v2
    assert train_script().unevaluated(folders, "v2") == [folders[1]]
    assert train_script().unevaluated(folders, "v1") == [folders[0]]


def test_a_finished_run_is_only_restarted_by_extending_it(tmp_path):
    env_config, train = plateau_run(4)
    assert evolution(tmp_path, env_config, train).run() == "finished"
    with pytest.raises(RuntimeError, match="finished"):
        evolution(tmp_path, env_config, train).run()
    assert is_finished(tmp_path)                          # refusing left it as it was


def test_extending_a_finished_run_equals_an_uninterrupted_run_with_the_larger_patience(tmp_path):
    env_config, short = plateau_run(4)
    _, long = plateau_run(10)
    assert evolution(tmp_path / "a", env_config, short).run() == "finished"          # stops at generation 5
    ext = evolution(tmp_path / "a", env_config, long, extend=True)
    assert ext.run() == "finished" and not any((tmp_path / "a").glob("*.partial"))
    full = evolution(tmp_path / "b", env_config, long)
    assert full.run() == "finished"
    assert ext.final.generation == full.final.generation > 5
    same_state(State.load(tmp_path / "a" / "state.npz"), State.load(tmp_path / "b" / "state.npz"))
    assert lineage_rows(tmp_path / "a") == lineage_rows(tmp_path / "b")
    assert is_finished(tmp_path / "a")


def test_a_crash_during_an_extension_leaves_a_resumable_run(tmp_path):
    env_config, short = plateau_run(4)
    _, long = plateau_run(1000)
    evolution(tmp_path, env_config, short).run()
    evolution(tmp_path, env_config, long, extend=True).run(pause_after=9)
    assert not is_finished(tmp_path)                       # unfinished while it runs, so a plain resume continues
    assert evolution(tmp_path, env_config, long).run(pause_after=11) == "paused"
    assert State.load(tmp_path / "state.npz").generation == 11


def test_a_run_finished_before_d29_is_restored_from_its_lineage(tmp_path):
    env_config, train = plateau_run(4)
    assert evolution(tmp_path, env_config, train).run() == "finished"
    kept = State.load(tmp_path / "state.npz")
    (tmp_path / "state.npz").unlink()                      # as for R020-R024, which finished without it
    restored = evolution(tmp_path, env_config, train, extend=True).restore_state()
    same_state(restored, kept)
    assert restored.settings == kept.settings and restored.wall_s == pytest.approx(kept.wall_s, abs=0.1)      # finished.json rounds it to 0.1 s
    _, long = plateau_run(10)
    assert evolution(tmp_path, env_config, long, extend=True).run() == "finished"      # extend restores by itself
    assert State.load(tmp_path / "state.npz").generation > 6


def test_maps_must_be_named():
    assert training_maps(make_env_config(trackset="train7")) == [
        "fast_flow", "technical", "training_ground", "city_blocks", "esses", "hairpin_valley", "oval_infield"]
    assert training_maps(make_env_config(["technical"])) == ["technical"]
    with pytest.raises(ValueError, match="generated pool"):
        training_maps(make_env_config(trackset="maps7"))


def test_workers_budget_and_evaluation_may_change_on_resume():
    env_config, train = small_run()
    other = copy.deepcopy(train)
    other["ga"].update(workers=8, max_generations=3000, evaluate={"top_validated": 3, "last": 1})
    assert settings_key(env_config, train) == settings_key(env_config, other)
    other["seed"] = 5
    assert settings_key(env_config, train) != settings_key(env_config, other)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    env_config = make_env_config(["training_ground"], env="rays7_lean")
    train = load_preset("train", "ga_default")
    env = RacingEnv(env_config)
    sizes = ne.layer_sizes(env.observation_space.shape[0], train["policy"]["net_arch"], env.action_space.shape[0])
    flat = ne.init_weights(sizes, 123)
    folder = tmp_path_factory.mktemp("models") / "S001_GA_test@1k"
    export_model(folder, flat, train, env.observation_space, env.action_space)
    snap = {"model_id": folder.name, "algorithm": "ga", "train": train, "env_config": env_config,
            "manifest": build_manifest(env_config, train)}
    (folder / "config.yaml").write_text(yaml.safe_dump(snap, sort_keys=False), encoding="utf-8")
    return folder, flat, sizes, env


def test_loaded_model_drives_like_the_numpy_network(exported):
    folder, flat, sizes, env = exported
    m = LoadedModel(folder)
    obs = np.stack([env.reset(seed=s)[0] for s in range(32)])
    obs[::4] = np.random.default_rng(0).uniform(-3, 3, obs[::4].shape)   # also outside the input range: clipping
    expected = ne.forward(ne.unflatten(flat, sizes), obs)
    np.testing.assert_allclose(m.act_batch(obs), expected, atol=1e-5)
    np.testing.assert_allclose(np.stack([m.act(o) for o in obs]), expected, atol=1e-5)


def test_folder_is_a_complete_model(exported):
    folder, *_ = exported
    from test_integrity import MODEL_FILES
    for f in MODEL_FILES:
        assert (folder / f).is_file()
    snap = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    assert check(snap) == []
    assert snap["manifest"]["n_params"] == 69_122 and snap["manifest"]["obs_dim"] == 10
    assert (folder / "model.zip").stat().st_size < 2_000_000


def test_inputs_are_not_normalised(exported):
    folder, *_ = exported
    m = LoadedModel(folder)
    x = np.linspace(-1, 1, 10, dtype=np.float32)[None]
    np.testing.assert_array_equal(m.normalise(x), x)


def test_folders_are_never_overwritten(exported):
    folder, flat, sizes, env = exported
    with pytest.raises(FileExistsError):
        export_model(folder, flat, load_preset("train", "ga_default"), env.observation_space, env.action_space)


def test_resume_lists_what_the_command_changed():
    a = {"train": {"ga": {"sigma": 0.02, "workers": 5}, "seed": 0}, "env_config": {"tracks": ["a"]}}
    b = {"train": {"ga": {"sigma": 0.05, "workers": 8}, "seed": 0}, "env_config": {"tracks": ["a", "b"]}}
    assert train_script()._diff_keys(a, b) == ["env_config.tracks", "train.ga.sigma", "train.ga.workers"]
    assert train_script()._diff_keys(a, a) == []


def train_script():
    import importlib.util

    from racer.config import PROJECT_ROOT
    spec = importlib.util.spec_from_file_location("train_script", PROJECT_ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evaluation_takes_the_best_validated_and_the_latest(tmp_path):
    vals = [0.2, 1.3, 0.4, 0.9, 1.1, 0.3, 0.5, 0.6]                  # generations 20, 40, ... 160
    folders = []
    for i, v in enumerate(vals):
        f = tmp_path / f"S001_GA_x@{i}k"
        f.mkdir()
        (f / "config.yaml").write_text(yaml.safe_dump({"generation": 20 * (i + 1), "validation": v}), encoding="utf-8")
        folders.append(f)
    picked = train_script().checkpoints_to_evaluate(folders, {"top_validated": 2, "last": 2})
    assert [p.name for p in picked] == ["S001_GA_x@1k", "S001_GA_x@4k", "S001_GA_x@6k", "S001_GA_x@7k"]
    only_best = train_script().checkpoints_to_evaluate(folders, {"top_validated": 1, "last": 0})
    assert [p.name for p in only_best] == ["S001_GA_x@1k"]


def test_any_generation_is_exported_as_a_working_model(tmp_path):
    """The export tool rebuilds a car from the lineage alone; it must drive exactly like the car
    the run checkpointed for that generation."""
    from racer.agents.evolve import export_car, generation_table, pick_car
    from racer.registry.registry import Registry
    env_config, train = small_run(max_generations=3)
    run_dir = tmp_path / "S900_GA_exp"
    saved = {}
    evolution(run_dir, env_config, train, lambda w, g, car, s, f, v: saved.update({g: (car, w.copy())})).run()
    snap = {"run_id": "S900", "algorithm": "ga", "train": train, "env_config": env_config,
            "manifest": build_manifest(env_config, train), "versions": {"numpy": np.__version__}}
    (run_dir / "snapshot.yaml").write_text(yaml.safe_dump(snap), encoding="utf-8")
    out, reg = tmp_path / "models", Registry(tmp_path / "registry")
    car, w = saved[2]
    assert pick_car(run_dir, 2) == car                                 # validated: the best-validated elite
    folder, created = export_car(run_dir, 2, out_dir=out, registry=reg)
    assert created and folder.name.startswith("S900_GA_exp@")
    cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["generation"] == 2 and cfg["car"] == car and cfg["validation"] is not None
    obs = np.random.default_rng(0).uniform(0, 1, (16, 10)).astype(np.float32)
    expected = ne.forward(ne.unflatten(w, ne.layer_sizes(10, [256, 256], 2)), obs)
    np.testing.assert_allclose(LoadedModel(folder).act_batch(obs), expected, atol=1e-5)
    assert reg.model(folder.name) is not None
    assert export_car(run_dir, 2, out_dir=out, registry=reg) == (folder, False)   # already saved
    other, created = export_car(run_dir, 0, out_dir=out, registry=reg)          # unvalidated: best on its map
    assert created and yaml.safe_load((other / "config.yaml").read_text(encoding="utf-8"))["validation"] is None
    assert [r["generation"] for r in generation_table(run_dir)] == [0, 1, 2]
    with pytest.raises(KeyError):
        export_car(run_dir, 7, out_dir=out, registry=reg)


def test_export_breaks_validation_ties_like_the_checkpoint(tmp_path):
    """Equal validation scores: the elite ranked higher on its map wins, in the run and in the export."""
    from racer.agents.evolve import Lineage, pick_car
    lin = Lineage(tmp_path / "lineage.sqlite")
    fitness = [0.5, 0.9, 0.7, 0.1]                      # ranking: cars 1, 2, 0 (the elites), then 3
    lin.write(0, [(0, k, -1, k, f, 0.0, 0.0, 1, "stuck") for k, f in enumerate(fitness)],
              {"map": "x", "start_s": 0, "lateral": 0, "heading_offset": 0, "speed": 0, "best": 0.9, "mean": 0.5,
               "median": 0.6, "finished": 0, "env_steps": 4, "validation": 0.3}, {1: 0.3, 2: 0.1, 0: 0.3})
    lin.close()
    assert pick_car(tmp_path, 0) == 1                   # not car 0, the lower index with the same score


@pytest.mark.parametrize("argv, message", [
    (["--train", "ga_default", "--tracks", "training_ground", "--env", "rays7_lean", "--steps", "1000"], "--generations"),
    (["--train", "ga_default", "--tracks", "training_ground", "--env", "rays7_lean", "--generations", "0"], "at least 1"),
    (["--tracks", "training_ground", "--generations", "10"], "budgeted with --steps"),
    (["--tracks", "training_ground", "--extend"], "--extend is for neuroevolution"),
])
def test_each_algorithm_refuses_the_other_budget(argv, message, monkeypatch, capsys):
    import sys
    mod = train_script()
    monkeypatch.setattr(sys, "argv", ["train.py", "--tag", "x", "--scratch", *argv])
    with pytest.raises(SystemExit):
        mod.main()
    assert message in capsys.readouterr().err



def test_line_start_puts_every_generation_on_the_start_line_with_the_same_maps(tmp_path):
    env_config = make_env_config(trackset="train7", env="rays7_lean")
    rand = evolution(tmp_path / "r", env_config, small_run(start="random")[1])
    line = evolution(tmp_path / "l", env_config, small_run(start="line")[1])
    standing = {"start_s": 0.0, "lateral": 0.0, "heading_offset": 0.0, "speed": 0.0}
    for g in range(20):
        assert line._start(g) == {"track": rand._start(g)["track"], **standing}
    assert len({rand._start(g)["start_s"] for g in range(20)}) > 1


def test_runs_without_a_start_setting_keep_random_starts(tmp_path):
    env_config, train = small_run()
    del train["ga"]["start"]
    assert evolution(tmp_path, env_config, train)._start(0)["start_s"] != 0.0


# ----------------------------------------------------------------- D31: fitness v2, σ schedule, smoothing
def small_v2(workers=1):
    env_config = make_env_config(["training_ground"], env="rays7_lean")
    train = load_preset("train", "ga_v2")
    train["ga"].update(population=6, elite_frac=0.34, workers=workers, state_every=3, sigma_g0=2.0)
    train["ga"]["validation"].update(every=3, tracks=1, patience=1000)
    return env_config, train


def test_v2_paused_and_resumed_equals_uninterrupted_and_cars_rebuild_exactly(tmp_path):
    env_config, train = small_v2(workers=2)
    assert evolution(tmp_path / "a", env_config, train).run(pause_after=7) == "paused"
    env1, train1 = small_v2(workers=1)
    assert evolution(tmp_path / "b", env1, train1).run(pause_after=3) == "paused"
    assert evolution(tmp_path / "b", env_config, train).run(pause_after=7) == "paused"
    a, b = State.load(tmp_path / "a" / "state.npz"), State.load(tmp_path / "b" / "state.npz")
    assert np.array_equal(a.weights, b.weights) and a.recent == b.recent and len(a.recent) >= 1
    assert lineage_rows(tmp_path / "a") == lineage_rows(tmp_path / "b")
    from racer.agents.evolve import Lineage, meta_sigma
    lin = Lineage(tmp_path / "a" / "lineage.sqlite")
    try:
        sizes, sig = ne.layer_sizes(10, [256, 256], 2), meta_sigma(lin.meta())

        def origin(g, c):
            return (int(a.parents[c]), int(a.seeds[c])) if g == 7 else lin.origin(g, c)

        assert all(np.array_equal(ne.rebuild(origin, 7, car, sizes, sig), a.weights[car]) for car in range(6))
        # the schedule is in effect: a constant sigma would rebuild a child differently
        assert not all(np.array_equal(ne.rebuild(origin, 7, car, sizes, train["ga"]["sigma"]), a.weights[car])
                       for car in range(6))
    finally:
        lin.close()


def test_a_run_restored_from_its_lineage_keeps_the_smoothing_window(tmp_path):
    env_config, train = small_v2()
    evolution(tmp_path, env_config, train).run(pause_after=7)
    live = State.load(tmp_path / "state.npz")
    (tmp_path / "state.npz").unlink()
    (tmp_path / "finished.json").write_text("{}", encoding="utf-8")
    restored = evolution(tmp_path, env_config, train, extend=True).restore_state()
    assert restored.recent == live.recent and restored.best_validation == live.best_validation
    assert restored.last_gain == live.last_gain


def test_older_snapshots_keep_their_meaning():
    from racer.agents.evolve import GAParams
    from racer.config import from_dict
    old = load_preset("train", "ga_default")["ga"]       # a snapshot made before D31 lacks the new keys
    for k in ("fitness", "ref_speed", "fail_speed_frac", "speed_k", "finish_bonus", "sigma_floor", "sigma_g0",
              "stage_window", "stage_min_finishers", "stage_share"):
        old.pop(k)
    old["validation"].pop("smooth")
    p = from_dict(GAParams, old, where="train.ga")
    assert p.fitness == 1 and p.sigma_floor is None and p.sigma_at(5000) == old["sigma"]
    assert p.fit_for(1) == p.fit_for(2) == {"version": 1, "speed_scale": old["speed_scale"]}
    assert load_preset("train", "ga_default")["ga"]["fitness"] == 1
    assert load_preset("train", "ga_v2")["ga"]["fitness"] == 2
    staged = load_preset("train", "ga_staged")["ga"]
    assert staged["fitness"] == 3 and staged["fail_speed_frac"] == 0.1 and staged["sigma_floor"] == 0.005


def test_a_state_saved_without_the_smoothing_window_still_loads(tmp_path):
    env_config, train = small_run()
    evolution(tmp_path, env_config, train).run(pause_after=4)
    with np.load(tmp_path / "state.npz") as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in ("weights", "parents", "seeds")}
    meta.pop("recent")
    np.savez(tmp_path / "state.npz", meta=json.dumps(meta), **arrays)
    assert State.load(tmp_path / "state.npz").recent == []


# ----------------------------------------------------------------- D32: staged fitness
def small_staged(workers=1):
    env_config, train = small_v2(workers)
    train["ga"].update(fitness=3, stage_window=3, stage_min_finishers=0, stage_share=1.0)   # switches after generation 2
    return env_config, train


def test_staged_run_switches_once_and_restarts_the_validation_baseline(tmp_path):
    env_config, train = small_staged()
    evolution(tmp_path, env_config, train).run(pause_after=2)
    assert State.load(tmp_path / "state.npz").stage == 1
    evolution(tmp_path, env_config, train).run(pause_after=6)
    st = State.load(tmp_path / "state.npz")
    assert st.stage == 2 and st.generation == 6
    _, gens, _ = lineage_rows(tmp_path)
    val = {g[0]: g[-1] for g in gens if g[-1] is not None}       # validated at generations 2 and 5
    assert set(val) == {2, 5} and st.last_gain >= 3 and st.recent == [val[5]]    # window restarted after the switch


def test_staged_run_paused_across_the_switch_equals_an_uninterrupted_run(tmp_path):
    env_config, train = small_staged(workers=2)
    evolution(tmp_path / "a", env_config, train).run(pause_after=7)
    env1, train1 = small_staged(workers=1)
    evolution(tmp_path / "b", env1, train1).run(pause_after=1)
    evolution(tmp_path / "b", env1, train1).run(pause_after=4)      # crosses the switch (after generation 2)
    evolution(tmp_path / "b", env_config, train).run(pause_after=7)
    a, b = State.load(tmp_path / "a" / "state.npz"), State.load(tmp_path / "b" / "state.npz")
    assert a.stage == b.stage == 2 and np.array_equal(a.weights, b.weights) and a.recent == b.recent
    assert lineage_rows(tmp_path / "a") == lineage_rows(tmp_path / "b")


def test_a_staged_run_restored_from_its_lineage_is_in_the_same_stage(tmp_path):
    env_config, train = small_staged()
    evolution(tmp_path, env_config, train).run(pause_after=7)
    live = State.load(tmp_path / "state.npz")
    (tmp_path / "state.npz").unlink()
    (tmp_path / "finished.json").write_text("{}", encoding="utf-8")
    restored = evolution(tmp_path, env_config, train, extend=True).restore_state()
    assert restored.stage == live.stage == 2 and restored.last_gain == live.last_gain
    assert restored.recent == live.recent and restored.best_validation == live.best_validation


def test_a_state_saved_before_stages_loads_as_stage_one(tmp_path):
    env_config, train = small_run()
    evolution(tmp_path, env_config, train).run(pause_after=4)
    with np.load(tmp_path / "state.npz") as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in ("weights", "parents", "seeds")}
    meta.pop("stage", None)
    np.savez(tmp_path / "state.npz", meta=json.dumps(meta), **arrays)
    assert State.load(tmp_path / "state.npz").stage == 1
