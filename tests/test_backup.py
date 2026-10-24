"""Model backups (T81): additive, never overwrite or delete."""
from __future__ import annotations

from pathlib import Path

from racer.config import load_preset
from racer.registry.backup import backup


def make_model(models: Path, name: str, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        p = models / name / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def test_copies_missing_folders_and_new_eval_files(tmp_path):
    models, dest = tmp_path / "models", tmp_path / "backup"
    make_model(models, "R001_a@100k", {"model.zip": b"m", "config.yaml": b"c", "eval_v2.json": b"{}"})
    make_model(models, "R002_b@100k", {"model.zip": b"mm"})
    (models / ".run_ids").mkdir()
    make_model(dest, "R001_a@100k", {"model.zip": b"m", "config.yaml": b"c"})   # backed up before eval_v2

    r = backup(models, dest)
    assert r.ok
    assert r.new_folders == ["R002_b@100k"]
    assert sorted(r.copied) == ["R001_a@100k/eval_v2.json", "R002_b@100k/model.zip"]
    assert r.up_to_date == 2
    assert (dest / "R002_b@100k" / "model.zip").read_bytes() == b"mm"
    assert not (dest / ".run_ids").exists()
    assert not list(dest.rglob("*.partial"))

    again = backup(models, dest)
    assert again.copied == [] and again.up_to_date == 4


def test_never_overwrites_or_deletes(tmp_path):
    models, dest = tmp_path / "models", tmp_path / "backup"
    make_model(models, "R001_a@100k", {"model.zip": b"new and longer"})
    make_model(dest, "R001_a@100k", {"model.zip": b"old"})
    make_model(dest, "R000_gone@100k", {"model.zip": b"x"})                    # deleted locally

    r = backup(models, dest)
    assert not r.ok and r.conflicts == ["R001_a@100k/model.zip"]
    assert (dest / "R001_a@100k" / "model.zip").read_bytes() == b"old"
    assert (dest / "R000_gone@100k" / "model.zip").exists()


def test_dry_run_and_single_run(tmp_path):
    models, dest = tmp_path / "models", tmp_path / "backup"
    make_model(models, "R001_a@100k", {"model.zip": b"m"})
    make_model(models, "R015_b@100k", {"model.zip": b"m"})
    r = backup(models, dest, only_run="R015", dry_run=True)
    assert r.copied == ["R015_b@100k/model.zip"]
    assert not dest.exists()


def test_backup_config_points_outside_the_project():
    from racer.config import PROJECT_ROOT
    dest = Path(load_preset("backup", "default")["dest"])
    assert dest.is_absolute() and not dest.resolve().is_relative_to(PROJECT_ROOT.resolve())


def test_only_finished_evolve_runs_are_backed_up(tmp_path):
    """A paused run's lineage keeps growing; copying it would leave a stale copy that is never updated."""
    from racer.registry.backup import finished_evolve_runs
    evolve, dest = tmp_path / "evolve", tmp_path / "backup" / "evolve"
    make_model(evolve, "R020_GA_main_s1", {"lineage.sqlite": b"done", "finished.json": b"{}", "state.npz": b"kept"})
    make_model(evolve, "R021_GA_ctrl_s1", {"lineage.sqlite": b"growing", "state.npz": b"s"})
    runs = finished_evolve_runs(evolve)
    assert [p.name for p in runs] == ["R020_GA_main_s1"]
    r = backup(evolve, dest, folders=runs)
    # a finished run keeps its state so it can be extended (D29): it is backed up with the lineage
    assert sorted(r.copied) == ["R020_GA_main_s1/finished.json", "R020_GA_main_s1/lineage.sqlite",
                                "R020_GA_main_s1/state.npz"]
    assert not (dest / "R021_GA_ctrl_s1").exists()


def test_an_extended_run_is_backed_up_under_a_versioned_name(tmp_path):
    """D29: extending changes files of a run that was backed up; the old copy stays, the new one is .v2, .v3, …"""
    from racer.registry.backup import finished_evolve_runs
    evolve, dest = tmp_path / "evolve", tmp_path / "backup" / "evolve"
    run = "R020_GA_main_s1"
    make_model(evolve, run, {"lineage.sqlite": b"900 generations", "state.npz": b"aaaa", "finished.json": b"{}"})

    def save():
        return backup(evolve, dest, folders=finished_evolve_runs(evolve), versioned=True)
    save()
    assert save().copied == []                                               # nothing changed: up to date
    (evolve / run / "lineage.sqlite").write_bytes(b"900 generations + 600 more")     # extended: the file grew
    (evolve / run / "state.npz").write_bytes(b"bbbb")                                # same size, other content
    r = save()
    assert r.ok and sorted(r.copied) == [f"{run}/lineage.sqlite.v2", f"{run}/state.npz.v2"]
    assert (dest / run / "lineage.sqlite").read_bytes() == b"900 generations"        # the first copy is untouched
    assert (dest / run / "lineage.sqlite.v2").read_bytes() == b"900 generations + 600 more"
    assert save().copied == []                                               # the latest version is current
    (evolve / run / "state.npz").write_bytes(b"cccc")                                # extended again
    assert save().copied == [f"{run}/state.npz.v3"]


def test_without_versioning_a_changed_file_is_still_a_conflict(tmp_path):
    evolve, dest = tmp_path / "evolve", tmp_path / "backup" / "models"
    make_model(evolve, "R001_a@100k", {"model.zip": b"old"})
    backup(evolve, dest)
    (evolve / "R001_a@100k" / "model.zip").write_bytes(b"older!")
    r = backup(evolve, dest)
    assert r.conflicts == ["R001_a@100k/model.zip"] and not r.ok
    assert (dest / "R001_a@100k" / "model.zip").read_bytes() == b"old"
