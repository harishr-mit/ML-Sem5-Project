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
