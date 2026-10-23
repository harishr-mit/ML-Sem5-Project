"""Back up model folders to a second location (T81; PROJECT_CONTEXT §9).

``models/`` is git-ignored, so the backup is the only second copy. The backup is additive,
like the model folders themselves (D10): files missing from the backup are copied, files
already there are never overwritten, and nothing is ever deleted. A file whose size differs
from the backup's copy is reported as a conflict and left alone.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BackupReport:
    new_folders: list[str] = field(default_factory=list)
    copied: list[str] = field(default_factory=list)       # relative paths of copied files
    bytes_copied: int = 0
    up_to_date: int = 0                                    # files already backed up
    conflicts: list[str] = field(default_factory=list)    # same path, different size

    @property
    def ok(self) -> bool:
        return not self.conflicts


def model_folders(models_dir: Path, only_run: str | None = None) -> list[Path]:
    """Model folders under ``models_dir`` (dot folders such as ``.run_ids`` are skipped)."""
    if not models_dir.is_dir():
        return []
    folders = [p for p in sorted(models_dir.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if only_run:
        folders = [p for p in folders if p.name.split("_", 1)[0] == only_run]
    return folders


def _copy_file(src: Path, dst: Path) -> None:
    """Copy via a temporary name, so an interrupted copy never looks like a finished one."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copy2(src, tmp)
    if tmp.stat().st_size != src.stat().st_size:
        tmp.unlink(missing_ok=True)
        raise OSError(f"size mismatch after copying {src}")
    os.replace(tmp, dst)


def finished_evolve_runs(evolve_dir: Path, only_run: str | None = None) -> list[Path]:
    """Neuroevolution run folders whose run has finished (a paused run's files still change)."""
    from racer.agents.evolve import FINISHED
    return [p for p in model_folders(evolve_dir, only_run) if (p / FINISHED).is_file()]


def backup(models_dir: Path, dest: Path, only_run: str | None = None, dry_run: bool = False,
           folders: list[Path] | None = None) -> BackupReport:
    """Copy what is missing from ``dest``; never overwrite or delete.

    ``folders`` (all under ``models_dir``) replaces the default list of model folders.
    """
    report = BackupReport()
    for folder in folders if folders is not None else model_folders(models_dir, only_run):
        target = dest / folder.name
        if not target.exists():
            report.new_folders.append(folder.name)
        for src in sorted(p for p in folder.rglob("*") if p.is_file()):
            rel = src.relative_to(models_dir)
            dst = dest / rel
            if dst.exists():
                if dst.stat().st_size == src.stat().st_size:
                    report.up_to_date += 1
                else:
                    report.conflicts.append(rel.as_posix())
                continue
            if not dry_run:
                _copy_file(src, dst)
            report.copied.append(rel.as_posix())
            report.bytes_copied += src.stat().st_size
    return report
