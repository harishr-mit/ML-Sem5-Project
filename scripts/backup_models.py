"""Back up trained models to the backup folder (T81; PROJECT_CONTEXT §9).

    .venv\\Scripts\\python scripts\\backup_models.py [--run R015] [--dry-run] [--dest <folder>]

Run it after every completed run (training + evaluation, or a whole queue). It copies model
folders and files missing from the backup (``configs/backup/default.yaml``: ``dest``), never
overwrites or deletes anything, and reports files whose size differs. Finished neuroevolution
runs (``outputs/evolve/<run>/``: lineage database) go to ``<dest>/evolve/``. It refuses to run while
a training, queue or evaluation is running, because those are still writing model folders.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from racer.agents.evolve import EVOLVE_DIR
from racer.agents.loader import MODELS_DIR
from racer.config import load_preset
from racer.registry.backup import backup, finished_evolve_runs
from racer.sandbox import procs


def main() -> int:
    cfg = load_preset("backup", "default")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=cfg["dest"], help=f"backup folder (default {cfg['dest']})")
    ap.add_argument("--run", default=None, help="only this run's checkpoints, e.g. R015")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be copied")
    ap.add_argument("--force", action="store_true", help="run even while a training/queue/evaluation is running")
    a = ap.parse_args()

    busy = procs.find_scripts(cfg["busy_scripts"])
    if busy and not a.force:
        names = ", ".join(sorted({Path(b["script"]).name for b in busy}))
        print(f"[backup] {names} is running and may still be writing model folders; "
              f"run the backup when it has finished (or --force)")
        return 2

    dest = Path(a.dest)
    r = backup(MODELS_DIR, dest, only_run=a.run, dry_run=a.dry_run)
    verb = "would copy" if a.dry_run else "copied"
    print(f"[backup] {dest}")
    print(f"[backup] {verb} {len(r.copied)} files ({r.bytes_copied / 1e6:.1f} MB), "
          f"{len(r.new_folders)} new model folders; {r.up_to_date} files already backed up")
    runs = finished_evolve_runs(EVOLVE_DIR, a.run)
    e = backup(EVOLVE_DIR, dest / "evolve", dry_run=a.dry_run, folders=runs)
    print(f"[backup] neuroevolution lineage: {verb} {len(e.copied)} files ({e.bytes_copied / 1e6:.1f} MB) "
          f"of {len(runs)} finished runs")
    for c in r.conflicts + [f"evolve/{c}" for c in e.conflicts]:
        print(f"[backup] CONFLICT (size differs, left alone): {c}")
    return 0 if r.ok and e.ok else 1


if __name__ == "__main__":
    sys.exit(main())
