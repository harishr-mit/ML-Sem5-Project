"""Evaluate model folders and record the results in the right registry (used by train.py and evaluate.py)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from racer.agents.evaluation import DEFAULT_PROTOCOL, SCRATCH_DIR, evaluate_many
from racer.registry.export_xlsx import export
from racer.registry.registry import (REGISTRY_DIR, SCRATCH_REGISTRY_DIR, Registry, metrics_row,
                                     newer_or_same_protocol, register_folder, run_row)


def is_scratch(folder: Path) -> bool:
    try:
        Path(folder).resolve().relative_to(SCRATCH_DIR.resolve())
        return True
    except ValueError:
        return False


def registry_for(scratch: bool) -> Registry:
    return Registry(SCRATCH_REGISTRY_DIR if scratch else REGISTRY_DIR)


def record(results: Sequence[dict[str, Any]], folders: Sequence[Path], reg: Registry) -> list[str]:
    """Write eval metrics for evaluated folders into ``reg``; returns the affected run ids."""
    run_ids = []
    for folder, doc in zip(folders, results):
        snap = register_folder(reg, Path(folder))
        if reg.run(snap["run_id"]) is None:
            reg.upsert_run(run_row(snap))
        if newer_or_same_protocol(reg.model(doc["model_id"]), doc):
            reg.upsert_model(metrics_row(doc))
        run_ids.append(snap["run_id"])
    return sorted(set(run_ids))


def evaluate_and_record(folders: Sequence[Path], protocol: str = DEFAULT_PROTOCOL, workers: int = 6,
                        do_export: bool = True) -> list[dict[str, Any]]:
    """Evaluate folders, update their registries, mark each run's best, regenerate the xlsx."""
    t0 = time.perf_counter()
    folders = [Path(f) for f in folders]
    results = evaluate_many(folders, protocol, workers)
    for scratch in (False, True):
        sel = [(f, r) for f, r in zip(folders, results) if is_scratch(f) == scratch]
        if not sel:
            continue
        reg = registry_for(scratch)
        for run_id in record([r for _, r in sel], [f for f, _ in sel], reg):
            print(f"  {run_id}: best checkpoint {reg.set_best(run_id)}", flush=True)
        if do_export:
            export(reg)
    print(f"  evaluated {len(folders)} model(s) in {time.perf_counter() - t0:.0f} s", flush=True)
    return results
