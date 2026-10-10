"""Model registry commands (PROJECT_CONTEXT §9).

    .venv\\Scripts\\python scripts\\registry.py backfill [--scratch]   register every model folder found
    .venv\\Scripts\\python scripts\\registry.py export [--scratch]     regenerate registry.xlsx from the CSVs
    .venv\\Scripts\\python scripts\\registry.py list [--run R001] [--best] [--scratch]
    .venv\\Scripts\\python scripts\\registry.py describe [--scratch]  fill obs_dim / action_layout / n_params
                                                                  of runs registered before D23, and print
                                                                  models the engine can't run

Evaluation results are added by scripts\\evaluate.py (and automatically after train.py).
"""
from __future__ import annotations

import argparse
import sys

from racer.agents.loader import MODELS_DIR, SCRATCH_DIR
from racer.registry.export_xlsx import export
from racer.registry.registry import REGISTRY_DIR, SCRATCH_REGISTRY_DIR, Registry, backfill, typed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["backfill", "export", "list", "describe"])
    ap.add_argument("--scratch", action="store_true", help="use the scratch registry (outputs/scratch/registry)")
    ap.add_argument("--run", default=None)
    ap.add_argument("--best", action="store_true", help="list: only each run's best checkpoint")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    reg = Registry(SCRATCH_REGISTRY_DIR if args.scratch else REGISTRY_DIR)

    if args.command == "backfill":
        runs = backfill(reg, SCRATCH_DIR if args.scratch else MODELS_DIR)
        print(f"registered runs {runs}; {len(reg.models())} checkpoints in {reg.models_path}")
    elif args.command == "export":
        print(f"wrote {export(reg)}")
    elif args.command == "describe":
        from racer.agents.compat import check
        from racer.config import PROJECT_ROOT, load_yaml
        from racer.registry.registry import run_row

        for run in reg.runs():
            path = next((m["path"] for m in reg.models() if m["run_id"] == run["run_id"] and m.get("path")), None)
            cfg = PROJECT_ROOT / path / "config.yaml" if path else None
            if cfg is None or not cfg.is_file():
                continue
            snap = load_yaml(cfg)
            problems = check(snap)
            row = run_row(snap)
            if not run.get("obs_dim") and row.get("obs_dim") is not None:
                reg.upsert_run({k: row[k] for k in ("run_id", "obs_dim", "action_layout", "n_params")})
            status = "OK" if not problems else "CAN'T RUN: " + "; ".join(problems)
            print(f"{run['run_id']:<6} {row.get('obs_dim') or '-':>4} inputs  {row.get('action_layout') or '-':<22} "
                  f"{row.get('n_params') or '-':>8} params  {status}")
    else:
        rows = [{k: typed(v) for k, v in r.items()} for r in reg.models()]
        rows = [r for r in rows if (not args.run or r["run_id"] == args.run) and (not args.best or r.get("is_best"))]
        rows.sort(key=lambda r: (str(r["run_id"]), r.get("steps") or 0))
        print(f"{'model_id':<34} {'best':>4} {'show':>5} {'held':>5} {'TG':>7} {'Tech':>7} {'Flow':>7} {'ratio':>6}")

        def f(v, fmt):
            return format(v, fmt) if v is not None else "-"
        for r in rows:
            print(f"{r['model_id']:<34} {'*' if r.get('is_best') else '':>4} {f(r.get('completion_rate'), '.0%'):>5} "
                  f"{f(r.get('heldout_completion'), '.0%'):>5} {f(r.get('best_lap_training_ground'), '.2f'):>7} "
                  f"{f(r.get('best_lap_technical'), '.2f'):>7} {f(r.get('best_lap_fast_flow'), '.2f'):>7} "
                  f"{f(r.get('lap_ratio'), '.3f'):>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
