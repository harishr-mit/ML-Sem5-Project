"""Evaluate models with the versioned protocol and record the results (PROJECT_CONTEXT §8, §9).

    .venv\\Scripts\\python scripts\\evaluate.py --model R001_ppo_rays5_tg@1200k [more ids or folders]
    .venv\\Scripts\\python scripts\\evaluate.py --run R001            every checkpoint of a run (+ marks is_best)
    .venv\\Scripts\\python scripts\\evaluate.py --baseline            the hand-coded baseline (BASE_pid)
                                            [--protocol v1] [--workers 6] [--no-export]

Results are added to each model folder once (eval.json, thumb_<track>.png; never
overwritten), copied to outputs/eval/<model_id>/, and written to the registry; then
registry.xlsx is regenerated. Models in outputs/scratch/ use the scratch registry.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from racer.agents.evaluation import evaluate_baseline, load_protocol
from racer.agents.loader import SCRATCH_DIR, resolve_model
from racer.config import load_yaml
from racer.registry.export_xlsx import export
from racer.registry.record import evaluate_and_record
from racer.registry.registry import Registry, metrics_row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", nargs="+", help="model ids or folders")
    g.add_argument("--run", help="every checkpoint of this run id (models/ or, with --scratch, outputs/scratch/)")
    g.add_argument("--baseline", action="store_true")
    ap.add_argument("--scratch", action="store_true", help="--run looks in outputs/scratch/")
    ap.add_argument("--protocol", default="v1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-export", action="store_true", help="don't regenerate registry.xlsx")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.baseline:
        protocol = load_protocol(args.protocol)
        doc = evaluate_baseline(protocol)
        reg = Registry()
        reg.upsert_run({"run_id": "BASE", "tag": "baseline", "algorithm": "pid", "obs_spec": "map (privileged)",
                        "physics_preset": "default", "notes": "hand-coded centreline + curvature speed plan (E8)"})
        reg.upsert_model({"model_id": protocol["baseline_id"], "run_id": "BASE", "steps": 0, "tags": "baseline"})
        reg.upsert_model(metrics_row(doc))
        reg.set_best("BASE")
        if not args.no_export:
            export(reg)
        print(f"baseline: {doc['metrics']}")
        return 0

    if args.run:
        base = SCRATCH_DIR if args.scratch else Path(__file__).resolve().parents[1] / "models"
        folders = sorted((p.parent for p in base.glob(f"{args.run}_*/config.yaml")),
                         key=lambda p: load_yaml(p / "config.yaml").get("steps", 0))
        if not folders:
            ap.error(f"no checkpoints of {args.run} in {base}")
    else:
        folders = [resolve_model(m) for m in args.model]
    print(f"evaluating {len(folders)} model(s) with protocol {args.protocol} on {args.workers} worker(s)", flush=True)
    evaluate_and_record(folders, args.protocol, args.workers, not args.no_export)
    return 0


if __name__ == "__main__":
    sys.exit(main())
