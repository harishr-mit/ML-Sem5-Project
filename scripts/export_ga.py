"""Save any car of a neuroevolution run as a model folder, rebuilt from its lineage (D26).

    .venv\\Scripts\\python scripts\\export_ga.py --run R020 --list
    .venv\\Scripts\\python scripts\\export_ga.py --run R020 --generation 750 [--car 3] [--evaluate]

Checkpoints are only saved every ``ga.validation.every`` generations; this saves any other
moment of the run afterwards, e.g. to hand-pick the steps of an Evolution Race. Without
``--car`` it takes the generation's best-validated elite if that generation was validated,
otherwise its best car on that generation's map. The folder is registered like a checkpoint
(``models/`` for R runs, ``outputs/scratch/`` for S runs) and races with ``demo.py --run``.
"""
from __future__ import annotations

import argparse
import sys

from racer.agents.evolve import export_car, generation_table, run_dir_of


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run id, e.g. R020 (S### for a scratch run)")
    ap.add_argument("--list", action="store_true", help="list the run's validated generations")
    ap.add_argument("--generation", type=int, help="generation to export")
    ap.add_argument("--car", type=int, default=None, help="car index in that generation (default: see above)")
    ap.add_argument("--evaluate", action="store_true", help="evaluate the exported model (protocol v2)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        run_dir = run_dir_of(a.run)
    except FileNotFoundError as e:
        ap.error(str(e))
    if a.list or a.generation is None:
        rows = generation_table(run_dir)
        print(f"{run_dir.name}: {len(rows)} generations driven")
        print(f"{'gen':>6} {'map':<16} {'best':>6} {'mean':>6} {'laps':>5} {'env steps':>12} {'validation':>10}")
        for r in rows:
            if r["validation"] is not None or r is rows[-1]:
                val = f"{r['validation']:.3f}" if r["validation"] is not None else "-"
                print(f"{r['generation']:>6} {r['map']:<16} {r['best']:>6.3f} {r['mean']:>6.3f} {r['finished']:>5} "
                      f"{r['env_steps']:>12,} {val:>10}")
        print("(validated generations, and the last one; any generation can be exported)")
        return 0
    try:
        folder, created = export_car(run_dir, a.generation, a.car)
    except (KeyError, RuntimeError, OSError) as e:
        print(f"[export] {e}")
        return 1
    print(f"[export] {'saved' if created else 'already saved as'} {folder}")
    if a.evaluate:
        from racer.agents.evaluation import DEFAULT_PROTOCOL
        from racer.registry.record import evaluate_and_record
        evaluate_and_record([folder], DEFAULT_PROTOCOL, 1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
