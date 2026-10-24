"""Write the report plots (PNG) for the write-up and slides (PROJECT_CONTEXT §11).

    .venv\\Scripts\\python scripts\\make_plots.py [--out outputs\\plots\\report] [--track technical]
                                                [--models baseline R015_t50_narrow7_s1@4600k ...] [--no-drive]

The same charts as the AI Sandbox's Workbench, as files: results scatter, ablation bars, the E10
method comparison, training curves, and (driving laps now) racing lines, speed traces and the g-g diagram of
the chosen models. Figures whose data is missing on this machine are skipped and listed.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    from racer.sandbox.report import DEFAULT_OUT, build_report

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--track", default="technical")
    ap.add_argument("--models", nargs="+", default=["baseline"], help="model ids to drive (and 'baseline')")
    ap.add_argument("--runs", nargs="+", default=["R021", "R022", "R023"], help="runs for curves and the E10 method comparison (like-for-like: same inputs and rules)")
    ap.add_argument("--no-drive", action="store_true", help="skip the figures that need a driven lap")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    written, skipped = build_report(args.out, args.models, args.track, args.runs, drive=not args.no_drive, log=print)
    print(f"{len(written)} file(s) in {args.out}")
    for s in skipped:
        print(f"skipped {s}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
