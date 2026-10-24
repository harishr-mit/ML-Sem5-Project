"""RL Racing Line demo (PROJECT_CONTEXT §10): a mode menu and the Model Arena (up to 20 ghost cars).

    .venv\\Scripts\\python scripts\\demo.py                       mode menu
    .venv\\Scripts\\python scripts\\demo.py --mode arena          straight to a mode's model picker
    .venv\\Scripts\\python scripts\\demo.py --track technical --models R001_ppo_rays5_tg@1200k R002_…
                                         [--baseline] [--laps 3] [--fair]
    .venv\\Scripts\\python scripts\\demo.py --track fast_flow --run R002 [--every 5 | --spaced 6 | --top 6]   a run's checkpoints
    .venv\\Scripts\\python scripts\\demo.py --track technical --best                   every run's best checkpoint
        [--sim-speed 1] [--screenshot out.png --frames 600]                         headless capture
        [--title "window caption"]                                                  (AI Sandbox)

The race window, its keys and the picker live in ``racer/render/arena.py``; the shell and the
mode registry in ``racer/render/app.py``.
"""
from __future__ import annotations

import argparse
import os
import sys

from racer.render.arena import arena_keys, blocked_models, list_tracks  # noqa: F401  (re-exported for tests/tools)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default=None, help="open this mode directly (see the menu), e.g. arena")
    ap.add_argument("--track", default="training_ground")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--models", nargs="+", help="model ids or folders (skips the picker)")
    sel.add_argument("--run", help="race the checkpoints of this run")
    sel.add_argument("--best", action="store_true", help="race every run's best checkpoint")
    ap.add_argument("--every", type=int, default=1, help="--run: take every N-th checkpoint")
    ap.add_argument("--spaced", type=int, default=None, help="--run: N checkpoints spread evenly over the run")
    ap.add_argument("--top", type=int, default=None, help="--run: the N best checkpoints by registry score")
    ap.add_argument("--sound", action=argparse.BooleanOptionalAction, default=None,
                    help="engine and tyre sound (default: on for races, off for headless --screenshot runs; S toggles)")
    ap.add_argument("--gradient", action="store_true", help="colour the cars from blue (first) to red (last)")
    ap.add_argument("--baseline", action="store_true", help="add the hand-coded PID baseline car")
    ap.add_argument("--laps", type=int, default=None)
    ap.add_argument("--fair", action="store_true", help="every car on the same physics (arena.yaml fair_physics)")
    ap.add_argument("--sim-speed", type=float, default=1.0)
    ap.add_argument("--screenshot", default=None, help="run headless and save a PNG after --frames frames")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--title", default=None, help="window caption (the AI Sandbox names each game window)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    from racer.render import app      # after SDL_VIDEODRIVER is set

    if args.mode and args.mode not in {m.key for m in app.MODES}:
        ap.error(f"unknown mode {args.mode!r}; choose from {', '.join(m.key for m in app.MODES)}")
    direct = bool(args.models or args.run or args.best or args.baseline)
    ctx = app.build_context(track=args.track, laps=args.laps, sim_speed=args.sim_speed, screenshot=args.screenshot,
                            frames=args.frames, title=args.title, direct=direct, sound=args.sound)
    try:
        preset = app.preset_selection(ctx, models=args.models, run=args.run, best=args.best, every=args.every,
                                      baseline=args.baseline, spaced=args.spaced, top=args.top)
    except ValueError as e:
        ap.error(str(e))
    if preset is not None:
        preset.fair = args.fair
        preset.gradient = preset.gradient or args.gradient
    return app.run(ctx, preset, args.mode)


if __name__ == "__main__":
    sys.exit(main())
