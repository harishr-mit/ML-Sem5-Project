"""Generate the held-out track set and report the generator's acceptance rate.

    .venv\\Scripts\\python scripts\\generate_tracks.py --heldout        # writes tracks/heldout/*.json
    .venv\\Scripts\\python scripts\\generate_tracks.py --stats 200      # acceptance rate on training seeds

Held-out tracks come from their own seed stream and are NEVER used in training (§6).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from racer.config import load_preset
from racer.track.generator import TRAIN_STREAM, generate
from racer.track.io import TRACKS_DIR, save_track_spec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heldout", action="store_true", help="(re)write tracks/heldout/")
    ap.add_argument("--stats", type=int, default=0, help="generate N training-stream tracks and report")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_preset("track", "default")

    if args.heldout:
        ho = cfg["heldout"]
        out_dir = TRACKS_DIR / "heldout"
        for k in range(ho["count"]):
            name = f"heldout_{k:02d}"
            track, attempts = generate(k, ho["stream"], name, cfg, build_field=False)
            path = save_track_spec(track.spec, out_dir / f"{name}.json")
            print(f"{path.name}: {track.length:.0f} m, {len(track.spec.control_points)} points, "
                  f"{track.spec.direction}, attempts {attempts}")

    if args.stats:
        t0 = time.perf_counter()
        attempts, lengths = [], []
        for seed in range(args.stats):
            track, a = generate(seed, TRAIN_STREAM, track_cfg=cfg, build_field=False)
            attempts.append(a)
            lengths.append(track.length)
        dt = time.perf_counter() - t0
        a = np.array(attempts)
        print(f"{args.stats} tracks: acceptance rate {len(a) / a.sum():.1%}, mean attempts {a.mean():.1f} "
              f"(max {a.max()}), length {np.mean(lengths):.0f} ± {np.std(lengths):.0f} m, "
              f"{dt / args.stats * 1000:.0f} ms per valid track (without field)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
