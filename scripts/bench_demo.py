"""Frame-rate check for the game window (PROJECT_CONTEXT §10.2: 60 FPS with 20 cars, effects and audio on).

    .venv\Scripts\python scripts\bench_demo.py [--cars 20] [--frames 600] [--track technical]

Runs the PID baseline as ``--cars`` scattered cars (no trained model needed) through the real race window
headless, and prints the mean frame rate and the 1 % low. Exit code 1 if the mean is below ``--min-fps``.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def measure(cars: int, frames: int, track: str) -> tuple[float, float]:
    """``(mean fps, 1 % low fps)`` of a headless swarm of ``cars`` baseline cars."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bench.png"
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "demo.py"), "--swarm", str(cars), "--baseline",
                            "--track", track, "--sound", "--screenshot", str(out), "--frames", str(frames)],
                           capture_output=True, text=True, timeout=900)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])
    m = re.search(r"mean (\d+) fps, 1% low (\d+) fps", r.stdout)
    if not m:
        raise RuntimeError("no frame-rate line in: " + r.stdout[-500:])
    return float(m.group(1)), float(m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cars", type=int, default=20)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--track", default="technical")
    ap.add_argument("--min-fps", type=float, default=60.0)
    args = ap.parse_args()
    mean, low = measure(args.cars, args.frames, args.track)
    print(f"{args.cars} cars on {args.track}: mean {mean:.0f} fps, 1% low {low:.0f} fps (target {args.min_fps:.0f})")
    return 0 if mean >= args.min_fps else 1


if __name__ == "__main__":
    sys.exit(main())
