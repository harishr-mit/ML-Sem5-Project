"""Build the hand-designed maps of the map library (PROJECT_CONTEXT §6, D19).

    .venv\\Scripts\\python scripts\\make_maps.py [--check] [names ...]

Each map is a rounded polygon: a closed list of corners ``(x, y, radius_m)``. The outline
is the polygon with every corner filleted by an arc of that radius, so corner radii are
exact. The outline is sampled every ``SPACING_M`` and saved as the control points of a
track JSON (the first point, the start/finish line, is the middle of the first edge).
``direction`` sets the driving direction, whatever the corner order.

Training maps go to ``tracks/train/``, test maps to ``tracks/test/`` (never trained on).
Every map is validated with the same rules as generated tracks; ``--check`` validates
without writing. Run ``scripts/preview_tracks.py train test`` to look at them.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from racer.track.io import TRACKS_DIR, TrackSpec, save_track_spec
from racer.track.track import Track

SPACING_M = 10.0   # control point spacing along the outline

# name: (folder, display name, width_m, direction, corners [(x, y, radius_m), ...])
MAPS: dict[str, tuple[str, str, float, str, list[tuple[float, float, float]]]] = {
    # A chicane, then a long straight into a 180° hairpin (hard braking), and a fast return.
    "hairpin_valley": ("train", "Hairpin Valley", 12.0, "ccw", [
        (0, 0, 22), (200, 0, 40), (240, 22, 40), (280, 0, 40), (640, 0, 20), (640, 42, 20),
        (440, 42, 30), (340, 150, 50), (0, 120, 22)]),
    # A snake of esses on a narrow road, closed by two straights.
    "esses": ("train", "Esses", 10.0, "cw", [
        (0, 0, 25), (520, 0, 25), (520, 230, 25), (430, 200, 35), (350, 270, 35),
        (270, 200, 35), (190, 270, 35), (110, 200, 35), (0, 240, 25)]),
    # Fast oval on a wide road with a tight infield section.
    "oval_infield": ("train", "Oval Infield", 14.0, "ccw", [
        (0, 0, 70), (720, 0, 70), (720, 230, 70), (500, 230, 30), (440, 110, 22),
        (310, 110, 22), (250, 230, 30), (0, 230, 70)]),
    # Right-angle street corners around city blocks.
    "city_blocks": ("train", "City Blocks", 12.0, "cw", [
        (0, 0, 20), (420, 0, 20), (420, 160, 20), (290, 160, 20), (290, 320, 20),
        (130, 320, 20), (130, 190, 20), (0, 190, 20)]),
    # TEST: mountain road, hairpins between fast sections.
    "alpine": ("test", "Alpine", 12.0, "ccw", [
        (0, 0, 30), (460, 0, 45), (560, 120, 22), (440, 170, 20), (580, 280, 30),
        (320, 350, 70), (90, 320, 22), (170, 200, 20), (0, 160, 28)]),
    # TEST: fast coastal sweepers with one heavy braking zone into a hairpin.
    "coastline": ("test", "Coastline", 13.0, "cw", [
        (0, 0, 50), (780, 0, 20), (790, 120, 20), (580, 150, 120), (390, 250, 90),
        (160, 215, 60), (60, 120, 40)]),
}


def fillet_outline(corners: list[tuple[float, float, float]], step: float = 1.0) -> np.ndarray:
    """Dense points (~``step`` apart) of the closed polygon with filleted corners.

    Starts at the middle of the edge from corner 0 to corner 1.
    """
    P = np.array([(x, y) for x, y, _ in corners], dtype=float)
    R = [float(r) for _, _, r in corners]
    n = len(P)
    arcs = []   # per corner: (tangent point in, tangent point out, centre, radius, turn sign, angles)
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        u = (b - a) / np.linalg.norm(b - a)
        v = (c - b) / np.linalg.norm(c - b)
        turn = math.atan2(u[0] * v[1] - u[1] * v[0], float(u @ v))
        if abs(turn) < 1e-9:
            raise ValueError(f"corner {i} is straight")
        r = R[i]
        t = r * math.tan(abs(turn) / 2.0)
        if t > np.linalg.norm(b - a) or t > np.linalg.norm(c - b):
            raise ValueError(f"corner {i}: radius {r} too big for its edges")
        p_in, p_out = b - u * t, b + v * t
        left = np.array([-u[1], u[0]]) * math.copysign(1.0, turn)
        centre = p_in + left * r
        arcs.append((p_in, p_out, centre, r, turn))
    for i in range(n):   # fillets of neighbouring corners must not overlap
        e_len = np.linalg.norm(P[(i + 1) % n] - P[i])
        used = np.linalg.norm(arcs[i][1] - P[i]) + np.linalg.norm(arcs[(i + 1) % n][0] - P[(i + 1) % n])
        if used > e_len + 1e-6:
            raise ValueError(f"edge {i}->{(i + 1) % n}: fillets overlap ({used:.1f} m > {e_len:.1f} m)")

    pts: list[np.ndarray] = []

    def line(p: np.ndarray, q: np.ndarray) -> None:
        k = max(int(np.linalg.norm(q - p) / step), 1)
        pts.extend(p + (q - p) * j / k for j in range(k))

    def arc(i: int) -> None:
        p_in, _, centre, r, turn = arcs[i]
        a0 = math.atan2(p_in[1] - centre[1], p_in[0] - centre[0])
        k = max(int(abs(turn) * r / step), 1)
        pts.extend(centre + r * np.array([math.cos(a0 + turn * j / k), math.sin(a0 + turn * j / k)])
                   for j in range(k))

    mid = (arcs[0][1] + arcs[1][0]) / 2.0
    line(mid, arcs[1][0])
    for i in list(range(1, n)) + [0]:
        arc(i)
        nxt = (i + 1) % n
        q = mid if nxt == 1 else arcs[nxt][0]
        line(arcs[i][1], q)
    return np.array(pts)


def resample(points: np.ndarray, spacing: float) -> np.ndarray:
    """Closed polyline → points evenly spaced along its arc length (first point kept)."""
    closed = np.vstack([points, points[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    k = max(int(round(s[-1] / spacing)), 3)
    t = np.arange(k) * s[-1] / k
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def build_spec(name: str) -> TrackSpec:
    _, display, width, direction, corners = MAPS[name]
    pts = resample(fillet_outline(corners), SPACING_M)
    return TrackSpec(display, pts.round(3).tolist(), width, 15.0, 1.2, direction,
                     meta={"source": "scripts/make_maps.py", "key": name})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="maps to build (default: all)")
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    names = args.names or list(MAPS)
    failed = 0
    for name in names:
        folder = MAPS[name][0]
        try:
            spec = build_spec(name)
            track = Track(spec, build_field=False)
            problems = track.validate()
        except ValueError as e:
            problems = [str(e)]
            track = None
        length = f"{track.length:6.0f} m" if track is not None else "     -"
        kmax = float(np.max(np.abs(track.centreline.curvature_smooth))) if track is not None else 0.0
        status = "OK  " if not problems else "FAIL"
        print(f"{status} {folder:<5} {name:<15} {length}  tightest r {1 / max(kmax, 1e-9):5.1f} m  "
              + "; ".join(problems))
        if problems:
            failed += 1
            continue
        if not args.check:
            save_track_spec(spec, TRACKS_DIR / folder / f"{name}.json")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
