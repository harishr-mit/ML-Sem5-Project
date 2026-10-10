"""Render PNG previews of tracks and print their validation results.

    .venv\\Scripts\\python scripts\\preview_tracks.py [showcase|heldout|user|<path.json> ...]

Saves outputs/plots/tracks/<name>.png: the field rasterised (grass, road, kerbs, outside),
the centreline coloured by curvature, control points and the start/finish line.
Exits non-zero if any track fails validation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from racer.config import PROJECT_ROOT
from racer.physics.surfaces import GRASS_ID, KERB_ID, OUTSIDE_ID, ROAD_ID
from racer.track import field as F
from racer.track.io import TRACKS_DIR, load_track_spec, resolve_track_path
from racer.track.track import Track

OUT_DIR = PROJECT_ROOT / "outputs" / "plots" / "tracks"
COLOURS = {OUTSIDE_ID: (0.22, 0.22, 0.25), GRASS_ID: (0.36, 0.55, 0.30),
           ROAD_ID: (0.45, 0.45, 0.47), KERB_ID: (0.85, 0.25, 0.25)}


def surface_image(track: Track, step: int = 2) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """RGB image of the surfaces (every ``step``-th field node) and its extent."""
    f = track.field
    h, w = f.shape
    ys = f.y0 + np.arange(0, h, step) * f.cell
    xs = f.x0 + np.arange(0, w, step) * f.cell
    gx, gy = np.meshgrid(xs, ys)
    ids = track.surfaces_at(gx.ravel(), gy.ravel()).reshape(gx.shape)
    # Territory walls (D3): mark nodes whose nearest sample jumps vs. their neighbours.
    idx = f.idx[::step, ::step]
    n, jump = int(f.geom[6]), int(f.geom[7])
    wall = np.zeros_like(ids, dtype=bool)
    for a, b in ((idx[:, 1:], idx[:, :-1]), (idx[1:, :], idx[:-1, :])):
        k = np.abs(a - b)
        k = np.minimum(k, n - k)
        bad = (k > jump) & (a >= 0) & (b >= 0)
        if a.shape[1] != idx.shape[1]:
            wall[:, 1:] |= bad
        else:
            wall[1:, :] |= bad
    img = np.zeros(ids.shape + (3,))
    for sid, col in COLOURS.items():
        img[ids == sid] = col
    img[wall & (ids == GRASS_ID)] = (0.95, 0.95, 0.95)
    return img, (xs[0], xs[-1], ys[0], ys[-1])


def preview(track: Track, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, ext = surface_image(track)
    cl = track.centreline
    fig, ax = plt.subplots(figsize=(10, 10 * (ext[3] - ext[2]) / max(ext[1] - ext[0], 1)))
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    radius = 1.0 / np.maximum(np.abs(cl.curvature_smooth), 1e-6)
    sc = ax.scatter(cl.points[:, 0], cl.points[:, 1], c=np.minimum(radius, 300), s=1.0,
                    cmap="viridis", vmin=0, vmax=300)
    fig.colorbar(sc, ax=ax, fraction=0.03, label="corner radius (m, capped at 300)")
    ctrl = np.asarray(track.spec.control_points)
    ax.plot(ctrl[:, 0], ctrl[:, 1], "o", ms=3, c="white", alpha=0.6)
    p0, n0 = cl.points[0], cl.normal[0]
    a, b = p0 + n0 * track.half_width, p0 - n0 * track.half_width
    ax.plot([a[0], b[0]], [a[1], b[1]], c="white", lw=3)
    t0 = cl.tangent[0] * 25
    ax.annotate("", xy=p0 + t0, xytext=p0, arrowprops=dict(color="yellow", width=2))
    problems = track.validate()
    clash = track.closest_sections()
    title = (f"{track.name}: {cl.length:.0f} m, width {track.spec.width_m:.0f} m, "
             f"min radius {1 / np.max(np.abs(cl.curvature_smooth)):.0f} m"
             + (f", closest sections {clash[0]:.0f} m" if clash else ""))
    ax.set_title(title + ("\n" + "; ".join(problems) if problems else ""), fontsize=9,
                 color="red" if problems else "black")
    ax.set_aspect("equal")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(track.spec.name).stem.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return path


def collect(targets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for t in targets:
        if (TRACKS_DIR / t).is_dir():
            paths += sorted((TRACKS_DIR / t).glob("*.json"))
        else:
            paths.append(resolve_track_path(t))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="*", default=["showcase"])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures = 0
    for path in collect(args.targets):
        track = Track(load_track_spec(path))
        problems = track.validate()
        failures += bool(problems)
        out = preview(track, OUT_DIR)
        status = "OK  " if not problems else "FAIL"
        print(f"{status} {path.name:<28} {track.length:7.0f} m  kerbs {track.kerb_mask.mean():4.0%}  "
              f"field {track.field.nbytes / 1e6:4.1f} MB  -> {out.relative_to(PROJECT_ROOT)}"
              + ("".join(f"\n      - {p}" for p in problems)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
