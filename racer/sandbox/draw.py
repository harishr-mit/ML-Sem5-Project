"""Draw a Track in the sandbox (D27, T119): canvas drawing → track → ``tracks/user/``.

The canvas (``streamlit-drawable-canvas``, Fabric.js) returns its objects as JSON in pixels,
y down. :func:`canvas_points` takes the last freehand stroke or clicked polygon,
:func:`to_metres` turns it into world metres (y up), and ``racer.track.sketch`` builds and
validates the track. :func:`save_user_track` writes it to ``tracks/user/`` with a preview PNG
next to the other track previews, so Launch lists and shows it like any map.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from racer.config import PROJECT_ROOT
from racer.track.io import TRACKS_DIR, save_track_spec
from racer.track.track import Track

PREVIEWS_DIR = PROJECT_ROOT / "outputs" / "plots" / "tracks"


def canvas_points(json_data: Mapping[str, Any] | None) -> np.ndarray | None:
    """Pixel points of the last freehand stroke or polygon on the canvas, or None."""
    objs = [o for o in (json_data or {}).get("objects", [])        # Fabric.js 6 writes "Path", "Polygon"
            if str(o.get("type")).lower() in ("path", "polygon")]
    if not objs:
        return None
    o = objs[-1]
    # shortcut: uses the drawn coordinates; an object moved or scaled with the canvas's edit
    # toggle is read where it was drawn. Upgrade if people start editing strokes.
    if str(o["type"]).lower() == "path":       # commands ["M", x, y], ["Q", cx, cy, x, y], ["L", x, y]: keep end points
        return np.array([c[-2:] for c in o["path"] if len(c) >= 3], dtype=np.float64)
    return np.array([[p["x"], p["y"]] for p in o["points"]], dtype=np.float64)


def to_metres(px: np.ndarray, height_px: float, metres_per_px: float) -> np.ndarray:
    """Canvas pixels (y down) → world metres (y up)."""
    return np.column_stack([px[:, 0], height_px - px[:, 1]]) * metres_per_px


def track_name(text: str) -> str:
    """A file-safe track name: lower case, letters, digits and underscores."""
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def save_user_track(track: Track, name: str, tracks_dir: Path = TRACKS_DIR,
                    previews_dir: Path = PREVIEWS_DIR) -> Path:
    """Save to ``<tracks_dir>/user/<name>.json`` (never overwrites; names are unique across
    every folder, since a bare name is looked up in all of them) plus a preview PNG."""
    from racer.sandbox import diagrams
    name = track_name(name)
    if not name:
        raise ValueError("give the track a name")
    taken = [d.name for d in tracks_dir.iterdir() if d.is_dir() and (d / f"{name}.json").is_file()] \
        if tracks_dir.is_dir() else []
    if taken:
        raise ValueError(f"a track called {name!r} already exists (tracks/{taken[0]}/)")
    spec = track.spec
    spec.name = name
    spec.meta = {"drawn": {"source": "sandbox", "date": datetime.now().isoformat(timespec="seconds")}}
    path = save_track_spec(spec, tracks_dir / "user" / f"{name}.json")
    previews_dir.mkdir(parents=True, exist_ok=True)
    fig = diagrams.track_figure(track)
    (previews_dir / f"{name}.png").write_bytes(diagrams.png(fig))
    return path
