"""Track JSON files (PROJECT_CONTEXT §6).

    {"name": str, "control_points": [[x, y], ...], "width_m": float, "runoff_m": float,
     "kerb_m": float, "direction": "ccw" | "cw"}

Coordinates are metres, x right and y up. ``direction`` is the driving direction; the first
control point is the start/finish line.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from racer.config import PROJECT_ROOT

TRACKS_DIR = PROJECT_ROOT / "tracks"
REQUIRED_KEYS = ("name", "control_points", "width_m", "runoff_m", "kerb_m", "direction")


@dataclass
class TrackSpec:
    name: str
    control_points: list[list[float]]
    width_m: float
    runoff_m: float
    kerb_m: float
    direction: str = "ccw"
    meta: dict[str, Any] = field(default_factory=dict)   # optional extras, e.g. generator seed

    def __post_init__(self) -> None:
        if self.direction not in ("ccw", "cw"):
            raise ValueError(f"track {self.name!r}: direction must be 'ccw' or 'cw'")
        if len(self.control_points) < 3:
            raise ValueError(f"track {self.name!r}: needs at least 3 control points")


def spec_from_dict(data: dict[str, Any]) -> TrackSpec:
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"track JSON missing keys {missing}")
    unknown = set(data) - set(REQUIRED_KEYS) - {"meta"}
    if unknown:
        raise ValueError(f"track JSON has unknown keys {sorted(unknown)}")
    return TrackSpec(
        name=str(data["name"]),
        control_points=[[float(x), float(y)] for x, y in data["control_points"]],
        width_m=float(data["width_m"]),
        runoff_m=float(data["runoff_m"]),
        kerb_m=float(data["kerb_m"]),
        direction=str(data["direction"]),
        meta=dict(data.get("meta", {})),
    )


def load_track_spec(path: Path | str) -> TrackSpec:
    with Path(path).open(encoding="utf-8") as f:
        return spec_from_dict(json.load(f))


def save_track_spec(spec: TrackSpec, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(spec)
    if not data["meta"]:
        del data["meta"]
    data["control_points"] = [[round(x, 3), round(y, 3)] for x, y in data["control_points"]]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    return path


def resolve_track_path(name_or_path: str) -> Path:
    """Accept a path, or a bare name looked up in tracks/{showcase,heldout,user}/."""
    p = Path(name_or_path)
    if p.suffix == ".json" and p.is_file():
        return p
    for sub in ("showcase", "heldout", "user"):
        candidate = TRACKS_DIR / sub / f"{p.stem}.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"track not found: {name_or_path}")
