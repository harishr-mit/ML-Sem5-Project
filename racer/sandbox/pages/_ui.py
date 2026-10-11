"""Small helpers shared by the pages (Streamlit only)."""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import streamlit as st

from racer.sandbox.data import STORE, Store


def store() -> Store:
    return STORE


def esc(text: Any) -> str:
    """Make text safe inside a markdown table cell."""
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """A markdown table (wraps long text, unlike st.dataframe)."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def pct(v: Any) -> str:
    return "—" if v is None or v == "" else f"{float(v) * 100:.0f} %"


def num(v: Any, digits: int = 3) -> str:
    return "—" if v is None or v == "" else f"{float(v):.{digits}g}"


# ------------------------------------------------------------------ cached figures (PNG bytes)
def freeze(rays) -> tuple:
    """A ray list as a hashable tuple (cache key)."""
    return tuple(tuple(sorted(dict(r).items())) for r in rays)


@st.cache_data(show_spinner=False, max_entries=64)
def straight_png(key: str, rays_sets: tuple, width: float, offset: float) -> bytes:
    from racer.sandbox import diagrams
    return diagrams.png(diagrams.ray_diagram_straight([(lbl, [dict(r) for r in rays]) for lbl, rays in rays_sets],
                                                      width, offset))


@st.cache_data(show_spinner=False, max_entries=64)
def map_png(key: str, track: str, s: float, lateral: float, heading: float, sets: tuple) -> bytes:
    from racer.sandbox import diagrams
    return diagrams.png(diagrams.ray_diagram_map(
        track, s, lateral, [(lbl, [dict(r) for r in rays], dict(rc)) for lbl, rays, rc in sets], heading))


@st.cache_data(show_spinner=False, max_entries=32)
def network_png(labels: tuple, arch: tuple, actions: tuple, activation: str) -> bytes:
    from racer.sandbox import diagrams
    return diagrams.png(diagrams.network_diagram(list(labels), list(arch), list(actions), activation))


@st.cache_data(show_spinner=False, ttl=60)
def track_names() -> list[str]:
    """Every track in tracks/ (any folder), by name."""
    from racer.track.io import track_folders
    names: list[str] = []
    for folder in track_folders():
        names += [p.stem for p in sorted(folder.glob("*.json")) if p.stem not in names]
    return names


@st.cache_data(show_spinner=False)
def track_length(name: str) -> float:
    from racer.sandbox import diagrams
    return diagrams.track_length(name)


def model_label(row: dict[str, Any]) -> str:
    """'R015_t50_narrow7_s1@4600k · unseen 88 % · lap 0.969' for pickers."""
    parts = [str(row["model_id"])]
    if row.get("unseen_completion") not in (None, ""):
        parts.append(f"unseen {pct(row['unseen_completion'])}")
    elif row.get("heldout_completion") not in (None, ""):
        parts.append(f"held-out {pct(row['heldout_completion'])}")
    if row.get("lap_ratio") not in (None, ""):
        parts.append(f"lap {float(row['lap_ratio']):.3f}")
    if row.get("is_best") is True:
        parts.append("★ best")
    return " · ".join(parts)
