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
