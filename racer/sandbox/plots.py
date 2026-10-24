"""Plots for the Workbench and the PNG report (T135-T137): data frames, Altair charts, matplotlib figures.

Pure functions, no Streamlit. Altair charts (interactive: zoom, tooltips) are for the sandbox; the
``fig_*`` functions draw the same things with matplotlib for ``scripts/make_plots.py`` and slides.
Inputs are telemetry (:mod:`racer.agents.telemetry`), registry rows and TensorBoard curves.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import altair as alt
import numpy as np
import pandas as pd

from racer.agents.telemetry import Telemetry

SPEED_SCALE = alt.Scale(scheme="turbo", domain=[0, 250])        # km/h
LABEL_SCHEME = "category10"


# ------------------------------------------------------------------ track outline
@lru_cache(maxsize=32)
def outline(track: str) -> dict[str, np.ndarray]:
    """Centreline and road edges of a track (no field is built): arrays of shape (n, 2) in world metres."""
    from racer.config import load_preset
    from racer.track.io import load_track_spec, resolve_track_path
    from racer.track.track import Track

    tr = Track(load_track_spec(resolve_track_path(track)), load_preset("track", "default"), build_field=False)
    cl = tr.centreline
    hw = tr.half_width
    return {"centre": cl.points, "left": cl.points + cl.normal * hw, "right": cl.points - cl.normal * hw,
            "s": cl.s, "length": np.asarray(cl.length)}


def _bounds(*arrays: np.ndarray, pad: float = 0.06) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = np.concatenate(arrays)
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (float(x0 - dx), float(x1 + dx)), (float(y0 - dy), float(y1 + dy))


# ------------------------------------------------------------------ telemetry tables
def lap_summary(tels: Sequence[Telemetry]) -> pd.DataFrame:
    """One row per driven lap: lap time, speeds, cornering and braking, off-road time."""
    rows = []
    for t in tels:
        lap = t.lap()
        d = lap.data
        n = max(len(lap), 1)
        rows.append({
            "driver": t.label, "lap time (s)": lap.best_lap, "top speed (km/h)": float(d["speed"].max() * 3.6),
            "mean speed (km/h)": float(d["speed"].mean() * 3.6), "max lateral g": float(np.abs(d["lat_g"]).max()),
            "braking (% of lap)": float(100.0 * (d["brake"] > 0.1).sum() / n),
            "off-road (% of lap)": float(100.0 * (d["offroad"] > 0.0).sum() / n),
            "sliding (% of lap)": float(100.0 * d["sliding"].mean()), "ended": t.termination or "lap done"})
    return pd.DataFrame(rows).round(2)


def telemetry_long(tels: Sequence[Telemetry], channels: Sequence[str], use_lap: bool = True) -> pd.DataFrame:
    """Long-form frame (driver, s, channel, value) of the chosen channels, speed in km/h."""
    frames = []
    for t in tels:
        src = t.lap() if use_lap else t
        for ch in channels:
            v = src.data[ch] * (3.6 if ch == "speed" else 1.0)
            frames.append(pd.DataFrame({"driver": t.label, "s": src.data["s"], "channel": _NAMES.get(ch, ch), "value": v}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["driver", "s", "channel", "value"])


_NAMES = {"speed": "speed (km/h)", "gas": "gas / brake", "steer": "steering", "lat_g": "lateral g", "long_g": "longitudinal g",
          "throttle": "throttle", "brake": "brake", "offroad": "off-road"}


# ------------------------------------------------------------------ Altair charts
def chart_racing_line(tels: Sequence[Telemetry], track: str, width: int = 760) -> alt.Chart:
    """The driven line coloured by speed on the road outline (one panel per driver when several)."""
    o = outline(track)
    (x0, x1), (y0, y1) = _bounds(o["left"], o["right"])
    height = int(width * (y1 - y0) / (x1 - x0))
    xs = alt.X("x:Q", scale=alt.Scale(domain=[x0, x1], nice=False), axis=None)
    ys = alt.Y("y:Q", scale=alt.Scale(domain=[y0, y1], nice=False), axis=None)
    edges = pd.concat([pd.DataFrame({"x": o[k][:, 0], "y": o[k][:, 1], "edge": k, "i": np.arange(len(o[k]))})
                       for k in ("left", "right")])
    road = alt.Chart(edges).mark_line(color="#555a66", strokeWidth=1.5).encode(xs, ys, detail="edge:N", order="i:Q")
    layers = []
    for t in tels:
        lap = t.lap()
        df = pd.DataFrame({"x": lap.data["x"], "y": lap.data["y"], "km/h": lap.data["speed"] * 3.6,
                           "s (m)": lap.data["s"], "gas": lap.data["gas"], "driver": t.label})
        line = alt.Chart(df).mark_point(filled=True, size=16, opacity=0.95).encode(
            xs, ys, color=alt.Color("km/h:Q", scale=SPEED_SCALE, legend=alt.Legend(title="km/h")),
            tooltip=["driver", alt.Tooltip("km/h:Q", format=".0f"), alt.Tooltip("s (m):Q", format=".0f"),
                     alt.Tooltip("gas:Q", format="+.2f")])
        panel = (road + line).properties(width=width if len(tels) == 1 else width // 2, height=height if len(tels) == 1
                                          else height // 2, title=t.label)
        layers.append(panel)
    if len(layers) == 1:
        return layers[0].configure_view(stroke=None)
    rows = [alt.hconcat(*layers[i:i + 2]) for i in range(0, len(layers), 2)]
    return alt.vconcat(*rows).configure_view(stroke=None)


def chart_telemetry(tels: Sequence[Telemetry], channels: Sequence[str] = ("speed", "gas", "steer", "lat_g"),
                    width: int = 820, height: int = 120) -> alt.VConcatChart:
    """Channels against lap distance, one stacked panel per channel, zoom and pan shared by all panels."""
    long = telemetry_long(tels, channels)
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    panels = []
    for ch in channels:
        name = _NAMES.get(ch, ch)
        sub = long[long["channel"] == name]
        panels.append(alt.Chart(sub).mark_line(strokeWidth=1.6).encode(
            x=alt.X("s:Q", title="lap distance (m)" if ch == channels[-1] else None),
            y=alt.Y("value:Q", title=name), color=alt.Color("driver:N", scale=alt.Scale(scheme=LABEL_SCHEME)),
            tooltip=["driver", alt.Tooltip("s:Q", format=".0f"), alt.Tooltip("value:Q", format=".2f")])
            .properties(width=width, height=height).add_params(zoom))
    return alt.vconcat(*panels).resolve_scale(color="shared")


def chart_gg(tels: Sequence[Telemetry], mu: float = 1.1, size: int = 420) -> alt.LayerChart:
    """Lateral against longitudinal g with the friction circle: how much of the grip each driver uses."""
    frames = []
    for t in tels:
        lap = t.lap()
        frames.append(pd.DataFrame({"lateral g": lap.data["lat_g"], "longitudinal g": lap.data["long_g"], "driver": t.label}))
    df = pd.concat(frames, ignore_index=True)
    th = np.linspace(0, 2 * np.pi, 100)
    circle = pd.DataFrame({"lateral g": mu * np.cos(th), "longitudinal g": mu * np.sin(th), "i": np.arange(100)})
    lim = [-1.5, 1.5]
    ring = alt.Chart(circle).mark_line(color="#888", strokeDash=[4, 3]).encode(
        x=alt.X("lateral g:Q", scale=alt.Scale(domain=lim)), y=alt.Y("longitudinal g:Q", scale=alt.Scale(domain=lim)), order="i:Q")
    pts = alt.Chart(df).mark_circle(size=14, opacity=0.5).encode(
        x="lateral g:Q", y="longitudinal g:Q", color=alt.Color("driver:N", scale=alt.Scale(scheme=LABEL_SCHEME)),
        tooltip=["driver", alt.Tooltip("lateral g:Q", format=".2f"), alt.Tooltip("longitudinal g:Q", format=".2f")])
    return (ring + pts).properties(width=size, height=size)


def chart_scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, tooltip: Sequence[str] = (),
                  width: int = 760, height: int = 420) -> alt.Chart:
    """Any two registry columns against each other, a point per model."""
    enc: dict[str, Any] = {"x": alt.X(f"{x}:Q", title=x), "y": alt.Y(f"{y}:Q", title=y),
                           "tooltip": list(tooltip) or [c for c in (color, x, y) if c]}
    if color:
        enc["color"] = alt.Color(f"{color}:N", scale=alt.Scale(scheme="tableau20"))
    return alt.Chart(df.dropna(subset=[x, y])).mark_circle(size=70, opacity=0.8).encode(**enc) \
        .properties(width=width, height=height).interactive()


def chart_learning(long: pd.DataFrame, title: str = "", width: int = 780, height: int = 320) -> alt.Chart:
    """Training curves: ``long`` has columns ``step``, ``value`` and ``run``."""
    zoom = alt.selection_interval(bind="scales")
    return alt.Chart(long).mark_line(strokeWidth=1.8).encode(
        x=alt.X("step:Q", title="env steps"), y=alt.Y("value:Q", title=title or "value"),
        color=alt.Color("run:N", scale=alt.Scale(scheme=LABEL_SCHEME)),
        tooltip=["run", alt.Tooltip("step:Q", format=",.0f"), alt.Tooltip("value:Q", format=".3f")]) \
        .properties(width=width, height=height).add_params(zoom)


def chart_bars(df: pd.DataFrame, category: str, value: str, color: str | None = None, width: int = 760,
               height: int = 340, percent: bool = False) -> alt.Chart:
    """Bars of ``value`` per ``category`` (optionally grouped by ``color``): ablations and method comparisons."""
    enc: dict[str, Any] = {"x": alt.X(f"{category}:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30)),
                           "y": alt.Y(f"{value}:Q", title=value, axis=alt.Axis(format="%" if percent else "g")),
                           "tooltip": [category, alt.Tooltip(f"{value}:Q", format=".1%" if percent else ".3f")]}
    if color:
        enc["color"] = alt.Color(f"{color}:N", scale=alt.Scale(scheme=LABEL_SCHEME))
        enc["xOffset"] = alt.XOffset(f"{color}:N")
    return alt.Chart(df).mark_bar().encode(**enc).properties(width=width, height=height)


# ------------------------------------------------------------------ registry tables
METRIC_LABELS = {"completion_rate": "seen maps", "test_completion": "test maps", "heldout_completion": "held-out tracks",
                 "unseen_completion": "all unseen", "overall_completion": "overall", "lap_ratio": "lap ratio vs baseline"}


def models_frame(rows: Iterable[Mapping[str, Any]], runs: Mapping[str, Mapping[str, Any]] | None = None) -> pd.DataFrame:
    """Registry model rows as a frame with the run's algorithm, tag and steps in thousands."""
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["steps_k"] = pd.to_numeric(df.get("steps"), errors="coerce") / 1000.0
    if runs is not None:
        df["algorithm"] = df["run_id"].map(lambda r: (runs.get(r) or {}).get("algorithm"))
        df["tag"] = df["run_id"].map(lambda r: (runs.get(r) or {}).get("tag"))
    for c in ("completion_rate", "test_completion", "heldout_completion", "unseen_completion", "overall_completion",
              "lap_ratio", "mean_speed", "score"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def best_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """The registry's best checkpoint of each run (``is_best``), one row per run."""
    if df.empty or "is_best" not in df:
        return df.iloc[0:0]
    return df[df["is_best"].astype(str).str.lower().isin(["true", "1", "1.0"])].drop_duplicates("run_id")


def comparison_long(df: pd.DataFrame, models: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    """Long frame (model, metric, value) for grouped bars, metric names made readable."""
    sub = df[df["model_id"].isin(models)]
    out = sub.melt(id_vars=["model_id"], value_vars=[m for m in metrics if m in sub], var_name="metric", value_name="value")
    out["metric"] = out["metric"].map(lambda m: METRIC_LABELS.get(m, m))
    out["model"] = out["model_id"].map(short_id)
    return out.dropna(subset=["value"])


def short_id(model_id: str) -> str:
    """``R015_t50_narrow7_s1@4600k`` → ``R015@4600k``."""
    run, _, rest = model_id.partition("_")
    return f"{run}@{rest.rpartition('@')[2]}" if "@" in rest else model_id


# ------------------------------------------------------------------ matplotlib figures (PNG report)
def _mpl():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def fig_racing_line(tels: Sequence[Telemetry], track: str):
    plt = _mpl()
    o = outline(track)
    n = len(tels)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 5.2), squeeze=False)
    for ax, t in zip(axes[0], tels):
        lap = t.lap()
        for k in ("left", "right"):
            ax.plot(o[k][:, 0], o[k][:, 1], color="#555a66", lw=1)
        sc = ax.scatter(lap.data["x"], lap.data["y"], c=lap.data["speed"] * 3.6, s=7, cmap="turbo", vmin=0, vmax=250)
        braking = lap.data["brake"] > 0.1
        ax.scatter(lap.data["x"][braking], lap.data["y"][braking], s=14, facecolors="none", edgecolors="k", lw=0.4)
        ax.set_aspect("equal")
        ax.axis("off")
        lt = f" - {lap.best_lap:.2f} s" if lap.best_lap else ""
        ax.set_title(f"{t.label}{lt}")
    fig.colorbar(sc, ax=axes[0].tolist(), label="km/h", shrink=0.7)
    fig.suptitle(f"{track}: speed along the driven line (circled = braking)")
    return fig


def fig_speed_trace(tels: Sequence[Telemetry]):
    plt = _mpl()
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for t in tels:
        lap = t.lap()
        a.plot(lap.data["s"], lap.data["speed"] * 3.6, label=t.label, lw=1.4)
        b.plot(lap.data["s"], lap.data["gas"], lw=1.0)
    a.set_ylabel("speed (km/h)")
    b.set_ylabel("gas / brake")
    b.set_xlabel("lap distance (m)")
    a.legend()
    a.grid(alpha=0.3)
    b.grid(alpha=0.3)
    return fig


def fig_gg(tels: Sequence[Telemetry], mu: float = 1.1):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for t in tels:
        lap = t.lap()
        ax.scatter(lap.data["lat_g"], lap.data["long_g"], s=5, alpha=0.5, label=t.label)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(mu * np.cos(th), mu * np.sin(th), "--", color="#888")
    ax.set_xlabel("lateral g")
    ax.set_ylabel("longitudinal g")
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("g-g diagram (dashed: grip limit)")
    return fig


def fig_learning(long: pd.DataFrame, title: str = ""):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for run, g in long.groupby("run", sort=False):
        ax.plot(g["step"], g["value"], label=run, lw=1.4)
    ax.set_xlabel("env steps")
    ax.set_ylabel(title or "value")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def fig_bars(df: pd.DataFrame, category: str, value: str, color: str | None = None, percent: bool = False, title: str = ""):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cats = list(dict.fromkeys(df[category]))
    if color:
        groups = list(dict.fromkeys(df[color]))
        w = 0.8 / len(groups)
        for i, g in enumerate(groups):
            sub = df[df[color] == g].set_index(category).reindex(cats)
            ax.bar(np.arange(len(cats)) + i * w - 0.4 + w / 2, sub[value], w, label=str(g))
        ax.legend(fontsize=8)
    else:
        ax.bar(np.arange(len(cats)), df.set_index(category).reindex(cats)[value])
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylabel(value)
    if percent:
        ax.set_ylim(0, 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    return fig
