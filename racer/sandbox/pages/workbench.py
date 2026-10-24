"""Workbench: plot, compare and analyse models live (T136, D28).

Everything runs in this process: registry tables and charts, training curves, and fresh laps driven by
any model or the baseline (telemetry), so a racing line, a speed trace or a g-g diagram is a button press
away. Charts are interactive (zoom, hover); the data can be downloaded as CSV.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from racer.agents.telemetry import Telemetry, drive
from racer.sandbox import curves, plots
from racer.sandbox.pages._ui import model_label, store, track_names

NUMERIC = ["steps_k", "completion_rate", "test_completion", "heldout_completion", "unseen_completion",
           "overall_completion", "lap_ratio", "mean_speed", "score"]
CHANNELS = {"speed": "speed", "gas / brake": "gas", "steering": "steer", "lateral g": "lat_g", "longitudinal g": "long_g",
            "throttle": "throttle", "brake": "brake", "off-road": "offroad"}


def _goto(page: str, **state) -> None:
    from racer.sandbox.pages import goto
    st.session_state.update(state)
    goto(page)


@st.cache_data(show_spinner=False, ttl=30)
def _registry(trials: bool) -> pd.DataFrame:
    s = store()
    return plots.models_frame(s.models(trials), {str(r["run_id"]): r for r in s.runs(trials)})


@st.cache_data(show_spinner="Driving…", max_entries=24)
def _drive(who: str, track: str, laps: int, start_speed: float) -> Telemetry:
    return drive(who, track, laps=laps, start_speed=start_speed, record_obs=True)


def render() -> None:
    s = store()
    st.title("Workbench")
    st.caption("Plot, compare and analyse models with live data: the registry, training curves, and laps driven right "
               "here (racing line, speed, g-g, braking, what the network does). Zoom with the mouse wheel, hover for values.")
    tabs = st.tabs(["Registry", "Compare results", "Learning curves", "Drive & analyse", "Brain"])
    with tabs[0]:
        _registry_tab()
    with tabs[1]:
        _compare_tab()
    with tabs[2]:
        _curves_tab()
    with tabs[3]:
        _drive_tab()
    with tabs[4]:
        _brain_tab()


# ------------------------------------------------------------------ tab 1: the registry as a data set
def _registry_tab() -> None:
    c1, c2, c3 = st.columns([1, 1, 2])
    trials = c1.toggle("Trial runs", key="wb_trials")
    best_only = c2.toggle("Best checkpoints only", value=True, key="wb_best")
    df = _registry(trials)
    if df.empty:
        st.info("No models in the registry yet.")
        return
    if best_only:
        df = plots.best_per_run(df)
    algos = sorted(a for a in df["algorithm"].dropna().unique())
    chosen = c3.multiselect("Algorithm", algos, default=algos, key="wb_algos")
    df = df[df["algorithm"].isin(chosen)] if chosen else df
    cols = [c for c in NUMERIC if c in df and df[c].notna().any()]
    a, b, c = st.columns(3)
    x = a.selectbox("X", cols, index=cols.index("steps_k") if "steps_k" in cols else 0, key="wb_x")
    y = b.selectbox("Y", cols, index=cols.index("unseen_completion") if "unseen_completion" in cols else 0, key="wb_y")
    color = c.selectbox("Colour by", ["algorithm", "run_id", "tag"], key="wb_color")
    st.altair_chart(plots.chart_scatter(df, x, y, color, tooltip=["model_id", "algorithm", x, y]), width="stretch")
    st.caption(f"{len(df)} model(s). Protocol v2 numbers; older models show blanks until re-evaluated.")
    shown = [c for c in ["model_id", "algorithm", "steps_k", *NUMERIC[1:], "tag"] if c in df]
    st.dataframe(df[shown], hide_index=True)
    st.download_button("Download CSV", df[shown].to_csv(index=False).encode("utf-8"), "registry_selection.csv", "text/csv")


# ------------------------------------------------------------------ tab 2: results side by side
def _compare_tab() -> None:
    s = store()
    df = _registry(False)
    if df.empty:
        st.info("No models yet.")
        return
    best = plots.best_per_run(df)
    options = list(best["model_id"])
    pre = [m for m in st.session_state.pop("wb_models", []) if m in options]
    default = pre or options[-3:]
    picked = st.multiselect("Models (each run's best checkpoint)", options, default=default, key="wb_cmp_models",
                            format_func=lambda m: model_label(s.model_row(m) or {"model_id": m}))
    metrics = st.multiselect("Metrics", list(plots.METRIC_LABELS), default=["completion_rate", "unseen_completion"],
                             format_func=lambda m: plots.METRIC_LABELS[m], key="wb_cmp_metrics")
    if not picked or not metrics:
        st.caption("Pick at least one model and one metric.")
        return
    pct = [m for m in metrics if m != "lap_ratio"]
    if pct:
        st.altair_chart(plots.chart_bars(plots.comparison_long(df, picked, pct), "metric", "value", "model", percent=True),
                        width="stretch")
    if "lap_ratio" in metrics:
        st.altair_chart(plots.chart_bars(plots.comparison_long(df, picked, ["lap_ratio"]), "model", "value", "model",
                                         height=240), width="stretch")
        st.caption("Lap ratio: best lap divided by the hand-coded baseline's (below 1 = faster). Only comparable "
                   "between models driven under the same rules.")
    long = plots.comparison_long(df, picked, metrics)
    st.dataframe(long.pivot(index="model", columns="metric", values="value").reset_index())


# ------------------------------------------------------------------ tab 3: learning curves
def _curves_tab() -> None:
    s = store()
    runs = [str(r["run_id"]) for r in s.explorable_runs(False)]
    picked = st.multiselect("Runs", runs, default=runs[-3:], key="wb_curve_runs",
                            format_func=lambda r: f"{r} · {(s.run(r) or {}).get('tag', '')}")
    if not picked:
        st.caption("Pick runs to overlay.")
        return
    first = curves.run_scalars(picked[0])
    tags = [t for t in curves.MAIN_CURVES if any(t in curves.run_scalars(r) for r in picked)] or sorted(first)
    c1, c2 = st.columns([3, 1])
    tag = c1.selectbox("Curve", tags, key="wb_curve_tag", format_func=lambda t: curves.MAIN_CURVES.get(t, t))
    weight = c2.slider("Smoothing", 0.0, 0.99, 0.6, key="wb_curve_smooth")
    wide = curves.table(picked, tag, weight=weight)
    if wide.empty:
        st.info("No data for this curve in the chosen runs (GA runs log different curves than PPO).")
        return
    long = wide.reset_index(names="step").melt(id_vars="step", var_name="run", value_name="value").dropna()
    st.altair_chart(plots.chart_learning(long, curves.MAIN_CURVES.get(tag, tag)), width="stretch")


# ------------------------------------------------------------------ tab 4: drive a lap and analyse it
def _drivers() -> dict[str, str]:
    """Driver label → who (a model id or ``baseline``), for models this engine can run."""
    s = store()
    out = {"PID baseline (hand-coded)": "baseline"}
    for r in s.explorable_runs(False):
        best = s.best(str(r["run_id"]))
        if best and not s.problems(best["model_id"]):
            out[model_label(best)] = best["model_id"]
    return out


def _drive_tab() -> None:
    drivers = _drivers()
    labels = list(drivers)
    pre = st.session_state.pop("wb_drivers", None)
    with st.form("wb_drive_form"):
        c1, c2, c3, c4 = st.columns([5, 2, 1, 1])
        default = [lbl for lbl in labels if drivers[lbl] in (pre or [])] or labels[:1]
        picked = c1.multiselect("Drivers (up to 4)", labels, default=default, max_selections=4)
        track = c2.selectbox("Map", track_names(), index=track_names().index("technical") if "technical" in track_names() else 0)
        laps = int(c3.number_input("Laps", 1, 3, 2, help="2 laps: the second is a flying lap, free of the standing start."))
        speed = float(c4.number_input("Start speed (m/s)", 0.0, 40.0, 0.0))
        go = st.form_submit_button("Drive", type="primary")
    if go and picked:
        st.session_state["wb_run"] = {"who": [drivers[p] for p in picked], "track": track, "laps": laps, "speed": speed}
    run = st.session_state.get("wb_run")
    if not run:
        st.caption("Pick drivers and press Drive: each drives the chosen laps under its own rules (its car, rays and "
                   "track limits) and the plots below show the last lap.")
        return
    tels: list[Telemetry] = []
    for who in run["who"]:
        try:
            tels.append(_drive(who, run["track"], run["laps"], run["speed"]))
        except Exception as e:                                       # a model this engine can't run, a bad track …
            st.error(f"{who}: {e}")
    st.session_state["wb_tels"] = tels
    if not tels:
        return
    st.subheader(f"{run['track']} · {run['laps']} lap(s)")
    st.dataframe(plots.lap_summary(tels), hide_index=True)
    line, trace, gg, zones = st.tabs(["Racing line", "Traces", "g-g diagram", "Braking zones"])
    with line:
        st.altair_chart(plots.chart_racing_line(tels, run["track"]), width="stretch")
        st.caption("Colour = speed (km/h). Hover for the speed, distance and pedal at a point.")
    with trace:
        chosen = st.multiselect("Channels", list(CHANNELS), default=["speed", "gas / brake", "steering", "lateral g"],
                                key="wb_channels")
        if chosen:
            st.altair_chart(plots.chart_telemetry(tels, [CHANNELS[c] for c in chosen]), width="stretch")
    with gg:
        st.altair_chart(plots.chart_gg(tels), width="content")
        st.caption("Each dot is one decision step; the dashed ring is the grip limit. Close to the ring all the time = "
                   "using the car fully.")
    with zones:
        rows = []
        for t in tels:
            z = t.lap().braking_zones()
            rows += [{"driver": t.label, "from (m)": round(a), "to (m)": round(b), "length (m)": round(b - a)} for a, b in z]
        st.dataframe(pd.DataFrame(rows), hide_index=True) if rows else st.caption("No braking in this lap.")
    frame = pd.concat([t.lap().frame().assign(driver=t.label) for t in tels], ignore_index=True)
    st.download_button("Download the lap data (CSV)", frame.to_csv(index=False).encode("utf-8"), "telemetry.csv", "text/csv")


# ------------------------------------------------------------------ tab 5: what the network does
def _brain_tab() -> None:
    from racer.agents.activations import ActivationProbe
    from racer.agents.loader import LoadedModel
    from racer.render.brain_panel import neuron_image

    tels = [t for t in st.session_state.get("wb_tels", []) if t.obs is not None and t.label != "PID baseline"]
    if not tels:
        st.caption("Drive a model in *Drive & analyse* first (the baseline has no network).")
        return
    t = st.selectbox("Driver", tels, format_func=lambda x: x.label, key="wb_brain_driver").lap()
    i = st.slider("Moment in the lap", 0, len(t) - 1, min(len(t) // 3, len(t) - 1), key="wb_brain_i",
                  help="Drag along the lap; the network's inputs, neurons and outputs at that moment are shown.")
    m = LoadedModel(t.label)
    probe = ActivationProbe(m.policy.policy, m.normalise)
    snap = probe.capture(t.obs[i])
    a, b = st.columns([2, 3])
    with a:
        st.markdown(f"**{t.data['s'][i]:.0f} m · {t.data['speed'][i] * 3.6:.0f} km/h · "
                    f"steer {snap.action[0]:+.2f} · {'brake' if snap.action[1] < 0 else 'gas'} {abs(snap.action[1]):.2f}**")
        labels = m.manifest["obs_labels"]
        bars = pd.DataFrame({"input": labels, "value": np.clip(t.obs[i][:len(labels)], -1, 1)})
        import altair as alt
        st.altair_chart(alt.Chart(bars).mark_bar().encode(
            y=alt.Y("input:N", sort=None, title=None), x=alt.X("value:Q", scale=alt.Scale(domain=[-1, 1]), title="what the car senses"),
            color=alt.condition(alt.datum.value > 0, alt.value("#ffaa32"), alt.value("#4696ff"))).properties(height=22 * len(labels) + 30),
            width="stretch")
    with b:
        if not snap.layers:
            st.caption("This network's shape isn't recognised, so only inputs and outputs are shown.")
        cols = st.columns(max(len(snap.layers), 1))
        for k, (col, layer) in enumerate(zip(cols, snap.layers), 1):
            img = neuron_image(layer, max(1.0, float(np.abs(layer).max())))
            col.image(np.kron(img, np.ones((10, 10, 1), dtype=np.uint8)), caption=f"layer {k}: {len(layer)} neurons "
                      "(blue negative, orange positive)", width="stretch")
    st.caption("Scrub the slider to watch the neurons change through a corner. The same view is live in the game (key B).")
