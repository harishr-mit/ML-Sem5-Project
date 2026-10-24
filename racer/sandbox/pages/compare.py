"""Compare 2–4 models side by side: what differs, shape, results, curves, driven lines (T69)."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from racer.sandbox import curves, diagrams, explain
from racer.sandbox.pages._ui import (freeze, map_png, md_table, model_label, num, pct, store, straight_png,
                                     track_length, track_names)

METRICS = [("completion_rate", "seen"), ("test_completion", "test maps"), ("heldout_completion", "held-out"),
           ("unseen_completion", "all unseen"), ("overall_completion", "overall")]


def short(model_id: str) -> str:
    """'R015_t50_narrow7_s1@4600k' → 'R015@4600k'."""
    run, _, rest = model_id.partition("_")
    return f"{run}@{rest.rpartition('@')[2]}" if "@" in rest else model_id


def _goto(page: str, **state) -> None:
    from racer.sandbox.pages import goto
    st.session_state.update(state)
    goto(page)


def render() -> None:
    s = store()
    st.title("Compare")
    st.caption("Pick 2–4 models. The first section lists only the settings that differ, each with what it affects; "
               "then their shape, results, learning curves and driven lines side by side.")
    pre = [m for m in st.session_state.pop("compare_models", []) if s.folder(m)]
    c1, c2 = st.columns([5, 1.4])
    trials = c2.toggle("Trial runs", key="cmp_trials")
    any_cp = c2.toggle("Any checkpoint", key="cmp_any", help="Off: only each run's best checkpoint.")
    runs = s.explorable_runs(trials or any(m.startswith("S") for m in pre))
    rows: dict[str, dict[str, Any]] = {}
    for r in runs:
        for cp in (s.checkpoints(str(r["run_id"])) if any_cp else [s.best(str(r["run_id"]))]):
            if cp:
                rows[cp["model_id"]] = cp
    for m in pre:
        rows.setdefault(m, s.model_row(m) or {"model_id": m})
    if pre:
        st.session_state["cmp_models"] = pre + [b for b in _default(s, runs) if b not in pre][: max(0, 2 - len(pre))]
    elif "cmp_models" not in st.session_state:
        st.session_state["cmp_models"] = _default(s, runs)
    st.session_state["cmp_models"] = [m for m in st.session_state["cmp_models"] if m in rows]
    picked = c1.multiselect("Models", list(rows), key="cmp_models", max_selections=4,
                            format_func=lambda m: model_label(rows[m]), placeholder="Choose 2–4 models")
    if len(picked) < 2:
        st.info("Choose at least two models.")
        return
    labels = [short(m) for m in picked]
    if len(set(labels)) < len(labels):
        labels = list(picked)
    snaps = [s.snapshot(m) for m in picked]
    mans = [s.manifest(m) for m in picked]

    st.button("Launch these in a race", type="primary", on_click=_goto, args=("launch",),
              kwargs={"launch_models": list(picked)}, disabled=any(s.problems(m) for m in picked),
              help="Opens the Launch page with these models selected.")

    st.button("Plot and analyse these in the Workbench", on_click=_goto, args=("workbench",),
              kwargs={"wb_models": list(picked), "wb_drivers": list(picked)}, key="cmp_wb",
              disabled=any(s.problems(m) for m in picked),
              help="Opens the Workbench: result bars for these models, and a lap each to compare racing lines.")

    st.header("What is different")
    flat = [explain.snapshot_rows(sn, mn or None) for sn, mn in zip(snaps, mans)]
    d = explain.diff(flat)
    c1, c2 = st.columns(2)
    show_all = c1.toggle("Show settings that are the same too", key="cmp_all")
    show_ids = c2.toggle("Show ids, dates, seeds and derived sizes", key="cmp_ids")
    keep = [r for r in d if (show_ids or r["kind"] == "config") and (show_all or r["differs"])]
    if not [r for r in d if r["kind"] == "config" and r["differs"]]:
        st.success("Same configuration: any difference in results comes from the seed or the training length.")
    if keep:
        body = []
        for r in keep:
            vals = [("—" if miss else explain.fmt(v)) for v, miss in zip(r["values"], r["missing"])]
            if r["differs"] and show_all:
                vals = [f"**{v}**" for v in vals]
            body.append([f"`{r['key']}`", *vals, r["note"]])
        st.markdown(md_table(["Setting", *labels, "What it affects"], body))
        if show_all:
            st.caption("Bold = differs between the models.")

    st.header("Shape")
    shape = [
        ("Algorithm", *[str(sn.get("algorithm", "ppo")).upper() for sn in snaps]),
        ("Inputs", *[f"{mn.get('obs_dim', '?')}: " + explain.fmt(mn.get("inputs")) if mn else "?" for mn in mans]),
        ("Outputs", *[f"{mn.get('action_dim', '?')} ({mn.get('action_layout', '?')})" if mn else "?" for mn in mans]),
        ("Hidden layers", *[" × ".join(str(n) for n in mn.get("net_arch", [])) + f" {mn.get('activation', '')}"
                            if mn else "?" for mn in mans]),
        ("Parameters (drive / total)", *[f"{mn['n_params']:,} / {mn['n_params_total']:,}" if mn else "?"
                                         for mn in mans]),
        ("Physics · reward · tracks", *[" · ".join(str(sn["env_config"].get("presets", {}).get(k, "?"))
                                                    for k in ("physics", "reward", "trackset")) for sn in snaps]),
        ("Trained steps", *[f"{int(sn.get('steps') or 0) / 1e6:.2f}M" for sn in snaps]),
        ("Engine can run it", *["yes" if not s.problems(m) else "no: " + "; ".join(s.problems(m)) for m in picked]),
    ]
    st.markdown(md_table(["", *labels], shape))
    ray_sets = [(lbl, diagrams.rays_of(mn.get("inputs", [])), sn["env_config"]["env"]["raycast"])
                for lbl, mn, sn in zip(labels, mans, snaps) if mn and diagrams.rays_of(mn.get("inputs", []))]
    if ray_sets:
        st.markdown("**Rays overlaid on a straight 12 m road**")
        st.image(straight_png("cmp", tuple((lbl, freeze(r)) for lbl, r, _ in ray_sets), 12.0, 0.0), width="stretch")
        with st.expander("Rays overlaid on a real map"):
            names = track_names()
            a, b = st.columns([1, 2])
            track = a.selectbox("Map", names, index=names.index("technical") if "technical" in names else 0,
                                key="cmp_map")
            pos = b.slider("Position along the lap (m)", 0.0, float(int(track_length(track))), 300.0 if
                           track_length(track) > 300 else 0.0, 5.0, key=f"cmp_pos_{track}")
            st.image(map_png("cmp", track, pos, 0.0, 0.0, tuple((lbl, freeze(r), tuple(sorted(rc.items())))
                                                                for lbl, r, rc in ray_sets)), width="stretch")

    st.header("Results")
    _results(s, picked, labels)

    st.header("Learning curves")
    run_ids, run_labels = [], []
    for m, sn in zip(picked, snaps):
        if sn["run_id"] not in run_ids:
            run_ids.append(sn["run_id"])
            run_labels.append(f"{sn['run_id']} {sn.get('tag', '')}")
    a, b = st.columns([2, 1])
    tag = a.selectbox("Curve", list(curves.MAIN_CURVES), format_func=curves.MAIN_CURVES.get, key="cmp_curve")
    w = b.slider("Smoothing", 0.0, 0.99, 0.8, 0.01, key="cmp_smooth")
    with st.spinner("Reading the training logs…"):
        df = curves.table(run_ids, tag, run_labels, weight=w)
    if df.empty:
        st.info("No training logs for these runs.")
    else:
        st.line_chart(df, x_label="training steps", y_label=curves.MAIN_CURVES[tag])
        if len(run_ids) < len(picked):
            st.caption("Checkpoints of the same run share one curve.")


def _default(s, runs) -> list[str]:
    best = [s.best(str(r["run_id"])) for r in runs[:2]]
    return [b["model_id"] for b in best if b]


def _results(s, picked: list[str], labels: list[str]) -> None:
    from racer.agents.evaluation import protocol_version
    protos = [set(s.eval_protocols(m)) for m in picked]
    common = sorted(set.intersection(*protos), key=protocol_version)
    if not common:
        st.warning("These models have no evaluation protocol in common, so their numbers can't be compared. "
                   "Evaluate them with the same protocol (`scripts/evaluate.py --protocol v2 --model …`).")
        return
    if any(p != set(common) for p in protos):
        st.caption("Only protocols every model was evaluated with are offered (numbers from different protocols are "
                   "not comparable).")
    proto = st.radio("Evaluation protocol", common, index=len(common) - 1, horizontal=True, key="cmp_proto")
    docs = [s.eval_doc(m, proto) for m in picked]
    ms = [doc["metrics"] for doc in docs]
    metrics = [(k, n) for k, n in METRICS if any(k in m for m in ms)]
    table = [(n, *[pct(m.get(k)) for m in ms]) for k, n in metrics]
    table += [("crash rate", *[pct(m.get("crash_rate")) for m in ms]),
              ("lap time vs baseline", *[num(m.get("lap_ratio")) for m in ms]),
              ("mean speed (km/h)", *[f"{float(m['mean_speed']) * 3.6:.0f}" if m.get("mean_speed") else "—"
                                      for m in ms]),
              ("off-road (% of wheels)", *[num(m.get("offroad_pct")) for m in ms])]
    st.markdown(md_table([f"Protocol {proto}", *labels], table))
    chart = pd.DataFrame({lbl: [float(m.get(k) or 0) for k, _ in metrics] for lbl, m in zip(labels, ms)},
                         index=[n for _, n in metrics])
    st.bar_chart(chart, stack=False, y_label="share of episodes completed", sort=False)
    base = s.baseline_laps(proto)
    maps = sorted({k[len("best_lap_"):] for m in ms for k in m if k.startswith("best_lap_")})
    laps = []
    for t in maps:
        vals = [m.get(f"best_lap_{t}") for m in ms]
        fastest = min((v for v in vals if v), default=None)
        cells = [("—" if not v else f"**{v:.2f}** ★" if v == fastest and sum(1 for x in vals if x) > 1 else f"{v:.2f}")
                 for v in vals]
        laps.append((t, *cells, num(base.get(t), 4) if base.get(t) else "—"))
    if laps:
        st.markdown("**Best lap per map (s)**; ★ = fastest of these models, — = never finished a lap there")
        st.markdown(md_table(["Map", *labels, "Baseline"], laps))
    thumbs = [s.thumbnails(m, proto) for m in picked]
    tracks = sorted(set().union(*[set(t) for t in thumbs]))
    if tracks:
        st.markdown("**Driven lines** (colour = speed)")
        t = st.selectbox("Map", tracks, key="cmp_thumb_map")
        cols = st.columns(len(picked))
        for col, lbl, th in zip(cols, labels, thumbs):
            if t in th:
                col.image(str(th[t]), caption=lbl, width="stretch")
            else:
                col.caption(f"{lbl}: no thumbnail")
