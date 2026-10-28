"""Model Explorer: understand one model, what it was trained with and why (T68)."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from racer.sandbox import curves, diagrams, explain
from racer.sandbox.data import experiment_group
from racer.sandbox.pages._ui import (freeze as _freeze, map_png as _map_png, md_table, network_png as _network_png,
                                     num, pct, store, straight_png as _straight_png, track_length as _track_length,
                                     track_names as _track_names)

RANGE_LIMIT = 100          # models one *Save a range* press may create
TABS = ["Summary & why", "What it senses", "Its brain", "The car", "Reward", "Where it trained",
        "How it trained", "Episode rules", "Results", "Learning curves"]


# ------------------------------------------------------------------ helpers
def run_label(r: dict[str, Any]) -> str:
    bits = [str(r["run_id"]), str(r.get("tag") or "")]
    if r.get("curriculum"):
        bits.append(str(r["curriculum"]))
    return " · ".join(b for b in bits if b) + ("  (trial)" if r.get("kind") == "trial" else "")


def _value_cell(row: dict[str, Any]) -> str:
    v = explain.fmt(row["value"])
    return f"**{v}** ●" if row["changed"] else v


def _settings_table(rows: list[dict[str, Any]], with_default: bool = True) -> None:
    if not rows:
        st.caption("Nothing recorded.")
        return
    headers = ["Setting", "Value"] + (["Default"] if with_default else []) + ["What it is", "What it affects"]
    body = []
    for r in rows:
        cells = [f"`{r['name']}`", _value_cell(r)]
        if with_default:
            cells.append("" if not r["changed"] else explain.fmt(r["default"]))
        cells += [r["short"], r["affects"] + (f" *Typical: {r['range']}.*" if r["range"] else "")]
        body.append(cells)
    st.markdown(md_table(headers, body))
    if with_default and any(r["changed"] for r in rows):
        st.caption("● = differs from the default preset (shown in the Default column).")


def _open_model(model_id: str) -> None:
    st.session_state["explorer_model"] = model_id


def _goto(page: str, **state) -> None:
    from racer.sandbox.pages import goto
    st.session_state.update(state)
    goto(page)


# ------------------------------------------------------------------ page
def render() -> None:
    s = store()
    st.title("Model Explorer")
    st.caption("Pick a run and a checkpoint to see what the model senses, how big its brain is, what it was trained "
               "with and why, and how well it drives. Every setting has a short note; the Glossary page has them all.")
    want = st.session_state.pop("explorer_model", None)
    c1, c2, c3 = st.columns([4, 5, 1.4])
    if want and want.startswith("S"):
        st.session_state["ex_trials"] = True     # keep showing trial runs on the next interaction too
    trials = c3.toggle("Trial runs", key="ex_trials", help="Include trial (scratch) runs, S-ids.")
    runs = s.explorable_runs(trials)
    if not runs:
        st.info("No trained models found in models/ yet.")
        return
    by_id = {r["run_id"]: r for r in runs}
    if want:
        row = s.model_row(want)
        if row and row["run_id"] in by_id:
            st.session_state["ex_run"] = row["run_id"]
            st.session_state[f"ex_cp_{row['run_id']}"] = want
    if st.session_state.get("ex_run") not in by_id:
        st.session_state["ex_run"] = runs[0]["run_id"]
    run_id = c1.selectbox("Run", list(by_id), format_func=lambda r: run_label(by_id[r]), key="ex_run")
    cps = s.checkpoints(run_id)
    best = s.best(run_id)
    ids = [c["model_id"] for c in cps]
    steps = {c["model_id"]: int(c.get("steps") or 0) for c in cps}
    key = f"ex_cp_{run_id}"
    if st.session_state.get(key) not in ids:
        st.session_state[key] = best["model_id"]
    if len(ids) > 1:
        model_id = c2.select_slider("Checkpoint (★ = best by evaluation)", ids, key=key,
                                    format_func=lambda m: f"{steps[m] // 1000}k" + (" ★" if m == best["model_id"] else ""))
    else:
        model_id = ids[0]
        c2.markdown(f"**Checkpoint:** {steps[model_id] // 1000}k (the only one)")

    snap = s.snapshot(model_id)
    man = s.manifest(model_id)
    problems = s.problems(model_id)
    rows = explain.snapshot_rows(snap, man or None)
    run = by_id[run_id]

    st.subheader(model_id)
    if problems:
        st.warning("This engine can't run this model yet (it can't be launched): " + "; ".join(problems))
    doc = s.eval_doc(model_id)
    m = (doc or {}).get("metrics", {})
    k = st.columns(6)
    k[0].metric("Algorithm", str(snap.get("algorithm", "ppo")).upper())
    k[1].metric("Trained for", f"{steps[model_id] / 1e6:.2f}M steps")
    k[2].metric("Inputs → outputs", f"{man.get('obs_dim', '?')} → {man.get('action_dim', '?')}")
    k[3].metric("Network size", f"{man.get('n_params', 0):,} params" if man else "?")
    if m:
        unseen = m.get("unseen_completion", m.get("heldout_completion"))
        k[4].metric("Unseen tracks completed" if "unseen_completion" in m else "Held-out completed", pct(unseen),
                    help="Share of evaluation episodes on tracks it never trained on that it finished.")
        k[5].metric("Lap time vs baseline", num(m.get("lap_ratio")), help="Below 1 = faster than the hand-coded "
                    "driver (best lap ÷ baseline best lap, averaged over the maps it lapped).")
    else:
        k[4].metric("Evaluation", "not evaluated")

    ga = snap.get("algorithm") == "ga"
    tabs = st.tabs(TABS + (["Evolution"] if ga else []))
    with tabs[0]:
        _summary(s, run, snap, model_id, m, doc)
    with tabs[1]:
        _senses(snap, man, model_id)
    with tabs[2]:
        _brain(snap, man)
    with tabs[3]:
        _car(rows, snap)
    with tabs[4]:
        _reward(rows)
    with tabs[5]:
        _tracks(s, rows, snap)
    with tabs[6]:
        _training(rows, snap, run)
    with tabs[7]:
        _episode(rows)
    with tabs[8]:
        _results(s, run_id, model_id)
    with tabs[9]:
        _curves(run_id)
    if ga:
        with tabs[10]:
            _evolution(s, run_id, snap)


# ------------------------------------------------------------------ tabs
def _summary(s, run, snap, model_id, m, doc) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Why this run was made")
        notes = s.notes().get(str(run["run_id"]), {})
        group = experiment_group(snap.get("tag"))
        st.markdown(md_table(["", ""], [
            ("Tag", f"`{snap.get('tag', '')}` (experiment group `{group}`)"),
            ("Purpose", snap.get("curriculum") or "—"),
            ("Notes when started", snap.get("notes") or "—"),
            ("Why (run notes)", notes.get("why") or "—"),
            ("What it showed", notes.get("outcome") or "—"),
        ]))
        with st.expander("Edit the run notes (saved to registry/run_notes.yaml)"):
            with st.form(f"notes_{run['run_id']}"):
                why = st.text_area("Why it was made", notes.get("why", ""), height=80)
                outcome = st.text_area("What it showed", notes.get("outcome", ""), height=100)
                if st.form_submit_button("Save notes"):
                    s.save_note(str(run["run_id"]), "why", why)
                    s.save_note(str(run["run_id"]), "outcome", outcome)
                    st.success("Saved.")
                    st.rerun()
        chain = s.parent_chain(model_id)
        if len(chain) > 1:
            st.markdown("**Fine-tuned from:** " + " ← ".join(f"`{c}`" for c in chain[1:]))
            st.button("Open the parent", on_click=_open_model, args=(chain[1],), key="ex_parent")
        else:
            st.markdown("**Trained from scratch** (no parent model).")
        same = [r for r in s.explorable_runs(trials=True)
                if experiment_group(r.get("tag")) == group and r["run_id"] != run["run_id"]]
        if same:
            st.markdown("**Same experiment, other seeds or repeats:** " + ", ".join(f"`{r['run_id']}`" for r in same))
    with right:
        st.markdown("#### Results at a glance")
        if m:
            proto = doc.get("protocol", "?")
            rows = [("Seen tracks completed", pct(m.get("completion_rate")))]
            if "test_completion" in m:
                rows += [("Test maps completed", pct(m.get("test_completion")))]
            rows += [("Held-out tracks completed", pct(m.get("heldout_completion")))]
            if "unseen_completion" in m:
                rows += [("All unseen tracks", pct(m.get("unseen_completion")))]
            rows += [("Crash rate", pct(m.get("crash_rate"))), ("Lap time vs baseline", num(m.get("lap_ratio"))),
                     ("Mean speed", f"{float(m['mean_speed']) * 3.6:.0f} km/h" if m.get("mean_speed") else "—")]
            st.markdown(md_table(["Evaluation " + proto, ""], rows))
        else:
            st.info("This checkpoint has not been evaluated.")
        st.markdown("#### Identity")
        st.markdown(md_table(["", ""], [
            ("Run", f"{run['run_id']} ({'trial' if run.get('kind') == 'trial' else 'real'} run)"),
            ("Started", snap.get("created_at", "—")), ("Saved", snap.get("saved_at", "—")),
            ("Code version", f"`{str(snap.get('git_commit') or '—')[:10]}`"),
            ("Wall clock", f"{run['wall_clock_min']:.0f} min" if run.get("wall_clock_min") else "—"),
        ]))
        if snap.get("command"):
            with st.expander("Exact training command"):
                st.code(str(snap["command"]), language="text")
        c1, c2, c3 = st.columns(3)
        c1.button("Compare with others", key="ex_cmp", on_click=_goto,
                  args=("compare",), kwargs={"compare_models": [model_id]})
        c2.button("Launch in a race", key="ex_launch", on_click=_goto,
                  args=("launch",), kwargs={"launch_models": [model_id]})
        c3.button("Drive a lap and analyse it", key="ex_wb", on_click=_goto,
                  args=("workbench",), kwargs={"wb_drivers": [model_id]},
                  help="Workbench: racing line, speed trace, g-g diagram and what the network does.")


def _senses(snap, man, model_id) -> None:
    from racer.env.features import FEATURES
    if not man:
        st.warning("The engine can't describe this model's inputs.")
        return
    inputs = man.get("inputs", [])
    st.markdown(f"The network receives **{man['obs_dim']} numbers** every decision (20 per second), from "
                f"{len(inputs)} features in this order. It never sees the map itself.")
    body = []
    for f in inputs:
        desc = FEATURES[f["feature"]].description if f["feature"] in FEATURES else "(unknown to this engine)"
        params = {k: v for k, v in (f.get("params") or {}).items() if k != "rays"}
        body.append((f"`{f['feature']}`", f.get("dims", ""), ", ".join(f.get("labels", [])),
                     ", ".join(f"{k} = {explain.fmt(v)}" for k, v in params.items()) or "—", desc))
    st.markdown(md_table(["Feature", "Numbers", "Labels", "Parameters", "What it senses"], body))
    rays = diagrams.rays_of(inputs)
    if not rays:
        st.info("This model has no ray sensors.")
        return
    st.markdown("#### Its rays on a straight road")
    st.caption("Drawn to scale. Each ray reads distance to the road edge ÷ its range (1 = nothing within range). "
               "Shallow rays see a bend early; wide rays see the edges beside the car.")
    c1, c2 = st.columns(2)
    width = c1.slider("Road width (m)", 8.0, 16.0, 12.0, 0.5, key="ex_width")
    offset = c2.slider("Car's distance left of the centre (m)", -width / 2 + 1, width / 2 - 1, 0.0, 0.5, key="ex_off")
    st.image(_straight_png(model_id, (("", _freeze(rays)),), width, offset), width="stretch")
    st.markdown("#### Its rays on a real map")
    st.caption("The model's own sensor code casts its rays on the chosen track, as during training and racing.")
    names = _track_names()
    c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
    track = c1.selectbox("Map", names, index=names.index("technical") if "technical" in names else 0, key="ex_map")
    length = _track_length(track)
    pos = c2.slider("Position along the lap (m)", 0.0, float(int(length)), min(300.0, float(int(length))), 5.0,
                    key=f"ex_pos_{track}")
    lat = c3.slider("Left of centre (m)", -5.0, 5.0, 0.0, 0.5, key="ex_lat")
    head = c4.slider("Heading offset (°)", -45, 45, 0, 5, key="ex_head")
    rc = snap["env_config"]["env"]["raycast"]
    st.image(_map_png(model_id, track, pos, lat, float(head), (("", _freeze(rays), tuple(sorted(rc.items()))),)),
             width="stretch")


def _brain(snap, man) -> None:
    from racer.agents.algorithms import ALGORITHMS
    from racer.env.actions import LAYOUTS
    if not man:
        st.warning("The engine can't describe this model's network.")
        return
    algo = man.get("algorithm", "ppo")
    st.markdown(f"**{algo.upper()}**: {ALGORITHMS[algo].description if algo in ALGORITHMS else ''}")
    st.image(_network_png(tuple(man["obs_labels"]), tuple(man["net_arch"]), tuple(man["action_labels"]),
                          man.get("activation", "tanh")), width="stretch")
    st.caption(f"The driving network (actor): {man['obs_dim']} inputs → "
               + " → ".join(str(n) for n in man["net_arch"]) + f" → {man['action_dim']} outputs, "
               f"{man.get('activation', 'tanh')} activations. {man['n_params']:,} parameters drive the car"
               + ("; they were evolved, not trained by gradients." if algo == "ga" else
                  f"; {man['n_params_total']:,} were trained in total (the critic only helps during training)."))
    try:
        layers = diagrams.layer_table(algo, int(man["obs_dim"]), int(man["action_dim"]), man["net_arch"])
        st.markdown(md_table(["Network", "Layer", "Shape", "Parameters"],
                             [(r["network"], r["layer"], r["shape"], f"{r['params']:,}") for r in layers]))
    except ValueError:
        pass
    vn = snap.get("train", {}).get("vecnormalize", {})
    if vn.get("norm_obs"):
        st.caption("Inputs are normalised by their running mean and spread before entering the network; those "
                   "statistics are saved with the model (vecnormalize.pkl).")
    st.markdown("#### What it controls")
    lay = LAYOUTS.get(man.get("action_layout", "steer_gas"))
    st.markdown(f"Layout `{man.get('action_layout')}`: {lay.description if lay else ''}")
    out_notes = {"steer": "−1 = full right, +1 = full left (the wheels turn at a limited rate).",
                 "gas": "+1 = full throttle, −1 = full brake; there is no reverse.",
                 "throttle": "−1 = released … +1 = floored.", "brake": "−1 = released … +1 = full brake."}
    st.markdown(md_table(["Output", "Meaning"], [(f"`{a}`", out_notes.get(a, "")) for a in man["action_labels"]]))


def _car(rows, snap) -> None:
    from racer.physics.versions import physics_version
    preset = snap["env_config"].get("presets", {}).get("physics", "?")
    st.markdown(f"Physics preset **`{preset}`**, equations version {physics_version(snap['env_config']['physics'])}. "
                "Grip (μ) per surface sets how fast the car can corner and brake: about μ × 9.81 m/s².")
    _settings_table(explain.section(rows, "physics", explain.reference_rows("physics", "default")))


def _reward(rows) -> None:
    rtype = rows.get("reward.type", "progress")
    ref = explain.reference_rows("reward", {"speed": "speed_naive"}.get(rtype, "progress"))
    if rtype == "progress":
        st.markdown("Every decision the car gets: **progress** × metres gained − **time** × seconds − "
                    "**steer_change** × steering change − **offroad** × wheels off × seconds; plus **lap_bonus** "
                    "per lap and − **crash** when the episode ends early.")
    elif rtype == "speed":
        st.markdown("**Naive speed reward (E3):** the car is paid for every metre it moves in any direction, minus "
                    "**crash** when the episode ends early. Driving in circles pays as well as racing.")
    _settings_table(explain.section(rows, "reward", ref, exclude=["reward.type"]))


def _tracks(s, rows, snap) -> None:
    tracks = snap["env_config"].get("tracks")
    if isinstance(tracks, list):
        fixed = [{"track": t, "weight": 1.0} for t in tracks]
        gen: dict[str, Any] = {}
        st.markdown("Trained on a fixed list of tracks (older snapshot format).")
    else:
        fixed = list((tracks or {}).get("fixed") or [])
        gen = dict((tracks or {}).get("generated") or {})
        st.markdown(f"Track set **`{(tracks or {}).get('trackset', '?')}`**. Each episode picks a track at random by "
                    "weight. Test maps and held-out tracks are never trained on.")
    total = sum(float(f.get("weight", 1)) for f in fixed) + float(gen.get("weight", 0) or 0)
    if fixed:
        cols = st.columns(4)
        for i, f in enumerate(fixed):
            name, w = f.get("track", "?"), float(f.get("weight", 1))
            with cols[i % 4]:
                img = s.track_preview(name)
                cap = f"{name} · {w / total * 100:.0f} % of episodes" if total else name
                if img:
                    st.image(str(img), caption=cap, width="stretch")
                else:
                    st.markdown(f"`{name}` ({cap})")
    if gen:
        share = float(gen.get("weight", 0)) / total * 100 if total else 0
        st.markdown(f"**Generated tracks:** {share:.0f} % of episodes come from a pool of random tracks.")
        _settings_table(explain.section(rows, "tracks.generated"), with_default=False)


def _training(rows, snap, run) -> None:
    from racer.agents.algorithms import ALGORITHMS
    algo = snap.get("algorithm", snap.get("train", {}).get("algorithm", "ppo"))
    st.markdown(f"**{str(algo).upper()}**: {ALGORITHMS[algo].description if algo in ALGORITHMS else ''}")
    other = "sac" if algo == "ppo" else "ppo"
    ref = explain.reference_rows("train", f"{algo}_default")
    rs = explain.section(rows, "train", ref)
    _settings_table([r for r in rs if not r["key"].startswith(f"train.{other}.")])


def _episode(rows) -> None:
    st.markdown("How each training episode starts and when it ends. Ending early costs the crash penalty.")
    _settings_table(explain.section(rows, "env.episode", explain.reference_rows("env", "rays5")))


def _results(s, run_id, model_id) -> None:
    protos = s.eval_protocols(model_id)
    if not protos:
        st.info("This checkpoint has not been evaluated. Run `scripts/evaluate.py --model " + model_id + "`.")
    else:
        proto = st.radio("Evaluation protocol", protos, index=len(protos) - 1, horizontal=True, key="ex_proto",
                         help="v2 uses the 9-map library; v1 the 3 showcase tracks. Compare models within one protocol.")
        doc = s.eval_doc(model_id, proto)
        m = doc["metrics"]
        base = s.baseline_laps(proto)
        eps = doc.get("episodes", [])
        groups: dict[str, list] = {}
        for e in eps:
            groups.setdefault(e.get("group", "?"), []).append(e)
        names = {"train": "Training maps (seen)", "showcase": "Showcase tracks (seen)", "test": "Test maps (unseen)",
                 "heldout": "Generated held-out (unseen)"}
        body = []
        for g, es in groups.items():
            ends: dict[str, int] = {}
            for e in es:
                if e.get("termination"):
                    ends[e["termination"]] = ends.get(e["termination"], 0) + 1
            body.append((names.get(g, g), f"{sum(e['completed'] for e in es)}/{len(es)}",
                         ", ".join(f"{k} ×{v}" for k, v in ends.items()) or "—"))
        st.markdown(md_table(["Group", "Episodes finished", "How the others ended"], body))
        laps = []
        for key, v in m.items():
            if key.startswith("best_lap_"):
                t = key[len("best_lap_"):]
                b = base.get(t)
                laps.append((t, num(v, 4) if v else "—", num(m.get(f"mean_lap_{t}"), 4) if v else "—",
                             num(b, 4) if b else "—", f"{v / b:.3f}" if v and b else "—"))
        if laps:
            st.markdown("**Lap times (s)** against the hand-coded baseline (ratio < 1 = faster):")
            st.markdown(md_table(["Map", "Best lap", "Mean lap", "Baseline", "Ratio"], laps))
        st.caption(f"Other metrics: off-road {num(m.get('offroad_pct'))} % of wheels, mean |lateral g| "
                   f"{num(m.get('mean_lat_g'))}, mean speed {num(m.get('mean_speed'))} m/s.")
        thumbs = s.thumbnails(model_id, proto)
        if thumbs:
            st.markdown("**Driven lines** (colour = speed):")
            cols = st.columns(3)
            for i, (t, p) in enumerate(thumbs.items()):
                cols[i % 3].image(str(p), caption=t, width="stretch")
    st.markdown("#### Over training")
    protos_run = sorted({p for c in s.checkpoints(run_id) for p in s.eval_protocols(c["model_id"])})
    if not protos_run:
        return
    proto = protos_run[-1] if len(protos_run) == 1 else st.radio(
        "Protocol for the chart", protos_run, index=len(protos_run) - 1, horizontal=True, key="ex_proto_run")
    pts = s.checkpoint_metrics(run_id, proto)
    if len(pts) < 2:
        st.caption(f"Only {len(pts)} checkpoint evaluated with {proto}; nothing to plot.")
        return
    df = pd.DataFrame(pts).set_index("steps")
    cols = [c for c in ("completion_rate", "unseen_completion", "heldout_completion", "test_completion") if c in df]
    nice = {"completion_rate": "seen", "unseen_completion": "unseen", "heldout_completion": "held-out",
            "test_completion": "test maps"}
    st.line_chart(df[cols].rename(columns=nice), x_label="training steps", y_label="share completed")
    if "lap_ratio" in df and df["lap_ratio"].notna().sum() > 1:
        st.line_chart(df[["lap_ratio"]], x_label="training steps", y_label="lap time vs baseline")
    best = next((p for p in pts if p["is_best"]), None)
    st.caption(f"{len(pts)} checkpoints evaluated with {proto}."
               + (f" Best: {best['steps'] // 1000}k." if best else ""))


def _curves(run_id) -> None:
    with st.spinner("Reading the training logs…"):
        sc = curves.run_scalars(run_id)
    if not sc:
        st.info("No TensorBoard logs found for this run (runs/ is local only).")
        return
    w = st.slider("Smoothing", 0.0, 0.99, 0.8, 0.01, key="ex_smooth")
    cols = st.columns(2)
    i = 0
    for tag, title in curves.MAIN_CURVES.items():
        if tag in sc:
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                st.line_chart(curves.downsample(curves.smooth(sc[tag], w)).rename(title), x_label="training steps")
            i += 1
    ends = {t[len(curves.END_PREFIX):]: sc[t] for t in sc if t.startswith(curves.END_PREFIX)}
    if ends:
        st.markdown("**How episodes ended** (share of episodes)")
        df = pd.DataFrame({k: curves.downsample(curves.smooth(v, w)) for k, v in ends.items()})
        st.line_chart(df, x_label="training steps")
    if st.toggle("Show optimiser statistics (losses, entropy, KL, …)", key="ex_optstats"):
        tags = sorted(t for t in sc if t.startswith(curves.TRAIN_PREFIX))
        cols = st.columns(3)
        for i, tag in enumerate(tags):
            with cols[i % 3]:
                st.markdown(f"`{tag}`")
                st.line_chart(curves.downsample(curves.smooth(sc[tag], w), 200), height=160)


def _export_car(root, run_id: str, generation: int, car: int | None) -> None:
    from racer.agents.evolve import export_car, run_dir_of
    from racer.registry.registry import Registry
    scratch = run_id.startswith("S")
    try:
        folder, created = export_car(run_dir_of(run_id, root), generation, car,
                                     out_dir=root / ("outputs/scratch" if scratch else "models"),
                                     registry=Registry(root / ("outputs/scratch/registry" if scratch else "registry")))
    except (KeyError, RuntimeError, OSError) as e:
        st.session_state["ex_ga_msg"] = ("error", f"Not saved: {e}")
        return
    st.session_state["ex_ga_msg"] = ("success", f"{'Saved' if created else 'Already saved as'} {folder.name}; "
                                                "it is selected above and races like any checkpoint.")
    st.session_state["explorer_model"] = folder.name


def _save_range(root, run_id: str, generations: list[int]) -> None:
    """*Save a range of generations*: each generation's Auto car (best-validated elite, else its best car)."""
    from racer.agents.evolve import export_car, run_dir_of
    from racer.registry.registry import Registry
    scratch = run_id.startswith("S")
    made = known = 0
    try:
        run_dir = run_dir_of(run_id, root)
        reg = Registry(root / ("outputs/scratch/registry" if scratch else "registry"))
        for g in generations:
            _, created = export_car(run_dir, g, None, out_dir=root / ("outputs/scratch" if scratch else "models"),
                                    registry=reg)
            made, known = made + created, known + (not created)
    except (KeyError, RuntimeError, OSError) as e:
        st.session_state["ex_ga_msg"] = ("error", f"Stopped after {made + known} of {len(generations)}: {e}")
        return
    st.session_state["ex_ga_msg"] = ("success", f"Saved {made} new model(s); {known} of the {len(generations)} "
                                                "were saved before. They are in the model lists and Launch → "
                                                "Evolution race.")


def _arg(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def _queue_extension(root, run_id: str, run_snapshot, patience: int, generations: int, workers: int) -> None:
    from racer.sandbox import builder as B
    from racer.sandbox.queue import QueueControl
    qc = QueueControl(root=root)
    scratch = run_id.startswith("S")
    busy = [e["name"] for e in qc.entries() if e["status"] in ("waiting", "running", "interrupted")
            and e["scratch"] == scratch and _arg(e["args"], "--tag") == run_snapshot.get("tag")]
    if busy:         # two processes on one run folder would corrupt its lineage and state
        st.session_state["ex_ga_msg"] = ("error", f"Not queued: {busy[0]} is already in the queue for this run.")
        return
    try:
        entry = B.extend_entry(run_snapshot, patience, generations or None, workers, qc.names())
        qc.add([entry])
    except ValueError as e:
        st.session_state["ex_ga_msg"] = ("error", f"Not queued: {e}")
        return
    st.session_state["ex_ga_msg"] = ("success", f"Queued as {entry['name']}. Press *Start queue* on Build & Queue to "
                                                "run it (a real run needs a clean git tree, D12).")


def _continue_run(s, run_id: str, run_dir, snap, finished: bool) -> None:
    import yaml
    if not finished:
        st.caption("This run hasn't finished (paused or running): resume it from the queue entry that started it, "
                   "or with the same command. *Continue this run* is for finished runs.")
        return
    st.markdown("**Continue this run** (D29). It stopped (validation plateau or generation budget), but it keeps its "
                "population, so it can go on from where it stopped, exactly as if it had been started with the "
                "larger patience.")
    try:
        run_snap = yaml.safe_load((run_dir / "snapshot.yaml").read_text(encoding="utf-8"))
    except OSError:
        run_snap = snap
    ga = run_snap["train"]["ga"]
    old = int(ga["validation"]["patience"])
    c1, c2, c3 = st.columns(3)
    patience = int(c1.number_input("Patience (generations)", 1, 100_000, 2 * old, step=100, key=f"ex_ext_pat_{run_id}",
                                   help=f"Now {old}. Generations without a 1 % validation gain before it stops."))
    gens = int(c2.number_input("Stop after (generations, 0 = none)", 0, 1_000_000, 0, step=100,
                               key=f"ex_ext_gen_{run_id}"))
    workers = int(c3.number_input("Processes", 1, 16, int(ga["workers"]), key=f"ex_ext_workers_{run_id}"))
    st.button("Add to the queue", key=f"ex_ext_add_{run_id}", on_click=_queue_extension,
              args=(s.root, run_id, run_snap, patience, gens, workers))
    st.caption("A run that finished before D29 has no saved population; it is rebuilt from the lineage when it starts "
               "(needs the same numpy version as the run).")


def _evolution(s, run_id: str, snap) -> None:
    from racer.agents.evolve import generation_table, is_finished, run_dir_of
    st.markdown("How this neuroevolution run improved, generation by generation, from its lineage database "
                "(every car of every generation is recorded there).")
    try:
        run_dir = run_dir_of(run_id, s.root)
    except FileNotFoundError:
        st.info("This run's lineage (outputs/evolve/) isn't on this PC.")
        return
    rows = generation_table(run_dir)
    if not rows:
        st.info("No generation has been driven yet.")
        return
    df = pd.DataFrame(rows).set_index("generation")
    status = "finished" if is_finished(run_dir) else "paused or still running"
    ga = snap["train"]["ga"]
    staged = ga.get("fitness", 1) == 3
    fitness = ("lap fraction / 1 + speed (fitness 1) until the cars complete laps consistently, then the lap-time score "
               "(fitness 2, stage 2)" if staged else
               f"lap-time score (fitness 2): convex in lap speed ÷ {ga['ref_speed']:g} m/s, crashes charged the rest of the lap "
               f"at {ga['fail_speed_frac']:g} × that, +{ga['finish_bonus']:g} for finishing" if ga.get("fitness", 1) == 2 else
               f"lap fraction (0–1) if the car failed, 1 + mean speed / {ga['speed_scale']:g} m/s if it finished the lap")
    st.caption(f"{len(df)} generations ({status}). Fitness: {fitness}. Validation: the best elite's mean "
               "fitness on generated tracks it never trains on (every "
               f"{snap['train']['ga']['validation']['every']} generations).")
    chart = df[["best", "mean"]].copy()
    chart["validation"] = df["validation"].interpolate(limit_area="inside")
    st.line_chart(chart)
    st.markdown("**Save any generation's car as a model** (rebuilt exactly from the lineage), e.g. to add a step "
                "to an Evolution Race. *Auto* takes the generation's best-validated elite if it was validated, "
                "otherwise its best car on that generation's map.")
    msg = st.session_state.pop("ex_ga_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    last = int(df.index.max())
    c1, c2, c3 = st.columns(3)
    gen = int(c1.number_input("Generation", 0, last, last, key=f"ex_ga_gen_{run_id}"))
    auto = c2.radio("Car", ["Auto", "By index"], key=f"ex_ga_auto_{run_id}", horizontal=True) == "Auto"
    car = None if auto else int(c3.number_input("Car index", 0, int(snap["train"]["ga"]["population"]) - 1, 0,
                                                key=f"ex_ga_car_{run_id}"))
    st.button("Save as a model", key=f"ex_ga_save_{run_id}", on_click=_export_car, args=(s.root, run_id, gen, car))
    st.markdown("**Save a range of generations**, one *Auto* car each, e.g. every 50th generation for an "
                "Evolution race (Launch → Evolution race).")
    r1, r2, r3 = st.columns(3)
    lo = int(r1.number_input("From", 0, last, 0, key=f"ex_rg_from_{run_id}"))
    hi = int(r2.number_input("To", 0, last, last, key=f"ex_rg_to_{run_id}"))
    every = int(r3.number_input("Every", 1, max(last, 1), max(last // 10, 1), key=f"ex_rg_every_{run_id}"))
    picked = list(range(lo, hi + 1, every))
    too_many = len(picked) > RANGE_LIMIT
    st.caption(f"{len(picked)} model(s)" + (f" — too many, at most {RANGE_LIMIT} at a time" if too_many else ""))
    st.button("Save the range", key=f"ex_rg_save_{run_id}", disabled=not picked or too_many,
              on_click=_save_range, args=(s.root, run_id, picked))
    st.divider()
    _continue_run(s, run_id, run_dir, snap, is_finished(run_dir))

