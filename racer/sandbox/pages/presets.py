"""Presets: every preset a model can be built from, what it contains, and which runs used it."""
from __future__ import annotations

import streamlit as st

from racer.config import load_preset
from racer.sandbox import builder as B
from racer.sandbox import explain
from racer.sandbox.pages._ui import freeze, md_table, store, straight_png


def render() -> None:
    st.title("Presets")
    st.caption("A model is built from five presets in `configs/`: its inputs, car, reward, algorithm and track set. "
               "Five more kinds are quick-picks on *Build & Queue* (ray sets, car readings, network sizes, learning "
               "styles, budgets). Pick one to see what it contains (every value with its note), a picture where it "
               "helps, and the runs trained with it.")
    kinds = list(explain.PRESET_KINDS)
    kind = st.radio("Kind", kinds, format_func=lambda k: explain.PRESET_KINDS[k][0], horizontal=True, key="pr_kind")
    title, flag, what = explain.PRESET_KINDS[kind]
    st.caption(what)
    names = B.preset_names(kind)
    name = st.selectbox(f"{title} preset ({len(names)})", names, key=f"pr_name_{kind}")
    preset = load_preset(kind, name)
    header = explain.preset_header(kind, name)
    if header:
        st.info(header)
    if not flag:
        st.caption("A quick-pick on *Build & Queue*: one click fills the form with these values.")
    elif not (kind == "train" and "policy" not in preset):      # the baseline driver isn't trained
        st.code(f".venv\\Scripts\\python scripts\\train.py --tag my_run {flag} {name} …", language="text")
    visual = {"env": _inputs, "train": _training, "trackset": _tracks, "rays": _rays}.get(kind)
    if visual:
        visual(name, preset)
    rows = explain.section(explain.preset_rows(kind, preset), {"trackset": "tracks"}.get(kind, kind))
    st.markdown("**Every value**")
    st.markdown(md_table(["Setting", "Value", "What it is", "What it affects"],
                         [(f"`{r['name']}`", explain.fmt(r["value"]), r["short"],
                           r["affects"] + (f" *Typical: {r['range']}.*" if r["range"] else "")) for r in rows]))
    _used_by(kind, name)


def _inputs(name: str, preset) -> None:
    from racer.env.actions import action_spec_of, build_action_layout
    from racer.env.obs_spec import build_observer
    layout = build_action_layout(action_spec_of(preset))
    obs = build_observer(preset["obs_spec"], preset["raycast"], layout)
    c1, c2 = st.columns([3, 2])
    c1.markdown(f"**{obs.dim} inputs:** " + ", ".join(f"`{x}`" for x in obs.labels))
    c1.markdown(f"**{layout.dim} outputs ({layout.name}):** " + ", ".join(f"`{x}`" for x in layout.labels))
    ep = preset["episode"]
    limits = "; **and when the car's centre crosses the road edge (track limits)**" if ep.get("track_limits") else ""
    c1.markdown(f"**A try ends** at the barrier, after {ep['offroad_limit_s']:g} s fully off the road, when stuck, "
                f"going the wrong way, or after {ep['max_steps']} decisions{limits}.")
    spec = preset["obs_spec"]
    rays = spec.get("rays") if spec.get("version") == 1 else next(
        (f["rays"] for f in spec.get("features", []) if f.get("feature") == "rays"), None)
    if rays:
        c2.image(straight_png(f"preset_{name}", (("", freeze(rays)),), 12.0, 0.0), width="stretch",
                 caption="The rays on a straight 12 m road, to scale.")


def _rays(name: str, preset) -> None:
    st.image(straight_png(f"preset_rays_{name}", (("", freeze(preset["rays"])),), 12.0, 0.0), width=520,
             caption="The rays on a straight 12 m road, to scale.")


def _training(name: str, preset) -> None:
    from racer.agents.algorithms import ALGORITHMS
    from racer.agents.compat import count_params
    algo = preset.get("algorithm")
    if "policy" not in preset:
        st.markdown("Settings of the hand-coded PID baseline driver (E8), the reference every model's lap time is "
                    "compared with. It isn't trained, so this isn't a `--train` preset.")
        return
    st.markdown(f"**{algo.upper()}**: {ALGORITHMS[algo].description if algo in ALGORITHMS else ''}")
    arch = [int(n) for n in preset["policy"]["net_arch"]]
    n_in, env = (10, "rays7_lean") if algo == "ga" else (15, "rays7_narrow")      # the inputs its runs usually use
    n = count_params(algo, n_in, 2, arch)["policy"]
    st.markdown(f"**Network:** hidden layers {' × '.join(map(str, arch))}, {preset['policy']['activation']}; "
                f"{n:,} parameters drive the car with {n_in} inputs ({env}) and 2 outputs.")
    if algo == "ga":
        gens = preset["ga"]["max_generations"]
        st.markdown(f"**Budget:** {gens or 'no limit on'} generations (`--generations`); without a limit it runs "
                    "until the validation score stops improving.")
    else:
        st.markdown(f"**Budget:** {int(preset['total_steps']):,} env steps (`--steps`).")


def _tracks(name: str, preset) -> None:
    from racer.env.trackset import resolve_trackset
    ts = resolve_trackset(name)
    fixed, gen = ts.get("fixed") or [], ts.get("generated") or {}
    pool = f" + a generated pool (weight {gen['weight']:g})" if gen else " and no generated pool (fits neuroevolution)"
    st.markdown(f"**{len(fixed)} named tracks**{pool}")
    cols = st.columns(7)
    for i, f in enumerate(fixed):
        label = f"{f['track']} ×{float(f.get('weight', 1)):g}"
        img = store().track_preview(f["track"])
        if img:
            cols[i % 7].image(str(img), caption=label, width="stretch")
        else:
            cols[i % 7].caption(label)


def _used_by(kind: str, name: str) -> None:
    runs = explain.runs_using(kind, name, store().runs())
    if kind in ("rays", "readings", "network", "learning", "budget"):
        return
    label = "Runs with this algorithm" if kind == "train" else "Runs trained with this preset"
    st.markdown(f"**{label}:** " + (", ".join(f"{r['run_id']} ({r.get('tag')})" for r in runs) if runs else "none yet."))
