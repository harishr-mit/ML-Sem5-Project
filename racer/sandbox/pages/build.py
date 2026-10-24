"""Build & Queue: configure a new model and add it to the training queue (T71)."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
import yaml

from racer.config import load_preset
from racer.sandbox import builder as B
from racer.sandbox import explain
from racer.sandbox.pages._ui import freeze, md_table, model_label, store, straight_png
from racer.sandbox.queue import QueueControl, command_line

SCALE_DEFAULTS = {"v_long": 70.0, "v_lat": 10.0, "yaw_rate": 3.0}
WIDTHS = [16, 32, 64, 128, 256, 512, 1024]
STEP_LABELS = {1_000_000: "1M", 2_000_000: "2M", 3_000_000: "3M", 5_000_000: "5M"}
GEN_CHOICES = [200, 500, 1000, 2000]


# ------------------------------------------------------------------ state
def _initial() -> B.Plan:
    best = store().best("R015")
    if best:
        p = B.plan_from_snapshot(store().snapshot(best["model_id"]), best["model_id"])
    else:
        p = B.plan_from_presets()
    p.tag, p.why = "", ""
    return p


def _base() -> B.Plan:
    if "bq_plan" not in st.session_state:
        st.session_state["bq_plan"] = _initial()
        st.session_state["bq_ver"] = 0
    return st.session_state["bq_plan"]


def _load(plan: B.Plan, what: str) -> None:
    plan.tag, plan.why = "", ""
    st.session_state["bq_plan"] = plan
    st.session_state["bq_ver"] = st.session_state.get("bq_ver", 0) + 1
    st.session_state["bq_msg"] = ("info", f"Form filled from {what}. Change what you want to test, then give it a "
                                          "tag and a reason.")


def _load_model(model_id: str) -> None:
    _load(B.plan_from_snapshot(store().snapshot(model_id), model_id), model_id)


def _load_presets(env: str, algorithm: str) -> None:
    _load(B.plan_from_presets(env, algorithm), f"presets {env} + {algorithm}_default")


def _qc() -> QueueControl:
    return QueueControl()


@st.cache_data(ttl=10, show_spinner=False)
def _busy() -> list[str]:
    """Trainings / queues already running (a process scan: slow the first time, so cached briefly)."""
    return QueueControl().busy()


# ------------------------------------------------------------------ page
def render() -> None:
    st.title("Build & Queue")
    st.caption("Configure a new model: what it senses, its outputs and network, how it learns, the car, reward, "
               "tracks and the training budget. Each value has a note on what it affects. Add it to the queue; "
               "the queue trains in its own window (two runs at a time) and every checkpoint is evaluated.")
    msg = st.session_state.pop("bq_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])
    base = _base()
    v = st.session_state["bq_ver"]

    with st.expander("Start from …", expanded=v == 0):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**A trained model's setup**")
            runs = store().explorable_runs(False)
            best = {r["run_id"]: store().best(str(r["run_id"])) for r in runs}
            ids = [b["model_id"] for b in best.values() if b]
            if ids:
                m = st.selectbox("Model", ids, key="bq_from_model",
                                 format_func=lambda i: model_label(store().model_row(i) or {"model_id": i}))
                st.button("Use this model's setup", key="bq_load_model", on_click=_load_model, args=(m,))
        with c2:
            st.markdown("**Presets**")
            envs = B.preset_names("env")
            env = st.selectbox("Inputs preset", envs, index=envs.index("rays7_narrow") if "rays7_narrow" in envs else 0,
                               key="bq_from_env")
            algo = st.selectbox("Algorithm", list(B.BUILDABLE), key="bq_from_algo")
            st.button("Use these presets", key="bq_load_presets", on_click=_load_presets, args=(env, algo))
    st.caption(f"Based on: **{base.started_from or 'presets'}**. Nothing is written until you press *Add to queue*.")

    tabs = st.tabs(["Inputs", "Outputs & network", "Algorithm", "Car, reward, tracks", "Budget & why"])
    plan = base.copy()
    with tabs[0]:
        plan.features = _inputs(base, v)
    with tabs[1]:
        plan.layout, plan.net_arch, plan.activation = _network(base, v, plan)
    with tabs[2]:
        plan.algorithm, plan.hparams, hp_errors = _algorithm(base, v)
    with tabs[3]:
        plan.physics, plan.reward, plan.trackset = _world(base, v)
    with tabs[4]:
        budget, plan.seeds, plan.trial, plan.tag, plan.why, seed_err = _budget(base, v, plan.algorithm)
        if plan.algorithm == "ga":
            plan.generations = budget
        else:
            plan.steps = budget

    st.header("Summary")
    _summary(plan, hp_errors + seed_err)
    st.header("Queue")
    _queue()


# ------------------------------------------------------------------ form sections
def _inputs(base: B.Plan, v: int) -> list[dict[str, Any]]:
    from racer.env.features import FEATURES, REQUIRED
    cur = {f["feature"]: f for f in base.features}
    feats: list[dict[str, Any]] = []
    st.markdown("**Rays**: distance to the road edge in each direction (angle 0 = straight ahead, + = left). "
                "Add or remove rows to change them.")
    use_rays = st.checkbox("Use rays", value="rays" in cur, key=f"bq_userays_{v}")
    if use_rays:
        rows_key, n_key = f"bq_raysrows_{v}", f"bq_raysn_{v}"
        rays = st.session_state.get(rows_key)
        if rays is None:
            rays = cur.get("rays", {}).get("rays") or [{"angle_deg": 0.0, "range_m": 200.0}]
        n_edit = st.session_state.get(n_key, 0)                 # a new editor key after each removal resets its edits
        table = pd.DataFrame(rays, columns=["angle_deg", "range_m"])
        table.insert(0, "remove", False)
        df = st.data_editor(table, num_rows="dynamic", key=f"bq_rays_{v}_{n_edit}", hide_index=True,
                            column_config={
                                "remove": st.column_config.CheckboxColumn("Remove", default=False,
                                                                          help="Tick rays to delete, then press the button below."),
                                "angle_deg": st.column_config.NumberColumn("Angle (°, + = left)", min_value=-180.0,
                                                                           max_value=180.0, step=1.0, required=True),
                                "range_m": st.column_config.NumberColumn("Range (m)", min_value=1.0, max_value=500.0,
                                                                         step=5.0, required=True)})
        marked = [bool(m) if pd.notna(m) else False for m in df["remove"]]
        if st.button(f"Remove {sum(marked)} ticked ray(s)", disabled=not any(marked), key=f"bq_rmrays_{v}"):
            kept = B.drop_marked([{"angle_deg": r.angle_deg, "range_m": r.range_m} for r in df.itertuples()], marked)
            st.session_state[rows_key] = [r for r in kept if pd.notna(r["angle_deg"]) and pd.notna(r["range_m"])]
            st.session_state[n_key] = n_edit + 1
            st.rerun()
        ray_list = [{"angle_deg": float(r.angle_deg), "range_m": float(r.range_m)} for r in df.itertuples()
                    if pd.notna(r.angle_deg) and pd.notna(r.range_m)]
        ray_list.sort(key=lambda r: r["angle_deg"])
        if ray_list:
            feats.append({"feature": "rays", "rays": ray_list})
            st.image(straight_png("build", (("", freeze(ray_list)),), 12.0, 0.0), width="stretch")
    st.markdown("**The car's own readings**")
    for name in B.FEATURE_ORDER[1:]:
        cls = FEATURES.get(name)
        if cls is None:
            continue
        c1, c2 = st.columns([2, 3])
        on = c1.checkbox(name, value=name in cur, key=f"bq_f_{name}_{v}", help=cls.description)
        c1.caption(cls.description)
        if not on:
            continue
        f: dict[str, Any] = {"feature": name}
        for k, default in cls.params.items():
            val = cur.get(name, {}).get(k)
            if val is None:
                val = SCALE_DEFAULTS.get(name) if default is REQUIRED else default
            if isinstance(val, list):
                txt = c2.text_input(f"{name}: {k}", ", ".join(f"{float(x):g}" for x in val), key=f"bq_p_{name}_{k}_{v}",
                                    help="Comma-separated numbers.")
                try:
                    f[k] = [float(x) for x in txt.replace(";", ",").split(",") if x.strip()]
                except ValueError:
                    c2.error("Numbers only, separated by commas.")
                    f[k] = val
            else:
                f[k] = float(c2.number_input(f"{name}: {k}", value=float(val), min_value=0.001, format="%g",
                                             key=f"bq_p_{name}_{k}_{v}"))
        feats.append(f)
    return B.ordered(feats)


def _network(base: B.Plan, v: int, plan: B.Plan) -> tuple[str, list[int], str]:
    from racer.agents.compat import count_params
    from racer.env.actions import LAYOUTS
    layouts = list(LAYOUTS)
    layout = st.radio("Outputs (what the network controls)", layouts, index=layouts.index(base.layout)
                      if base.layout in layouts else 0, key=f"bq_layout_{v}", horizontal=True,
                      captions=[LAYOUTS[n].description for n in layouts])
    st.markdown("**Hidden layers** (the same shape is used for the critic)")
    c1, c2 = st.columns([1, 3])
    n = int(c1.number_input("Layers", 1, 4, len(base.net_arch), key=f"bq_nlayers_{v}"))
    cols = c2.columns(n)
    arch = []
    for i in range(n):
        default = base.net_arch[i] if i < len(base.net_arch) else base.net_arch[-1]
        opts = sorted(set(WIDTHS + [int(default)]))
        arch.append(int(cols[i].selectbox(f"Layer {i + 1}", opts, index=opts.index(int(default)),
                                          key=f"bq_w{i}_{v}")))
    acts = ["tanh", "relu"]
    act = st.radio("Activation", acts, index=acts.index(base.activation) if base.activation in acts else 0,
                   key=f"bq_act_{v}", horizontal=True, help=explain.note("train.policy.activation"))
    try:
        probe = plan.copy()
        probe.layout, probe.net_arch, probe.activation = layout, arch, act
        man = B.preview(probe)
        ref = count_params("ppo", 15, 2, [256, 256])
        st.metric("Parameters that drive the car", f"{man['n_params']:,}",
                  delta=f"{man['n_params'] - ref['policy']:+,} vs R015's network", delta_color="off",
                  help=f"{man['n_params_total']:,} trained in total, including the critic.")
    except (ValueError, KeyError, TypeError):
        pass
    return layout, arch, act


def _algorithm(base: B.Plan, v: int) -> tuple[str, dict[str, Any], list[str]]:
    from racer.agents.algorithms import ALGORITHMS
    algos = [a for a in ALGORITHMS if a in B.BUILDABLE]
    algo = st.radio("Algorithm", algos, index=algos.index(base.algorithm) if base.algorithm in algos else 0,
                    key=f"bq_algo_{v}", horizontal=True, captions=[ALGORITHMS[a].description for a in algos])
    default = load_preset("train", B.train_preset_name(algo))[algo]
    start = base.hparams if algo == base.algorithm else default
    st.caption(f"Defaults come from `configs/train/{B.train_preset_name(algo)}.yaml`; changed values become "
               "`--set` arguments of train.py.")
    errors: list[str] = []
    cols = st.columns(3)
    fields = [(k, d, start.get(k, d)) for k, d in default.items() if not isinstance(d, dict) and k not in B.BUDGET_KEYS]
    fields += [(f"{k}.{sk}", sd, (start.get(k) or {}).get(sk, sd))      # nested sections, e.g. ga.validation
               for k, d in default.items() if isinstance(d, dict) for sk, sd in d.items()]
    flat: dict[str, Any] = {}
    for i, (k, d, val) in enumerate(fields):
        flat[k] = _hp_input(cols[i % 3], algo, k, d, val, v, errors)
    hp: dict[str, Any] = {}
    hp.update({k: start.get(k, d) for k, d in default.items() if k in B.BUDGET_KEYS})
    for k, val in flat.items():
        sec, _, sub = k.partition(".")
        if sub:
            hp.setdefault(sec, {})[sub] = val
        else:
            hp[k] = val
    return algo, hp, errors


def _hp_input(col, algo: str, k: str, d: Any, val: Any, v: int, errors: list[str]) -> Any:
    e = explain.lookup(f"train.{algo}.{k}") or {}
    tip = " ".join(x for x in (e.get("short"), e.get("affects"), f"Typical: {e['range']}." if e.get("range")
                               else "", f"Default: {d}.") if x)
    key = f"bq_hp_{algo}_{k}_{v}"
    if k == "lr_schedule":
        out = col.selectbox(k, ["linear", "constant"], index=0 if val == "linear" else 1, key=key, help=tip)
    elif isinstance(d, bool):
        out = col.checkbox(k, value=bool(val), key=key, help=tip)
    elif isinstance(d, int):
        out = int(col.number_input(k, value=int(val), min_value=1, step=1, key=key, help=tip))
    else:                                   # floats (and SAC's ent_coef, which may be 'auto')
        txt = col.text_input(k, f"{val:g}" if isinstance(val, float) else str(val), key=key, help=tip).strip()
        try:
            out = float(txt)
        except ValueError:
            if isinstance(d, str) and txt:
                out = txt
            else:
                errors.append(f"{k}: not a number ({txt!r}).")
                out = d
    if out != d:
        col.caption(f"changed (default {d})")
    return out


def _world(base: B.Plan, v: int) -> tuple[str, str, str]:
    from racer.env.trackset import resolve_trackset
    c1, c2 = st.columns(2)
    phys = B.preset_names("physics")
    physics = c1.selectbox("Car (physics preset)", phys, index=phys.index(base.physics) if base.physics in phys else 0,
                           key=f"bq_phys_{v}")
    diff = [r for r in explain.section(explain.reference_rows("physics", physics), "physics",
                                       explain.reference_rows("physics", "default"), exclude=["physics.extends"])
            if r["changed"]]
    c1.caption("Same as the default car." if not diff else "Differs from the default car: " + "; ".join(
        f"{r['name']} = {explain.fmt(r['value'])} (default {explain.fmt(r['default'])})" for r in diff))
    rews = B.preset_names("reward")
    reward = c2.selectbox("Reward preset", rews, index=rews.index(base.reward) if base.reward in rews else 0,
                          key=f"bq_rew_{v}")
    rrows = explain.reference_rows("reward", reward)
    terms = " · ".join(f"{k.split('.', 1)[1]} {explain.fmt(val)}" for k, val in rrows.items() if k != "reward.type")
    if rrows.get("reward.type") == "speed":
        terms += " — naive speed reward (E3): pays for circling as much as racing."
    c2.caption(terms)
    sets = B.preset_names("trackset")
    trackset = st.selectbox("Track set (test maps and held-out tracks are never trained on)", sets,
                            index=sets.index(base.trackset) if base.trackset in sets else 0, key=f"bq_ts_{v}")
    try:
        ts = resolve_trackset(trackset)
        fixed = ts.get("fixed") or []
        gen = ts.get("generated") or {}
        total = sum(float(f.get("weight", 1)) for f in fixed) + float(gen.get("weight", 0) or 0)
        st.caption(", ".join(f"{f['track']} {float(f.get('weight', 1)) / total * 100:.0f} %" for f in fixed)
                   + (f"; generated tracks {float(gen['weight']) / total * 100:.0f} %" if gen else ""))
        cols = st.columns(7)
        for i, f in enumerate(fixed[:7]):
            img = store().track_preview(f["track"])
            if img:
                cols[i].image(str(img), caption=f["track"], width="stretch")
    except Exception as e:     # a broken preset shows up in the validation below
        st.caption(f"Can't read this track set: {e}")
    return physics, reward, trackset


def _budget(base: B.Plan, v: int, algorithm: str):
    c1, c2 = st.columns([2, 1])
    if algorithm == "ga":
        opts = [None, *GEN_CHOICES, "custom"]
        cur = base.generations if base.generations in opts else "custom"
        pick = c1.radio("Generations", opts, index=opts.index(cur),
                        format_func=lambda o: "No limit" if o is None else "Custom" if o == "custom" else str(o),
                        key=f"bq_gens_{v}", horizontal=True,
                        help="Neuroevolution runs until its validation score stops improving (or you pause it); a "
                             "number of generations stops it earlier. A checkpoint is saved at every validation.")
        budget = pick if pick != "custom" else int(c1.number_input("Generations", 1, 100_000,
                                                                  int(base.generations or 1000), step=100,
                                                                  key=f"bq_gens_custom_{v}"))
    else:
        opts = list(STEP_LABELS) + ["custom"]
        cur = base.steps if base.steps in STEP_LABELS else "custom"
        pick = c1.radio("Training steps", opts, index=opts.index(cur),
                        format_func=lambda o: STEP_LABELS.get(o, "Custom") if o != "custom" else "Custom",
                        key=f"bq_steps_{v}", horizontal=True,
                        help="More steps = longer training. Every 100k steps a checkpoint is saved and evaluated, "
                             "so the best moment of the run can be picked afterwards.")
        budget = pick if pick != "custom" else int(c1.number_input("Steps", 100_000, 500_000_000,
                                                                  int(base.steps or 2_000_000), step=100_000,
                                                                  key=f"bq_steps_custom_{v}"))
    seeds_txt = c2.text_input("Seeds", ", ".join(str(s) for s in base.seeds), key=f"bq_seeds_{v}",
                              help="One run per seed. Two seeds show how much of a result is luck.")
    err = []
    try:
        seeds = [int(s) for s in seeds_txt.replace(";", ",").split(",") if s.strip()]
    except ValueError:
        seeds, err = [], ["Seeds: whole numbers separated by commas."]
    trial = st.checkbox("Trial run (scratch: S-id, kept out of the main registry and results)", value=base.trial,
                        key=f"bq_trial_{v}")
    tag = st.text_input("Tag (short name; lowercase letters, digits, _)", base.tag, key=f"bq_tag_{v}",
                        placeholder="e.g. narrow9_ent").strip()
    why = st.text_area("Why this run? (required, stored with the model)", base.why, key=f"bq_why_{v}",
                       placeholder="What should this run show, compared with which model?")
    return budget, seeds, trial, tag, why, err


# ------------------------------------------------------------------ summary + add
def _taken_tags(trial: bool) -> set[str]:
    return {str(r.get("tag")) for r in store().runs(trials=True) if (r.get("kind") == "trial") == trial}


def _summary(plan: B.Plan, extra_errors: list[str]) -> None:
    qc = _qc()
    errors = extra_errors + B.validate(plan, _taken_tags(plan.trial), qc.names())
    warns = B.warnings(plan)
    try:
        man = B.preview(plan)
    except (ValueError, KeyError, TypeError):
        man = None
    waiting = [e for e in qc.entries() if e["status"] in ("waiting", "interrupted", "running")]
    par = qc.load()["parallel"]
    est = B.estimate_minutes(plan.steps, len(waiting) + len(plan.seeds), par, algorithm=plan.algorithm,
                             generations=plan.generations)
    k = st.columns(5)
    k[0].metric("Inputs → outputs", f"{man['obs_dim']} → {man['action_dim']}" if man else "?")
    k[1].metric("Network parameters", f"{man['n_params']:,}" if man else "?")
    if plan.algorithm == "ga":
        budget = f"{plan.generations} generations" if plan.generations else "until it stops improving"
    else:
        budget = f"{plan.steps / 1e6:g}M steps"
    k[2].metric("Runs", f"{len(plan.seeds)} × {budget}")
    k[3].metric("Time per run", _fmt_min(est["run"]) if est else "open-ended",
                help=("Generations × the measured time per generation (T96; later generations drive longer "
                      "laps, so it's rough), plus evaluating the chosen checkpoints." if plan.algorithm == "ga" else
                      "Training + evaluating its checkpoints, with 2 runs in parallel (measured T62/T64).") if est else
                "Ends at the validation plateau; pause it any time and it resumes where it stopped.")
    k[4].metric("Queue done in", _fmt_min(est["queue"]) if est else "?",
                help=f"Everything waiting in the queue plus this, {par} at a time.")
    match = B.matching_env_preset(plan) if man else None
    if match:
        inputs = f"existing preset `{match}`"
    elif plan.tag:
        inputs = f"new preset `configs/env/{B.env_preset_name(plan)}.yaml` (written when you add it)"
    else:
        inputs = "a new preset (needs a tag)"
    n_sets = len(B.train_sets(plan)) if man else 0
    st.markdown(f"Inputs: {inputs} · train preset `{B.train_preset_name(plan.algorithm)}`"
                + (f" with {n_sets} change(s)" if n_sets else " unchanged")
                + (" · **trial run**" if plan.trial else " · real run (R-id)"))
    for w in warns:
        st.warning(w)
    for e in errors:
        st.error(e)
    if not errors:
        with st.expander("Exact training command(s)"):
            env_name = match or B.env_preset_name(plan)
            for e in B.queue_entries(plan, env_name):
                st.code(command_line(e["args"]), language="text")
    st.button("Add to queue", type="primary", key="bq_add", disabled=bool(errors), on_click=_add, args=(plan,))


def _add(plan: B.Plan) -> None:
    try:
        match = B.matching_env_preset(plan)
        new_file = None
        if not match:
            new_file = B.write_env_preset(plan)
        env_name = match or B.env_preset_name(plan)
        entries = B.queue_entries(plan, env_name)
        _qc().add(entries)
        text = f"Added {', '.join(e['name'] for e in entries)} to the queue."
        if new_file:
            text += (f" New input preset `configs/env/{new_file.name}` written (untracked: real runs still start; "
                     "commit it if you want to keep it with the code).")
        st.session_state["bq_msg"] = ("success", text)
    except (ValueError, FileExistsError, OSError) as e:
        st.session_state["bq_msg"] = ("error", f"Not added: {e}")


def _fmt_min(m: float) -> str:
    return f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h"


# ------------------------------------------------------------------ queue
def _start() -> None:
    try:
        pid = _qc().start()
        st.session_state["bq_msg"] = ("success", f"Queue started in its own window (PID {pid}). It keeps running if "
                                                 "you close the sandbox; stop it here or with Ctrl+C in its window.")
    except (RuntimeError, OSError) as e:
        st.session_state["bq_msg"] = ("error", f"Not started: {e}")


def _stop() -> None:
    ok = _qc().request_stop()
    st.session_state["bq_msg"] = ("info", "Stop requested: running trainings stop within a few seconds and are "
                                          "marked interrupted (when the queue starts again, PPO/SAC runs restart "
                                          "from scratch; neuroevolution resumes from its last saved generation)."
                                          if ok else "The queue isn't running.")


def _remove(name: str) -> None:
    try:
        _qc().remove(name)
        st.session_state["bq_msg"] = ("info", f"Removed {name} from the queue.")
    except ValueError as e:
        st.session_state["bq_msg"] = ("error", str(e))


def _set_parallel() -> None:
    _qc().set_parallel(int(st.session_state["bq_parallel"]))


def _open(model_id: str) -> None:
    from racer.sandbox.pages import goto
    st.session_state["explorer_model"] = model_id
    goto("explorer")


def _queue() -> None:
    qc = _qc()
    runner = qc.runner()
    entries = qc.entries()
    st.caption(f"File `{qc.path.relative_to(qc.root).as_posix()}` (git-ignored). Logs in "
               f"`outputs/queue_logs/{qc.path.stem}/`.")
    if runner:
        st.success(f"The queue is running (PID {runner['pid']}, since {str(runner.get('started', ''))[11:]}, "
                   f"{runner.get('parallel', '?')} at a time). New entries are picked up when a slot frees.")
    else:
        st.info("The queue is not running.")
    if not entries:
        st.caption("The queue is empty. Build a run above and press *Add to queue*.")
    tags = {str(r.get("tag")): r for r in store().runs(trials=True)}
    for e in entries:
        c1, c2, c3 = st.columns([5, 2, 1.3])
        mins = f" · {e['minutes']} min" if e.get("minutes") else ""
        steps = B.budget_label(e["steps"], e.get("generations"))
        c1.markdown(f"**{e['name']}** · {steps}{' · trial' if e['scratch'] else ''}{mins}")
        icon = {"waiting": "⏳", "running": "🏃", "done": "✅", "failed": "❌", "interrupted": "⏸"}.get(e["status"], "")
        c2.markdown(f"{icon} {e['status']}")
        if e["status"] in ("waiting", "interrupted", "failed"):
            c3.button("Remove", key=f"bq_rm_{e['name']}", on_click=_remove, args=(e["name"],))
        elif e["status"] == "done" and e["name"] in tags:
            best = store().best(str(tags[e["name"]]["run_id"]))
            if best:
                c3.button("Explore", key=f"bq_ex_{e['name']}", on_click=_open, args=(best["model_id"],))
        if e["status"] in ("running", "failed"):
            tail = qc.log_tail(e["name"], 8)
            if tail:
                with st.expander(f"{e['name']}: last log lines"):
                    st.code(tail, language="text")
    c1, c2, c3 = st.columns([1.3, 1.6, 3])
    data = qc.load()
    c1.selectbox("Runs at a time", [1, 2, 3], index=[1, 2, 3].index(min(max(data["parallel"], 1), 3)),
                 key="bq_parallel", on_change=_set_parallel, help="2 is fastest on this CPU (T62).")
    if runner:
        sure = c2.checkbox("Yes, stop the running trainings", key="bq_stop_sure")
        c3.button("Stop queue", key="bq_stop", disabled=not sure, on_click=_stop,
                  help="Like Ctrl+C in the queue window: checkpoints already saved stay.")
    else:
        busy = _busy()
        todo = [e for e in entries if e["status"] in ("waiting", "interrupted")]
        c2.button("Start queue", type="primary", key="bq_start", on_click=_start, disabled=bool(busy) or not todo)
        if busy:
            c3.caption("Can't start: " + "; ".join(busy))
        elif not todo:
            c3.caption("Nothing waiting.")
        else:
            c3.caption(f"{len(todo)} run(s) waiting. Real runs need committed code (D12); untracked new presets are fine.")
