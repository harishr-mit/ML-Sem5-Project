"""Launch: pick trained models and open game windows racing them, several at once (T70)."""
from __future__ import annotations

import streamlit as st

from racer.agents.checkpoints import STRATEGIES, choose
from racer.config import load_preset
from racer.sandbox.builder import preset_names
from racer.sandbox.launcher import Launcher
from racer.sandbox.pages._ui import model_label, store, track_names


def _launcher() -> Launcher:
    return Launcher()


def _do_launch(**kw) -> None:
    try:
        item = _launcher().launch(**kw)
        st.session_state["ln_msg"] = ("success", f"Opened game window #{item['id']}: {item['title']}. "
                                      "It appears in a few seconds (models load first).")
    except (ValueError, OSError) as e:
        st.session_state["ln_msg"] = ("error", f"Could not launch: {e}")


def _do_stop(instance_id: int) -> None:
    res = _launcher().stop(instance_id)
    st.session_state["ln_msg"] = ("info", f"Window #{instance_id}: {res}.")


def _do_stop_all() -> None:
    n = _launcher().stop_all()
    st.session_state["ln_msg"] = ("info", f"Stopped {n} window(s).")


def render() -> None:
    s = store()
    acfg = load_preset("demo", "arena")
    max_cars = int(acfg["max_cars"])
    st.title("Launch")
    st.caption("Pick trained models and press Launch: a game window opens and they race as ghost cars. Every Launch "
               "opens its own window, so several races can run side by side; closing the sandbox leaves them open. "
               "In the game: Tab follows the next car, F overview, R rays, Space pause, Esc closes the window.")
    msg = st.session_state.pop("ln_msg", None)
    if msg:
        getattr(st, msg[0])(msg[1])

    pre = st.session_state.pop("launch_models", None)
    pre_track = st.session_state.pop("launch_track", None)           # handed over by Draw a Track
    tab_models, tab_evo, tab_drive = st.tabs(["Race chosen models", "Evolution race (one run)", "Drive it yourself"])
    names = track_names()
    with tab_models:
        c1, c2 = st.columns([5, 1.4])
        trials = c2.toggle("Trial runs", key="ln_trials") or bool(pre and any(m.startswith("S") for m in pre))
        any_cp = c2.toggle("Any checkpoint", key="ln_any", help="Off: only each run's best checkpoint.")
        runs = s.explorable_runs(trials)
        rows, blocked = {}, {}
        for r in runs:
            for cp in (s.checkpoints(str(r["run_id"])) if any_cp else [s.best(str(r["run_id"]))]):
                if not cp:
                    continue
                problems = s.problems(cp["model_id"])
                if problems:
                    blocked[cp["model_id"]] = "; ".join(problems)
                else:
                    rows[cp["model_id"]] = cp
        for m in pre or []:
            if m not in rows and s.folder(m) and not s.problems(m):
                rows[m] = s.model_row(m) or {"model_id": m}
        if pre is not None:
            st.session_state["ln_models"] = [m for m in pre if m in rows]
        st.session_state["ln_models"] = [m for m in st.session_state.get("ln_models", []) if m in rows]
        picked = c1.multiselect(f"Models (up to {max_cars} cars)", list(rows), key="ln_models",
                                max_selections=max_cars, format_func=lambda m: model_label(rows[m]),
                                placeholder="Choose models to race")
        if blocked:
            with st.expander(f"{len(blocked)} model(s) can't run on this engine and are hidden"):
                for m, why in blocked.items():
                    st.markdown(f"- `{m}`: {why}")
        track, laps, baseline, fair, sound = _race_options("ln", names, acfg, pre_track)
        n_cars = len(picked) + int(baseline)
        too_many = n_cars > max_cars
        if too_many:
            st.error(f"{n_cars} cars: the arena takes at most {max_cars}.")
        st.button("Launch", type="primary", key="ln_go", disabled=not n_cars or too_many, on_click=_do_launch,
                  kwargs={"track": track, "models": list(picked), "laps": laps, "baseline": baseline, "fair": fair,
                          "extra": _sound_args(sound)})

    with tab_evo:
        st.markdown("The checkpoints of one run race each other: early ones crash, later ones are faster. "
                    "Shows learning in about 20 seconds.")
        real = [r for r in s.explorable_runs(False)]
        if not real:
            st.info("No runs yet.")
        else:
            by_id = {str(r["run_id"]): r for r in real}
            run_id = st.selectbox("Run", list(by_id), key="evo_run",
                                  format_func=lambda r: f"{r} · {by_id[r].get('tag', '')}")
            cps = s.checkpoints(run_id)
            c_a, c_b = st.columns([3, 2])
            strategy = c_a.selectbox("Which checkpoints", list(STRATEGIES), key="evo_strategy",
                                     format_func=lambda k: STRATEGIES[k])
            count = int(c_b.number_input("N", 1, max(len(cps), 1), min(6, max(len(cps), 1)), key=f"evo_n_{run_id}"))
            track, laps, baseline, fair, sound = _race_options("evo", names, acfg)
            usable = [c for c in cps if not s.problems(c["model_id"])]
            chosen = choose(usable, strategy, count)[: max_cars - int(baseline)]
            st.caption(f"{len(chosen)} cars: " + ", ".join(f"{int(c.get('steps') or 0) // 1000}k" for c in chosen))
            st.button("Launch evolution race", type="primary", key="evo_go", disabled=not chosen,
                      on_click=_do_launch, kwargs={"track": track, "models": [c["model_id"] for c in chosen],
                                                   "laps": laps, "baseline": baseline, "fair": fair,
                                                   "label": f"evolution of {run_id}",
                                                   "extra": ["--gradient", *_sound_args(sound)]})

    with tab_drive:
        st.markdown("Drive a car yourself in a game window (arrows or WASD, or a gamepad) against the models you pick "
                    "above. Hard braking is assisted by ABS (K toggles it in the game). Your car uses the physics you choose, "
                    "so you can feel what a model was trained on.")
        physics_names = preset_names("physics")
        physics = st.selectbox("Your car's physics", physics_names, key="dr_physics",
                               index=physics_names.index("default") if "default" in physics_names else 0)
        opponents = st.multiselect("Opponents (from the models listed in the first tab)", list(rows), key="dr_models",
                                   max_selections=max_cars - 1, format_func=lambda m: model_label(rows[m]))
        track, laps, baseline, fair, sound = _race_options("dr", names, acfg)
        st.button("Drive!", type="primary", key="dr_go", on_click=_do_launch,
                  kwargs={"track": track, "models": list(opponents), "laps": laps, "baseline": baseline, "fair": fair,
                          "extra": _sound_args(sound), "human": physics})

    st.header("Game windows")
    items = _launcher().instances()
    live = [it for it in items if it["running"]]
    if not live:
        st.caption("No game window is open.")
    for it in live:
        c1, c2 = st.columns([6, 1])
        cars = len(it.get("models") or []) + int(bool(it.get("baseline")))
        c1.markdown(f"**#{it['id']}** {it['title']}  \n{cars} car(s) · {it.get('laps') or 'default'} laps"
                    f"{' · fair physics' if it.get('fair') else ''} · started {it['started'][11:]} · PID {it['pid']}")
        c2.button("Stop", key=f"stop_{it['id']}", on_click=_do_stop, args=(it["id"],))
    if len(live) > 1:
        st.button("Stop all", key="stop_all", on_click=_do_stop_all)
    done = [it for it in items if not it["running"]]
    if done:
        with st.expander(f"Closed windows ({len(done)})"):
            for it in reversed(done):
                st.markdown(f"**#{it['id']}** {it['title']} · {it['started'][11:]}–{str(it.get('ended', ''))[11:]}")
                tail = Launcher.log_tail(it, 6)
                if tail:
                    st.code(tail, language="text")


def _sound_args(sound: bool) -> list[str]:
    return [] if sound else ["--no-sound"]


def _race_options(prefix: str, names: list[str], acfg: dict,
                  chosen: str | None = None) -> tuple[str, int, bool, bool, bool]:
    s = store()
    c1, c2, c3, c4, c5 = st.columns([3, 1.2, 1.4, 1.4, 1.0])
    if chosen in names:
        st.session_state.pop(f"{prefix}_track", None)     # the hand-over wins over the last choice
    default = chosen if chosen in names else "technical"
    track = c1.selectbox("Map", names, key=f"{prefix}_track",
                         index=names.index(default) if default in names else 0)
    laps = int(c2.number_input("Laps", 1, 20, int(acfg["race"]["laps"]), key=f"{prefix}_laps"))
    baseline = c3.checkbox("Add the baseline car", key=f"{prefix}_base",
                           help="The hand-coded PID driver (E8), as a reference.")
    fair = c4.checkbox("Same physics for all", key=f"{prefix}_fair",
                       help=f"Every car on the '{acfg['fair_physics']}' physics preset instead of its own.")
    sound = c5.checkbox("Sound", value=True, key=f"{prefix}_sound",
                        help="Engine and tyre sound in the game window (M toggles it there). On when watching a race.")
    img = s.track_preview(track)
    if img:
        with st.expander("Map preview"):
            st.image(str(img), width=420)
    return track, laps, baseline, fair, sound
