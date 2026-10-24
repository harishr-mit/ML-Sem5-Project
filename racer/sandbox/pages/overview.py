"""Overview (home): what the sandbox does, the leaderboard, and what is running (T72)."""
from __future__ import annotations

import datetime as dt
import pandas as pd
import streamlit as st

from racer.sandbox.data import experiment_group
from racer.sandbox.launcher import Launcher
from racer.sandbox.pages._ui import store
from racer.sandbox.queue import QueueControl

GOALS = [
    ("explorer", "Understand a model", "What it senses (its rays on a real map), its network, the car, reward, "
                                       "tracks and hyperparameters it trained with, each with a note, and why it was made."),
    ("compare", "Compare models", "Only the settings that differ, side by side with results, curves and driven lines."),
    ("build", "Build & queue a model", "Change inputs, network, algorithm, car, tracks and steps; queue it for training."),
    ("launch", "Launch a race", "Pick models and open a game window; several windows can run at once."),
]


def _goto(page: str, **state) -> None:
    from racer.sandbox.pages import goto
    st.session_state.update(state)
    goto(page)


@st.cache_data(ttl=10, show_spinner=False)
def _busy() -> list[str]:
    return QueueControl().busy()


def leaderboard(protocol: str, trials: bool = False) -> pd.DataFrame:
    """Best checkpoint of every run, with its metrics under ``protocol`` (blank if not evaluated)."""
    s = store()
    rows = []
    for r in s.explorable_runs(trials):
        best = s.best(str(r["run_id"]))
        if not best:
            continue
        doc = s.eval_doc(best["model_id"], protocol) if protocol in s.eval_protocols(best["model_id"]) else None
        m = (doc or {}).get("metrics", {})
        other = [f"{k}: {r.get(f'{k}_preset')}" for k, default in (("physics", "default"), ("reward", "progress"))
                 if r.get(f"{k}_preset") not in (None, "", default)]
        rows.append({"model": best["model_id"], "run": r["run_id"], "experiment": experiment_group(r.get("tag")),
                     "seen": m.get("completion_rate"),
                     "unseen": m.get("unseen_completion", m.get("heldout_completion")),
                     "lap ratio": m.get("lap_ratio"), "steps (M)": int(best.get("steps") or 0) / 1e6,
                     "not comparable": ", ".join(other), "purpose": r.get("curriculum") or "",
                     "runs here": "no" if s.problems(best["model_id"]) else "yes"})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_other"] = df["not comparable"] != ""
    df = df.sort_values(["_other", "unseen", "lap ratio"], ascending=[True, False, True], na_position="last")
    return df.drop(columns="_other").reset_index(drop=True)


def render() -> None:
    s = store()
    st.title("Overview")
    st.caption("The AI Sandbox explains the trained racing models, compares them, configures and queues new ones, "
               "and launches races. It runs on this PC only; game windows and the queue run as separate programs.")
    cols = st.columns(4)
    for col, (page, title, text) in zip(cols, GOALS):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)
            st.button("Open", key=f"ov_go_{page}", on_click=_goto, args=(page,))

    runs = s.explorable_runs(False)
    cps = sum(len(s.checkpoints(str(r["run_id"]))) for r in runs)
    blocked = sum(1 for r in runs if s.best(str(r["run_id"])) and s.problems(s.best(str(r["run_id"]))["model_id"]))
    trials = len(s.explorable_runs(True)) - len(runs)
    k = st.columns(4)
    k[0].metric("Runs", len(runs))
    k[1].metric("Checkpoints", cps)
    k[2].metric("Trial runs", trials)
    k[3].metric("Can't run on this engine", blocked)

    st.header("Leaderboard")
    c1, c2 = st.columns([3, 1])
    proto = c2.radio("Protocol", ["v2", "v1"], horizontal=True, key="ov_proto",
                     help="v2: 7 training maps, 2 test maps, 10 held-out tracks. v1: 3 showcase tracks + 10 held-out. "
                          "Numbers from different protocols aren't comparable.")
    c1.caption("Each run's best checkpoint, sorted by completion on tracks it never trained on, then by lap time "
               "against the hand-coded baseline (below 1 = faster). Runs with another car or reward (e.g. no grip "
               "limit) are listed last: their numbers aren't comparable. Blank = not evaluated with this protocol "
               "(yet). Select rows to explore, compare or race them.")
    df = leaderboard(proto)
    if df.empty:
        st.info("No trained models yet.")
    else:
        ev = st.dataframe(df, hide_index=True, key="ov_table", on_select="rerun", selection_mode="multi-row",
                          column_config={
                              "seen": st.column_config.ProgressColumn("seen", format="percent", min_value=0, max_value=1),
                              "unseen": st.column_config.ProgressColumn("unseen", format="percent", min_value=0,
                                                                        max_value=1),
                              "lap ratio": st.column_config.NumberColumn("lap ratio", format="%.3f"),
                              "steps (M)": st.column_config.NumberColumn("steps (M)", format="%.1f")})
        picked = [df.iloc[i]["model"] for i in ev.selection.rows] if ev and ev.selection else []
        b = st.columns([1, 1, 1, 4])
        b[0].button("Explore", key="ov_ex", disabled=len(picked) != 1, on_click=_goto, args=("explorer",),
                    kwargs={"explorer_model": picked[0] if picked else ""})
        b[1].button("Compare", key="ov_cmp", disabled=not 2 <= len(picked) <= 4, on_click=_goto, args=("compare",),
                    kwargs={"compare_models": picked})
        b[2].button("Race them", key="ov_race", disabled=not picked or len(picked) > 20, on_click=_goto,
                    args=("launch",), kwargs={"launch_models": picked})

        _plots(df)

    st.header("Running now")
    _running()


def _plots(df: pd.DataFrame) -> None:
    """Two default plots of the leaderboard: who is both general and fast, and who generalises best."""
    from racer.sandbox import plots
    ok = df[(df["not comparable"] == "")].rename(columns={"lap ratio": "lap_ratio", "steps (M)": "steps_M"})
    ok = ok.dropna(subset=["unseen"])
    if ok.empty:
        return
    st.header("Results at a glance")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Faster and more general is down-right** (lap ratio below 1 = faster than the baseline)")
        st.altair_chart(plots.chart_scatter(ok.dropna(subset=["lap_ratio"]), "unseen", "lap_ratio", "experiment",
                                            tooltip=["model", "run", "unseen", "lap_ratio"], width=420, height=320),
                        width="stretch")
    with c2:
        st.markdown("**Completion on tracks never trained on** (top 10 runs)")
        top = ok.sort_values("unseen", ascending=False).head(10).assign(run_model=lambda d: d["run"] + " " + d["experiment"].astype(str))
        st.altair_chart(plots.chart_bars(top, "run_model", "unseen", None, width=420, height=320, percent=True),
                        width="stretch")
    st.button("Open the Workbench", key="ov_wb", on_click=_goto, args=("workbench",))


def _running() -> None:
    qc = QueueControl()
    runner = qc.runner()
    entries = qc.entries()
    live = [e for e in entries if e["status"] == "running"]
    waiting = [e for e in entries if e["status"] in ("waiting", "interrupted")]
    q = st.container(border=True)
    q.markdown("**Training queue** (`configs/queue/sandbox.yaml`)")
    if runner:
        par = int(runner.get("parallel") or 1)
        from racer.config import load_preset
        per_m = float(load_preset("sandbox", "default")["queue"]["minutes_per_million_parallel" if par > 1
                                                                 else "minutes_per_million_alone"])
        for e in live:
            started = dt.datetime.fromisoformat(e["started"]) if e.get("started") else dt.datetime.now()
            elapsed = (dt.datetime.now() - started).total_seconds() / 60
            from racer.sandbox import builder as B
            if e.get("generations") or "ga_default" in e["args"]:      # neuroevolution: a generation budget, or none
                est = B.estimate_minutes(None, 1, 1, algorithm="ga", generations=e.get("generations"))
                total = est["run"] if est else 0
            else:
                total = (e["steps"] or 0) / 1e6 * per_m
            q.markdown(f"🏃 **{e['name']}** · {B.budget_label(e['steps'], e.get('generations'))} · "
                       f"running {elapsed:.0f} min"
                       + (f" · ≈ {max(total - elapsed, 0):.0f} min of training left" if total else ""))
        if waiting:
            q.caption(f"Waiting: {', '.join(e['name'] for e in waiting)}")
    else:
        q.caption("Not running." + (f" {len(waiting)} run(s) waiting: start them on the Build & Queue page."
                                    if waiting else ""))
        others = _busy()                       # trainings or queues started from a terminal
        if others:
            q.caption("Also running: " + "; ".join(others))
    q.button("Open Build & Queue", key="ov_queue", on_click=_goto, args=("build",))
    g = st.container(border=True)
    g.markdown("**Game windows**")
    wins = Launcher().running()
    if not wins:
        g.caption("None open. Use the Launch page.")
    for it in wins:
        a, b = g.columns([6, 1])
        a.markdown(f"#{it['id']} {it['title']}")
        b.button("Stop", key=f"ov_stop_{it['id']}", on_click=_stop_game, args=(it["id"],))


def _stop_game(instance_id: int) -> None:
    Launcher().stop(instance_id)
