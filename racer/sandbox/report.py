"""The PNG report set for the write-up and the slides (T137): the Workbench plots, written as files.

``build_report`` draws from the registry, TensorBoard curves and (unless ``drive=False``) laps driven
right now, with the same functions as the Workbench (:mod:`racer.sandbox.plots`). Figures whose data
is not available on this machine (a model folder or TensorBoard logs that were not copied over) are
skipped and listed, never faked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from racer.config import PROJECT_ROOT
from racer.sandbox import curves, plots
from racer.sandbox.data import STORE, Store, experiment_group

DEFAULT_OUT = PROJECT_ROOT / "outputs" / "plots" / "report"


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def registry_frame(store: Store = STORE) -> pd.DataFrame:
    return plots.models_frame(store.models(False), {str(r["run_id"]): r for r in store.runs(False)})


def results_scatter(df: pd.DataFrame):
    """Best checkpoint of every comparable run: completion on unseen tracks against lap ratio."""
    import matplotlib.pyplot as plt
    best = plots.best_per_run(df).dropna(subset=["unseen_completion", "lap_ratio"])
    best = best[~best["tag"].astype(str).str.contains("nogrip|naive", regex=True)]      # other car / reward: not comparable
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for algo, g in best.groupby("algorithm"):
        ax.scatter(g["unseen_completion"], g["lap_ratio"], label=str(algo), s=45)
        for _, r in g.iterrows():
            ax.annotate(r["run_id"], (r["unseen_completion"], r["lap_ratio"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(1.0, color="#888", lw=0.8, ls="--")
    ax.set_xlabel("completion on tracks never trained on")
    ax.set_ylabel("lap ratio vs the hand-coded baseline (below 1 = faster)")
    ax.set_title("Best checkpoint of each run")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def ablation_bars(df: pd.DataFrame):
    """Seen and unseen completion of each run's best checkpoint, runs grouped by experiment."""
    best = plots.best_per_run(df)
    best = best.assign(group=best["tag"].map(experiment_group)).sort_values(["group", "run_id"])
    best["run"] = best["run_id"] + " " + best["group"].astype(str)
    long = best.melt(id_vars=["run"], value_vars=["completion_rate", "unseen_completion"], var_name="metric", value_name="value")
    long["metric"] = long["metric"].map(plots.METRIC_LABELS)
    return plots.fig_bars(long.dropna(subset=["value"]), "run", "value", "metric", percent=True,
                          title="Completion by run (best checkpoint, protocol v2 where evaluated)")


def method_comparison(df: pd.DataFrame, runs: Sequence[str]):
    """E10: grouped bars of the given runs' best checkpoints."""
    best = plots.best_per_run(df)
    ids = list(best[best["run_id"].isin(runs)]["model_id"])
    if len(ids) < 2:
        return None
    metrics = ["completion_rate", "test_completion", "heldout_completion", "unseen_completion"]
    long = plots.comparison_long(df, ids, metrics)
    return plots.fig_bars(long, "metric", "value", "model", percent=True, title="Method comparison (best checkpoints)")


def learning_figure(run_ids: Sequence[str], tag: str, store: Store = STORE):
    wide = curves.table(run_ids, tag, weight=0.6, store=store)
    if wide.empty:
        return None
    long = wide.reset_index(names="step").melt(id_vars="step", var_name="run", value_name="value").dropna()
    return plots.fig_learning(long, curves.MAIN_CURVES.get(tag, tag))


def build_report(out_dir: Path | str = DEFAULT_OUT, models: Sequence[str] = ("baseline",), track: str = "technical",
                 compare_runs: Sequence[str] = ("R021", "R022", "R023"), drive: bool = True,
                 log: Callable[[str], None] = lambda s: None, store: Store = STORE) -> tuple[list[Path], list[str]]:
    """Write the figures; returns ``(files written, reasons for the ones skipped)``."""
    out = Path(out_dir)
    written: list[Path] = []
    skipped: list[str] = []

    def put(name: str, make: Callable[[], object]) -> None:
        try:
            fig = make()
        except Exception as e:                                       # one broken figure must not stop the rest
            skipped.append(f"{name}: {e}")
            return
        if fig is None:
            skipped.append(f"{name}: no data on this machine")
            return
        written.append(_save(fig, out / name))
        log(f"wrote {written[-1]}")

    df = registry_frame(store)
    if df.empty:
        return written, ["registry is empty"]
    put("results_scatter.png", lambda: results_scatter(df))
    put("ablations.png", lambda: ablation_bars(df))
    put("method_comparison.png", lambda: method_comparison(df, compare_runs))
    for tag, name in (("rollout/ep_rew_mean", "learning_reward.png"), ("ga/validation", "learning_ga_validation.png")):
        put(name, lambda tag=tag: learning_figure([r for r in compare_runs], tag, store))
    if drive:
        from racer.agents.telemetry import drive as drive_lap
        tels = []
        for who in models:
            try:
                tels.append(drive_lap(who, track, laps=2))
            except Exception as e:
                skipped.append(f"drive {who}: {e}")
        if tels:
            put(f"racing_line_{track}.png", lambda: plots.fig_racing_line(tels, track))
            put(f"speed_trace_{track}.png", lambda: plots.fig_speed_trace(tels))
            put(f"gg_{track}.png", lambda: plots.fig_gg(tels))
            summary = plots.lap_summary(tels)
            out.mkdir(parents=True, exist_ok=True)
            summary.to_csv(out / f"laps_{track}.csv", index=False)
            written.append(out / f"laps_{track}.csv")
    return written, skipped
