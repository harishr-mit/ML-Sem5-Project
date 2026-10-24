"""Workbench plots (T135): tables, Altair specs that compile, matplotlib figures that draw."""
import numpy as np
import pandas as pd
import pytest

from racer.agents.telemetry import drive
from racer.sandbox import plots as P


@pytest.fixture(scope="module")
def tels():
    a = drive("baseline", "training_ground", laps=1)
    b = drive("baseline", "training_ground", laps=1, start_speed=10.0)
    b.label = "baseline (rolling start)"
    return [a, b]


def test_outline_has_edges_a_road_width_apart():
    o = P.outline("training_ground")
    assert o["left"].shape == o["right"].shape == o["centre"].shape
    width = np.linalg.norm(o["left"] - o["right"], axis=1)
    assert width.min() > 5 and np.ptp(width) < 1e-6                         # constant road width
    assert float(o["length"]) > 1000
    assert P.outline("training_ground") is P.outline("training_ground")   # cached


def test_lap_summary_has_one_row_per_driver_and_sane_numbers(tels):
    df = P.lap_summary(tels)
    assert list(df["driver"]) == [t.label for t in tels] and len(df) == 2
    assert df["lap time (s)"].between(30, 80).all() and df["top speed (km/h)"].between(100, 300).all()
    assert df["max lateral g"].between(0.3, 1.6).all() and (df["off-road (% of lap)"] < 20).all()
    assert (df["ended"] == "lap done").all()


def test_telemetry_long_form_scales_speed_to_kmh(tels):
    long = P.telemetry_long(tels, ["speed", "gas"])
    assert set(long["channel"]) == {"speed (km/h)", "gas / brake"} and set(long["driver"]) == {t.label for t in tels}
    sp = long[(long["driver"] == tels[0].label) & (long["channel"] == "speed (km/h)")]["value"]
    assert sp.max() == pytest.approx(tels[0].lap().data["speed"].max() * 3.6)
    assert P.telemetry_long([], ["speed"]).empty


@pytest.mark.parametrize("make", [
    lambda t: P.chart_racing_line(t[:1], "training_ground"),
    lambda t: P.chart_racing_line(t, "training_ground"),
    lambda t: P.chart_telemetry(t),
    lambda t: P.chart_gg(t),
])
def test_altair_telemetry_charts_compile_to_a_vega_spec(tels, make):
    spec = make(tels).to_dict()
    assert "$schema" in spec and ("layer" in spec or "vconcat" in spec or "hconcat" in spec or "data" in spec)


def registry_frame():
    rows = [{"model_id": f"R0{r:02d}_x_{t}@{s}k", "run_id": f"R0{r:02d}", "steps": s * 1000, "is_best": s == 200,
             "completion_rate": c, "test_completion": c / 2, "heldout_completion": c, "unseen_completion": c * 0.8,
             "overall_completion": c * 0.9, "lap_ratio": 1.0 + r / 100}
            for r, t, c in ((15, "ppo", 0.64), (21, "ga", 0.9), (23, "ppo2", 1.0)) for s in (100, 200)]
    runs = {"R015": {"algorithm": "ppo", "tag": "t50"}, "R021": {"algorithm": "ga", "tag": "main"},
            "R023": {"algorithm": "ppo", "tag": "lean"}}
    return P.models_frame(rows, runs)


def test_registry_frame_best_and_comparison():
    df = registry_frame()
    assert {"steps_k", "algorithm", "tag"} <= set(df.columns) and df["steps_k"].max() == 200
    best = P.best_per_run(df)
    assert list(best["run_id"]) == ["R015", "R021", "R023"] and (best["steps_k"] == 200).all()
    long = P.comparison_long(df, list(best["model_id"]), ["completion_rate", "unseen_completion", "lap_ratio"])
    assert set(long["metric"]) == {"seen maps", "all unseen", "lap ratio vs baseline"} and len(long) == 9
    assert set(long["model"]) == {"R015@200k", "R021@200k", "R023@200k"}
    assert P.best_per_run(pd.DataFrame()).empty


def test_altair_registry_charts_compile():
    df = registry_frame()
    assert "$schema" in P.chart_scatter(df, "steps_k", "completion_rate", "run_id").to_dict()
    long = P.comparison_long(df, list(P.best_per_run(df)["model_id"]), ["completion_rate", "lap_ratio"])
    assert "$schema" in P.chart_bars(long, "metric", "value", "model").to_dict()
    curves = pd.DataFrame({"run": ["a"] * 3 + ["b"] * 3, "step": [0, 1, 2] * 2, "value": [0.0, 1.0, 2.0, 0.0, 0.5, 0.7]})
    assert "$schema" in P.chart_learning(curves, "reward").to_dict()


def test_matplotlib_figures_draw(tels):
    import matplotlib.pyplot as plt
    df = registry_frame()
    long = P.comparison_long(df, list(P.best_per_run(df)["model_id"]), ["completion_rate", "lap_ratio"])
    curves = pd.DataFrame({"run": ["a"] * 3 + ["b"] * 3, "step": [0, 1, 2] * 2, "value": [0.0, 1.0, 2.0, 0.0, 0.5, 0.7]})
    figs = [P.fig_racing_line(tels, "training_ground"), P.fig_speed_trace(tels), P.fig_gg(tels),
            P.fig_learning(curves, "reward"), P.fig_bars(long, "metric", "value", "model", title="t")]
    assert all(f.axes for f in figs)
    for f in figs:
        plt.close(f)
