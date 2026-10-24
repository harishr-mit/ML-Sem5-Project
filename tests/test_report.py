"""The PNG report (T137): files are written from the real registry; missing data is skipped, not faked."""
from pathlib import Path

import pytest

from racer.sandbox import report


def test_report_writes_the_registry_figures_and_a_driven_lap(tmp_path):
    written, skipped = report.build_report(tmp_path, ["baseline"], "training_ground", drive=True)
    names = {p.name for p in written}
    assert {"results_scatter.png", "ablations.png", "racing_line_training_ground.png", "speed_trace_training_ground.png",
            "gg_training_ground.png", "laps_training_ground.csv"} <= names
    assert all(p.stat().st_size > (3000 if p.suffix == ".png" else 50) for p in written)
    csv = (tmp_path / "laps_training_ground.csv").read_text(encoding="utf-8")
    assert "PID baseline" in csv and "lap time" in csv
    # learning curves need TensorBoard logs that are not tracked: skipped with a reason, never invented
    assert all(isinstance(s, str) and ":" in s for s in skipped)


def test_report_without_driving_skips_the_lap_figures(tmp_path):
    written, _ = report.build_report(tmp_path, ["baseline"], "training_ground", drive=False)
    names = {p.name for p in written}
    assert "results_scatter.png" in names and not any(n.startswith("racing_line") for n in names)


def test_unknown_driver_is_reported_not_fatal(tmp_path):
    written, skipped = report.build_report(tmp_path, ["R999_nope@1k"], "training_ground", drive=True)
    assert any("R999_nope@1k" in s for s in skipped)
    assert not any(p.name.startswith("racing_line") for p in written)


def test_script_exits_zero_and_lists_what_it_wrote(tmp_path):
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, str(root / "scripts" / "make_plots.py"), "--out", str(tmp_path), "--no-drive"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "results_scatter.png" in r.stdout
