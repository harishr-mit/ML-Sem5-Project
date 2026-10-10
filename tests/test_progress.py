"""ProgressTracker: forwards, backwards, wrap-around, laps and territory jumps."""
import numpy as np
import pytest

from racer.track.io import TrackSpec
from racer.track.progress import ProgressTracker
from racer.track.track import Track


@pytest.fixture(scope="module")
def track():
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    pts = np.column_stack([150 * np.cos(a), 150 * np.sin(a)])
    return Track(TrackSpec("circle150", pts.tolist(), 12.0, 15.0, 1.2, "ccw"))


def drive(tracker, track, s_values, lateral=0.0):
    out = []
    for s in s_values:
        x, y, _ = track.pose_at(s, lateral)
        out.append(tracker.update(x, y))
    return out


def test_forward_lap_with_wrap(track):
    pt = ProgressTracker(track)
    x, y, _ = track.pose_at(track.length - 50.0)
    pt.reset(x, y)
    steps = np.arange(track.length - 50.0 + 2.0, 2 * track.length + 10.0, 2.0)
    res = drive(pt, track, steps % track.length)
    assert all(not r[2] for r in res)
    assert all(r[0] > 0 for r in res)
    assert pt.total == pytest.approx(steps[-1] - (track.length - 50.0), abs=1.5)
    assert pt.crossings == 2


def test_backwards_is_negative(track):
    pt = ProgressTracker(track)
    x, y, _ = track.pose_at(20.0)
    pt.reset(x, y)
    res = drive(pt, track, [(20.0 - k) % track.length for k in range(1, 40)])
    assert all(r[0] < 0 for r in res)
    assert pt.total == pytest.approx(-39.0, abs=1.5)
    assert pt.crossings == -1


def test_lateral_offset_sign(track):
    pt = ProgressTracker(track)
    x, y, _ = track.pose_at(100.0, lateral=3.0)
    pt.reset(x, y)
    assert pt.lateral == pytest.approx(3.0, abs=0.05)


def test_heading_error(track):
    pt = ProgressTracker(track)
    x, y, th = track.pose_at(100.0)
    pt.reset(x, y)
    assert pt.heading_error(th) == pytest.approx(0.0, abs=0.02)
    assert abs(pt.heading_error(th + np.pi)) == pytest.approx(np.pi, abs=0.02)


def test_territory_jump(track):
    pt = ProgressTracker(track)
    x, y, _ = track.pose_at(100.0)
    pt.reset(x, y)
    x2, y2, _ = track.pose_at(300.0)
    assert pt.update(x2, y2)[2]
    assert pt.update(1e5, 1e5)[2]
