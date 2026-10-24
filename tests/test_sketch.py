"""Draw-a-Track: a drawing (metres) becomes a validated track (T111)."""
import numpy as np
import pytest

from racer.config import load_preset
from racer.track.sketch import sketch_track

CFG = load_preset("demo", "editor")["sketch"]


def circle(radius, n=200, cw=False, start_deg=0.0):
    a = np.radians(start_deg) + np.linspace(0, 2 * np.pi, n, endpoint=False) * (-1 if cw else 1)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])


def test_circle_is_valid_and_starts_where_drawing_started():
    pts = circle(200, start_deg=90)
    track, problems = sketch_track(pts, "c", CFG)
    assert problems == []
    assert track.spec.direction == "ccw"
    assert np.allclose(track.spec.control_points[0], [0, 200], atol=15)
    assert track.length == pytest.approx(2 * np.pi * 200, rel=0.05)   # right size: not rescaled


def test_direction_follows_the_drawing():
    track, problems = sketch_track(circle(200, cw=True), "c", CFG)
    assert problems == [] and track.spec.direction == "cw"


def test_small_doodle_is_scaled_up_to_a_valid_track():
    track, problems = sketch_track(circle(20), "d", CFG)
    assert problems == []
    assert track.length >= track.cfg["validation"]["min_length_m"]


def test_huge_drawing_is_scaled_down():
    track, problems = sketch_track(circle(2000), "h", CFG)
    assert problems == []
    assert track.length <= track.cfg["validation"]["max_length_m"]


def test_freehand_square_with_sharp_corners_and_jitter_becomes_valid():
    side = np.linspace(0, 100, 50, endpoint=False)
    sq = np.vstack([np.column_stack([side, 0 * side]), np.column_stack([100 + 0 * side, side]),
                    np.column_stack([100 - side, 100 + 0 * side]), np.column_stack([0 * side, 100 - side])])
    sq += np.random.default_rng(0).normal(0, 1.0, sq.shape)
    track, problems = sketch_track(sq, "sq", CFG)
    assert problems == []


def test_clicked_points_work_too():
    track, problems = sketch_track([[0, 0], [300, 0], [300, 200], [0, 200]], "click", CFG)
    assert problems == []


def test_figure_eight_is_reported_not_fixed():
    t = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    eight = np.column_stack([300 * np.sin(t), 150 * np.sin(2 * t)])
    track, problems = sketch_track(eight, "8", CFG)
    assert track is not None and any("apart" in p for p in problems)


@pytest.mark.parametrize("pts", [[], [[0, 0], [10, 0]], [[0, 0], [1, 0], [1, 1]], [[5, 5]] * 50,
                                 [[0, 0], [150, 1], [300, 0]], [[x, 0] for x in range(0, 300, 10)]])
def test_too_little_drawing_gives_a_message_and_no_track(pts):
    track, problems = sketch_track(pts, "x", CFG)
    assert track is None and len(problems) == 1


def test_jagged_random_polygons_never_raise_and_mostly_become_valid():
    rng = np.random.default_rng(1)
    valid = 0
    for _ in range(20):
        n, base = rng.integers(5, 12), rng.uniform(30, 400)
        a, r = np.sort(rng.uniform(0, 2 * np.pi, n)), base * rng.uniform(0.5, 1, n)
        ctrl = np.column_stack([r * np.cos(a), r * np.sin(a)])
        pts = np.vstack([np.linspace(ctrl[i], ctrl[(i + 1) % n], 30, endpoint=False) for i in range(n)])
        track, problems = sketch_track(pts + rng.normal(0, base * 0.01, pts.shape), "j", CFG)
        assert track is not None
        valid += not problems
    assert valid >= 17
