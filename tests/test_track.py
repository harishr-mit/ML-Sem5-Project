"""Spline, field and surface tests on simple synthetic tracks."""
import math

import numpy as np
import pytest

from racer.physics.surfaces import GRASS_ID, KERB_ID, OUTSIDE_ID, ROAD_ID
from racer.track.io import TrackSpec
from racer.track.track import Track


def circle_spec(radius=100.0, n=48, direction="ccw", width=12.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([radius * np.cos(a), radius * np.sin(a)])
    return TrackSpec("circle", pts.tolist(), width, 15.0, 1.2, direction)


def stadium_spec(width=12.0):
    """Two 300 m straights (y = ±40) joined by R = 40 m semicircles (kerbs only on the bends)."""
    pts = []
    for x in np.linspace(-150, 150, 7):
        pts.append([x, -40.0])
    for a in np.linspace(-np.pi / 2, np.pi / 2, 7)[1:-1]:
        pts.append([150 + 40 * np.cos(a), 40 * np.sin(a)])
    for x in np.linspace(150, -150, 7):
        pts.append([x, 40.0])
    for a in np.linspace(np.pi / 2, 3 * np.pi / 2, 7)[1:-1]:
        pts.append([-150 + 40 * np.cos(a), 40 * np.sin(a)])
    return TrackSpec("stadium", pts, width, 15.0, 1.2, "ccw")


@pytest.fixture(scope="module")
def circle():
    return Track(circle_spec())


def test_circle_length_and_curvature(circle):
    cl = circle.centreline
    assert cl.length == pytest.approx(2 * math.pi * 100, rel=2e-3)
    assert cl.ds == pytest.approx(1.0, abs=0.01)
    assert np.allclose(cl.curvature_smooth, 1 / 100, rtol=0.03)   # ccw = left turns = positive
    assert np.allclose(np.hypot(*cl.tangent.T), 1.0)
    assert Track(circle_spec(radius=150.0)).validate() == []   # R = 100 m is shorter than min_length


def test_direction_cw_reverses(circle):
    cw = Track(circle_spec(direction="cw"))
    assert np.allclose(cw.centreline.curvature_smooth, -1 / 100, rtol=0.03)
    assert np.allclose(cw.centreline.points[0], circle.centreline.points[0])  # same start line


def test_field_matches_brute_force(circle):
    rng = np.random.default_rng(1)
    pts = circle.centreline.points
    for _ in range(500):
        # Within the field reach (hw + runoff + margin = 25 m), and > 1 m from the centreline:
        # bilinear interpolation rounds off the kink of |d| there (irrelevant: edges are at 6 m).
        r = 100 + rng.choice([-1, 1]) * rng.uniform(1.0, 24.0)
        a = rng.uniform(0, 2 * np.pi)
        x, y = r * np.cos(a), r * np.sin(a)
        d, i = circle.sample(x, y)
        dd = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        # brute force against the polyline (the samples are 1 m apart; true circle distance is |r - R|)
        assert d == pytest.approx(abs(r - 100), abs=0.05)
        assert dd[i] <= dd.min() + 0.6


def test_surfaces_on_stadium():
    t = Track(stadium_spec())
    assert t.validate() == []
    hw = t.half_width
    # straight bottom (y = -60): no kerbs
    assert t.surface_at(0.0, -40.0) == ROAD_ID
    assert t.surface_at(0.0, -40.0 + hw - 0.5) == ROAD_ID
    assert t.surface_at(0.0, -40.0 - hw - 3.0) == GRASS_ID
    assert t.surface_at(0.0, -40.0 - hw - t.runoff - 1.0) == OUTSIDE_ID
    assert t.surface_at(5000.0, 0.0) == OUTSIDE_ID
    # middle of the right-hand bend: kerb band just inside the edge
    x_bend = 150 + 40
    assert t.surface_at(x_bend + hw - 0.5, 0.0) == KERB_ID
    assert t.surface_at(x_bend - hw + 0.5, 0.0) == KERB_ID
    assert t.surface_at(x_bend, 0.0) == ROAD_ID
    assert t.kerb_mask.any() and not t.kerb_mask.all()


def test_validation_rejects_close_sections():
    # Thin hairpin "paperclip": straights 20 m apart (< width 12 + clearance 6 + margin)
    pts = [[-200, -10], [0, -10], [200, -10], [215, 0], [200, 10], [0, 10], [-200, 10], [-215, 0]]
    t = Track(TrackSpec("clip", pts, 12.0, 15.0, 1.2, "ccw"))
    problems = t.validate()
    assert any("apart" in p for p in problems) or any("radius" in p for p in problems)


def test_field_memory_is_reasonable(circle):
    assert circle.field.nbytes < 20e6
