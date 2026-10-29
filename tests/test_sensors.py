"""Ray casting vs exact ray-circle intersections on a circular track."""
import math
import time

import numpy as np
import pytest

from racer.config import load_preset
from racer.env.sensors import cast_ray, cast_rays, pack_raycast_params
from racer.track.io import TrackSpec
from racer.track.track import Track

R = 150.0


@pytest.fixture(scope="module")
def setup():
    a = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    pts = np.column_stack([R * np.cos(a), R * np.sin(a)])
    track = Track(TrackSpec("circle", pts.tolist(), 12.0, 15.0, 1.2, "ccw"))
    params = pack_raycast_params(load_preset("env", "rays5")["raycast"])
    return track, params


def exact(ox, oy, dx, dy, hw, rng, slack=0.0):
    """First crossing of the annulus boundary (radii R ± hw), or rng.

    ``slack`` widens the road by that much on both sides, which ignores rays that only
    graze an edge by less than ``slack`` (a touch, not a crossing).
    """
    best = rng
    for rho in (R - hw - slack, R + hw + slack):
        b = ox * dx + oy * dy
        c = ox * ox + oy * oy - rho * rho
        disc = b * b - c
        if disc < 0:
            continue
        for t in (-b - math.sqrt(disc), -b + math.sqrt(disc)):
            if 1e-9 < t < best:
                best = t
    return best


@pytest.mark.parametrize("start", ["road", "grass"])
def test_rays_match_exact(setup, start):
    track, params = setup
    f = track.field
    hw = track.half_width
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(3000):
        if start == "road":
            r = R + rng.uniform(-hw + 0.3, hw - 0.3)
        else:
            r = R + rng.choice([-1, 1]) * rng.uniform(hw + 0.5, hw + track.runoff - 0.5)
        a = rng.uniform(0, 2 * np.pi)
        ox, oy = r * math.cos(a), r * math.sin(a)
        th = rng.uniform(0, 2 * np.pi)
        dx, dy = math.cos(th), math.sin(th)
        rmax = 200.0
        got = cast_ray(f.dist, f.idx, f.geom, ox, oy, dx, dy, rmax, params)
        want = exact(ox, oy, dx, dy, hw, rmax)
        # Rays from grass that would pass the field's outer limit only see roads it covers;
        # skip those (the exact answer may lie across the infield, outside the field reach).
        mid_r = np.hypot(ox + dx * want / 2, oy + dy * want / 2)
        if start == "grass" and abs(mid_r - R) > f.reach and want < rmax:
            continue
        err = abs(got - want)
        if err > 0.1 and start == "road":
            # allowed only for a ray that dips < 5 cm past an edge (a grazing touch)
            err = min(err, abs(got - exact(ox, oy, dx, dy, hw, rmax, slack=0.05)))
        errs.append(err)
    errs = np.array(errs)
    assert len(errs) > 200
    assert np.percentile(errs, 99) < 0.05
    assert errs.max() < 0.1, f"max error {errs.max():.3f} m"


def test_ray_speed(setup):
    track, params = setup
    f = track.field
    angles = np.radians([-90.0, -45.0, 0.0, 45.0, 90.0])
    ranges = np.array([60.0, 60.0, 200.0, 60.0, 60.0])
    out = np.empty(5)
    x, y, th = track.pose_at(10.0, 4.0)
    cast_rays(f.dist, f.idx, f.geom, x, y, th, angles, ranges, params, out)  # compile
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        cast_rays(f.dist, f.idx, f.geom, x, y, th, angles, ranges, params, out)
    us = (time.perf_counter() - t0) / n * 1e6
    assert us < 15.0, f"{us:.1f} µs for 5 rays"


def exact_crossings(ox, oy, dx, dy, hw):
    """Every positive distance at which the ray meets the annulus boundary (radii R ± hw), sorted."""
    ts = []
    for rho in (R - hw, R + hw):
        b = ox * dx + oy * dy
        disc = b * b - (ox * ox + oy * oy - rho * rho)
        if disc >= 0:
            ts += [t for t in (-b - math.sqrt(disc), -b + math.sqrt(disc)) if t > 1e-9]
    return sorted(ts)


def test_rays_through_the_edge_match_exact(setup):
    """Rays that continue past the road edge find the 2nd (and 3rd) crossing as well (D33)."""
    from racer.env.sensors import cast_rays_multi
    track, params = setup
    f = track.field
    hw = track.half_width
    rng = np.random.default_rng(1)
    rmax, n_cross = 200.0, 3
    out = np.empty((1, n_cross))
    errs, seconds = [], 0
    for _ in range(1500):
        r = R + rng.uniform(-hw + 0.3, hw - 0.3)
        a = rng.uniform(0, 2 * np.pi)
        ox, oy = r * math.cos(a), r * math.sin(a)
        th = rng.uniform(0, 2 * np.pi)
        ts = exact_crossings(ox, oy, math.cos(th), math.sin(th), hw)
        if len(ts) < n_cross or ts[n_cross - 1] > rmax or min(np.diff(ts)) < 0.5:
            continue          # grazing edges, or crossings beyond the range, are not what this tests
        cast_rays_multi(f.dist, f.idx, f.geom, ox, oy, 0.0, np.array([th]), np.array([rmax]), params, n_cross, out)
        errs.append(np.abs(out[0] - np.array(ts[:n_cross])))
        seconds += 1
    assert seconds > 100
    assert np.max(errs) < 0.2, f"max error {np.max(errs):.3f} m"


def test_rays_through_the_edge_report_range_when_nothing_is_there(setup):
    from racer.env.sensors import cast_rays_multi
    track, params = setup
    f = track.field
    out = np.empty((1, 2))
    x, y, th = track.pose_at(10.0, 0.0)
    # to the left (into the infield): leaves the road once, and the far side is out of range
    cast_rays_multi(f.dist, f.idx, f.geom, x, y, 0.0, np.array([th + math.pi / 2]), np.array([60.0]), params, 2, out)
    assert 5.0 < out[0, 0] < 7.0 and out[0, 1] == 60.0
