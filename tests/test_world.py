"""Track world visuals (T130): textures, scenery placement, the pre-rendered surface."""
import os

import numpy as np
import pytest

from racer.config import load_preset
from racer.render import textures as T
from racer.render.scenery import place_props

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
DCFG = load_preset("demo", "render")["debug"]


@pytest.fixture(scope="module")
def track():
    from racer.env.trackset import get_track
    return get_track("technical", load_preset("track", "default"))


def test_grade_darkens_and_desaturates():
    rgb = np.array([[[200, 100, 50]]], dtype=np.uint8)
    assert T.grade(rgb, 0.5, 0.0)[0, 0].tolist() == [100, 50, 25]
    grey = T.grade(rgb, 0.0, 1.0)[0, 0]
    assert grey.max() - grey.min() <= 1                                   # fully desaturated
    assert T.grade(rgb, -1.0, 0.0)[0, 0].tolist() == [255, 200, 100]      # negative darken brightens, clipped


def test_tiles_wrap_seamlessly_and_missing_files_fall_back():
    rng = np.random.default_rng(0)
    tile = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    assert T.tile_at(tile, 5, 6).tolist() == tile[1, 2].tolist()
    assert T.tile_at(tile, np.array([[0], [4]]), np.array([[0, 4]])).shape == (2, 2, 3)
    pygame_tile = T.load_tile("no_such_texture.jpg", 32, 0.0, 0.0, (80, 120, 60), rng)
    assert pygame_tile.shape == (32, 32, 3) and abs(int(pygame_tile[..., 1].mean()) - 120) < 12
    real = T.load_tile("grass.jpg", 64, 0.3, 0.2, (0, 0, 0), rng)
    assert real.shape == (64, 64, 3) and real.mean() > 10                  # the real file loaded, not the black fallback


def test_bilinear_is_exact_on_nodes_and_linear_between():
    g = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = T.bilinear(g, np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 3.0]))
    assert out.shape == (3, 3) and out[1, 1] == g[1, 1] and out[2, 2] == g[2, 3]
    mid = T.bilinear(g, np.array([0.5]), np.array([0.5]))
    assert mid[0, 0] == pytest.approx(g[:2, :2].mean())


def test_low_frequency_multiplier_stays_near_one():
    m = T.low_frequency((120, 200), 20, 0.1, np.random.default_rng(1))
    assert m.shape == (120, 200) and 0.88 < m.min() and m.max() < 1.12 and abs(m.mean() - 1.0) < 0.05


def test_props_stay_outside_the_barrier_and_do_not_overlap(track):
    cfg = DCFG["scenery"]
    props = place_props(track, cfg, np.random.default_rng(3))
    assert len(props) > 20
    f = track.field
    barrier = track.half_width + track.runoff
    for p in props:
        c, r = int(round((p.x - f.x0) / f.cell)), int(round((p.y - f.y0) / f.cell))
        assert 0 <= r < f.dist.shape[0] and 0 <= c < f.dist.shape[1]
        assert f.dist[r, c] >= barrier + 0.5, f"{p.kind} at {p.x:.0f},{p.y:.0f} is inside the barrier"
    kinds = {p.kind for p in props}
    assert any(k.startswith("grandstand") for k in kinds) and any(k.startswith("tree") for k in kinds)


def test_props_are_deterministic_per_seed(track):
    a = place_props(track, DCFG["scenery"], np.random.default_rng(7))
    b = place_props(track, DCFG["scenery"], np.random.default_rng(7))
    c = place_props(track, DCFG["scenery"], np.random.default_rng(8))
    assert a == b and a != c


def test_grandstands_sit_by_the_start_line(track):
    props = [p for p in place_props(track, DCFG["scenery"], np.random.default_rng(3)) if p.kind.startswith("grandstand")]
    start = track.centreline.points[0]
    reach = DCFG["scenery"]["grandstand_span_m"] + 60
    assert props and all(np.hypot(p.x - start[0], p.y - start[1]) < reach for p in props)


@pytest.mark.parametrize("textured", [True, False])
def test_world_surface_builds_with_and_without_textures(track, textured):
    import pygame
    from racer.render.world import TrackWorld

    pygame.init()
    pygame.display.set_mode((64, 64))
    cfg = dict(DCFG)
    cfg["textures"] = dict(DCFG["textures"], enabled=textured)
    cfg["scenery"] = dict(DCFG["scenery"], enabled=textured)
    colours = {k: tuple(v) for k, v in DCFG["colours"].items()}
    world = TrackWorld(cfg, colours)
    world.set_track(track)
    x0, x1, y0, y1 = track.field.extent
    assert world.surface.get_size() == (int((x1 - x0) * cfg["world_px_per_m"]), int((y1 - y0) * cfg["world_px_per_m"]))
    assert bool(world.props) == textured
    # the road is where the physics says: the surface pixel on the centreline is asphalt-ish (not grass green)
    px, py = world.wpx(*track.centreline.points[10])
    r, g, b, _ = world.surface.get_at((int(px), int(py)))
    assert not (g > r + 25 and g > b + 25)
    pygame.quit()
