"""Race effects (T129): particle pool, skid marks, shake, and the step logic on fake cars."""
from types import SimpleNamespace

import numpy as np
import pytest

from racer.config import load_preset
from racer.physics.surfaces import GRASS_ID, ROAD_ID
from racer.render.effects import Effects, ParticleField, Shake, SkidMarks

CFG = load_preset("demo", "arena")["effects"]


def test_particles_age_expire_and_the_pool_is_capped():
    p = ParticleField(5)
    for i in range(8):                                   # more than the capacity: the oldest slots are reused
        p.spawn(float(i), 0.0, 1.0, 0.0, 1.0, 0.5, 1.0, "smoke", 0.5)
    assert p.count() == 5
    p.update(0.6)
    assert p.count() == 5 and p.x[p.alive].min() > 2.5                      # they moved (and drag slowed them a little)
    p.update(0.6)
    assert p.count() == 0
    p.spawn(0, 0, 0, 0, 1.0, 0.5, 1.0, "dust", 1.0)
    assert p.count() == 1


def test_particle_view_filters_by_rectangle():
    p = ParticleField(10)
    p.spawn(1.0, 1.0, 0, 0, 5.0, 1, 1, "smoke", 1)
    p.spawn(50.0, 1.0, 0, 0, 5.0, 1, 1, "smoke", 1)
    assert list(p.view(0, 10, 0, 10)) == [0] and len(p.view(-100, 100, -100, 100)) == 2


def test_skid_marks_connect_consecutive_sliding_points_only():
    s = SkidMarks(max_segments=3)
    s.add(0, 0, 0.0, 0.0, True, 0.0)                     # first point: nothing to connect yet
    assert not s.segments
    s.add(0, 0, 1.0, 0.0, True, 0.1)
    s.add(0, 0, 2.0, 0.0, True, 0.2)
    assert len(s.segments) == 2
    s.add(0, 0, 3.0, 0.0, False, 0.3)                    # the slide ends: the chain is cut
    s.add(0, 0, 9.0, 0.0, True, 0.4)
    assert len(s.segments) == 2                          # no segment from (3,0) to (9,0)
    for k in range(10):                                  # the cap forgets the oldest
        s.add(1, 0, float(k), 5.0, True, 1.0)
    assert len(s.segments) == 3


def test_shake_decays_to_zero():
    sh = Shake()
    sh.kick(1.5)
    sh.kick(0.5)                                         # a weaker kick does not reduce it
    assert sh.amplitude_m == 1.5
    for _ in range(120):
        sh.decay(1 / 60)
    assert sh.amplitude_m == 0.0
    assert Shake().offset(np.random.default_rng(0)) == (0.0, 0.0)


def fake_car(index=0, state="racing", sliding=True, speed=30.0, surface=ROAD_ID):
    phys = SimpleNamespace(speed=speed, sliding=sliding, vx=speed, vy=0.0,
                           wheel_positions=lambda: [(1.0, 0.8), (1.0, -0.8), (-1.0, 0.8), (-1.0, -0.8)])
    env = SimpleNamespace(car=phys, _wheels=np.full(4, surface))
    return SimpleNamespace(index=index, state=state, env=env, pose=(0.0, 0.0, 0.0))


def test_sliding_on_tarmac_makes_smoke_and_marks_but_grass_makes_dust_only():
    fx = Effects(CFG, seed=0)
    car = fake_car()
    for _ in range(120):
        car.env.car.wheel_positions = (lambda t=fx.t: [(t * 30 + dx, dy) for dx, dy in ((1, .8), (1, -.8), (-1, .8), (-1, -.8))])
        fx.step(1 / 60, [car])
    assert len(fx.skids.segments) > 50 and fx.particles.count() > 10
    assert set(fx.particles.kind[fx.particles.alive]) == {0}               # smoke only

    fx2 = Effects(CFG, seed=0)
    grass = fake_car(sliding=True, surface=GRASS_ID)
    for _ in range(120):
        fx2.step(1 / 60, [grass])
    assert not fx2.skids.segments                                            # no marks on grass
    assert set(fx2.particles.kind[fx2.particles.alive]) == {1}               # dust


def test_slow_cars_leave_nothing_and_a_crash_throws_sparks_and_shakes_the_followed_car():
    fx = Effects(CFG, seed=0)
    slow = fake_car(speed=1.0, sliding=True)
    for _ in range(60):
        fx.step(1 / 60, [slow])
    assert not fx.skids.segments
    car = fake_car(sliding=False)
    fx.step(1 / 60, [car], follow=0)
    car.state = "respawn"
    fx.step(1 / 60, [car], follow=0)
    assert 0.8 * CFG["crash_shake_m"] < fx.shake.amplitude_m <= CFG["crash_shake_m"]      # kicked, then one step of decay
    assert set(fx.particles.kind[fx.particles.alive]) == {2}
    other = Effects(CFG, seed=0)
    c2 = fake_car(sliding=False)
    other.step(1 / 60, [c2], follow=5)
    c2.state = "respawn"
    other.step(1 / 60, [c2], follow=5)
    assert other.shake.amplitude_m == 0.0                                    # only the followed car shakes the view


def test_effects_draw_headless_without_error():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from racer.render.camera import Camera

    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    dcfg = load_preset("demo", "render")["debug"]
    cam = Camera((640, 480), dcfg)
    cam.snap(0.0, 0.0)
    fx = Effects(CFG, seed=0)
    fx.step(1 / 60, [fake_car()])
    fx.step(1 / 60, [fake_car()])
    fx.particles.spawn(1.0, 1.0, 0, 0, 1.0, 1.0, 2.0, "smoke", 0.6)
    fx.particles.spawn(2.0, 1.0, 1, 1, 0.5, 0.1, 0.05, "spark", 1.0)
    before = pygame.image.tobytes(screen, "RGB")
    fx.draw_under(screen, cam)
    fx.draw_over(screen, cam)
    assert pygame.image.tobytes(screen, "RGB") != before
    pygame.quit()
