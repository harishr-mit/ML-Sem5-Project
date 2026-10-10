"""Unit tests for surfaces and the car step (quick checks; scripts/physics_sanity.py is the full suite)."""
import math
import subprocess
import sys

import pytest

from racer.config import load_preset
from racer.physics.car import Car, PhysicsParams
from racer.physics.surfaces import GRASS_ID, KERB_ID, ROAD_ID, Kerb, Road, Surface, SurfaceSet

ROAD4 = (ROAD_ID,) * 4


@pytest.fixture(scope="module")
def params():
    return PhysicsParams.from_config(load_preset("physics", "default"))


def test_surface_inheritance(params):
    sf = SurfaceSet(params.surfaces)
    assert type(sf.grass) is Surface and isinstance(sf.road, Road) and isinstance(sf.kerb, Kerb)
    assert sf.kerb.is_road and sf.road.is_road and not sf.grass.is_road
    assert sf.get(-1) is sf.grass
    assert sf.road.mu_static == pytest.approx(1.10)


def test_surface_fallback_to_parent():
    sf = SurfaceSet({"grass": {"mu_static": 0.5, "mu_kinetic": 0.4, "rolling_res": 0.1},
                     "road": {"mu_static": 1.0, "mu_kinetic": 0.9, "rolling_res": 0.01},
                     "kerb": {"mu_static": 0.8}})
    assert sf.kerb.mu_static == 0.8 and sf.kerb.mu_kinetic == 0.9 and sf.kerb.rolling_res == 0.01


def test_full_throttle_accelerates_straight(params):
    car = Car(params)
    for _ in range(params.physics_hz * 5):
        car.step(0.0, 1.0, ROAD4)
    assert car.speed > 25.0
    assert abs(car.y) < 1e-9 and car.x > 50.0
    assert not car.sliding


def test_no_reverse_when_braking(params):
    car = Car(params)
    car.reset(0, 0, 0, 5.0)
    for _ in range(params.physics_hz * 3):
        car.step(0.0, -1.0, ROAD4)
    assert car.speed == 0.0
    assert car.x > 0.0


def test_steering_turns_left_for_positive_steer(params):
    car = Car(params)
    car.reset(0, 0, 0, 10.0)
    for _ in range(params.physics_hz):
        car.step(1.0, 0.2, ROAD4)
    assert car.theta > 0.3 and car.y > 0.0


def test_grass_is_slower(params):
    speeds = {}
    for name, sid in (("road", ROAD_ID), ("grass", GRASS_ID), ("kerb", KERB_ID)):
        car = Car(params)
        for _ in range(params.physics_hz * 20):
            car.step(0.0, 1.0, (sid,) * 4)
        speeds[name] = car.speed
    assert speeds["grass"] < 0.8 * speeds["road"]
    assert speeds["kerb"] < speeds["road"]


def test_physics_does_not_import_pygame():
    code = ("import sys, racer.physics.car, racer.physics.surfaces; "
            "sys.exit(1 if 'pygame' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
