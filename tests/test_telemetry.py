"""Telemetry (T135): a recorded lap of the baseline, lap slicing, braking zones."""
import numpy as np
import pytest

from racer.agents.telemetry import Telemetry, braking_zones, drive


@pytest.fixture(scope="module")
def lap():
    return drive("baseline", "training_ground", laps=2, record_obs=True)


def test_the_baseline_drives_two_laps_and_every_channel_is_recorded(lap):
    assert len(lap.lap_times) == 2 and lap.termination is None
    assert 30 < lap.lap_times[1] < 80 and abs(lap.lap_times[1] - lap.lap_times[0]) < 1.0   # a steady driver: equal laps
    n = len(lap)
    assert all(len(v) == n for v in lap.data.values()) and n > 500
    assert lap.obs.shape[0] == n
    assert lap.data["t"][0] == pytest.approx(lap.dt) and np.all(np.diff(lap.data["t"]) > 0)
    assert lap.data["speed"].max() > 25 and np.abs(lap.data["lat_g"]).max() < 1.5       # grip-limited, roughly 1 g
    assert lap.data["gas"].min() < 0 < lap.data["gas"].max()                            # it brakes and accelerates
    assert lap.length > 1000


def test_lap_slices_cut_at_the_line_and_keep_the_lap_time(lap):
    second = lap.lap(1)
    first = lap.lap(0)
    assert len(first) + len(second) <= len(lap) and len(first) > 100
    assert second.lap_times == [lap.lap_times[1]]
    assert second.data["t"][0] == 0.0                                                    # time restarts at the line
    assert second.data["s"][0] < 30 and second.data["s"][-1] > lap.length - 30         # one lap of distance
    assert np.all(np.diff(second.data["s"]) > 0)                                        # monotonic: no wrap inside a lap
    assert lap.lap(-1).lap_times == second.lap_times
    assert abs(len(second) * second.dt - second.lap_times[0]) < 0.5


def test_braking_zones_are_stretches_of_the_lap_distance(lap):
    zones = lap.lap(1).braking_zones()
    assert zones and all(0 <= a < b <= lap.length for a, b in zones)
    assert all(b - a >= 5.0 for a, b in zones)
    s = np.arange(0, 100.0, 1.0)
    brake = np.zeros_like(s)
    brake[10:30] = 1.0
    brake[50:52] = 1.0                                                                  # too short to count
    assert braking_zones(s, brake) == [(10.0, 29.0)]
    assert braking_zones(s, np.zeros_like(s)) == []
    assert braking_zones(s, np.ones_like(s)) == [(0.0, 99.0)]


def test_frame_has_kmh_and_the_whole_drive_is_returned_without_a_completed_lap():
    crashy = drive("baseline", "training_ground", laps=1, max_steps=40)               # 2 s: no lap finished
    assert crashy.lap_times == [] and crashy.lap() is crashy
    assert crashy.frame()["kmh"].iloc[-1] == pytest.approx(crashy.data["speed"][-1] * 3.6)
    assert isinstance(crashy, Telemetry)
