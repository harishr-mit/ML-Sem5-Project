"""Audio logic (T125): cosmetic gearbox, engine and tyre sound commands, the mixer controller (dummy device)."""
import json
import os

import pytest

from racer.config import PROJECT_ROOT, load_preset
from racer.render.engine_sound import Cmd, Curve, EngineSound, TyreSound
from racer.render.gearbox import Gearbox

CFG = load_preset("demo", "audio")
MAP = json.loads((PROJECT_ROOT / "assets" / "sounds" / "engine_map.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- gearbox
def drive(box, speeds, dt=1 / 60):
    out = []
    for v in speeds:
        out.append(box.update(dt, v, True))
    return out


def test_gearbox_top_gear_reaches_the_top_of_rev_up_at_top_speed():
    box = Gearbox(CFG["gearbox"])
    assert len(box.factors) == 5
    assert box.factors[-1] * 70.0 == pytest.approx(CFG["gearbox"]["shift_up_ratio"])
    ratios = [box.factors[i + 1] / box.factors[i] for i in range(4)]
    assert ratios == pytest.approx([CFG["gearbox"]["after_shift_ratio"] / CFG["gearbox"]["shift_up_ratio"]] * 4)   # taller gears


def test_gearbox_accelerating_shifts_up_and_rpm_stays_in_band():
    box = Gearbox(CFG["gearbox"])
    res = drive(box, [i * 0.05 for i in range(1400)])          # 0 → 70 m/s over 23 s
    gears = [box.gear]
    rpms = [r for r, _ in res]
    assert box.gear == 5 and sum(s for _, s in res) == 4         # four upshifts, no hunting
    assert max(rpms) <= CFG["gearbox"]["shift_up_ratio"] + 0.2 and min(rpms) >= 1.0
    after = [r for (r, s) in res if s]
    assert all(a == pytest.approx(CFG["gearbox"]["after_shift_ratio"], abs=0.15) for a in after)
    assert gears[0] == 5


def test_gearbox_idle_clutch_and_downshift():
    box = Gearbox(CFG["gearbox"])
    assert box.update(0.016, 0.0, False)[0] == 1.0               # standstill, no throttle: idle
    assert box.update(0.016, 0.0, True)[0] == CFG["gearbox"]["clutch_min_ratio"]    # throttle: clutch slip
    drive(box, [i * 0.05 for i in range(1200)])
    top = box.gear
    box.update(1.0, 5.0, False)                                  # braking hard to 5 m/s
    assert box.gear < top


# ----------------------------------------------------------------------------- curves
def test_curve_round_trip_for_rising_and_falling_clips():
    up, down = Curve(MAP["rev_up"]["curve"]), Curve(MAP["rev_down"]["curve"])
    for t in (0.5, 1.5, 2.5):
        assert up.time_for(up.ratio_at(t)) == pytest.approx(t, abs=0.03)
    for t in (0.5, 3.0, 6.0):
        assert down.time_for(down.ratio_at(t)) == pytest.approx(t, abs=0.1)
    assert up.time_for(0.5) == up.t[0] and down.time_for(9.0) == down.t[0]     # clamped to the ends


# ----------------------------------------------------------------------------- engine
def run_engine(eng, steps, throttle, rpm, dt=1 / 60):
    cmds = []
    for _ in range(steps):
        cmds += eng.step(dt, throttle, rpm() if callable(rpm) else rpm)
    return cmds


def test_engine_starts_at_idle_and_revs_up_from_standstill():
    eng = EngineSound(MAP, CFG["engine"])
    first = eng.step(1 / 60, 0.0, 1.0)
    assert first == [Cmd("engine", "idle_loop", 0.0, 0.0, True)]
    nxt = eng.step(1 / 60, 1.0, 1.2)                              # throttle pressed from standstill
    assert nxt[0].clip == "rev_up" and nxt[0].offset_s == pytest.approx(0.0, abs=0.05)


def test_engine_throttle_while_moving_seeks_into_the_middle_of_rev_up():
    eng = EngineSound(MAP, CFG["engine"])
    eng.step(1 / 60, 0.0, 2.0)
    cmds = eng.step(1 / 60, 1.0, 2.0)
    assert cmds[0].clip == "rev_up" and 1.0 < cmds[0].offset_s < 2.5     # rpm 2.0 is mid-clip


def test_engine_reaches_the_top_then_holds_the_high_loop():
    eng = EngineSound(MAP, CFG["engine"])
    eng.step(1 / 60, 0.0, 1.0)
    cmds = run_engine(eng, 60 * 4, 1.0, 2.88)                    # rpm at the top of the clip
    assert [c.clip for c in cmds][-1] == "high_loop" and cmds[-1].loop
    assert eng.state == "high"


def test_engine_upshift_drops_the_rpm_and_reseeks_rev_up():
    eng = EngineSound(MAP, CFG["engine"])
    eng.step(1 / 60, 0.0, 1.0)
    run_engine(eng, 60 * 4, 1.0, 2.88)
    cmds = run_engine(eng, 30, 1.0, 2.0)                         # after the shift the rpm is 2.0
    assert cmds and cmds[0].clip == "rev_up" and cmds[0].offset_s < 2.5


def test_engine_lifting_off_runs_rev_down_into_idle():
    eng = EngineSound(MAP, CFG["engine"])
    eng.step(1 / 60, 0.0, 1.0)
    run_engine(eng, 60 * 4, 1.0, 2.88)
    off = eng.step(1 / 60, 0.0, 2.88)
    assert off[0].clip == "rev_down" and off[0].offset_s == 0.0   # above the clip's top: start at 0
    rest = run_engine(eng, 60 * 9, 0.0, 1.0)
    assert any(c.clip == "idle_loop" and c.loop for c in rest) and eng.state == "idle"


def test_engine_does_not_chatter_with_a_wobbling_throttle():
    eng = EngineSound(MAP, CFG["engine"])
    eng.step(1 / 60, 0.0, 1.0)
    cmds = []
    for i in range(60 * 3):
        gas = 0.4 + 0.08 * (-1) ** i                              # 0.32 / 0.48: always above the hysteresis
        cmds += eng.step(1 / 60, gas, 1.6 + 0.4 * i / 180)
    assert len(cmds) <= 12                                        # a few re-seeks, not one per frame


def test_throttle_gain_is_louder_under_load():
    eng = EngineSound(MAP, CFG["engine"])
    assert eng.gain(1.0) == pytest.approx(1.0) and eng.gain(0.0) < eng.gain(1.0)


# ----------------------------------------------------------------------------- tyres
def test_tyres_loop_follows_slip_and_one_shot_fires_once_per_onset():
    t = TyreSound(CFG["tyres"])
    vols = [t.step(1 / 60, 0.0, False, 0.0, 20.0)[0] for _ in range(30)]
    assert max(vols) == 0.0
    shots = []
    for i in range(120):
        v, s = t.step(1 / 60, 4.0, True, 0.0, 30.0)
        shots.append(s)
    assert v > 0.9 and [s for s in shots if s is not None] == [0]          # one shot at the onset
    for _ in range(60):
        v, _ = t.step(1 / 60, 0.0, False, 0.0, 30.0)
    assert v < 0.05
    again = [t.step(1 / 60, 4.0, True, 0.0, 30.0)[1] for _ in range(200)]
    assert [s for s in again if s is not None] == [1]                     # next onset alternates the clip


def test_hard_braking_also_triggers_a_shot():
    t = TyreSound(CFG["tyres"])
    shots = [t.step(1 / 60, 0.0, False, 1.0, 40.0)[1] for _ in range(10)]
    assert [s for s in shots if s is not None] == [0]
    assert t.step(1 / 60, 0.0, False, 1.0, 5.0)[1] is None                # too slow to count


# ----------------------------------------------------------------------------- mixer
def test_audio_controller_with_a_dummy_device(monkeypatch):
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from racer.render.audio import AudioController, CarAudio

    ctl = AudioController(CFG, enabled=True)
    assert ctl.available
    state = CarAudio(speed=0.0, throttle=0.0, brake=0.0, v_lat=0.0, sliding=False)
    for i in range(30):
        ctl.update(1 / 60, state)
    fast = CarAudio(speed=30.0, throttle=1.0, brake=0.0, v_lat=5.0, sliding=True)
    for i in range(240):
        ctl.update(1 / 60, fast)
    assert ctl.engine.state in ("up", "high")
    ctl.toggle()
    assert not ctl.enabled
    ctl.update(1 / 60, fast)                                       # silent and harmless while off
    ctl.toggle()
    assert ctl.enabled
    ctl.close()


def test_audio_controller_degrades_without_a_device(monkeypatch):
    monkeypatch.setenv("SDL_AUDIODRIVER", "no-such-driver")
    from racer.render.audio import AudioController, CarAudio

    ctl = AudioController(CFG, enabled=True)
    assert not ctl.available and not ctl.enabled
    ctl.update(1 / 60, CarAudio(10.0, 1.0, 0.0, 0.0, False))      # never raises
    ctl.toggle()
    assert not ctl.enabled
    ctl.close()
    assert os.environ["SDL_AUDIODRIVER"] == "no-such-driver"
