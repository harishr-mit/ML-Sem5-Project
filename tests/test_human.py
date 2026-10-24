"""Human vs AI (T127): input reading, ABS, the picker and a headless race with a human car."""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from racer.config import load_preset
from racer.render.human import HumanDriver, read_input

ROOT = Path(__file__).resolve().parents[1]
HCFG = load_preset("demo", "arena")["human"]


class Keys(dict):
    """Stand-in for pygame.key.get_pressed(): missing keys are not pressed."""

    def __getitem__(self, k):
        return self.get(k, False)


def keys(*names):
    import pygame
    return Keys({getattr(pygame, f"K_{n}"): True for n in names})


def test_keyboard_arrows_and_wasd_give_the_same_inputs():
    assert read_input(keys("UP"), None, HCFG) == (0.0, 1.0)
    assert read_input(keys("w", "a"), None, HCFG) == (1.0, 1.0)          # left = positive steer (counter-clockwise)
    assert read_input(keys("DOWN", "RIGHT"), None, HCFG) == (-1.0, -1.0)
    assert read_input(keys("a", "d"), None, HCFG)[0] == 0.0               # both cancel
    assert read_input(Keys(), None, HCFG) == (0.0, 0.0)


def test_gamepad_stick_triggers_and_deadzone():
    dz = HCFG["deadzone"]
    assert read_input(Keys(), {"steer": 0.5 * dz, "gas": 0.0, "brake": 0.0}, HCFG) == (0.0, 0.0)
    steer, gas = read_input(Keys(), {"steer": 1.0, "gas": 1.0, "brake": 0.0}, HCFG)
    assert steer == -1.0 and gas == 1.0                                  # stick right = clockwise = negative steer
    assert read_input(Keys(), {"steer": 0.0, "gas": 0.0, "brake": 0.5}, HCFG)[1] == pytest.approx(-0.5)
    assert read_input(keys("UP"), {"steer": 1.0, "gas": 0.0, "brake": 0.0}, HCFG)[0] == 0.0     # keys win over the pad


def fake_env(limit):
    car = SimpleNamespace(abs_brake_limit=lambda wheels, margin: limit)
    return SimpleNamespace(car=car, _wheels=np.zeros(4))


def test_abs_caps_a_full_brake_but_never_below_the_minimum():
    d = HumanDriver(HCFG, abs_cfg={"enabled": True, "margin": 0.97, "min_brake": 0.25})
    d.set_input(0.0, -1.0)
    out = d.act(np.zeros((1, 3)), [fake_env(0.6)])
    assert out.shape == (1, 2) and out[0, 1] == pytest.approx(-0.6) and d.abs_active
    assert d.act(np.zeros((1, 3)), [fake_env(0.1)])[0, 1] == pytest.approx(-0.25)       # mid-slide: min brake
    assert d.act(np.zeros((1, 3)), [fake_env(1.0)])[0, 1] == pytest.approx(-1.0) and not d.abs_active
    d.set_input(0.3, 1.0)
    assert d.act(np.zeros((1, 3)), [fake_env(0.2)])[0].tolist() == pytest.approx([0.3, 1.0])    # gas untouched
    d.toggle_abs()
    d.set_input(0.0, -1.0)
    assert d.act(np.zeros((1, 3)), [fake_env(0.2)])[0, 1] == -1.0 and not d.abs_active


def test_human_picker_cycles_the_cars_physics_and_starts_without_models():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from racer.render.human import HumanPicker

    pygame.init()
    colours = {k: tuple(v) for k, v in load_preset("demo", "render")["debug"]["colours"].items()}
    pk = HumanPicker([], [("showcase", "technical")], (1280, 800), pygame.font.Font(None, 20), colours,
                     load_preset("demo", "arena")["palette"], max_cars=8,
                     physics_options=["default", "downforce"])
    screen = pygame.display.set_mode((1280, 800))
    pk.draw(screen)
    assert pk.physics == "default"
    pk.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c, unicode="C", mod=pygame.KMOD_SHIFT))
    assert pk.physics == "downforce"
    pk.handle(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r", mod=0))
    assert pk.done is not None and pk.done.human == "downforce" and pk.done.models == []
    pygame.quit()


def test_human_race_headless_with_the_baseline(tmp_path):
    out = tmp_path / "human.png"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "demo.py"), "--human", "downforce", "--baseline",
                        "--track", "technical", "--screenshot", str(out), "--frames", "300"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.is_file() and out.stat().st_size > 10_000
