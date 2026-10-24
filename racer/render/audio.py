"""Sound for the game windows (PROJECT_CONTEXT §10.4): the followed car's engine and tyres on the mixer.

The decisions live in :mod:`racer.render.engine_sound` and :mod:`racer.render.gearbox`; this module
decodes the clips once, executes the engine commands (seeking by slicing the sample arrays) and sets
the volumes. Without an audio device or without the assets the controller reports ``available =
False`` and every call is a no-op: the game never depends on sound.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pygame

from racer.config import PROJECT_ROOT
from racer.render.engine_sound import Cmd, EngineSound, TyreSound
from racer.render.gearbox import Gearbox

SOUNDS = PROJECT_ROOT / "assets" / "sounds"
FREQ = 44100
CLIP_GAIN = {"idle_loop": "idle_loop", "rev_up": "rev_up", "high_loop": "high_loop", "rev_down": "rev_down"}


@dataclass(frozen=True)
class CarAudio:
    """What the sound needs from the followed car."""

    speed: float
    throttle: float
    brake: float
    v_lat: float
    sliding: bool


class AudioController:
    def __init__(self, cfg: Mapping[str, Any], enabled: bool = True, sounds: Path = SOUNDS) -> None:
        self.cfg = cfg
        self.available = False
        self.enabled = False
        self.arrays: dict[str, np.ndarray] = {}
        self._key: Any = None
        self._clip = "idle_loop"
        try:
            self._load(sounds)
            self.available = True
        except (pygame.error, OSError, KeyError, ValueError) as e:
            print(f"sound off: {e}", flush=True)
            return
        self.gearbox = Gearbox(cfg["gearbox"])
        self.engine = EngineSound(self.engine_map, cfg["engine"])
        self.tyres = TyreSound(cfg["tyres"])
        self.ch = [pygame.mixer.Channel(0), pygame.mixer.Channel(1)]     # engine voices (crossfade)
        self.ch_tyre = pygame.mixer.Channel(2)
        self.ch_shot = pygame.mixer.Channel(3)
        self._voice = 0
        self.enabled = bool(enabled)

    # ------------------------------------------------------------------ setup
    def _load(self, sounds: Path) -> None:
        if pygame.mixer.get_init():
            pygame.mixer.quit()
        pygame.mixer.init(FREQ, -16, 1, 512)
        pygame.mixer.set_num_channels(8)
        self.freq = pygame.mixer.get_init()[0]
        self.engine_map = json.loads((sounds / "engine_map.json").read_text(encoding="utf-8"))
        files = {k: self.engine_map[k]["file"] for k in ("idle_loop", "high_loop", "rev_up", "rev_down")}
        files.update({"tyre_loop": "tyre_screech_loop.ogg", "tyre_short_1": "tyre_screech_short_1.ogg",
                      "tyre_short_2": "tyre_screech_short_2.ogg"})
        for name, f in files.items():
            self.arrays[name] = pygame.sndarray.array(pygame.mixer.Sound(str(sounds / f)))

    def close(self) -> None:
        if self.available and pygame.mixer.get_init():
            pygame.mixer.stop()
            pygame.mixer.quit()
        self.available = self.enabled = False

    # ------------------------------------------------------------------ control
    def toggle(self) -> None:
        if not self.available:
            return
        self.enabled = not self.enabled
        if self.enabled:
            self.reset()
        else:
            pygame.mixer.stop()

    def follow(self, key: Any) -> None:
        """Call every frame with the followed car's key; a new car restarts engine and gearbox."""
        if key != self._key:
            self._key = key
            if self.available:
                self.reset()

    def reset(self) -> None:
        if not self.available:
            return
        pygame.mixer.stop()
        self.gearbox.reset()
        self.engine = EngineSound(self.engine_map, self.cfg["engine"])
        self.tyres = TyreSound(self.cfg["tyres"])

    def pause(self, paused: bool) -> None:
        """Silence while the race is paused; resume where it was."""
        if self.available and self.enabled:
            (pygame.mixer.pause if paused else pygame.mixer.unpause)()

    def stop(self) -> None:
        if self.available:
            pygame.mixer.stop()

    @property
    def label(self) -> str:
        if not self.available:
            return "sound: no device"
        return "sound: on" if self.enabled else "sound: off (M)"

    @property
    def gear(self) -> int:
        return self.gearbox.gear if self.available else 0

    # ------------------------------------------------------------------ per frame
    def update(self, dt: float, car: CarAudio) -> None:
        if not (self.available and self.enabled):
            return
        master = float(self.cfg["master_volume"])
        gains = self.cfg["clips"]
        throttle_on = car.throttle > float(self.cfg["engine"]["throttle_off"])
        rpm, _ = self.gearbox.update(dt, car.speed, throttle_on)
        for cmd in self.engine.step(dt, car.throttle, rpm):
            self._run(cmd)
        vol = master * float(gains[CLIP_GAIN[self._clip]]) * self.engine.gain(car.throttle)
        for ch in self.ch:
            if ch.get_busy():
                ch.set_volume(vol)
        loop_v, shot = self.tyres.step(dt, car.v_lat, car.sliding, car.brake, car.speed)
        if not self.ch_tyre.get_busy():
            self.ch_tyre.play(pygame.sndarray.make_sound(self.arrays["tyre_loop"]), loops=-1)
        self.ch_tyre.set_volume(master * float(gains["tyre_loop"]) * loop_v)
        if shot is not None:
            snd = pygame.sndarray.make_sound(self.arrays[f"tyre_short_{shot + 1}"])
            snd.set_volume(master * float(gains["tyre_short"]))
            self.ch_shot.play(snd)

    def _run(self, cmd: Cmd) -> None:
        if cmd.kind != "engine":
            return
        arr = self.arrays[cmd.clip]
        i = min(int(cmd.offset_s * self.freq), max(len(arr) - 1, 0))
        snd = pygame.sndarray.make_sound(np.ascontiguousarray(arr[i:]))
        fade = int(cmd.fade_s * 1000)
        new, old = self.ch[self._voice], self.ch[self._voice ^ 1]
        new.play(snd, loops=-1 if cmd.loop else 0, fade_ms=fade)
        if old.get_busy():
            old.fadeout(fade)
        self._voice ^= 1
        self._clip = cmd.clip
