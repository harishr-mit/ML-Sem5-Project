"""Engine and tyre sound logic (PROJECT_CONTEXT §10.4) without any audio device: pure and testable.

``EngineSound.step`` turns the car's throttle and rpm ratio into *commands* (play this clip from this
offset with this fade). :class:`racer.render.audio.AudioController` executes them on the mixer.

Clips (``assets/sounds/engine_map.json``): ``rev_up`` rises 1.19 → 2.88 in 3.06 s, ``high_loop`` holds
2.88, ``rev_down`` falls 2.34 → 0.98 and joins ``idle_loop`` at ``idle_loop_entry_s``. A clip plays at
its own recorded speed, so when its pitch drifts from the car's rpm by more than ``drift_tol`` the clip
is re-sought to the offset that matches (with a short crossfade).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Cmd:
    """One thing for the mixer to do. ``kind``: ``engine`` (replace the engine voice) or ``stop``."""

    kind: str
    clip: str = ""            # idle_loop | rev_up | high_loop | rev_down
    offset_s: float = 0.0
    fade_s: float = 0.0
    loop: bool = False


class Curve:
    """A monotonic rpm-ratio curve of a clip: time → ratio and ratio → time."""

    def __init__(self, points: Sequence[Sequence[float]]) -> None:
        pts = np.asarray(points, dtype=float)
        self.t, self.r = pts[:, 0], pts[:, 1]
        self._inc = self.r[-1] >= self.r[0]

    def ratio_at(self, t: float) -> float:
        return float(np.interp(t, self.t, self.r))

    def time_for(self, ratio: float) -> float:
        """Offset whose ratio is ``ratio``; clamped to the clip's ends."""
        if self._inc:
            return float(np.interp(ratio, self.r, self.t))
        return float(np.interp(ratio, self.r[::-1], self.t[::-1]))


class EngineSound:
    def __init__(self, engine_map: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
        self.up = Curve(engine_map["rev_up"]["curve"])
        self.down = Curve(engine_map["rev_down"]["curve"])
        self.up_len = float(engine_map["rev_up"]["duration_s"])
        self.down_len = float(engine_map["rev_down"]["duration_s"])
        self.idle_entry = float(engine_map["rev_down"]["idle_loop_entry_s"])
        self.high_ratio = float(engine_map["high_loop"]["rpm_ratio"])
        self.cfg = cfg
        self.state = "idle"
        self.clip_t = 0.0
        self._pressed = False
        self._since_seek = 1e9
        self.started = False

    # ---------------------------------------------------------------- helpers
    def throttle_pressed(self, throttle: float) -> bool:
        c = self.cfg
        self._pressed = throttle > (c["throttle_off"] if self._pressed else c["throttle_on"])
        return self._pressed

    def clip_ratio(self) -> float:
        """The pitch (rpm ratio) of what is playing now."""
        if self.state == "up":
            return self.up.ratio_at(self.clip_t)
        if self.state == "down":
            return self.down.ratio_at(self.clip_t)
        return self.high_ratio if self.state == "high" else 1.0

    def _go_up(self, rpm: float, fade: float) -> Cmd:
        self.state, self.clip_t, self._since_seek = "up", self.up.time_for(rpm), 0.0
        return Cmd("engine", "rev_up", self.clip_t, fade)

    def _go_down(self, rpm: float, fade: float) -> Cmd:
        self.state, self._since_seek = "down", 0.0
        self.clip_t = 0.0 if rpm >= self.down.r[0] else self.down.time_for(rpm)
        return Cmd("engine", "rev_down", self.clip_t, fade)

    # ---------------------------------------------------------------- one step
    def step(self, dt: float, throttle: float, rpm: float) -> list[Cmd]:
        c = self.cfg
        on = self.throttle_pressed(throttle)
        self._since_seek += dt
        cmds: list[Cmd] = []
        if not self.started:                           # first call: start from the right place, no fade
            self.started = True
            if on:
                return [self._go_up(rpm, 0.0)]
            if rpm <= c["idle_ratio_max"]:
                self.state = "idle"
                return [Cmd("engine", "idle_loop", 0.0, 0.0, True)]
            return [self._go_down(rpm, 0.0)]

        if self.state in ("up", "down"):
            self.clip_t += dt
        drift = abs(self.clip_ratio() - rpm) / max(rpm, 1e-6)
        can_seek = self._since_seek >= c["min_reseek_s"]

        if on:
            if self.state in ("idle", "down"):
                cmds.append(self._go_up(rpm, c["crossfade_s"]))
            elif self.state == "up":
                if self.clip_t >= self.up_len - 0.05:
                    self.state = "high"
                    cmds.append(Cmd("engine", "high_loop", 0.0, c["high_crossfade_s"], True))
                elif drift > c["drift_tol"] and can_seek:
                    cmds.append(self._go_up(rpm, c["crossfade_s"]))
            elif self.state == "high" and rpm < self.high_ratio * (1 - c["drift_tol"]) and can_seek:
                cmds.append(self._go_up(rpm, c["crossfade_s"]))      # an upshift dropped the rpm
        else:
            if self.state in ("up", "high"):
                cmds.append(self._go_down(rpm, c["crossfade_s"]))
            elif self.state == "down":
                if self.clip_t >= self.idle_entry or rpm <= c["idle_ratio_max"] and self.clip_ratio() <= 1.02:
                    self.state = "idle"
                    cmds.append(Cmd("engine", "idle_loop", 0.0, c["crossfade_s"], True))
                elif drift > c["drift_tol"] and can_seek:
                    cmds.append(self._go_down(rpm, c["crossfade_s"]))
            elif self.state == "idle" and rpm > c["idle_ratio_max"] * 1.1:
                cmds.append(self._go_down(rpm, c["crossfade_s"]))    # rolling: the engine follows the wheels
        return cmds

    def gain(self, throttle: float) -> float:
        """Volume factor from the throttle (louder under load)."""
        g = float(self.cfg["throttle_gain"])
        return 1.0 - g + g * float(np.clip(throttle, 0.0, 1.0))


class TyreSound:
    """Screech loop volume follows the slip; a short one-shot on the onset of a slide or hard braking."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.volume = 0.0
        self._above = False
        self._cool = 0.0
        self._n = 0

    def intensity(self, v_lat: float, sliding: bool) -> float:
        c = self.cfg
        i = min(abs(v_lat) / float(c["slip_ref_ms"]), 1.0)
        return max(i, float(c["sliding_floor"])) if sliding else i

    def step(self, dt: float, v_lat: float, sliding: bool, brake: float, speed: float) -> tuple[float, int | None]:
        """Returns ``(loop volume 0-1, one-shot clip index or None)``."""
        c = self.cfg
        target = self.intensity(v_lat, sliding)
        k = min(dt / max(float(c["fade_s"]), 1e-3), 1.0)
        self.volume += (target - self.volume) * k
        self._cool = max(self._cool - dt, 0.0)
        hard = brake >= c["hard_brake"] and speed > c["hard_brake_speed_ms"]
        above = target >= c["onset"] or hard
        shot = None
        if above and not self._above and self._cool <= 0.0:
            shot = self._n % 2
            self._n += 1
            self._cool = float(c["oneshot_cooldown_s"])
        self._above = above
        return float(np.clip(self.volume, 0.0, 1.0)), shot
