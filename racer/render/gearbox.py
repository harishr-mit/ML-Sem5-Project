"""Cosmetic gearbox (PROJECT_CONTEXT §10.4): engine rpm from speed, for sound and HUD only.

The physics has no gears (it is power-limited). ``rpm_ratio`` is the engine pitch over the idle pitch
(idle = 1.0), the unit of ``assets/sounds/engine_map.json``. In gear *g* it is ``f_g · speed``; the
factors are chosen so the top gear reaches ``shift_up_ratio`` at ``top_speed_ms`` and every upshift
lands at ``after_shift_ratio``.
"""
from __future__ import annotations

from typing import Any, Mapping


class Gearbox:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.n = int(cfg["gears"])
        self.up = float(cfg["shift_up_ratio"])
        self.after = float(cfg["after_shift_ratio"])
        self.down = float(cfg["shift_down_ratio"])
        self.clutch = float(cfg["clutch_min_ratio"])
        self.cooldown = float(cfg["shift_cooldown_s"])
        step = self.after / self.up                      # factor between neighbouring gears
        top = self.up / float(cfg["top_speed_ms"])
        self.factors = [top / step ** (self.n - 1 - g) for g in range(self.n)]     # 1st gear first
        self.gear = 1
        self._since_shift = self.cooldown

    def reset(self) -> None:
        self.gear = 1
        self._since_shift = self.cooldown

    def update(self, dt: float, speed: float, throttle_on: bool) -> tuple[float, bool]:
        """Advance the gearbox; returns ``(rpm_ratio, shifted_this_step)``."""
        self._since_shift += dt
        speed = max(speed, 0.0)
        shifted = False
        r = self.factors[self.gear - 1] * speed
        if self._since_shift >= self.cooldown:
            if r >= self.up and self.gear < self.n:
                self.gear += 1
                shifted = True
            elif r < self.down and self.gear > 1:
                self.gear -= 1
                shifted = True
            if shifted:
                self._since_shift = 0.0
                r = self.factors[self.gear - 1] * speed
        floor = self.clutch if (self.gear == 1 and throttle_on) else 1.0
        return max(r, floor), shifted
