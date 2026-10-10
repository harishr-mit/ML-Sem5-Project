"""Arcade-sim car dynamics (PROJECT_CONTEXT §5.2–5.4).

One scalar car in plain Python floats: cheaper than numpy for a single body. World axes
are x right, y up; heading ``theta`` is counter-clockwise from +x. The car knows nothing
about tracks: the caller passes the surface id under each wheel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from racer.config import from_dict
from racer.physics.surfaces import GRASS_ID, SurfaceSet


@dataclass(frozen=True)
class PhysicsParams:
    """Mirror of ``configs/physics/*.yaml``. Every field is required."""

    physics_hz: int
    mass_kg: float
    gravity: float
    wheelbase_m: float
    car_length_m: float
    car_width_m: float
    wheel_track_m: float
    max_steer_low_deg: float
    max_steer_high_deg: float
    steer_fade_speed: float
    steer_rate_deg_s: float
    engine_force_max: float
    engine_power_max: float
    min_drive_speed: float
    brake_force_max: float
    brake_min_speed: float
    stop_speed: float
    air_density: float
    cda_m2: float
    c_lat: float
    k_down: float
    sliding_threshold: float
    surfaces: Mapping[str, Mapping[str, float]]

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "PhysicsParams":
        return from_dict(cls, cfg, where="physics config")

    @property
    def dt(self) -> float:
        return 1.0 / self.physics_hz

    def max_steer(self, speed: float) -> float:
        """Steering lock in radians, fading linearly from low to high speed."""
        f = min(speed / self.steer_fade_speed, 1.0)
        deg = self.max_steer_low_deg + (self.max_steer_high_deg - self.max_steer_low_deg) * f
        return math.radians(deg)


class Car:
    """State and one fixed-step update. Telemetry fields are refreshed by every ``step``."""

    def __init__(self, params: PhysicsParams, surfaces: SurfaceSet | None = None) -> None:
        self.p = params
        self.surfaces = surfaces if surfaces is not None else SurfaceSet(params.surfaces)
        hl, hw = params.wheelbase_m / 2.0, params.wheel_track_m / 2.0
        bl, bw = params.car_length_m / 2.0, params.car_width_m / 2.0
        # Car-frame offsets (forward, left): wheels FL, FR, RL, RR; body corners likewise.
        self.wheel_offsets = ((hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw))
        self.corner_offsets = ((bl, bw), (bl, -bw), (-bl, bw), (-bl, -bw))
        self._drag_k = 0.5 * params.air_density * params.cda_m2
        self._steer_rate = math.radians(params.steer_rate_deg_s)
        self.reset(0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ state
    def reset(self, x: float, y: float, theta: float, speed: float = 0.0) -> None:
        """Place the car, moving straight ahead at ``speed``."""
        self.x, self.y, self.theta = float(x), float(y), float(theta)
        self.vx = speed * math.cos(theta)
        self.vy = speed * math.sin(theta)
        self.steer = 0.0          # current front-wheel angle (rad)
        self.sliding = False
        self.yaw_rate = 0.0
        self.grip_scale = 1.0     # friction-circle scale of the last step (1 = within grip)
        self.a_long = 0.0         # last acceleration in the car frame (m/s²)
        self.a_lat = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.mu_eff = 0.0

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def v_long(self) -> float:
        return self.vx * math.cos(self.theta) + self.vy * math.sin(self.theta)

    @property
    def v_lat(self) -> float:
        return -self.vx * math.sin(self.theta) + self.vy * math.cos(self.theta)

    def _offset_points(self, offsets: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return [(self.x + f * c - l * s, self.y + f * s + l * c) for f, l in offsets]

    def wheel_positions(self) -> list[tuple[float, float]]:
        """World positions of the wheels FL, FR, RL, RR."""
        return self._offset_points(self.wheel_offsets)

    def corner_positions(self) -> list[tuple[float, float]]:
        """World positions of the body corners FL, FR, RL, RR."""
        return self._offset_points(self.corner_offsets)

    # ------------------------------------------------------------------- step
    def step(self, steer_cmd: float, gas_cmd: float, wheel_surfaces: Sequence[int] = (GRASS_ID,) * 4) -> None:
        """Advance one physics step (``1 / physics_hz``).

        ``steer_cmd`` and ``gas_cmd`` are in [−1, 1]; negative gas brakes (no reverse).
        ``wheel_surfaces`` holds the surface id under each of the 4 wheels.
        """
        p = self.p
        dt = p.dt
        m = p.mass_kg
        g = p.gravity
        steer_cmd = min(max(steer_cmd, -1.0), 1.0)
        gas_cmd = min(max(gas_cmd, -1.0), 1.0)

        c, s = math.cos(self.theta), math.sin(self.theta)
        vx, vy = self.vx, self.vy
        v_long = vx * c + vy * s
        v_lat = -vx * s + vy * c
        speed = math.hypot(vx, vy)

        # --- controls: speed-dependent lock + steering rate limit
        target = steer_cmd * p.max_steer(speed)
        max_delta = self._steer_rate * dt
        self.steer += min(max(target - self.steer, -max_delta), max_delta)
        throttle = gas_cmd if gas_cmd > 0.0 else 0.0
        brake = -gas_cmd if gas_cmd < 0.0 else 0.0

        # --- grip available: mean over the 4 wheels' surfaces
        sf = self.surfaces
        mu_table = sf.mu_kinetic_l if self.sliding else sf.mu_static_l
        mu = 0.0
        crr = 0.0
        for sid in wheel_surfaces:
            i = sid if sid >= 0 else GRASS_ID
            mu += mu_table[i]
            crr += sf.rolling_res_l[i]
        mu *= 0.25
        crr *= 0.25
        f_max = mu * (m * g + p.k_down * v_long * v_long)

        # --- tyre force demand (car frame)
        f_drive = throttle * min(p.engine_force_max, p.engine_power_max / max(v_long, p.min_drive_speed))
        f_brake = brake * p.brake_force_max if v_long > p.brake_min_speed else 0.0
        f_long = f_drive - f_brake
        f_lat = -m * v_lat / dt * p.c_lat

        # --- friction circle
        demand = math.hypot(f_long, f_lat)
        scale = min(1.0, f_max / demand) if demand > 0.0 else 1.0
        self.sliding = scale < p.sliding_threshold
        f_long *= scale
        f_lat *= scale

        nvx = vx + (f_long * c - f_lat * s) / m * dt
        nvy = vy + (f_long * s + f_lat * c) / m * dt

        # --- resistances (drag + rolling), not limited by grip. Applied as a speed loss
        # that can never exceed the current speed, so the car can't jitter around v = 0.
        nspeed = math.hypot(nvx, nvy)
        if nspeed > 0.0:
            loss = (self._drag_k * nspeed * nspeed + crr * m * g) / m * dt
            k = max(nspeed - loss, 0.0) / nspeed
            nvx *= k
            nvy *= k

        # --- braking (or coasting) can't send the car backwards
        if throttle == 0.0:
            vf = nvx * c + nvy * s
            if vf < 0.0:
                nvx -= vf * c
                nvy -= vf * s
            if math.hypot(nvx, nvy) < p.stop_speed:
                nvx = nvy = 0.0

        # --- yaw: kinematic bicycle, reduced when the tyres are saturated (understeer)
        yaw_rate = v_long * math.tan(self.steer) / p.wheelbase_m * scale
        self.theta += yaw_rate * dt
        if self.theta > math.pi:
            self.theta -= 2.0 * math.pi
        elif self.theta < -math.pi:
            self.theta += 2.0 * math.pi

        # --- telemetry: world acceleration projected on the new heading
        ax, ay = (nvx - vx) / dt, (nvy - vy) / dt
        c2, s2 = math.cos(self.theta), math.sin(self.theta)
        self.a_long = ax * c2 + ay * s2
        self.a_lat = -ax * s2 + ay * c2
        self.yaw_rate = yaw_rate
        self.grip_scale = scale
        self.throttle, self.brake = throttle, brake
        self.mu_eff = mu

        self.vx, self.vy = nvx, nvy
        self.x += nvx * dt
        self.y += nvy * dt
