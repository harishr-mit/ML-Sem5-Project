"""Race effects (T129, PROJECT_CONTEXT §10.2): skid marks, tyre smoke, grass dust, crash sparks.

The state (:class:`ParticleField`, :class:`SkidMarks`) is plain numpy and testable without a window;
:class:`Effects` feeds it from the race's cars and draws it with pygame, under the cars (marks, smoke,
dust) and over them (sparks). Everything is cosmetic: the physics and the models never see it.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pygame

from racer.physics.surfaces import GRASS_ID, KERB_ID, ROAD_ID

KINDS = {"smoke": 0, "dust": 1, "spark": 2}


class ParticleField:
    """A fixed-size pool of particles stored in arrays (oldest are replaced when it is full)."""

    def __init__(self, capacity: int) -> None:
        self.cap = int(capacity)
        n = self.cap
        self.x = np.zeros(n); self.y = np.zeros(n)
        self.vx = np.zeros(n); self.vy = np.zeros(n)
        self.age = np.zeros(n); self.life = np.zeros(n)        # life 0 = free slot
        self.size0 = np.zeros(n); self.size1 = np.zeros(n)     # radius in metres at birth and at death
        self.kind = np.zeros(n, dtype=np.int8)
        self.alpha = np.zeros(n)                               # peak opacity 0-1
        self._next = 0

    @property
    def alive(self) -> np.ndarray:
        return (self.life > 0) & (self.age < self.life)

    def count(self) -> int:
        return int(self.alive.sum())

    def spawn(self, x: float, y: float, vx: float, vy: float, life: float, size0: float, size1: float,
              kind: str, alpha: float) -> None:
        i = self._next
        self._next = (self._next + 1) % self.cap
        self.x[i], self.y[i], self.vx[i], self.vy[i] = x, y, vx, vy
        self.age[i], self.life[i] = 0.0, life
        self.size0[i], self.size1[i] = size0, size1
        self.kind[i], self.alpha[i] = KINDS[kind], alpha

    def update(self, dt: float, drag: float = 1.2) -> None:
        a = self.alive
        self.age[a] += dt
        k = max(1.0 - drag * dt, 0.0)
        self.vx[a] *= k
        self.vy[a] *= k
        self.x[a] += self.vx[a] * dt
        self.y[a] += self.vy[a] * dt
        self.life[~self.alive & (self.life > 0)] = 0.0         # expired slots become free

    def view(self, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
        """Indices of the live particles inside a world rectangle."""
        a = self.alive & (self.x > x0) & (self.x < x1) & (self.y > y0) & (self.y < y1)
        return np.nonzero(a)[0]

    def frac(self, i: np.ndarray) -> np.ndarray:
        return self.age[i] / self.life[i]


class SkidMarks:
    """Line segments laid down by sliding wheels, newest last, the oldest forgotten beyond ``max_segments``."""

    def __init__(self, max_segments: int) -> None:
        self.segments: deque[tuple[float, float, float, float, float]] = deque(maxlen=int(max_segments))
        self._last: dict[tuple[int, int], tuple[float, float]] = {}
        self.fresh: list[tuple[float, float, float, float]] = []      # segments not yet baked into the world

    def drain(self) -> list[tuple[float, float, float, float]]:
        """New segments since the last call (the world bakes them into its surface, so they last for the race)."""
        out, self.fresh = self.fresh, []
        return out

    def add(self, car: int, wheel: int, x: float, y: float, sliding: bool, now: float) -> None:
        key = (car, wheel)
        last = self._last.get(key)
        if sliding:
            if last is not None and (x - last[0]) ** 2 + (y - last[1]) ** 2 > 0.0004:
                self.segments.append((last[0], last[1], x, y, now))
                self.fresh.append((last[0], last[1], x, y))
            self._last[key] = (x, y)
        else:
            self._last.pop(key, None)

    def clear(self) -> None:
        self.segments.clear()
        self.fresh.clear()
        self._last.clear()


@dataclass
class Shake:
    """Camera shake: an impulse that decays; the follow camera's lerp pulls the view back by itself."""

    amplitude_m: float = 0.0

    def kick(self, metres: float) -> None:
        self.amplitude_m = max(self.amplitude_m, metres)

    def offset(self, rng: np.random.Generator) -> tuple[float, float]:
        a = self.amplitude_m
        return float(rng.uniform(-a, a)), float(rng.uniform(-a, a))

    def decay(self, dt: float, rate: float = 7.0) -> None:
        self.amplitude_m *= max(1.0 - rate * dt, 0.0)
        if self.amplitude_m < 0.01:
            self.amplitude_m = 0.0


class Effects:
    def __init__(self, cfg: Mapping[str, Any], seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.particles = ParticleField(int(cfg["max_particles"]))
        self.skids = SkidMarks(int(cfg["max_skid_segments"]))
        self.shake = Shake()
        self.t = 0.0
        self._state: dict[int, str] = {}
        self._puffs: dict[tuple[int, int, int], pygame.Surface] = {}

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.particles = ParticleField(int(self.cfg["max_particles"]))
        self.skids.clear()
        self.shake.amplitude_m = 0.0
        self._state.clear()
        self.t = 0.0

    def step(self, dt: float, cars: Sequence[Any], follow: int | None = None) -> None:
        """Advance by ``dt`` seconds; ``cars`` are the race's cars (``.env.car``, ``.state``, ``.pose``)."""
        c = self.cfg
        self.t += dt
        for car in cars:
            phys = car.env.car
            prev = self._state.get(car.index)
            self._state[car.index] = car.state
            if prev == "racing" and car.state == "respawn":                    # a crash: sparks and a kick
                self._sparks(car)
                if follow == car.index:
                    self.shake.kick(float(c["crash_shake_m"]))
            if car.state not in ("racing", "finished"):
                for w in range(2):
                    self.skids.add(car.index, w, 0.0, 0.0, False, self.t)
                continue
            speed = phys.speed
            wheels = phys.wheel_positions()
            surf = car.env._wheels                                              # surface id under each wheel
            sliding = bool(phys.sliding) and speed > float(c["skid_min_speed_ms"])
            for w in (2, 3):                                                   # rear wheels leave the marks
                x, y = wheels[w]
                on_tarmac = int(surf[w]) in (ROAD_ID, KERB_ID)
                self.skids.add(car.index, w - 2, x, y, sliding and on_tarmac, self.t)
            self._emit(car, phys, wheels, surf, speed, sliding, dt)
        self.particles.update(dt)
        self.shake.decay(dt)

    def _emit(self, car, phys, wheels, surf, speed: float, sliding: bool, dt: float) -> None:
        c = self.cfg
        p = self.particles
        for w in range(4):
            x, y = wheels[w]
            grass = int(surf[w]) == GRASS_ID
            if grass and speed > float(c["dust_min_speed_ms"]):
                if self.rng.random() < float(c["dust_rate"]) * dt * 60.0:
                    ang = self.rng.uniform(0, 2 * math.pi)
                    s = self.rng.uniform(0.3, 1.5)
                    p.spawn(x, y, phys.vx * 0.12 + s * math.cos(ang), phys.vy * 0.12 + s * math.sin(ang),
                            float(c["dust_life_s"]), 0.5, 1.9, "dust", 0.5)
            elif sliding and w >= 2:
                if self.rng.random() < float(c["smoke_rate"]) * dt * 60.0:
                    ang = self.rng.uniform(0, 2 * math.pi)
                    p.spawn(x, y, phys.vx * 0.08 + 0.6 * math.cos(ang), phys.vy * 0.08 + 0.6 * math.sin(ang),
                            float(c["smoke_life_s"]), 0.5, 2.4, "smoke", 0.55)

    def _sparks(self, car) -> None:
        x, y, _ = car.pose
        v = car.env.car
        for _ in range(int(self.cfg["crash_sparks"])):
            ang = self.rng.uniform(0, 2 * math.pi)
            s = self.rng.uniform(3.0, 14.0)
            self.particles.spawn(x, y, v.vx * 0.2 + s * math.cos(ang), v.vy * 0.2 + s * math.sin(ang),
                                 float(self.rng.uniform(0.25, 0.7)), 0.12, 0.04, "spark", 1.0)

    # ------------------------------------------------------------------ drawing
    def _puff(self, kind: int, radius_px: int, alpha_bucket: int) -> pygame.Surface:
        key = (kind, radius_px, alpha_bucket)
        s = self._puffs.get(key)
        if s is None:
            r = max(radius_px, 2)
            s = pygame.Surface((2 * r, 2 * r), pygame.SRCALPHA)
            col = {0: (205, 205, 210), 1: (150, 128, 88), 2: (255, 190, 70)}[kind]
            a = alpha_bucket / 8.0
            for k in range(r, 0, -max(r // 8, 1)):                              # soft disc: stacked circles
                t = 1.0 - k / r
                pygame.draw.circle(s, (*col, int(255 * a * (0.25 + 0.75 * t) * (0.35 if kind < 2 else 1.0))), (r, r), k)
            if len(self._puffs) > 600:
                self._puffs.clear()
            self._puffs[key] = s
        return s

    def draw_under(self, screen: pygame.Surface, cam) -> None:
        """Smoke and dust (call before the cars). Skid marks are baked into the world, see ``TrackWorld.bake_marks``."""
        z = cam.view_zoom
        x0, y0 = cam.to_world(0, screen.get_height())
        x1, y1 = cam.to_world(screen.get_width(), 0)
        p = self.particles
        idx = p.view(x0 - 3, x1 + 3, y0 - 3, y1 + 3)
        idx = idx[p.kind[idx] != KINDS["spark"]]
        if not len(idx):
            return
        order = idx[np.argsort(p.age[idx])[::-1]]                               # old (big, faint) first
        f = p.age[order] / p.life[order]
        radius = np.maximum(((p.size0[order] + (p.size1[order] - p.size0[order]) * f) * z).astype(np.int32) // 2 * 2, 2)
        alpha = np.clip(p.alpha[order] * (1.0 - f) * 8.0, 1, 8).astype(np.int32)
        sx = (p.x[order] - cam.cx) * z + cam.w / 2
        sy = (cam.cy - p.y[order]) * z + cam.h / 2
        for kind, r, a, x, y in zip(p.kind[order].tolist(), radius.tolist(), alpha.tolist(), sx.tolist(), sy.tolist()):
            spr = self._puff(kind, r, a)
            screen.blit(spr, (x - spr.get_width() / 2, y - spr.get_height() / 2))

    def draw_over(self, screen: pygame.Surface, cam) -> None:
        """Crash sparks (call after the cars)."""
        p = self.particles
        x0, y0 = cam.to_world(0, screen.get_height())
        x1, y1 = cam.to_world(screen.get_width(), 0)
        idx = p.view(x0, x1, y0, y1)
        idx = idx[p.kind[idx] == KINDS["spark"]]
        for i in idx:
            f = p.age[i] / p.life[i]
            sx, sy = cam.to_screen(p.x[i], p.y[i])
            tail = cam.to_screen(p.x[i] - p.vx[i] * 0.03, p.y[i] - p.vy[i] * 0.03)
            col = (255, int(210 - 120 * f), int(90 - 70 * f))
            pygame.draw.line(screen, col, tail, (sx, sy), 2)
