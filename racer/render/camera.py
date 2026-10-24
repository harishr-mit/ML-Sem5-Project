"""2D camera: world metres (x right, y up) ↔ screen pixels. Follow a car or show the whole track."""
from __future__ import annotations

import math
from typing import Any, Mapping


class Camera:
    def __init__(self, size: tuple[int, int], cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self.w, self.h = size
        self.zoom = float(cfg["zoom_px_per_m"])
        self.cx = self.cy = 0.0
        self.overview = False
        self.zoom_fit = self.zoom
        self.dyn = 1.0                      # speed-dependent zoom factor (1 = none)

    @property
    def view_zoom(self) -> float:
        """Pixels per metre currently on screen (the fitted zoom in overview)."""
        return self.zoom_fit if self.overview else self.zoom * self.dyn

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        z = self.view_zoom
        return (x - self.cx) * z + self.w / 2, (self.cy - y) * z + self.h / 2

    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        z = self.view_zoom
        return (sx - self.w / 2) / z + self.cx, self.cy - (sy - self.h / 2) / z

    def zoom_by(self, factor: float) -> None:
        self.overview = False
        self.zoom = min(max(self.zoom * factor, self.cfg["zoom_min"]), self.cfg["zoom_max"])

    def fit(self, extent: tuple[float, float, float, float], width: int | None = None) -> None:
        """Overview: centre on ``extent`` (x0, x1, y0, y1) and fit it into ``width`` px (default: window)."""
        x0, x1, y0, y1 = extent
        w = width or self.w
        self.zoom_fit = min(w / (x1 - x0), self.h / (y1 - y0))
        self.cx = (x0 + x1) / 2 + (self.w - w) / 2 / self.zoom_fit
        self.cy = (y0 + y1) / 2

    def update(self, x: float, y: float, vx: float, vy: float, sim_speed: float = 1.0,
               extent: tuple[float, float, float, float] | None = None, fit_width: int | None = None) -> None:
        """Follow a car. ``sim_speed`` = simulated physics steps shown per frame, so the
        camera follows per *simulated* time and keeps up at ×4 as well as at ×1."""
        if self.overview:
            if extent is not None:
                self.fit(extent, fit_width)
            return
        k = float(self.cfg.get("camera_speed_zoom", 0.0))
        if k > 0.0:
            target = 1.0 - k * min(math.hypot(vx, vy) / float(self.cfg["camera_speed_ref"]), 1.0)
            ease = 1.0 - (1.0 - float(self.cfg["camera_speed_ease"])) ** max(sim_speed, 0.0)
            self.dyn += (target - self.dyn) * ease
        lead = self.cfg["camera_lead_s"]
        tx, ty = x + vx * lead, y + vy * lead
        a = 1.0 - (1.0 - self.cfg["camera_lerp"]) ** max(sim_speed, 0.0)
        self.cx += (tx - self.cx) * a
        self.cy += (ty - self.cy) * a

    def snap(self, x: float, y: float) -> None:
        self.cx, self.cy = x, y
