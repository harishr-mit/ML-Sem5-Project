"""Brain panel (T128, PROJECT_CONTEXT §10.2): the followed car's network, live.

Inputs as labelled bars, each hidden layer as a grid of neurons (one square per neuron, blue =
negative, orange = positive, brighter = stronger), the outputs as a steering wheel and gas/brake
bars. The lines between columns brighten with the activity they carry. Everything is drawn from a
:class:`racer.agents.activations.Snapshot`, so it shows what really drives the car.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pygame

from racer.agents.activations import Snapshot
from racer.render.hud import TextCache, panel

NEG = np.array([70, 150, 255], dtype=np.float32)
POS = np.array([255, 170, 50], dtype=np.float32)


def grid_shape(n: int) -> tuple[int, int]:
    """Rows and columns of a near-square grid holding ``n`` neurons."""
    cols = max(math.ceil(math.sqrt(max(n, 1))), 1)
    return math.ceil(max(n, 1) / cols), cols


def neuron_image(values: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """RGB image (rows, cols, 3) of one layer: dark for zero, blue for negative, orange for positive."""
    rows, cols = grid_shape(len(values))
    v = np.zeros(rows * cols, dtype=np.float32)
    v[: len(values)] = np.clip(np.asarray(values, dtype=np.float32) / max(scale, 1e-6), -1.0, 1.0)
    k = np.abs(v)[:, None] ** 0.7                                   # brighten weak activity a little
    rgb = np.where(v[:, None] >= 0, POS, NEG) * k + np.array([18, 20, 26], dtype=np.float32) * (1 - k)
    return rgb.reshape(rows, cols, 3).astype(np.uint8)


@dataclass
class ModelInfo:
    title: str
    input_labels: Sequence[str]
    action_labels: Sequence[str]
    arch: str                                                       # e.g. "10 → 256 → 256 → 2 · tanh · 69,124 parameters"


@dataclass
class BrainPanel:
    fonts: tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]
    colours: Mapping[str, Any]
    size: tuple[int, int] = (640, 300)
    smoothing: float = 0.45                                         # 0 = instant, towards 1 = slow (flicker-free at 20 Hz decisions)
    _shown: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.text = TextCache(self.fonts[2])
        self.title = TextCache(self.fonts[0])

    def reset(self) -> None:
        self._shown = []

    def _smooth(self, snap: Snapshot) -> list[np.ndarray]:
        arrays = [snap.inputs, *snap.layers, snap.action]
        if len(self._shown) != len(arrays) or any(a.shape != b.shape for a, b in zip(self._shown, arrays)):
            self._shown = [a.copy() for a in arrays]
        else:
            k = self.smoothing
            self._shown = [k * old + (1 - k) * new for old, new in zip(self._shown, arrays)]
        return self._shown

    # ------------------------------------------------------------------ drawing
    def draw(self, screen: pygame.Surface, pos: tuple[int, int], info: ModelInfo, snap: Snapshot | None,
             raw_obs: np.ndarray | None = None) -> pygame.Rect:
        c = self.colours
        rect = pygame.Rect(pos, self.size)
        panel(screen, rect, c["hud_bg"])
        screen.blit(self.title.get(info.title, c["hud_text"]), (rect.x + 10, rect.y + 6))
        screen.blit(self.text.get(info.arch, c["hud_dim"]), (rect.x + 10, rect.y + 6 + self.title.font.get_linesize()))
        top = rect.y + 8 + 2 * self.title.font.get_linesize() + 4
        body = pygame.Rect(rect.x + 8, top, rect.w - 16, rect.bottom - top - 8)
        if snap is None:
            msg = self.text.get("no neural network: hand-coded controller or human driver", c["hud_dim"])
            screen.blit(msg, msg.get_rect(center=body.center))
            return rect
        shown = self._smooth(snap)
        inputs, action = shown[0], shown[-1]
        layers = shown[1:-1]
        raw = np.asarray(raw_obs if raw_obs is not None else inputs, dtype=np.float32)
        in_rect = self._inputs(screen, body, info.input_labels, raw)
        grids = self._grids(screen, body, layers, in_rect.right)
        out_rect = self._outputs(screen, body, info.action_labels, action)
        self._edges(screen, in_rect, grids, out_rect, raw, layers, action)
        return rect

    def _inputs(self, screen: pygame.Surface, body: pygame.Rect, labels: Sequence[str], values: np.ndarray) -> pygame.Rect:
        c = self.colours
        n = len(values)
        w = 150
        row_h = min(max(body.h // max(n, 1), 12), 22)
        y0 = body.y + (body.h - n * row_h) // 2
        for i, v in enumerate(values):
            y = y0 + i * row_h
            label = labels[i] if i < len(labels) else f"input {i}"
            screen.blit(self.text.get(label[:20], c["hud_dim"]), (body.x, y + 1))
            bar = pygame.Rect(body.x + 100, y + 3, 48, row_h - 6)
            pygame.draw.rect(screen, (40, 44, 52), bar)
            v = float(np.clip(v, -1.0, 1.0))
            half = bar.w // 2
            w_fill = int(abs(v) * half)
            col = tuple(int(x) for x in (POS if v >= 0 else NEG))
            pygame.draw.rect(screen, col, (bar.x + half if v >= 0 else bar.x + half - w_fill, bar.y, w_fill, bar.h))
            pygame.draw.line(screen, (120, 125, 135), (bar.centerx, bar.y), (bar.centerx, bar.bottom))
        return pygame.Rect(body.x, y0, w, n * row_h)

    def _grids(self, screen: pygame.Surface, body: pygame.Rect, layers: list[np.ndarray], left: int) -> list[pygame.Rect]:
        if not layers:
            return []
        shown = layers if len(layers) <= 3 else [layers[0], layers[1], layers[-1]]
        side = min(body.h - 24, 112)
        gap = 34
        x = left + 30
        rects = []
        for k, vals in enumerate(shown):
            rows, cols = grid_shape(len(vals))
            img = neuron_image(vals, max(1.0, float(np.abs(vals).max())))
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(img.transpose(1, 0, 2)))
            cell = max(side // max(rows, cols), 1)
            surf = pygame.transform.scale(surf, (cols * cell, rows * cell))
            r = surf.get_rect(topleft=(x, body.y + 14 + (body.h - 24 - surf.get_height()) // 2))
            screen.blit(surf, r)
            pygame.draw.rect(screen, (90, 95, 105), r.inflate(2, 2), 1)
            cap = self.text.get(f"layer {k + 1 if len(layers) <= 3 or k < 2 else len(layers)} · {len(vals)}", self.colours["hud_dim"])
            screen.blit(cap, (r.x, body.y))
            rects.append(r)
            x = r.right + gap
        return rects

    def _outputs(self, screen: pygame.Surface, body: pygame.Rect, labels: Sequence[str], action: np.ndarray) -> pygame.Rect:
        c = self.colours
        cx, cy, rad = body.right - 62, body.y + 62, 42
        steer = float(np.clip(action[0], -1, 1)) if len(action) else 0.0
        gas = float(np.clip(action[1], -1, 1)) if len(action) > 1 else 0.0
        pygame.draw.circle(screen, (60, 64, 74), (cx, cy), rad, 4)
        ang = -steer * math.radians(120)                           # +steer = left = counter-clockwise
        for a in (0.0, math.pi * 2 / 3, math.pi * 4 / 3):
            x2 = cx + rad * math.sin(ang + a)
            y2 = cy - rad * math.cos(ang + a)
            pygame.draw.line(screen, (230, 230, 235), (cx, cy), (x2, y2), 3)
        pygame.draw.circle(screen, (230, 230, 235), (cx, cy), 6)
        screen.blit(self.text.get(f"steer {steer:+.2f}", c["hud_text"]), (cx - 34, cy + rad + 6))
        pedal = pygame.Rect(cx - 50, body.bottom - 74, 100, 12)
        pygame.draw.rect(screen, (40, 44, 52), pedal)
        mid = pedal.centerx
        w = int(abs(gas) * 50)
        col = c.get("brake_bar", (240, 90, 80)) if gas < 0 else c.get("hud_good", (90, 220, 120))
        pygame.draw.rect(screen, col, (mid if gas >= 0 else mid - w, pedal.y, w, pedal.h))
        pygame.draw.line(screen, (120, 125, 135), (mid, pedal.y - 2), (mid, pedal.bottom + 2))
        screen.blit(self.text.get(("brake " if gas < 0 else "gas ") + f"{abs(gas):.2f}", c["hud_text"]), (pedal.x, pedal.bottom + 4))
        return pygame.Rect(cx - rad, cy - rad, 2 * rad, pedal.bottom - (cy - rad))

    def _edges(self, screen: pygame.Surface, in_rect: pygame.Rect, grids: list[pygame.Rect], out_rect: pygame.Rect,
               raw: np.ndarray, layers: list[np.ndarray], action: np.ndarray) -> None:
        """Lines between the columns; their brightness follows the activity they carry."""
        bg = np.array(self.colours["hud_bg"][:3], dtype=np.float32)
        def line(a, b, level, col):
            k = float(np.clip(level, 0.0, 1.0)) ** 0.8
            colour = tuple(int(x) for x in bg + (np.asarray(col, dtype=np.float32) - bg) * k * 0.9)
            if k > 0.02:
                pygame.draw.line(screen, colour, a, b, 1)
        n = len(raw)
        if grids:
            first = grids[0]
            for i, v in enumerate(raw):
                y = in_rect.y + (i + 0.5) * in_rect.h / max(n, 1)
                line((in_rect.right - 2, y), (first.x, first.y + (i + 0.5) * first.h / max(n, 1)), abs(v), POS if v >= 0 else NEG)
            for a, b, vals in zip(grids, grids[1:], layers):
                for j in range(6):
                    y = a.y + (j + 0.5) * a.h / 6
                    chunk = vals[j * len(vals) // 6:(j + 1) * len(vals) // 6]
                    line((a.right, y), (b.x, y), float(np.abs(chunk).mean()) if len(chunk) else 0.0, POS)
            last, lvals = grids[-1], layers[-1]
            for j in range(len(action)):
                y = out_rect.y + (j + 0.5) * out_rect.h / max(len(action), 1)
                line((last.right, last.y + (j + 0.5) * last.h / max(len(action), 1)), (out_rect.x, y), abs(float(action[j])), POS)


def model_info(m: Any) -> ModelInfo:
    """Panel header and labels for a :class:`racer.agents.loader.LoadedModel` (from its manifest)."""
    mf = m.manifest
    arch = " > ".join(str(n) for n in [mf["obs_dim"], *mf["net_arch"], mf["action_dim"]])
    return ModelInfo(m.model_id, mf["obs_labels"], mf["action_labels"],
                     f"{arch} · {mf['activation']} · {mf['n_params']:,} parameters")
