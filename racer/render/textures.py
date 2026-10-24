"""Ground textures for the track world (PROJECT_CONTEXT §14.2): load, grade, tile.

The photographic textures are darkened and slightly desaturated so they sit next to Kenney's flat
sprites (assets/LICENSES.md). A missing file falls back to noise around a flat colour, so the demo
never fails for want of an asset. All functions are pure numpy plus pygame image loading.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pygame

from racer.config import PROJECT_ROOT

GROUND = PROJECT_ROOT / "assets" / "textures" / "ground"


def grade(rgb: np.ndarray, darken: float, desaturate: float) -> np.ndarray:
    """Darken (0 = unchanged, 0.3 = 30 % darker) and desaturate (0–1) an RGB uint8 image."""
    x = rgb.astype(np.float32)
    grey = x.mean(axis=2, keepdims=True)
    x = grey + (x - grey) * (1.0 - desaturate)
    return np.clip(x * (1.0 - darken), 0, 255).astype(np.uint8)


def noise_tile(size: int, colour: Sequence[int], rng: np.random.Generator, amount: float = 0.08) -> np.ndarray:
    """Fallback tile: the colour with fine noise (seamless because it has no structure)."""
    base = np.asarray(colour, dtype=np.float32)
    n = rng.normal(0.0, amount, (size, size, 1)).astype(np.float32)
    return np.clip(base * (1.0 + n), 0, 255).astype(np.uint8)


def load_tile(name: str, tile_px: int, darken: float, desaturate: float, fallback: Sequence[int],
              rng: np.random.Generator) -> np.ndarray:
    """A ``tile_px`` square (h, w, 3) uint8 tile of ground texture ``name``."""
    tile_px = max(int(tile_px), 8)
    try:
        img = pygame.image.load(str(GROUND / name))
        img = pygame.transform.smoothscale(img, (tile_px, tile_px))
        arr = pygame.surfarray.array3d(img).transpose(1, 0, 2)          # (h, w, 3)
        return grade(np.ascontiguousarray(arr), darken, desaturate)
    except (FileNotFoundError, OSError, pygame.error):
        return noise_tile(tile_px, fallback, rng)


def tile_at(tile: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Texels for pixel coordinates (any shape, broadcast together), tiling seamlessly."""
    th, tw = tile.shape[:2]
    return tile[np.asarray(rows) % th, np.asarray(cols) % tw]


def low_frequency(shape: tuple[int, int], cell_px: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    """A smooth multiplier field around 1.0 (breaks up visible tiling over large areas), float32 (h, w)."""
    h, w = shape
    ch, cw = max(h // cell_px, 1) + 2, max(w // cell_px, 1) + 2
    coarse = rng.uniform(1.0 - strength, 1.0 + strength, (ch, cw)).astype(np.float32)
    surf = pygame.Surface((cw, ch))
    pygame.surfarray.blit_array(surf, np.repeat(coarse.T[..., None] * 127.5, 3, axis=2).clip(0, 255).astype(np.uint8))
    big = pygame.transform.smoothscale(surf, (cw * cell_px, ch * cell_px))
    arr = pygame.surfarray.array3d(big)[:w, :h, 0].T.astype(np.float32) / 127.5
    if arr.shape != (h, w):                                              # tiny worlds: pad by edge
        arr = np.pad(arr, ((0, max(h - arr.shape[0], 0)), (0, max(w - arr.shape[1], 0))), mode="edge")[:h, :w]
    return arr


def bilinear(grid: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinear samples of ``grid`` on the outer product of fractional ``rows`` x ``cols`` (float32, (h, w))."""
    h, w = grid.shape
    r = np.clip(rows, 0, h - 1)
    c = np.clip(cols, 0, w - 1)
    r0, c0 = np.minimum(np.floor(r).astype(np.int32), h - 2), np.minimum(np.floor(c).astype(np.int32), w - 2)
    fr, fc = (r - r0).astype(np.float32)[:, None], (c - c0).astype(np.float32)[None, :]
    g = grid
    top = g[r0[:, None], c0[None, :]] * (1 - fc) + g[r0[:, None], c0[None, :] + 1] * fc
    bot = g[r0[:, None] + 1, c0[None, :]] * (1 - fc) + g[r0[:, None] + 1, c0[None, :] + 1] * fc
    return (top * (1 - fr) + bot * fr).astype(np.float32)
