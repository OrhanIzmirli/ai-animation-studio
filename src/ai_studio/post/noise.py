"""Deterministic, spatially coherent noise and small image helpers.

Principle of the style layer: hand-made imperfection is consistent habit plus
content-dependent variation, never uniform randomness. So nothing here is
per-pixel independent noise except `fine_noise`, which is used only for grain.

Every random field is derived from three integers:
    seed   the artist profile's seed (the "hand")
    salt   which habit the field belongs to (line weight, wobble, ...)
    index  which drawing it belongs to
The same (seed, salt, index) always gives the same field, on any machine
with the same numpy.
"""

from __future__ import annotations

import numpy as np

# Salts: one per use, so different habits never share a field.
SALT_TIMING, SALT_PRESSURE, SALT_WOBBLE_X, SALT_WOBBLE_Y, SALT_HALFTONE, SALT_GRAIN, SALT_PALETTE = range(1, 8)


def rng_for(seed: int, salt: int, index: int = 0) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(salt), int(index)])


def smooth_noise(shape: tuple[int, int], cell: float, seed: int, salt: int, index: int = 0) -> np.ndarray:
    """Low-frequency value noise in [-1, 1], shape (height, width).

    Random values on a coarse grid with `cell` pixels between grid points,
    interpolated with a smoothstep, so neighbouring pixels are strongly
    correlated: the field varies over tens of pixels, like the pressure of a
    hand, not from pixel to pixel."""
    height, width = shape
    cell = max(float(cell), 1.0)
    rows, cols = int(np.ceil(height / cell)) + 2, int(np.ceil(width / cell)) + 2
    grid = rng_for(seed, salt, index).uniform(-1.0, 1.0, size=(rows, cols)).astype(np.float32)
    y = np.arange(height, dtype=np.float32) / cell
    x = np.arange(width, dtype=np.float32) / cell
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    fy, fx = y - y0, x - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)  # smoothstep
    top = grid[y0][:, x0] * (1 - fx) + grid[y0][:, x0 + 1] * fx
    bottom = grid[y0 + 1][:, x0] * (1 - fx) + grid[y0 + 1][:, x0 + 1] * fx
    return top * (1 - fy[:, None]) + bottom * fy[:, None]


def fine_noise(shape: tuple[int, int], seed: int, salt: int, index: int = 0) -> np.ndarray:
    """Per-pixel standard normal noise, shape (height, width). Grain only."""
    return rng_for(seed, salt, index).standard_normal(size=shape, dtype=np.float32)


def luma(frame: np.ndarray) -> np.ndarray:
    """(H, W) float32 luma 0-255 of an (H, W, 3) RGB frame (BT.601)."""
    frame = np.asarray(frame)
    return (frame[..., 0] * np.float32(0.299) + frame[..., 1] * np.float32(0.587)
            + frame[..., 2] * np.float32(0.114)).astype(np.float32)


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a (2r+1) x (2r+1) window, edges replicated. (H, W) float32."""
    if radius <= 0:
        return image.astype(np.float32)
    out = image.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius + 1, radius)
        summed = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis, dtype=np.float64)
        size = out.shape[axis]
        upper = np.take(summed, np.arange(2 * radius + 1, 2 * radius + 1 + size), axis=axis)
        lower = np.take(summed, np.arange(0, size), axis=axis)
        out = ((upper - lower) / (2 * radius + 1)).astype(np.float32)
    return out


def max_filter(image: np.ndarray, radius: int) -> np.ndarray:
    """Maximum over a (2r+1) x (2r+1) window (grey dilation), edges replicated."""
    out = image
    for axis in (0, 1):
        if radius <= 0:
            break
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        size = out.shape[axis]
        out = np.maximum.reduce([np.take(padded, np.arange(k, k + size), axis=axis) for k in range(2 * radius + 1)])
    return out


def shift_edge(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift by whole pixels (positive = right / down), edges replicated."""
    height, width = image.shape[:2]
    ys = np.clip(np.arange(height) - dy, 0, height - 1)
    xs = np.clip(np.arange(width) - dx, 0, width - 1)
    return image[ys][:, xs]
