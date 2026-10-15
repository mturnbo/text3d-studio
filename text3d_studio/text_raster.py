"""Rasterizes system-font text into a mask that drives the layer-stack extrusion.

Output RGBA channels:
  R, G  outward edge direction (local x-right, y-up), encoded 0..255 -> -1..1
  B     bevel factor: 1 at the silhouette, falling to 0 one bevel-width inside
  A     glyph coverage
"""

import math
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter

from .image_io import qimage_to_rgba
from .model import TextParams

FONT_PIXEL_SIZE = 384
MAX_TEXTURE_WIDTH = 8192


@dataclass
class GlyphRaster:
    rgba: np.ndarray  # (H, W, 4) uint8, top row first
    origin: tuple[float, float]  # bottom-left corner of the quad, in em units; ink bottom sits at y=0
    size: tuple[float, float]  # quad size in em units


def rasterize_text(params: TextParams) -> GlyphRaster | None:
    text = params.content
    if not text.strip():
        return None

    pad = int(math.ceil(params.bevel_width * 3)) + 8
    font_px = FONT_PIXEL_SIZE
    font = _make_font(params, font_px)
    bounds = QFontMetricsF(font).boundingRect(text)
    needed_width = bounds.width() + 2 * pad + font_px * 0.5
    if needed_width > MAX_TEXTURE_WIDTH:
        font_px = max(16, int(font_px * MAX_TEXTURE_WIDTH / needed_width))
        font = _make_font(params, font_px)
        bounds = QFontMetricsF(font).boundingRect(text)

    alpha = _paint_alpha(text, font, bounds, pad, font_px)
    cropped = _crop_to_ink(alpha, pad)
    if cropped is None:
        return None

    height, width = cropped.shape
    return GlyphRaster(
        rgba=_encode_mask(cropped, params.bevel_width),
        origin=(-(width / 2) / font_px, -pad / font_px),
        size=(width / font_px, height / font_px),
    )


def _make_font(params: TextParams, pixel_size: int) -> QFont:
    font = QFont(params.font_family)
    font.setPixelSize(pixel_size)
    font.setBold(params.bold)
    font.setItalic(params.italic)
    return font


def _paint_alpha(text, font, bounds, pad, font_px) -> np.ndarray:
    slack = int(font_px * 0.25)  # room for italic overhang the bounding rect may miss
    width = int(math.ceil(bounds.width())) + 2 * (pad + slack)
    height = int(math.ceil(bounds.height())) + 2 * (pad + slack)
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(QPointF(pad + slack - bounds.left(), pad + slack - bounds.top()), text)
    painter.end()
    return qimage_to_rgba(image)[..., 3].astype(np.float32) / 255.0


def _crop_to_ink(alpha: np.ndarray, pad: int) -> np.ndarray | None:
    rows = np.flatnonzero(alpha.max(axis=1) > 0.02)
    cols = np.flatnonzero(alpha.max(axis=0) > 0.02)
    if rows.size == 0 or cols.size == 0:
        return None
    ink = alpha[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return np.pad(ink, pad)


def _encode_mask(alpha: np.ndarray, bevel_width: float) -> np.ndarray:
    small = _gaussian(alpha, 1)
    large = _gaussian(alpha, max(1, int(round(bevel_width / 2))))

    # Rows grow downward, so outward (toward lower coverage) in a y-up frame is (-d/dx, +d/drow).
    small_gy, small_gx = np.gradient(small)
    large_gy, large_gx = np.gradient(large)
    small_dir = np.stack([-small_gx, small_gy], axis=-1)
    large_dir = np.stack([-large_gx, large_gy], axis=-1)
    small_mag = np.linalg.norm(small_dir, axis=-1, keepdims=True)
    large_mag = np.linalg.norm(large_dir, axis=-1, keepdims=True)
    direction = np.where(
        small_mag > 1e-3,
        small_dir / np.maximum(small_mag, 1e-6),
        np.where(large_mag > 1e-4, large_dir / np.maximum(large_mag, 1e-6), 0.0),
    )

    bevel = 1.0 - np.clip((large - 0.5) * 2.0, 0.0, 1.0)

    rgba = np.empty(alpha.shape + (4,), dtype=np.uint8)
    rgba[..., 0:2] = np.round((direction * 0.5 + 0.5) * 255)
    rgba[..., 2] = np.round(bevel * 255)
    rgba[..., 3] = np.round(np.clip(alpha, 0, 1) * 255)
    return rgba


def _gaussian(values: np.ndarray, radius: int) -> np.ndarray:
    """Three box blurs approximate a Gaussian; cumulative sums keep it O(n) per pass."""
    out = values
    for _ in range(3):
        out = _box_blur(_box_blur(out, radius, axis=0), radius, axis=1)
    return out


def _box_blur(values: np.ndarray, radius: int, axis: int) -> np.ndarray:
    pad = [(0, 0), (0, 0)]
    pad[axis] = (radius + 1, radius)
    sums = np.cumsum(np.pad(values, pad), axis=axis)
    n = values.shape[axis]
    upper = np.take(sums, np.arange(2 * radius + 1, 2 * radius + 1 + n), axis=axis)
    lower = np.take(sums, np.arange(n), axis=axis)
    return (upper - lower) / (2 * radius + 1)
