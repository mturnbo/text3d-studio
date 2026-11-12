"""Rasterizes system-font text into a mask that drives the layer-stack extrusion.

Output RGBA channels:
  R, G  outward edge direction (local x-right, y-up), encoded 0..255 -> -1..1
  B     bevel factor: 1 at the silhouette, falling to 0 one bevel-width inside
  A     glyph coverage
"""

import math
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QGlyphRun, QImage, QPainter, QTextLayout, QTransform

from .image_io import qimage_to_rgba
from .model import TextParams

FONT_PIXEL_SIZE = 384
MAX_TEXTURE_WIDTH = 8192  # the mask's limit on both sides; larger text is rasterized at a smaller font size
MIN_FONT_PIXEL_SIZE = 16
ALIGNMENTS = {"left": 0.0, "center": 0.5, "right": 1.0}  # TextParams.align -> share of the slack put before a line
WALL_STEP = 2  # outline tracing resolution (texels per step); the shadow is blurred, so coarse is fine


@dataclass
class GlyphRaster:
    rgba: np.ndarray  # (H, W, 4) uint8, top row first
    origin: tuple[float, float]  # bottom-left corner of the quad, in em units; ink bottom sits at y=0
    size: tuple[float, float]  # quad size in em units
    walls: np.ndarray  # (N, 3) float32 side-wall triangles used by the shadow pass

    def covers(self, x: float, y: float) -> bool:
        """Whether the glyphs' ink covers the point (x, y) in em units (the text's local space)."""
        u = (x - self.origin[0]) / self.size[0]
        v = (y - self.origin[1]) / self.size[1]
        if not (0.0 <= u < 1.0 and 0.0 < v <= 1.0):
            return False
        height, width = self.rgba.shape[:2]
        return bool(self.rgba[int((1.0 - v) * height), int(u * width), 3] >= 128)  # row 0 is the top


def rasterize_text(params: TextParams) -> GlyphRaster | None:
    """Rasterizes every line of the text, stacked downward and optionally bent along an arc. Blank lines keep
    their height; the mask is cropped to the ink, so leading and trailing blank lines don't count."""
    if params.align not in ALIGNMENTS:
        raise ValueError(f"Unknown text alignment: {params.align!r}")
    if not params.content.strip():
        return None

    pad = int(math.ceil(params.bevel_width * 3)) + 8
    font_px = FONT_PIXEL_SIZE
    lines, glyphs, bounds = _layout(params, font_px)
    limit = MAX_TEXTURE_WIDTH - 2 * pad
    while max(bounds.width(), bounds.height()) > limit and font_px > MIN_FONT_PIXEL_SIZE:
        font_px = max(MIN_FONT_PIXEL_SIZE, int(font_px * limit / max(bounds.width(), bounds.height())))
        lines, glyphs, bounds = _layout(params, font_px)

    alpha = _paint_alpha(lines, glyphs, _make_font(params, font_px), bounds, pad)
    cropped = _crop_to_ink(alpha, pad)
    if cropped is None:
        return None

    height, width = cropped.shape
    origin = (-(width / 2) / font_px, -pad / font_px)
    size = (width / font_px, height / font_px)
    return GlyphRaster(
        rgba=_encode_mask(cropped, params.bevel_width),
        origin=origin,
        size=size,
        walls=extract_walls(cropped, origin, size),
    )


def extract_walls(alpha: np.ndarray, origin, size, step: int = WALL_STEP) -> np.ndarray:
    """Traces the glyph outline into side-wall quads, as (N, 3) float32 triangle vertices.

    Each vertex is (local x, local y, back flag); the shader maps the flag to z = -depth.
    Layer slices alone cast no shadow when the light runs parallel to them, so the shadow
    pass uses these walls (plus the front/back slices) as a closed caster.
    """
    inside = np.pad(alpha[::step, ::step] >= 0.5, 1)
    tex_h, tex_w = alpha.shape

    # Vertical walls: a pixel boundary at x=c between padded columns c and c+1, spanning rows.
    cols, starts, ends = _runs(inside[:, 1:] != inside[:, :-1])
    vertical = np.stack([cols, starts - 1, cols, ends - 1], axis=1)
    # Horizontal walls: a boundary at y=r between padded rows r and r+1, spanning columns.
    rows, starts, ends = _runs((inside[1:, :] != inside[:-1, :]).T)
    horizontal = np.stack([starts - 1, rows, ends - 1, rows], axis=1)

    segments = np.concatenate([vertical, horizontal]).astype(np.float64) * step
    xs = origin[0] + segments[:, [0, 2]] / tex_w * size[0]
    ys = origin[1] + (1.0 - segments[:, [1, 3]] / tex_h) * size[1]
    p0 = np.stack([xs[:, 0], ys[:, 0]], axis=1)
    p1 = np.stack([xs[:, 1], ys[:, 1]], axis=1)

    front, back = np.zeros((len(p0), 1)), np.ones((len(p0), 1))
    corners = [
        np.hstack([p0, front]), np.hstack([p1, front]), np.hstack([p1, back]),
        np.hstack([p0, front]), np.hstack([p1, back]), np.hstack([p0, back]),
    ]
    return np.stack(corners, axis=1).reshape(-1, 3).astype(np.float32)


def _runs(boundary: np.ndarray):
    """Finds runs of True down each column. Returns (column, start row, end row exclusive)."""
    edges = np.diff(np.pad(boundary, ((1, 1), (0, 0))).astype(np.int8), axis=0)
    start_rows, start_cols = np.nonzero(edges == 1)
    end_rows, end_cols = np.nonzero(edges == -1)
    start_order = np.lexsort((start_rows, start_cols))
    end_order = np.lexsort((end_rows, end_cols))
    return start_cols[start_order], start_rows[start_order], end_rows[end_order]


def _make_font(params: TextParams, pixel_size: int) -> QFont:
    font = QFont(params.font_family)
    font.setPixelSize(pixel_size)
    font.setBold(params.bold)
    font.setItalic(params.italic)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, params.letter_spacing * pixel_size)
    return font


def _layout(params: TextParams, font_px: int):
    """Places each line's text at its baseline start, in pixels (y down), aligning the lines' ink.

    Returns (lines, glyphs, bounds): straight text is drawn a line at a time as `lines` [(text, start)]; arced
    text a glyph at a time as `glyphs` [(glyph run, anchor, degrees)]. `bounds` is the box their ink can reach,
    with slack for italic overhang the metrics may miss.
    """
    font = _make_font(params, font_px)
    metrics = QFontMetricsF(font)
    lines = params.content.split("\n")
    inks = [metrics.tightBoundingRect(line) if line.strip() else None for line in lines]
    widest = max(ink.width() for ink in inks if ink is not None)
    pitch = metrics.lineSpacing() * params.line_height
    slack = font_px * 0.25
    placed, bounds = [], QRectF()
    for index, (line, ink) in enumerate(zip(lines, inks)):
        if ink is None:
            continue
        start = QPointF((widest - ink.width()) * ALIGNMENTS[params.align] - ink.left(), index * pitch)
        placed.append((line, start))
        bounds = bounds.united(ink.translated(start).adjusted(-slack, -slack, slack, slack))
        if ink.width() == widest:
            widest_baseline = start.y()
    if params.arc == 0:
        return placed, [], bounds
    glyphs, bounds = _bend(placed, font, widest, widest_baseline, math.radians(params.arc), slack)
    return [], glyphs, bounds


def _bend(placed, font, widest, widest_baseline, angle, slack):
    """Lays the lines' glyphs along concentric arcs, each turned to its arc's tangent.

    The widest line's ink spans `angle` on a circle through its baseline; the other lines keep their letter
    spacing on circles one line pitch nearer or farther. The circle's center lies below the text for a positive
    angle (the middle bends up) and above it for a negative one. Positions stay measured along the arc, so
    alignment carries over.
    """
    metrics = QFontMetricsF(font)
    sign = 1.0 if angle > 0 else -1.0
    radius = widest / abs(angle)
    center = QPointF(widest / 2, widest_baseline + sign * radius)
    glyphs, bounds = [], QRectF()
    for line, start in placed:
        line_radius = max(radius - sign * (start.y() - widest_baseline), 1.0)
        layout = QTextLayout(line, font)
        layout.beginLayout()
        layout.createLine().setLineWidth(1e9)
        layout.endLayout()
        for run in layout.glyphRuns():
            indexes = run.glyphIndexes()
            advances = run.rawFont().advancesForGlyphIndexes(indexes)
            for index, position, advance in zip(indexes, run.positions(), advances):
                half = advance.x() / 2
                turn = (start.x() + position.x() + half - center.x()) / line_radius
                anchor = QPointF(
                    center.x() + line_radius * math.sin(turn),
                    center.y() - sign * line_radius * math.cos(turn),
                )
                degrees = math.degrees(sign * turn)
                glyph = QGlyphRun()
                glyph.setRawFont(run.rawFont())
                glyph.setGlyphIndexes([index])
                glyph.setPositions([QPointF(-half, 0.0)])
                glyphs.append((glyph, anchor, degrees))
                box = QRectF(-half, -metrics.ascent(), 2 * half, metrics.ascent() + metrics.descent())
                placement = QTransform().translate(anchor.x(), anchor.y()).rotate(degrees)
                bounds = bounds.united(placement.mapRect(box.adjusted(-slack, -slack, slack, slack)))
    return glyphs, bounds


def _paint_alpha(lines, glyphs, font, bounds, pad) -> np.ndarray:
    width = int(math.ceil(bounds.width())) + 2 * pad
    height = int(math.ceil(bounds.height())) + 2 * pad
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255))
    painter.translate(pad - bounds.left(), pad - bounds.top())
    for text, start in lines:
        painter.drawText(start, text)
    for glyph, anchor, degrees in glyphs:
        painter.save()
        painter.translate(anchor)
        painter.rotate(degrees)
        painter.drawGlyphRun(QPointF(0.0, 0.0), glyph)
        painter.restore()
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
