"""Occlusion masks: photo pixels that stand in front of the text, painted with a soft round brush. No Qt here.

A mask is an (H, W) uint8 array at the photo's size: 255 where the photo covers the text, 0 where the text shows,
and values between on feathered edges.
"""

import zlib

import numpy as np

Rect = tuple[int, int, int, int]  # (x0, y0, x1, y1) in mask pixels, end exclusive


def paint_segment(mask: np.ndarray, a, b, radius: float, feather: float, erase: bool = False) -> Rect | None:
    """Paints (or erases) a brush stroke from `a` to `b` (x, y in mask pixels, origin top left) into `mask`.

    Every pixel within `radius` of the segment is covered; the outer `feather` fraction of the radius fades out
    smoothly. Painting keeps the larger value, erasing the smaller, so going over a stroke again changes nothing.
    Returns the rectangle that may have changed, or None when the stroke misses the mask entirely.
    """
    radius = max(float(radius), 0.5)
    height, width = mask.shape
    x0 = max(int(np.floor(min(a[0], b[0]) - radius)), 0)
    y0 = max(int(np.floor(min(a[1], b[1]) - radius)), 0)
    x1 = min(int(np.ceil(max(a[0], b[0]) + radius)) + 1, width)
    y1 = min(int(np.ceil(max(a[1], b[1]) + radius)) + 1, height)
    if x0 >= x1 or y0 >= y1:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5  # pixel centers
    ax, ay = float(a[0]), float(a[1])
    dx, dy = float(b[0]) - ax, float(b[1]) - ay
    length_sq = dx * dx + dy * dy
    t = np.zeros_like(xs) if length_sq < 1e-12 else np.clip(((xs - ax) * dx + (ys - ay) * dy) / length_sq, 0.0, 1.0)
    distance = np.hypot(xs - (ax + t * dx), ys - (ay + t * dy))
    soft = max(radius * min(max(feather, 0.0), 1.0), 0.5)  # at least half a pixel, so edges stay antialiased
    alpha = np.clip((radius - distance) / soft, 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    stamp = np.round(alpha * 255.0).astype(np.uint8)
    region = mask[y0:y1, x0:x1]
    if erase:
        np.minimum(region, 255 - stamp, out=region)
    else:
        np.maximum(region, stamp, out=region)
    return (x0, y0, x1, y1)


def encode(mask: np.ndarray | None):
    """A compact, immutable snapshot of a mask for the undo history (None for no mask)."""
    if mask is None:
        return None
    return (mask.shape, zlib.compress(mask.tobytes(), 1))


def decode(snapshot) -> np.ndarray | None:
    if snapshot is None:
        return None
    shape, data = snapshot
    return np.frombuffer(zlib.decompress(data), np.uint8).reshape(shape).copy()
