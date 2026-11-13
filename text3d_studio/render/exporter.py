"""Exporter: renders the composition offscreen at the background's native resolution (or a scale of it)."""

import os
from dataclasses import dataclass

import numpy as np
from OpenGL import GL

from ..model import Document
from .compositor import Compositor
from .gl_utils import ColorTarget, RenderTarget

EXPORT_SAMPLES = 8


def scaled_size(size: tuple[int, int], scale: float) -> tuple[int, int]:
    return (max(1, round(size[0] * scale)), max(1, round(size[1] * scale)))


def render_document(
    compositor: Compositor, document: Document, transparent: bool = False, scale: float = 1.0
) -> np.ndarray:
    """Returns an (H, W, 4) uint8 RGBA array, top row first. Requires a current GL context.

    `transparent` leaves out the photo: straight-alpha text and shadow over transparency (see compose_transparent).
    Glass still refracts the photo and shadow behind it, since that is what it shows.
    `scale` renders at that fraction of the native size. The scene is resolution independent (the projection
    follows the aspect ratio, shadow softness is relative to the image height), so text, shadow and every
    other effect scale with the image, as in the preview.
    """
    width, height = scaled_size(document.native_size, scale)
    max_size = int(GL.glGetIntegerv(GL.GL_MAX_RENDERBUFFER_SIZE))
    if max(width, height) > max_size:
        raise ValueError(f"Export size {width}x{height} exceeds GPU limit of {max_size}px per side.")

    samples = min(EXPORT_SAMPLES, int(GL.glGetIntegerv(GL.GL_MAX_SAMPLES)))
    multisampled = RenderTarget(width, height, samples)
    resolved = RenderTarget(width, height, 0)
    try:
        compositor.render(document.scene, multisampled.fbo, (0, 0, width, height))
        if transparent:
            return _render_transparent(compositor, document, width, height)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, multisampled.fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, resolved.fbo)
        GL.glBlitFramebuffer(
            0, 0, width, height, 0, 0, width, height, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, resolved.fbo)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        data = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
    finally:
        multisampled.delete()
        resolved.delete()

    pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)[::-1].copy()
    pixels[..., 3] = 255
    return pixels


def _render_transparent(compositor: Compositor, document: Document, width: int, height: int) -> np.ndarray:
    """Reads the text layer the render just made, then renders the shadow's factor twice (colored, and strength
    only) into a float target, and combines them."""
    scene = document.scene
    if any(obj.id in compositor.glyphs for obj in scene.objects):
        compositor.text_layer.color.bind(0)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        data = GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        text = np.frombuffer(data, np.uint8).reshape(height, width, 4)[::-1]
    else:
        text = np.zeros((height, width, 4), np.uint8)

    factors = []
    target = ColorTarget(GL.GL_RGBA16F, GL.GL_RGBA)
    try:
        target.resize(width, height)
        for strength_only in (False, True):
            if not compositor.render_shadow_factor(scene, target.fbo, (0, 0, width, height), strength_only):
                break
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.fbo)
            data = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_FLOAT)
            factors.append(np.frombuffer(data, np.float32).reshape(height, width, 4)[::-1, :, :3])
    finally:
        target.delete()
    if not factors:
        factors = [np.ones((height, width, 3), np.float32)] * 2
    pixels = compose_transparent(text, factors[0], factors[1][..., :1])
    occlusion = compositor.read_occlusion(width, height)
    if occlusion is not None:  # the photo is in front there: the text and shadow are hidden
        pixels[..., 3] = np.round(pixels[..., 3] * (1.0 - occlusion)).astype(np.uint8)
    return pixels


def compose_transparent(text: np.ndarray, shadow_factor: np.ndarray, strength_factor: np.ndarray) -> np.ndarray:
    """Straight-alpha RGBA: the text over its shadow over nothing.

    `text`: the premultiplied text layer, (H, W, 4) uint8. `shadow_factor`: what the shadow multiplies the photo
    by per channel, 1 − opacity × strength × (1 − shadow color), (H, W, 3). `strength_factor`: 1 − opacity ×
    strength, (H, W, 1). The shadow becomes its color at alpha = opacity × strength.
    """
    text = text.astype(np.float32) / 255.0
    text_alpha = text[..., 3:]
    shadow_alpha = np.clip(1.0 - strength_factor, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shadow_color = np.where(shadow_alpha > 0, 1.0 - (1.0 - shadow_factor) / shadow_alpha, 0.0)
    shadow_color = np.clip(shadow_color, 0.0, 1.0)
    alpha = text_alpha + shadow_alpha * (1.0 - text_alpha)
    premultiplied = text[..., :3] + shadow_color * shadow_alpha * (1.0 - text_alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        color = np.where(alpha > 0, premultiplied / alpha, 0.0)
    pixels = np.empty(text.shape, np.uint8)
    pixels[..., :3] = np.round(np.clip(color, 0.0, 1.0) * 255)
    pixels[..., 3:] = np.round(alpha * 255)
    return pixels


# ----- Batch export -------------------------------------------------------------------------------

BATCH_FORMATS = ("png", "jpg", "transparent")  # "transparent" is a straight-alpha PNG of the text and shadow


@dataclass(frozen=True)
class BatchJob:
    path: str
    size: tuple[int, int]
    scale: float
    transparent: bool


def long_edge_scale(native_size: tuple[int, int], long_edge: int) -> float:
    """The scale that makes the image's longer side `long_edge` pixels."""
    return long_edge / max(native_size)


def batch_jobs(
    stem: str, native_size: tuple[int, int], scales, formats, folder: str
) -> list[BatchJob]:
    """One file per size and format, named <stem>_3d_<width>x<height>.<ext> (transparent PNGs end in
    _transparent.png). Scales that give the same size are exported once. Existing files are never replaced:
    the new file gets -2, -3, … before its extension instead."""
    jobs, taken = [], set()
    for scale in scales:
        width, height = scaled_size(native_size, scale)
        for kind in formats:
            if kind not in BATCH_FORMATS:
                raise ValueError(f"Unknown export format: {kind!r}")
            transparent = kind == "transparent"
            name = f"{stem}_3d_{width}x{height}" + ("_transparent.png" if transparent else f".{kind}")
            if name in taken:
                continue
            taken.add(name)
            jobs.append(BatchJob(_free_path(os.path.join(folder, name)), (width, height), scale, transparent))
    return jobs


def _free_path(path: str) -> str:
    root, extension = os.path.splitext(path)
    number = 2
    while os.path.exists(path):
        path = f"{root}-{number}{extension}"
        number += 1
    return path
