"""Exporter: renders the composition offscreen at the background's native resolution."""

import numpy as np
from OpenGL import GL

from ..model import Document
from .compositor import Compositor
from .gl_utils import RenderTarget

EXPORT_SAMPLES = 8


def render_document(compositor: Compositor, document: Document) -> np.ndarray:
    """Returns an (H, W, 4) uint8 RGBA array, top row first. Requires a current GL context."""
    width, height = document.native_size
    max_size = int(GL.glGetIntegerv(GL.GL_MAX_RENDERBUFFER_SIZE))
    if max(width, height) > max_size:
        raise ValueError(f"Export size {width}x{height} exceeds GPU limit of {max_size}px per side.")

    samples = min(EXPORT_SAMPLES, int(GL.glGetIntegerv(GL.GL_MAX_SAMPLES)))
    multisampled = RenderTarget(width, height, samples)
    resolved = RenderTarget(width, height, 0)
    try:
        compositor.render(document.scene, multisampled.fbo, (0, 0, width, height))
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
