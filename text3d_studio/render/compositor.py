"""Compositor: owns all GL resources and renders a full frame into any framebuffer.

It is independent of Qt widgets, so the on-screen canvas and the offscreen exporter
share exactly the same pipeline:  background -> soft shadow (multiply) -> layer-stacked text.
"""

import numpy as np
from OpenGL import GL

from .. import math3d
from ..lighting import LightingEngine
from ..model import Scene
from ..text_raster import GlyphRaster
from . import shaders
from .gl_utils import ShaderProgram, Texture2D, UnitQuad
from .shadow_renderer import ShadowRenderer
from .text3d_renderer import Text3DRenderer

NEAR, FAR = 0.1, 1000.0


class Compositor:
    def __init__(self):
        self.quad = UnitQuad()
        self.background_program = ShaderProgram(shaders.FULLSCREEN_VS, shaders.BACKGROUND_FS)
        self.background = Texture2D(mipmaps=True)
        self.background.upload(np.zeros((1, 1, 4), np.uint8))  # keeps the sampler complete before a photo loads
        self.has_background = False
        self.text = Text3DRenderer(self.quad)
        self.shadows = ShadowRenderer(self.quad)
        self.lighting = LightingEngine()
        self.max_texture_size = int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE))

    def set_background(self, pixels: np.ndarray | None) -> None:
        if pixels is None:
            self.has_background = False
            return
        height, width = pixels.shape[:2]
        if max(width, height) > self.max_texture_size:
            raise ValueError(
                f"Image is {width}x{height}; this GPU supports up to {self.max_texture_size}px per side."
            )
        self.background.upload(pixels)
        self.has_background = True

    def set_glyph(self, raster: GlyphRaster | None) -> None:
        self.text.set_raster(raster)

    def render(self, scene: Scene, target_fbo: int, viewport: tuple[int, int, int, int]) -> None:
        x, y, width, height = viewport
        proj = math3d.perspective(scene.camera.fov, width / height, NEAR, FAR)
        lighting = self.lighting.compute(scene.light, scene.ground)

        draw_shadow = scene.shadow.enabled and scene.shadow.opacity > 0 and self.text.has_text
        if draw_shadow:
            self.shadows.render_mask(self.text, scene, proj, lighting, viewport)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
        GL.glViewport(x, y, width, height)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glScissor(x, y, width, height)
        GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self._draw_background()
        if draw_shadow:
            self.shadows.composite_onto(scene)

        if self.text.has_text:
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LESS)
            GL.glEnable(GL.GL_SAMPLE_ALPHA_TO_COVERAGE)
            # Coverage drives antialiasing but must not reach the stored alpha: Qt composites the canvas as
            # premultiplied color, so alpha < 1 under bright pixels overflows into wrapped colors.
            GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_FALSE)
            reflect = self.has_background and scene.light.reflect_photo
            self.text.draw_beauty(scene, proj, lighting, height, self.background if reflect else None)
            GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
            GL.glDisable(GL.GL_SAMPLE_ALPHA_TO_COVERAGE)
            GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glDisable(GL.GL_SCISSOR_TEST)

    def _draw_background(self) -> None:
        program = self.background_program
        program.use()
        program.set_int("u_flip_y", 1)
        program.set_int("u_has_image", int(self.has_background))
        program.set_int("u_image", 0)
        self.background.bind(0)
        self.quad.draw()

    def delete(self) -> None:
        self.text.delete()
        self.shadows.delete()
        self.background_program.delete()
        self.background.delete()
        self.quad.delete()
