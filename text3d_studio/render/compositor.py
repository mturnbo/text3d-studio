"""Compositor: owns all GL resources and renders a full frame into any framebuffer.

It is independent of Qt widgets, so the on-screen canvas and the offscreen exporter
share exactly the same pipeline:  background -> soft shadow (multiply) -> text layer (composited over).
The text is drawn into its own multisampled layer, resolved, and composited over the target, so later
passes can work on the text alone. Transmissive text (glass) first copies the background and shadow into
a mipmapped backdrop texture, which it samples, refracted and blurred, to show what is behind it.
"""

import numpy as np
from OpenGL import GL

from .. import math3d
from ..lighting import LightingEngine
from ..materials import TRANSMISSION
from ..model import Scene
from ..text_raster import GlyphRaster
from . import shaders
from .gl_utils import ColorTarget, LayerTarget, ShaderProgram, Texture2D, UnitQuad
from .shadow_renderer import ShadowRenderer
from .text3d_renderer import Text3DRenderer

NEAR, FAR = 0.1, 1000.0
MIN_LAYER_SAMPLES = 4  # the text layer antialiases even when the target isn't multisampled


class Compositor:
    def __init__(self):
        self.quad = UnitQuad()
        self.background_program = ShaderProgram(shaders.FULLSCREEN_VS, shaders.BACKGROUND_FS)
        self.background = Texture2D(mipmaps=True)
        self.background.upload(np.zeros((1, 1, 4), np.uint8))  # keeps the sampler complete before a photo loads
        self.has_background = False
        self.backdrop = ColorTarget(GL.GL_RGBA8, GL.GL_RGBA, mipmaps=True)
        self.text_layer = LayerTarget()
        self.layer_program = ShaderProgram(shaders.FULLSCREEN_VS, shaders.LAYER_COMPOSITE_FS)
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
        target_samples = int(GL.glGetIntegerv(GL.GL_SAMPLES))
        self.background_viewport = viewport
        GL.glViewport(x, y, width, height)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glScissor(x, y, width, height)
        GL.glClearColor(0, 0, 0, 1)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        self._draw_background()
        if draw_shadow:
            self.shadows.composite_onto(scene)

        if self.text.has_text:
            backdrop = None
            if scene.material.kind in TRANSMISSION:
                self._capture_backdrop(target_fbo, viewport)
                backdrop = self.backdrop.texture
            GL.glDisable(GL.GL_SCISSOR_TEST)
            self._draw_text_layer(scene, proj, lighting, backdrop, max(MIN_LAYER_SAMPLES, target_samples))
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
            GL.glViewport(x, y, width, height)
            GL.glEnable(GL.GL_SCISSOR_TEST)
            self._composite_text_layer(viewport)

        GL.glDisable(GL.GL_SCISSOR_TEST)

    def _draw_text_layer(self, scene: Scene, proj, lighting, backdrop, samples: int) -> None:
        """Draws the text alone into self.text_layer and resolves it."""
        layer = self.text_layer
        layer.resize(self.background_viewport[2], self.background_viewport[3], samples)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, layer.draw.fbo)
        GL.glViewport(0, 0, layer.width, layer.height)
        GL.glClearColor(0, 0, 0, 0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        # Coverage picks the samples; each sample written is then fully opaque, so resolving averages the
        # text over the transparent clear into premultiplied color and a coverage alpha.
        GL.glEnable(GL.GL_SAMPLE_ALPHA_TO_COVERAGE)
        GL.glEnable(GL.GL_SAMPLE_ALPHA_TO_ONE)
        reflect = self.has_background and scene.light.reflect_photo
        self.text.draw_beauty(
            scene, proj, lighting, layer.height, self.background if reflect else None, backdrop
        )
        GL.glDisable(GL.GL_SAMPLE_ALPHA_TO_ONE)
        GL.glDisable(GL.GL_SAMPLE_ALPHA_TO_COVERAGE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        layer.resolve()

    def _composite_text_layer(self, viewport) -> None:
        """Puts the resolved layer over the bound target, writing its depth for the grid overlay.

        Alpha writes stay off: Qt composites the canvas as premultiplied color, so stored alpha below 1
        would let bright pixels overflow into wrapped colors.
        """
        x, y, _, _ = viewport
        program = self.layer_program
        program.use()
        self.text_layer.color.bind(0)
        self.text_layer.depth.bind(1)
        program.set_int("u_color", 0)
        program.set_int("u_depth", 1)
        program.set_vec2("u_origin", (x, y))
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_ALWAYS)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_FALSE)
        self.quad.draw()
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)

    def _capture_backdrop(self, target_fbo: int, viewport) -> None:
        """Copies the viewport (resolving any multisampling) into self.backdrop and builds its mipmaps."""
        x, y, width, height = viewport
        self.backdrop.resize(width, height)
        GL.glDisable(GL.GL_SCISSOR_TEST)  # the scissor would clip the copy, which lands at the origin
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, target_fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.backdrop.fbo)
        GL.glBlitFramebuffer(
            x, y, x + width, y + height, 0, 0, width, height, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        self.backdrop.texture.bind(0)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

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
        self.backdrop.delete()
        self.text_layer.delete()
        self.layer_program.delete()
        self.background_program.delete()
        self.background.delete()
        self.quad.delete()
