"""EffectsRenderer: finishing passes the Compositor runs around the text layer.

- Ground reflection: floor text drawn again, mirrored across the ground plane (light mirrored too), into a
  half-resolution layer; faded with depth below the ground and blurred by roughness, then put over the floor before
  the text.
"""

import dataclasses

import numpy as np
from OpenGL import GL

from .. import ground as ground_model
from .. import math3d
from ..model import Scene
from . import shaders
from .gl_utils import ColorTarget, LayerTarget, ShaderProgram, UnitQuad

BLUR_TAPS = 12  # one-sided tap count in BLUR_RGBA_FS
BLUR_ITERATIONS = 2
REFLECTION_SAMPLES = 4
REFLECTION_BLUR = 0.04  # the reflection's blur radius at roughness 1, as a fraction of the image height


class EffectsRenderer:
    def __init__(self, quad: UnitQuad):
        self.quad = quad
        self.blur = ShaderProgram(shaders.FULLSCREEN_VS, shaders.BLUR_RGBA_FS)
        self.scaled = ShaderProgram(shaders.FULLSCREEN_VS, shaders.SCALED_TEXTURE_FS)
        self.reflection_fade = ShaderProgram(shaders.FULLSCREEN_VS, shaders.REFLECTION_FADE_FS)
        self.reflection_layer = LayerTarget()
        self.reflection = ColorTarget(GL.GL_RGBA8, GL.GL_RGBA)  # faded and blurred, premultiplied
        self.half_scratch = ColorTarget(GL.GL_RGBA8, GL.GL_RGBA)
        self.has_reflection = False

    # ----- Ground reflection --------------------------------------------------------------------

    def render_reflection(self, text, scene: Scene, drawn, proj, lighting, environment, viewport) -> bool:
        """Draws the floor text's reflection into self.reflection (half the viewport's size). Wall text isn't
        reflected: the floor's reflection of the wall itself isn't in the photo, so its text's would float alone.
        Returns False, drawing nothing, when the reflection is off or there is no floor text."""
        effects = scene.effects
        on_floor = [(obj, glyph) for obj, glyph in drawn if obj.transform.mount == "floor"]
        self.has_reflection = effects.reflection_strength > 0 and bool(on_floor)
        if not self.has_reflection:
            return False
        _, _, width, height = viewport
        w, h = max(1, width // 2), max(1, height // 2)
        layer = self.reflection_layer
        layer.resize(w, h, REFLECTION_SAMPLES)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, layer.draw.fbo)
        GL.glViewport(0, 0, w, h)
        GL.glClearColor(0, 0, 0, 0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glEnable(GL.GL_SAMPLE_ALPHA_TO_COVERAGE)
        GL.glEnable(GL.GL_SAMPLE_ALPHA_TO_ONE)
        # Mirroring the light with the text shades each mirrored point as the real one looks from the mirrored camera.
        mirror = math3d.reflection_matrix(lighting.ground_plane)
        mirrored = dataclasses.replace(lighting, light_dir=mirror[:3, :3] @ lighting.light_dir)
        for obj, glyph in on_floor:
            model = mirror @ ground_model.text_model_matrix(scene.ground, obj.transform, scene.wall)
            text.draw_beauty(scene, obj, glyph, proj, mirrored, h, environment, None, model=model)
        GL.glDisable(GL.GL_SAMPLE_ALPHA_TO_ONE)
        GL.glDisable(GL.GL_SAMPLE_ALPHA_TO_COVERAGE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        layer.resolve()

        self.reflection.resize(w, h)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.reflection.fbo)
        GL.glViewport(0, 0, w, h)
        program = self.reflection_fade
        program.use()
        program.set_int("u_flip_y", 0)
        layer.color.bind(0)
        layer.depth.bind(1)
        program.set_int("u_color", 0)
        program.set_int("u_depth", 1)
        program.set_mat4("u_inverse_proj", math3d.to_gl(np.linalg.inv(proj)))
        program.set_vec4("u_ground_plane", lighting.ground_plane)
        program.set_float("u_fade", effects.reflection_fade)
        self.quad.draw()
        self._blur(self.reflection, self.half_scratch, effects.reflection_roughness * REFLECTION_BLUR * h)
        return True

    def composite_reflection(self, scene: Scene, write_alpha: bool = False) -> None:
        """Puts the reflection over the bound target and viewport (premultiplied over). Alpha writes stay off unless
        asked for (reading the reflection back), as for every pass into the canvas."""
        self._composite(
            self.reflection, scene.effects.reflection_strength, (GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA), write_alpha
        )

    # ----- Shared -------------------------------------------------------------------------------

    def _composite(self, source: ColorTarget, scale: float, blend, write_alpha: bool = False) -> None:
        program = self.scaled
        program.use()
        program.set_int("u_flip_y", 0)
        program.set_int("u_image", 0)
        program.set_float("u_scale", scale)
        source.texture.bind(0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(*blend)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, write_alpha)
        self.quad.draw()
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)

    def _blur(self, target: ColorTarget, scratch: ColorTarget, radius_px: float) -> None:
        """Blurs `target` in place (through `scratch`) by about `radius_px` of its pixels; nothing below half a pixel."""
        if radius_px < 0.5:
            return
        scratch.resize(target.width, target.height)
        step = radius_px / BLUR_TAPS / BLUR_ITERATIONS
        program = self.blur
        program.use()
        program.set_int("u_image", 0)
        program.set_int("u_flip_y", 0)
        GL.glViewport(0, 0, target.width, target.height)
        for _ in range(BLUR_ITERATIONS):
            for source, dest, direction in ((target, scratch, (step / target.width, 0.0)),
                                            (scratch, target, (0.0, step / target.height))):
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dest.fbo)
                source.texture.bind(0)
                program.set_vec2("u_step", direction)
                self.quad.draw()

    def delete(self) -> None:
        self.blur.delete()
        self.scaled.delete()
        self.reflection_fade.delete()
        self.reflection_layer.delete()
        self.reflection.delete()
        self.half_scratch.delete()
