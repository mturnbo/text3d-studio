"""EffectsRenderer: finishing passes the Compositor runs around the text layer.

- Ground reflection: floor text drawn again, mirrored across the ground plane (light mirrored too), into a
  half-resolution layer; faded with depth below the ground and blurred by roughness, then put over the floor before
  the text.
- Text finishing: passes on the resolved text layer alone, each into a full-size copy (ping-pong), so the composite,
  the bloom and the transparent export all see the finished text, in this order: depth of field (a blur radius per
  pixel from the layer's depth, spread past the text's outline, then a separable blur by it), blur (premultiplied, so the edge
  softens into transparency), then colour grade and film grain (seeded value noise, fixed to the image) in one pass.
- Bloom: what is brighter than the threshold in the text layer, at half resolution, blurred and added over the
  composite. It is light, with no alpha of its own; the transparent export gives it its brightest channel as alpha.
"""

import dataclasses

import numpy as np
from OpenGL import GL

from .. import ground as ground_model
from .. import math3d
from ..model import EffectsParams, Scene
from . import shaders
from .gl_utils import ColorTarget, LayerTarget, ShaderProgram, Texture2D, UnitQuad

BLUR_TAPS = 12  # one-sided tap count in BLUR_RGBA_FS
BLUR_ITERATIONS = 2
REFLECTION_SAMPLES = 4
REFLECTION_BLUR = 0.04  # the reflection's blur radius at roughness 1, as a fraction of the image height
BLOOM_SPREAD = 0.15  # the glow's blur radius at bloom_radius 1, as a fraction of the image height
DOF_BLUR = 0.04  # the blur radius of text far out of focus at aperture 1, as a fraction of the image height
TEXT_BLUR = 0.02  # the text's blur radius at blur_radius 1, as a fraction of the image height
WARMTH_SHIFT = 0.2  # how much warmth 1 raises red and lowers blue (as gains)
GRAIN_CONTRAST = 0.3  # how far grain at amount 1 moves a channel (0-1) either way
GRAIN_SEED = 0x7E47  # the same grain in every frame and export


def graded_neutrally(effects) -> bool:
    neutral = EffectsParams()
    return all(getattr(effects, name) == getattr(neutral, name)
               for name in ("grade_brightness", "grade_contrast", "grade_saturation", "grade_warmth"))


class EffectsRenderer:
    def __init__(self, quad: UnitQuad):
        self.quad = quad
        self.blur = ShaderProgram(shaders.FULLSCREEN_VS, shaders.BLUR_RGBA_FS)
        self.scaled = ShaderProgram(shaders.FULLSCREEN_VS, shaders.SCALED_TEXTURE_FS)
        self.reflection_fade = ShaderProgram(shaders.FULLSCREEN_VS, shaders.REFLECTION_FADE_FS)
        self.bloom_extract = ShaderProgram(shaders.FULLSCREEN_VS, shaders.BLOOM_EXTRACT_FS)
        self.text_finish = ShaderProgram(shaders.FULLSCREEN_VS, shaders.TEXT_FINISH_FS)
        self.dof_coc = ShaderProgram(shaders.FULLSCREEN_VS, shaders.DOF_COC_FS)
        self.dof_blur = ShaderProgram(shaders.FULLSCREEN_VS, shaders.DOF_BLUR_FS)
        self.reflection_layer = LayerTarget()
        self.reflection = ColorTarget(GL.GL_RGBA8, GL.GL_RGBA)  # faded and blurred, premultiplied
        self.half_scratch = ColorTarget(GL.GL_RGBA8, GL.GL_RGBA)
        self.bloom = ColorTarget(GL.GL_RGBA8, GL.GL_RGBA)
        self.finished = (ColorTarget(GL.GL_RGBA8, GL.GL_RGBA), ColorTarget(GL.GL_RGBA8, GL.GL_RGBA))  # ping-pong
        self.coc = ColorTarget(GL.GL_R16F, GL.GL_RED)  # depth of field: blur radius per pixel
        self.has_reflection = False
        self.has_bloom = False

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

    # ----- Text finishing ----------------------------------------------------------------------------

    def finish_text_layer(self, scene: Scene, layer: LayerTarget, near: float, far: float) -> Texture2D:
        """Runs the passes that change the text itself on the resolved `layer` and returns the texture to composite:
        the layer's own color, untouched, when they are all off. `near` and `far`: the projection's clip planes."""
        effects = scene.effects
        color = layer.color
        if effects.dof_aperture > 0:
            max_coc = effects.dof_aperture * DOF_BLUR * layer.height
            self.coc.resize(layer.width, layer.height)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.coc.fbo)
            GL.glViewport(0, 0, layer.width, layer.height)
            program = self.dof_coc
            program.use()
            program.set_int("u_depth", 0)
            program.set_float("u_near", near)
            program.set_float("u_far", far)
            program.set_float("u_focus", effects.dof_focus)
            program.set_float("u_max_coc", max_coc)
            layer.depth.bind(0)
            self.quad.draw()
            target, scratch = self.finished
            target.resize(layer.width, layer.height)
            scratch.resize(layer.width, layer.height)
            program = self.dof_blur
            program.use()
            program.set_int("u_image", 0)
            program.set_int("u_coc", 1)
            program.set_float("u_step", 1.0 / BLUR_TAPS / BLUR_ITERATIONS)
            self.coc.texture.bind(1)
            for iteration in range(BLUR_ITERATIONS):
                first = color if iteration == 0 else target.texture
                for image, dest, direction in ((first, scratch, (1.0, 0.0)), (scratch.texture, target, (0.0, 1.0))):
                    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dest.fbo)
                    image.bind(0)
                    program.set_vec2("u_direction", direction)
                    self.quad.draw()
            color = target.texture
        if effects.blur_radius > 0:
            target, scratch = self.finished if color is not self.finished[1].texture else self.finished[::-1]
            target.resize(layer.width, layer.height)
            self._blur(target, scratch, max(0.5, effects.blur_radius * TEXT_BLUR * layer.height), source=color)
            color = target.texture
        if effects.grain_amount > 0 or not graded_neutrally(effects):
            target = self._spare(color, layer)
            program = self.text_finish
            program.use()
            program.set_int("u_image", 0)
            program.set_float("u_brightness", effects.grade_brightness)
            program.set_float("u_contrast", effects.grade_contrast)
            program.set_float("u_saturation", effects.grade_saturation)
            warmth = effects.grade_warmth * WARMTH_SHIFT
            program.set_vec3("u_warmth", (1.0 + warmth, 1.0, 1.0 - warmth))
            program.set_float("u_grain_amount", effects.grain_amount * GRAIN_CONTRAST)
            program.set_float("u_grain_cell", effects.grain_size / 1000 * layer.height)
            program.set_uint("u_seed", GRAIN_SEED)
            color.bind(0)
            self.quad.draw()
            color = target.texture
        return color

    def _spare(self, current: Texture2D, layer: LayerTarget) -> ColorTarget:
        """The finishing target not holding `current`, sized to the layer and bound as the draw target."""
        target = self.finished[1] if current is self.finished[0].texture else self.finished[0]
        target.resize(layer.width, layer.height)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.fbo)
        GL.glViewport(0, 0, layer.width, layer.height)
        return target

    # ----- Bloom ----------------------------------------------------------------------------------

    def render_bloom(self, scene: Scene, drawn, text_layer, viewport) -> bool:
        """Extracts and blurs the text layer's glow (`text_layer`: a texture the viewport's size) into self.bloom.
        False, drawing nothing, when bloom is off or no text was drawn."""
        effects = scene.effects
        self.has_bloom = effects.bloom_strength > 0 and bool(drawn)
        if not self.has_bloom:
            return False
        _, _, width, height = viewport
        w, h = max(1, width // 2), max(1, height // 2)
        self.bloom.resize(w, h)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.bloom.fbo)
        GL.glViewport(0, 0, w, h)
        program = self.bloom_extract
        program.use()
        program.set_int("u_flip_y", 0)
        program.set_int("u_image", 0)
        program.set_float("u_threshold", effects.bloom_threshold)
        text_layer.bind(0)
        self.quad.draw()
        self._blur(self.bloom, self.half_scratch, 0.5 + effects.bloom_radius * BLOOM_SPREAD * h)
        return True

    def composite_bloom(self, scene: Scene) -> None:
        """Adds the glow onto the bound target and viewport (alpha writes off)."""
        self._composite(self.bloom, scene.effects.bloom_strength, (GL.GL_ONE, GL.GL_ONE))


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

    def _blur(self, target: ColorTarget, scratch: ColorTarget, radius_px: float, source: Texture2D | None = None) -> None:
        """Blurs `target` in place (through `scratch`) by about `radius_px` of its pixels; nothing below half a pixel.
        With a `source` (the target's size), blurs that into `target` instead."""
        if radius_px < 0.5:
            return
        scratch.resize(target.width, target.height)
        step = radius_px / BLUR_TAPS / BLUR_ITERATIONS
        program = self.blur
        program.use()
        program.set_int("u_image", 0)
        program.set_int("u_flip_y", 0)
        GL.glViewport(0, 0, target.width, target.height)
        for iteration in range(BLUR_ITERATIONS):
            first = source if source is not None and iteration == 0 else target.texture
            for image, dest, direction in ((first, scratch, (step / target.width, 0.0)),
                                           (scratch.texture, target, (0.0, step / target.height))):
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dest.fbo)
                image.bind(0)
                program.set_vec2("u_step", direction)
                self.quad.draw()

    def delete(self) -> None:
        self.blur.delete()
        self.scaled.delete()
        self.reflection_fade.delete()
        self.bloom_extract.delete()
        self.text_finish.delete()
        self.dof_coc.delete()
        self.dof_blur.delete()
        self.coc.delete()
        for target in self.finished:
            target.delete()
        self.bloom.delete()
        self.reflection_layer.delete()
        self.reflection.delete()
        self.half_scratch.delete()
