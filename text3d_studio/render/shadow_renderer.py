"""ShadowRenderer: planar shadow catcher -> separable Gaussian blur -> multiply onto the background.

With a wall in the photo (wall.enabled) the casters are also flattened onto the wall's plane into a second mask, and
the composite gives each pixel the shadow of the surface it shows (SHADOW_SURFACES_FS). The contact shadow is a third
mask: floor text flattened straight down onto the floor, fading out within shadow.contact_size em of the ground, with a
short blur, so it darkens the floor tightly around the text's base.
"""

import dataclasses
import math

import numpy as np
from OpenGL import GL

from .. import ground as ground_model
from .. import math3d
from ..lighting import MIN_SHADOW_ELEVATION, LightingState
from ..materials import TRANSMISSION
from ..model import MaterialParams, Scene, ShadowParams, TextObject
from . import shaders
from .gl_utils import ColorTarget, ShaderProgram, UnitQuad
from .text3d_renderer import Glyph, Text3DRenderer

BLUR_TAPS = 12  # one-sided tap count in BLUR_FS
BLUR_ITERATIONS = 2
CONTACT_SPREAD = 0.5  # the contact shadow's blur radius, as a fraction of contact_size (in on-screen em)


class ShadowRenderer:
    def __init__(self, quad: UnitQuad):
        self.quad = quad
        self.blur = ShaderProgram(shaders.FULLSCREEN_VS, shaders.BLUR_FS)
        self.composite = ShaderProgram(shaders.FULLSCREEN_VS, shaders.SHADOW_COMPOSITE_FS)
        self.surfaces = ShaderProgram(shaders.FULLSCREEN_VS, shaders.SHADOW_SURFACES_FS)
        # RGB: each channel's darkening, so every object's shadow keeps its own tint (see shadow_color).
        self.ping = ColorTarget(GL.GL_RGBA8, GL.GL_RGBA)  # the floor's shadow
        self.pong = ColorTarget(GL.GL_RGBA8, GL.GL_RGBA)  # blur scratch
        self.wall = ColorTarget(GL.GL_RGBA8, GL.GL_RGBA)  # the wall's shadow
        self.contact = ColorTarget(GL.GL_RGBA8, GL.GL_RGBA)  # the contact shadow
        # For the composite, from the last render_mask: (tan_half, ground plane, wall plane or None, contact drawn),
        # or None when only the floor's shadow was drawn.
        self._surfaces = None

    def render_mask(
        self, text: Text3DRenderer, scene: Scene, casters: list[tuple[TextObject, Glyph]], proj,
        lighting: LightingState, viewport, strength_only: bool = False,
    ) -> None:
        """Renders every caster's shadow into one blurred mask at half the viewport resolution, in self.ping.

        `strength_only` darkens every channel fully, so the mask holds each pixel's shadow strength alone (the
        transparent export divides the colored mask by it to recover the shadow's color)."""
        _, _, width, height = viewport
        sw, sh = max(1, width // 2), max(1, height // 2)
        self.ping.resize(sw, sh)
        self.pong.resize(sw, sh)

        self._draw_casters(self.ping, text, scene, casters, proj, lighting, strength_only)
        # Softness is relative to image height so preview and full-res export match.
        radius_px = 0.5 + scene.shadow.softness * sh * 0.08
        self._blur(self.ping, radius_px)

        if scene.wall.enabled:
            self.wall.resize(sw, sh)
            wall_lighting = self.wall_lighting(scene, lighting)
            self._draw_casters(self.wall, text, scene, casters if wall_lighting else [], proj, wall_lighting, strength_only)
            if wall_lighting is not None:
                self._blur(self.wall, radius_px)

        on_floor = [(obj, glyph) for obj, glyph in casters if obj.transform.mount == "floor"]
        contact = scene.shadow.contact_strength > 0 and bool(on_floor)
        if contact:
            self.contact.resize(sw, sh)
            contact_scene = dataclasses.replace(
                scene, shadow=dataclasses.replace(scene.shadow, falloff=1.0 / scene.shadow.contact_size)
            )
            down = dataclasses.replace(lighting, shadow_matrix=math3d.planar_shadow_matrix(
                lighting.ground_plane, ground_model.ground_up(scene.ground)
            ))
            self._draw_casters(self.contact, text, contact_scene, on_floor, proj, down, strength_only)
            self._blur(self.contact, 0.5 + CONTACT_SPREAD * scene.shadow.contact_size * _mean_em_px(scene, on_floor, sh))

        self._surfaces = None
        if scene.wall.enabled or contact:
            tan_half = math.tan(math.radians(scene.camera.fov) / 2)
            wall_plane = ground_model.wall_plane(scene.ground, scene.wall) if scene.wall.enabled else None
            self._surfaces = ((tan_half * width / height, tan_half), ground_model.ground_plane(scene.ground), wall_plane, contact)

    @staticmethod
    def wall_lighting(scene: Scene, lighting: LightingState) -> LightingState | None:
        """The lighting with the shadow flattened onto the wall's plane, and heights measured from the wall (so the
        falloff fades with distance from it). None when the light is behind the wall or grazes it: the wall then
        catches no shadow."""
        normal = ground_model.wall_normal(scene.ground, scene.wall)
        if float(np.dot(lighting.light_dir, normal)) < math.sin(math.radians(MIN_SHADOW_ELEVATION)):
            return None
        plane = ground_model.wall_plane(scene.ground, scene.wall)
        return dataclasses.replace(
            lighting, ground_plane=plane, shadow_matrix=math3d.planar_shadow_matrix(plane, lighting.light_dir)
        )

    def _draw_casters(self, target: ColorTarget, text, scene, casters, proj, lighting, strength_only) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.fbo)
        GL.glViewport(0, 0, target.width, target.height)
        GL.glClearColor(0, 0, 0, 0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendEquation(GL.GL_MAX)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
        for obj, glyph in casters:
            darken = (1.0, 1.0, 1.0) if strength_only else tuple(1.0 - c for c in shadow_color(scene.shadow, obj.material))
            text.draw_shadow_caster(scene, obj, glyph, proj, lighting, darken)
        GL.glBlendEquation(GL.GL_FUNC_ADD)
        GL.glDisable(GL.GL_BLEND)

    def _blur(self, target: ColorTarget, radius_px: float) -> None:
        """Blurs the target in place (through self.pong) by about radius_px of its pixels."""
        step = radius_px / BLUR_TAPS / BLUR_ITERATIONS
        self.blur.use()
        self.blur.set_int("u_image", 0)
        self.blur.set_int("u_flip_y", 0)
        GL.glViewport(0, 0, target.width, target.height)
        for _ in range(BLUR_ITERATIONS):
            self._blur_pass(target, self.pong, (step / target.width, 0.0))
            self._blur_pass(self.pong, target, (0.0, step / target.height))

    def _blur_pass(self, source: ColorTarget, dest: ColorTarget, step) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dest.fbo)
        source.texture.bind(0)
        self.blur.set_vec2("u_step", step)
        self.quad.draw()

    def composite_onto(self, scene: Scene) -> None:
        """Multiplies the blurred masks from the last render_mask onto whatever framebuffer/viewport is bound."""
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_DST_COLOR, GL.GL_ZERO)
        if self._surfaces is None:
            self.composite.use()
            self.ping.texture.bind(0)
            self.composite.set_int("u_image", 0)
            self.composite.set_int("u_flip_y", 0)
            self.composite.set_float("u_opacity", scene.shadow.opacity)
        else:
            tan_half, ground_plane, wall_plane, contact = self._surfaces
            program = self.surfaces
            program.use()
            self.ping.texture.bind(0)
            self.wall.texture.bind(1)
            self.contact.texture.bind(2)
            program.set_int("u_floor", 0)
            program.set_int("u_wall", 1)
            program.set_int("u_contact", 2)
            program.set_int("u_flip_y", 0)
            program.set_float("u_opacity", scene.shadow.opacity)
            program.set_float("u_contact_strength", scene.shadow.contact_strength if contact else 0.0)
            program.set_int("u_has_wall", int(wall_plane is not None))
            program.set_vec2("u_tan_half", tan_half)
            program.set_vec4("u_ground_plane", ground_plane)
            program.set_vec4("u_wall_plane", wall_plane if wall_plane is not None else (0.0, 0.0, 0.0, 0.0))
        self.quad.draw()
        GL.glDisable(GL.GL_BLEND)

    def delete(self) -> None:
        self.blur.delete()
        self.composite.delete()
        self.surfaces.delete()
        self.ping.delete()
        self.pong.delete()
        self.wall.delete()
        self.contact.delete()


def casts_shadow(scene: Scene) -> bool:
    """Whether the shadow pass has anything to draw: the shadow is on, with some opacity or a contact shadow."""
    return scene.shadow.enabled and (scene.shadow.opacity > 0 or scene.shadow.contact_strength > 0)


def _mean_em_px(scene: Scene, casters, height: int) -> float:
    """The casters' average on-screen em, in pixels of a target height pixels tall."""
    focal = (height / 2) / math.tan(math.radians(scene.camera.fov) / 2)
    sizes = [obj.transform.size * focal / ground_model.text_distance(scene.ground, obj.transform, scene.wall)
             for obj, _ in casters]
    return sum(sizes) / len(sizes)


def shadow_color(shadow: ShadowParams, material: MaterialParams) -> tuple[float, float, float]:
    """The color of an object's shadow at full strength. Light passing through a transmissive material lightens
    the shadow and takes on the material's tint."""
    transmission = TRANSMISSION.get(material.kind, 0.0)
    tint = material.face_color
    return tuple(s * (1 - transmission) + t * transmission for s, t in zip(shadow.color, tint))
