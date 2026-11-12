"""ShadowRenderer: planar shadow catcher -> separable Gaussian blur -> multiply onto the background."""

from OpenGL import GL

from ..lighting import LightingState
from ..materials import TRANSMISSION
from ..model import MaterialParams, Scene, ShadowParams, TextObject
from . import shaders
from .gl_utils import ColorTarget, ShaderProgram, UnitQuad
from .text3d_renderer import Glyph, Text3DRenderer

BLUR_TAPS = 12  # one-sided tap count in BLUR_FS
BLUR_ITERATIONS = 2


class ShadowRenderer:
    def __init__(self, quad: UnitQuad):
        self.quad = quad
        self.blur = ShaderProgram(shaders.FULLSCREEN_VS, shaders.BLUR_FS)
        self.composite = ShaderProgram(shaders.FULLSCREEN_VS, shaders.SHADOW_COMPOSITE_FS)
        # RGB: each channel's darkening, so every object's shadow keeps its own tint (see shadow_color).
        self.ping = ColorTarget(GL.GL_RGBA8, GL.GL_RGBA)
        self.pong = ColorTarget(GL.GL_RGBA8, GL.GL_RGBA)

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

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.ping.fbo)
        GL.glViewport(0, 0, sw, sh)
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

        # Softness is relative to image height so preview and full-res export match.
        radius_px = 0.5 + scene.shadow.softness * sh * 0.08
        step = radius_px / BLUR_TAPS / BLUR_ITERATIONS
        self.blur.use()
        self.blur.set_int("u_image", 0)
        self.blur.set_int("u_flip_y", 0)
        for _ in range(BLUR_ITERATIONS):
            self._blur_pass(self.ping, self.pong, (step / sw, 0.0))
            self._blur_pass(self.pong, self.ping, (0.0, step / sh))

    def _blur_pass(self, source: ColorTarget, dest: ColorTarget, step) -> None:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dest.fbo)
        source.texture.bind(0)
        self.blur.set_vec2("u_step", step)
        self.quad.draw()

    def composite_onto(self, scene: Scene) -> None:
        """Multiplies the blurred mask onto whatever framebuffer/viewport is currently bound."""
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_DST_COLOR, GL.GL_ZERO)
        self.composite.use()
        self.ping.texture.bind(0)
        self.composite.set_int("u_image", 0)
        self.composite.set_int("u_flip_y", 0)
        self.composite.set_float("u_opacity", scene.shadow.opacity)
        self.quad.draw()
        GL.glDisable(GL.GL_BLEND)

    def delete(self) -> None:
        self.blur.delete()
        self.composite.delete()
        self.ping.delete()
        self.pong.delete()


def shadow_color(shadow: ShadowParams, material: MaterialParams) -> tuple[float, float, float]:
    """The color of an object's shadow at full strength. Light passing through a transmissive material lightens
    the shadow and takes on the material's tint."""
    transmission = TRANSMISSION.get(material.kind, 0.0)
    tint = material.face_color
    return tuple(s * (1 - transmission) + t * transmission for s, t in zip(shadow.color, tint))
