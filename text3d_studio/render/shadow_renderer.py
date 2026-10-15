"""ShadowRenderer: planar shadow catcher -> separable Gaussian blur -> multiply onto the background."""

from OpenGL import GL

from ..lighting import LightingState
from ..model import Scene
from . import shaders
from .gl_utils import ColorTarget, ShaderProgram, UnitQuad
from .text3d_renderer import Text3DRenderer

BLUR_TAPS = 12  # one-sided tap count in BLUR_FS
BLUR_ITERATIONS = 2


class ShadowRenderer:
    def __init__(self, quad: UnitQuad):
        self.quad = quad
        self.blur = ShaderProgram(shaders.FULLSCREEN_VS, shaders.BLUR_FS)
        self.composite = ShaderProgram(shaders.FULLSCREEN_VS, shaders.SHADOW_COMPOSITE_FS)
        self.ping = ColorTarget()
        self.pong = ColorTarget()

    def render_mask(self, text: Text3DRenderer, scene: Scene, proj, lighting: LightingState, viewport) -> None:
        """Renders the blurred shadow mask at half the viewport resolution into self.ping."""
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
        text.draw_shadow_caster(scene, proj, lighting)
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
        self.composite.set_vec3("u_color", scene.shadow.color)
        self.composite.set_float("u_opacity", scene.shadow.opacity)
        self.quad.draw()
        GL.glDisable(GL.GL_BLEND)

    def delete(self) -> None:
        self.blur.delete()
        self.composite.delete()
        self.ping.delete()
        self.pong.delete()
