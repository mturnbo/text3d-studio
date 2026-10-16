"""GPU tests of the GLSL noise library, rendered to an offscreen target and read back."""

import numpy as np
import pytest
from OpenGL import GL

from text3d_studio.render import shaders
from text3d_studio.render.gl_utils import RenderTarget, ShaderProgram, UnitQuad

SIZE = 256


def render_field(expression: str) -> np.ndarray:
    """Evaluates a GLSL float expression of `p` (8 cells across) over a square; returns it mapped to [0, 1]."""
    fragment = (
        "#version 410 core\nin vec2 v_uv;\nout vec4 frag;\n" + shaders.NOISE_GLSL
        + "void main() { vec3 p = vec3(v_uv * 8.0, 0.37); float n = " + expression + ";"
        + " frag = vec4(vec3(n * 0.5 + 0.5), 1.0); }\n"
    )
    program, quad, target = ShaderProgram(shaders.FULLSCREEN_VS, fragment), UnitQuad(), RenderTarget(SIZE, SIZE)
    try:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.fbo)
        GL.glViewport(0, 0, SIZE, SIZE)
        program.use()
        program.set_int("u_flip_y", 0)
        quad.draw()
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        data = GL.glReadPixels(0, 0, SIZE, SIZE, GL.GL_RED, GL.GL_UNSIGNED_BYTE)
    finally:
        program.delete()
        quad.delete()
        target.delete()
    return np.frombuffer(data, np.uint8).reshape(SIZE, SIZE).astype(float) / 255.0


@pytest.fixture(scope="module")
def noise(gl_context):
    return render_field("gradient_noise(p)")


@pytest.fixture(scope="module")
def fbm5(gl_context):
    return render_field("fbm(p, 5)")


def test_noise_is_centered_and_spans_a_useful_range(noise):
    assert abs(noise.mean() - 0.5) < 0.05
    assert noise.min() < 0.3 and noise.max() > 0.7
    assert ((noise <= 0.0) | (noise >= 1.0)).mean() < 0.001  # stays inside [-1, 1]


def test_noise_is_smooth(noise):
    # 8 cells over 256 px: neighbouring pixels must differ only slightly.
    assert np.abs(np.diff(noise, axis=0)).max() < 0.08
    assert np.abs(np.diff(noise, axis=1)).max() < 0.08


def test_noise_is_deterministic(gl_context, noise):
    np.testing.assert_array_equal(render_field("gradient_noise(p)"), noise)


def test_noise_is_not_trivially_periodic_at_cell_scale(noise):
    cell = SIZE // 8
    assert np.abs(noise[:, :cell] - noise[:, cell : 2 * cell]).mean() > 0.05


def test_fbm_adds_finer_detail(noise, fbm5):
    # Fine-scale variation relative to overall variation (fBm is normalized, so compare shares, not sizes).
    roughness = lambda field: np.abs(np.diff(field, axis=1)).mean() / field.std()
    assert roughness(fbm5) > 1.5 * roughness(noise)
    assert abs(fbm5.mean() - 0.5) < 0.05
