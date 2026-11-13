"""GPU tests of the GLSL noise library, rendered to an offscreen target and read back."""

import numpy as np
import pytest
from OpenGL import GL

from text3d_studio.render import shaders
from text3d_studio.render.gl_utils import RenderTarget, ShaderProgram, UnitQuad

SIZE = 256


def render_field(expression: str, position="vec3(v_uv * 8.0, 0.37)", noise=None, volume=None) -> np.ndarray:
    """Evaluates a GLSL float expression of `p` (by default 8 cells across) over a square; returns it mapped to
    [0, 1]. `noise` is the GLSL noise library (procedural by default); `volume`, a baked noise texture for it."""
    fragment = (
        "#version 410 core\nin vec2 v_uv;\nout vec4 frag;\n" + (noise or shaders.NOISE_GLSL)
        + "void main() { vec3 p = " + position + "; float n = " + expression + ";"
        + " frag = vec4(vec3(n * 0.5 + 0.5), 1.0); }\n"
    )
    program, quad, target = ShaderProgram(shaders.FULLSCREEN_VS, fragment), UnitQuad(), RenderTarget(SIZE, SIZE)
    try:
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.fbo)
        GL.glViewport(0, 0, SIZE, SIZE)
        program.use()
        program.set_int("u_flip_y", 0)
        if volume is not None:
            volume.bind(0)
            program.set_int("u_noise", 0)
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


# ----- Baked noise (text3d_renderer.BAKED_NOISE) --------------------------------------------------


@pytest.fixture(scope="module")
def volume(gl_context):
    from text3d_studio.render.text3d_renderer import NoiseVolume

    quad = UnitQuad()
    baked = NoiseVolume(quad)
    yield baked
    baked.delete()
    quad.delete()


def baked_field(volume, expression, position="vec3(v_uv * 8.0, 0.37)"):
    return render_field(expression, position, shaders.BAKED_NOISE_GLSL, volume)


def zero_crossings(field):
    """Fraction of horizontal neighbours on opposite sides of zero (0.5 once mapped)."""
    signs = field > 0.5
    return np.mean(signs[:, 1:] != signs[:, :-1])


@pytest.mark.parametrize("procedural, baked", [("gradient_noise(p)", "octave_noise(p)"), ("fbm(p, 5)", "fbm(p, 5)")])
def test_baked_noise_has_the_procedural_noises_statistics(gl_context, volume, procedural, baked):
    reference, field = render_field(procedural), baked_field(volume, baked)
    assert abs(field.mean() - reference.mean()) < 0.03
    assert field.std() == pytest.approx(reference.std(), rel=0.15)
    assert zero_crossings(field) == pytest.approx(zero_crossings(reference), rel=0.2)
    assert ((field <= 0.0) | (field >= 1.0)).mean() < 0.001  # stays inside [-1, 1]


def test_baked_noise_matches_the_procedural_noise_inside_its_first_period(gl_context, volume):
    # Inside [0, period) no lattice cell wraps, so the texture holds the procedural noise itself, interpolated.
    reference, field = render_field("gradient_noise(p)"), baked_field(volume, "octave_noise(p)")
    assert np.abs(field - reference).max() < 0.03


def test_baked_noise_tiles_seamlessly(gl_context, volume):
    period = float(shaders.NOISE_PERIOD)
    # 8 cells across, straddling the wrap at x = period in the middle of the square.
    across_seam = f"vec3({period - 4.0} + v_uv.x * 8.0, v_uv.y * 8.0, 2.37)"
    field = baked_field(volume, "octave_noise(p)", across_seam)
    step = np.abs(np.diff(field, axis=1))
    seam = SIZE // 2 - 1
    assert step[:, seam - 1 : seam + 2].max() <= step.max()  # nothing special at the seam
    assert step.max() < 0.08  # as smooth as the procedural noise
    shifted = baked_field(volume, "octave_noise(p)", f"vec3({2 * period - 4.0} + v_uv.x * 8.0, v_uv.y * 8.0, 2.37)")
    assert np.abs(shifted - field).max() <= 1.0 / 255.0  # one period on, the same noise


def test_baked_noise_wraps_negative_cells_too(gl_context, volume):
    period = float(shaders.NOISE_PERIOD)
    here = baked_field(volume, "octave_noise(p)", "vec3(v_uv * 8.0, 5.0)")
    there = baked_field(volume, "octave_noise(p)", f"vec3(v_uv * 8.0 + vec2({-period}, {3 * period}), {5.0 - period})")
    assert np.abs(here - there).max() <= 1.0 / 255.0  # negative cells wrap too
