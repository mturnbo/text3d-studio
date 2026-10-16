"""GridRenderer: editor-only overlay of the ground grid, the text's lanes and the horizon.

The canvas draws it after Compositor.render, reusing that frame's depth buffer so the text hides the
grid behind it. The exporter never calls it, so the grid can't leak into exported images.
"""

import math

import numpy as np
from OpenGL import GL

from .. import ground as ground_model
from .. import math3d
from ..model import Scene
from ..text_raster import GlyphRaster
from . import shaders
from .compositor import FAR, NEAR
from .gl_utils import ShaderProgram, UnitQuad

GRID_EXTENT = 400.0  # world units each way from the ground origin; lanes visually reach the horizon


class GridRenderer:
    def __init__(self, quad: UnitQuad):
        self.quad = quad
        self.grid = ShaderProgram(shaders.GRID_VS, shaders.GRID_FS)
        self.horizon = ShaderProgram(shaders.HORIZON_VS, shaders.HORIZON_FS)

    def render(
        self, scene: Scene, raster: GlyphRaster | None, target_fbo: int, viewport, pixel_ratio: float = 1.0
    ) -> None:
        x, y, width, height = viewport
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target_fbo)
        GL.glViewport(x, y, width, height)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthMask(GL.GL_FALSE)

        proj = math3d.perspective(scene.camera.fov, width / height, NEAR, FAR)
        self._draw_grid(scene, raster, proj, pixel_ratio)
        self._draw_horizon(scene, viewport, pixel_ratio)

        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)

    def _draw_grid(self, scene: Scene, raster: GlyphRaster | None, proj: np.ndarray, pixel_ratio: float) -> None:
        g, t, grid = scene.ground, scene.transform, scene.grid
        ground_to_camera = math3d.translation(*ground_model.ground_origin(g)) @ ground_model.ground_rotation(g)
        program = self.grid
        program.use()
        program.set_mat4("u_proj", math3d.to_gl(proj))
        program.set_mat4("u_ground", math3d.to_gl(ground_to_camera))
        program.set_float("u_extent", GRID_EXTENT)
        program.set_float("u_cell", grid.cell_size)
        program.set_vec3("u_color", grid.color)
        program.set_vec3("u_accent", grid.accent_color)
        program.set_float("u_opacity", grid.opacity)
        program.set_float("u_pixel_ratio", pixel_ratio)
        program.set_int("u_has_text", int(raster is not None))
        program.set_vec2("u_text_xz", (t.x, t.z))
        program.set_float("u_text_yaw", math.radians(t.rot_y))
        program.set_vec4("u_footprint", footprint(raster, t.size, t.depth))
        self.quad.draw()

    def _draw_horizon(self, scene: Scene, viewport, pixel_ratio: float) -> None:
        x, y, width, height = viewport
        g, t = scene.ground, scene.transform
        text_axes = ground_model.ground_rotation(g)[:3, :3] @ math3d.rotation_y(t.rot_y)[:3, :3]
        program = self.horizon
        program.use()
        program.set_vec3("u_up", ground_model.ground_up(g))
        program.set_vec4("u_viewport", (x, y, width, height))
        program.set_float("u_focal", math3d.focal_length_px(scene.camera.fov, height))
        program.set_vec3("u_color", scene.grid.color)
        program.set_vec3("u_accent", scene.grid.accent_color)
        program.set_float("u_opacity", scene.grid.opacity)
        program.set_float("u_pixel_ratio", pixel_ratio)
        # The text's baseline and depth directions: where its lanes meet on the horizon.
        for name, axis in (("u_vp0", text_axes[:, 0]), ("u_vp1", text_axes[:, 2])):
            vp = math3d.vanishing_point(axis, scene.camera.fov, (width, height))
            program.set_int(f"{name}_valid", int(vp is not None))
            if vp is not None:
                program.set_vec2(name, (x + vp[0], y + height - vp[1]))  # top-left pixels -> GL coords
        self.quad.draw()

    def delete(self) -> None:
        self.grid.delete()
        self.horizon.delete()


def footprint(raster: GlyphRaster | None, size: float, depth: float) -> tuple[float, float, float, float]:
    """The text's base rectangle on the ground, in its own frame: (x min, x max, z min, z max)."""
    if raster is None:
        return (0.0, 0.0, 0.0, 0.0)
    pad = -raster.origin[1]  # the raster pads the ink equally on every side, and the ink bottom sits at y = 0
    x_min = (raster.origin[0] + pad) * size
    x_max = (raster.origin[0] + raster.size[0] - pad) * size
    return (x_min, x_max, -depth * size, 0.0)
