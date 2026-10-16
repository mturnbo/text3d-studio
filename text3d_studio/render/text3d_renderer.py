"""Text3DRenderer: draws the glyph mask as an instanced stack of duplicated layers."""

import math

import numpy as np
from OpenGL import GL

from .. import ground as ground_model
from .. import math3d
from ..lighting import LightingState, srgb_to_linear
from ..model import Scene
from ..text_raster import GlyphRaster
from . import shaders
from .gl_utils import ShaderProgram, Texture2D, TriangleBuffer, UnitQuad

IDENTITY_GL = math3d.to_gl(math3d.identity())
PATTERNS = {"stone": 1, "wood": 2}  # material kinds with a procedural texture (see LAYER_STACK_FS)
MAX_LAYERS = 2048
LAYERS_PER_PIXEL = 1.5  # oversample so neighbouring slices always overlap on screen

# Depth pre-pass: draw the stack once into depth only, then shade with GL_EQUAL so each pixel is shaded for
# its visible slice rather than every slice behind it. Measured gain is modest on Apple GPUs, which already
# reject most hidden slices: ~11% for stone without MSAA, ~0 with it, where the real cost is side-wall pixels
# whose samples belong to several slices. (An early synthetic benchmark suggested 2x; real textures did not.)
# The shading pass keeps its alpha test. False renders the stack in one pass.
DEPTH_PREPASS = True


class Text3DRenderer:
    def __init__(self, quad: UnitQuad):
        self.quad = quad
        self.depth_prepass = DEPTH_PREPASS
        self.beauty = ShaderProgram(shaders.LAYER_STACK_VS, shaders.LAYER_STACK_FS)
        self.depth = ShaderProgram(shaders.LAYER_STACK_VS, shaders.LAYER_DEPTH_FS)
        self.cap_shadow = ShaderProgram(shaders.LAYER_STACK_VS, shaders.SHADOW_FS)
        self.wall_shadow = ShaderProgram(shaders.WALL_VS, shaders.SHADOW_FS)
        self.mask = Texture2D(mipmaps=True)
        self.walls = TriangleBuffer()
        self.raster: GlyphRaster | None = None

    def set_raster(self, raster: GlyphRaster | None) -> None:
        self.raster = raster
        if raster is not None:
            self.mask.upload(raster.rgba)
            self.walls.upload(raster.walls)

    @property
    def has_text(self) -> bool:
        return self.raster is not None

    def draw_beauty(
        self, scene: Scene, proj: np.ndarray, lighting: LightingState, viewport_height: int,
        environment: Texture2D | None = None,
    ) -> None:
        """`environment`: the background photo to reflect, or None to reflect the ambient color instead."""
        layers = self.effective_layers(scene, viewport_height)
        if self.depth_prepass:
            self._bind_common(self.depth, scene, proj, IDENTITY_GL, lighting)
            self.depth.set_int("u_layers", layers)
            GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
            self.quad.draw(layers)
            GL.glDepthFunc(GL.GL_EQUAL)
            GL.glDepthMask(GL.GL_FALSE)

        # Coverage drives antialiasing but must not reach the stored alpha: Qt composites the canvas as
        # premultiplied color, so alpha < 1 under bright pixels overflows into wrapped colors.
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_FALSE)
        program = self.beauty
        self._bind_common(program, scene, proj, IDENTITY_GL, lighting)
        program.set_int("u_layers", layers)
        m = scene.material
        program.set_vec3("u_light_dir", lighting.light_dir)
        program.set_vec3("u_light_color", lighting.light_color)
        program.set_vec3("u_ambient", lighting.ambient)
        program.set_vec3("u_face_color", srgb_to_linear(m.face_color))
        program.set_vec3("u_side_color", srgb_to_linear(m.side_color))
        program.set_float("u_side_darkening", m.side_darkening)
        program.set_float("u_bevel_strength", m.bevel_strength)
        program.set_float("u_metallic", m.metallic)
        program.set_float("u_roughness", m.roughness)
        program.set_int("u_pattern", PATTERNS.get(m.kind, 0))
        program.set_vec3("u_secondary_color", srgb_to_linear(m.secondary_color))
        program.set_float("u_texture_scale", m.texture_scale)
        program.set_float("u_texture_strength", m.texture_strength)
        program.set_float("u_bump_strength", m.bump_strength)
        program.set_float("u_grain_angle", math.radians(m.grain_angle))
        program.set_float("u_varnish", m.varnish)
        program.set_int("u_has_environment", int(environment is not None))
        if environment is not None:
            environment.bind(1)
            program.set_int("u_environment", 1)
            program.set_float("u_environment_lod", environment.max_lod)
            program.set_float("u_environment_strength", scene.light.reflection_strength)
        self.quad.draw(layers)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthMask(GL.GL_TRUE)

    def draw_shadow_caster(self, scene: Scene, proj: np.ndarray, lighting: LightingState) -> None:
        """Front and back slices plus the traced side walls form a closed caster for any light angle."""
        flatten = math3d.to_gl(lighting.shadow_matrix)
        caps = min(2, self._layer_count(scene))

        program = self.cap_shadow
        self._bind_common(program, scene, proj, flatten, lighting)
        program.set_int("u_layers", caps)
        program.set_int("u_use_mask", 1)
        program.set_float("u_falloff", scene.shadow.falloff)
        self.quad.draw(caps)

        if caps > 1:
            program = self.wall_shadow
            self._bind_common(program, scene, proj, flatten, lighting)
            program.set_int("u_use_mask", 0)
            program.set_float("u_falloff", scene.shadow.falloff)
            self.walls.draw()

    @staticmethod
    def effective_layers(scene: Scene, viewport_height: int) -> int:
        """The user's layer count, raised if needed so slices never separate into visible bands."""
        t = scene.transform
        if t.depth <= 1e-4:
            return 1
        focal_px = (viewport_height / 2) / math.tan(math.radians(scene.camera.fov) / 2)
        depth_px = t.depth * t.size * focal_px / ground_model.text_distance(scene.ground, t)
        needed = math.ceil(depth_px * LAYERS_PER_PIXEL)
        return min(MAX_LAYERS, max(int(t.layers), needed))

    @staticmethod
    def _layer_count(scene: Scene) -> int:
        t = scene.transform
        return max(1, int(t.layers)) if t.depth > 1e-4 else 1

    def _bind_common(self, program, scene, proj, flatten_gl, lighting) -> None:
        t = scene.transform
        model = ground_model.text_model_matrix(scene.ground, t)
        program.use()
        self.mask.bind(0)
        program.set_int("u_mask", 0)
        program.set_mat4("u_model", math3d.to_gl(model))
        program.set_mat4("u_proj", math3d.to_gl(proj))
        program.set_mat4("u_flatten", flatten_gl)
        program.set_vec2("u_origin", self.raster.origin)
        program.set_vec2("u_size", self.raster.size)
        program.set_float("u_depth", t.depth)
        program.set_vec4("u_ground_plane", lighting.ground_plane)
        program.set_float("u_world_size", t.size)

    def delete(self) -> None:
        self.beauty.delete()
        self.depth.delete()
        self.cap_shadow.delete()
        self.wall_shadow.delete()
        self.mask.delete()
        self.walls.delete()
