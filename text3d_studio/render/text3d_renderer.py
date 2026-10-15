"""Text3DRenderer: draws the glyph mask as an instanced stack of duplicated layers."""

import numpy as np

from .. import math3d
from ..lighting import LightingState
from ..model import Scene
from ..text_raster import GlyphRaster
from . import shaders
from .gl_utils import ShaderProgram, Texture2D, UnitQuad

IDENTITY_GL = math3d.to_gl(math3d.identity())


class Text3DRenderer:
    def __init__(self, quad: UnitQuad):
        self.quad = quad
        self.beauty = ShaderProgram(shaders.LAYER_STACK_VS, shaders.LAYER_STACK_FS)
        self.shadow = ShaderProgram(shaders.LAYER_STACK_VS, shaders.SHADOW_FS)
        self.mask = Texture2D(mipmaps=True)
        self.raster: GlyphRaster | None = None

    def set_raster(self, raster: GlyphRaster | None) -> None:
        self.raster = raster
        if raster is not None:
            self.mask.upload(raster.rgba)

    @property
    def has_text(self) -> bool:
        return self.raster is not None

    def draw_beauty(self, scene: Scene, proj: np.ndarray, lighting: LightingState) -> None:
        program = self.beauty
        layers = self._bind_common(program, scene, proj, IDENTITY_GL, lighting)
        m = scene.material
        program.set_vec3("u_light_dir", lighting.light_dir)
        program.set_vec3("u_light_color", lighting.light_color)
        program.set_vec3("u_ambient", lighting.ambient)
        program.set_vec3("u_face_color", m.face_color)
        program.set_vec3("u_side_color", m.side_color)
        program.set_float("u_side_darkening", m.side_darkening)
        program.set_float("u_bevel_strength", m.bevel_strength)
        program.set_float("u_specular", m.specular)
        program.set_float("u_shininess", m.shininess)
        self.quad.draw(layers)

    def draw_shadow_caster(self, scene: Scene, proj: np.ndarray, lighting: LightingState) -> None:
        program = self.shadow
        flatten = math3d.to_gl(lighting.shadow_matrix)
        layers = self._bind_common(program, scene, proj, flatten, lighting)
        program.set_float("u_falloff", scene.shadow.falloff)
        self.quad.draw(layers)

    def _bind_common(self, program, scene, proj, flatten_gl, lighting) -> int:
        t = scene.transform
        layers = max(1, int(t.layers)) if t.depth > 1e-4 else 1
        model = math3d.model_matrix(t.x, t.y, t.z, t.rot_x, t.rot_y, t.rot_z, t.size)
        program.use()
        self.mask.bind(0)
        program.set_int("u_mask", 0)
        program.set_mat4("u_model", math3d.to_gl(model))
        program.set_mat4("u_proj", math3d.to_gl(proj))
        program.set_mat4("u_flatten", flatten_gl)
        program.set_vec2("u_origin", self.raster.origin)
        program.set_vec2("u_size", self.raster.size)
        program.set_float("u_depth", t.depth)
        program.set_int("u_layers", layers)
        program.set_vec4("u_ground_plane", lighting.ground_plane)
        program.set_float("u_world_size", t.size)
        return layers

    def delete(self) -> None:
        self.beauty.delete()
        self.shadow.delete()
        self.mask.delete()
