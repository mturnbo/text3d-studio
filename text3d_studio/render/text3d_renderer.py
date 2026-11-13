"""Text3DRenderer: draws a text object's glyph mask as an instanced stack of duplicated layers."""

import math

import numpy as np
from OpenGL import GL

from .. import ground as ground_model
from .. import math3d
from ..lighting import LightingState, srgb_to_linear
from ..materials import PRESETS, TRANSMISSION, transmits
from ..model import Scene, TextObject, TransformParams
from ..text_raster import GlyphRaster
from . import shaders
from .gl_utils import ColorTarget, RenderTarget, ShaderProgram, Texture2D, TriangleBuffer, UnitQuad

IDENTITY_GL = math3d.to_gl(math3d.identity())
PATTERNS = {"stone": 1, "wood": 2}  # material kinds with a procedural texture (see LAYER_STACK_FS)
MAX_LAYERS = 2048
LAYERS_PER_PIXEL = 1.5  # oversample so neighbouring slices always overlap on screen
SIDE_GLOW = 0.35  # the side walls and back cap glow with this fraction of their material's emission

# Depth pre-pass: draw the stack once into depth only, then shade with GL_EQUAL so each pixel is shaded for
# its visible slice rather than every slice behind it. Measured gain is modest on Apple GPUs, which already
# reject most hidden slices: ~11% for stone without MSAA, ~0 with it, where the real cost is side-wall pixels
# whose samples belong to several slices. (An early synthetic benchmark suggested 2x; real textures did not.)
# The shading pass keeps its alpha test. False renders the stack in one pass.
DEPTH_PREPASS = True

# Stone and wood read their noise from a texture baked once per renderer (shaders.BAKED_NOISE_GLSL) instead of
# computing it per pixel. False computes it procedurally.
BAKED_NOISE = True

# Glass shows its own back cap and side walls (drawn into the backdrop before it is shaded); False shows only
# what is behind the text.
GLASS_FAR_WALLS = True


class Glyph:
    """One text object's glyph raster on the GPU: the mask texture and the traced side walls."""

    def __init__(self, raster: GlyphRaster):
        self.mask = Texture2D(mipmaps=True)
        self.walls = TriangleBuffer()
        self.upload(raster)

    def upload(self, raster: GlyphRaster) -> None:
        self.raster = raster
        self.mask.upload(raster.rgba)
        self.walls.upload(raster.walls)

    def delete(self) -> None:
        self.mask.delete()
        self.walls.delete()


class NoiseVolume:
    """gradient_noise baked into a tiling 3D texture on the GPU, once (see shaders.BAKED_NOISE_GLSL)."""

    def __init__(self, quad: UnitQuad):
        size = shaders.NOISE_PERIOD * shaders.NOISE_TEXELS_PER_CELL
        self.size = size
        self.handle = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self.handle)
        for name, value in (
            (GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
            (GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT), (GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT),
            (GL.GL_TEXTURE_WRAP_R, GL.GL_REPEAT),
        ):
            GL.glTexParameteri(GL.GL_TEXTURE_3D, name, value)
        GL.glTexImage3D(GL.GL_TEXTURE_3D, 0, GL.GL_R16F, size, size, size, 0, GL.GL_RED, GL.GL_FLOAT, None)

        previous_fbo = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        previous_viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        program = ShaderProgram(shaders.FULLSCREEN_VS, shaders.NOISE_BAKE_FS)
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, size, size)
        program.use()
        program.set_int("u_flip_y", 0)
        for z in range(size):
            GL.glFramebufferTextureLayer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, self.handle, 0, z)
            program.set_float("u_slice", z)
            quad.draw()
        GL.glDeleteFramebuffers(1, [fbo])
        program.delete()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(previous_fbo))
        GL.glViewport(*map(int, previous_viewport))

    def bind(self, unit: int) -> None:
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_3D, self.handle)

    def delete(self) -> None:
        GL.glDeleteTextures(1, [self.handle])


class Text3DRenderer:
    """Draws one text object at a time, with the glyph uploaded for it; holds no per-object state."""

    def __init__(self, quad: UnitQuad):
        self.quad = quad
        self.depth_prepass = DEPTH_PREPASS
        self.glass_far_walls = GLASS_FAR_WALLS
        self.far_target: RenderTarget | None = None  # sized to the backdrop when glass is drawn
        self.noise = NoiseVolume(quad) if BAKED_NOISE else None
        fragment = shaders.LAYER_STACK_FS
        if self.noise is not None:
            fragment = shaders.with_define(fragment, "BAKED_NOISE")
        self.beauty = ShaderProgram(shaders.LAYER_STACK_VS, fragment)
        self.depth = ShaderProgram(shaders.LAYER_STACK_VS, shaders.LAYER_DEPTH_FS)
        self.cap_shadow = ShaderProgram(shaders.LAYER_STACK_VS, shaders.SHADOW_FS)
        self.wall_shadow = ShaderProgram(shaders.WALL_VS, shaders.SHADOW_FS)

    def draw_beauty(
        self, scene: Scene, obj: TextObject, glyph: Glyph, proj: np.ndarray, lighting: LightingState,
        viewport_height: int, environment: Texture2D | None = None, backdrop: Texture2D | None = None,
    ) -> None:
        """`environment`: the background photo to reflect, or None to reflect the ambient color instead.
        `backdrop`: what is behind the text in this viewport, for transmissive materials to show through."""
        layers = self.effective_layers(scene, obj.transform, viewport_height)
        if self.depth_prepass:
            self._bind_common(self.depth, scene, obj.transform, glyph, proj, IDENTITY_GL, lighting)
            self.depth.set_int("u_layers", layers)
            GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
            self.quad.draw(layers)
            GL.glDepthFunc(GL.GL_EQUAL)
            GL.glDepthMask(GL.GL_FALSE)

        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        program = self.beauty
        self._bind_common(program, scene, obj.transform, glyph, proj, IDENTITY_GL, lighting)
        program.set_int("u_layers", layers)
        self._bind_material(program, scene, obj.material, lighting, environment, backdrop)
        program.set_int("u_far_walls", 0)
        self.quad.draw(layers)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthMask(GL.GL_TRUE)

    def draw_far_walls(
        self, scene: Scene, drawn: list[tuple[TextObject, Glyph]], proj: np.ndarray, lighting: LightingState,
        backdrop: ColorTarget, environment: Texture2D | None = None,
    ) -> None:
        """Draws what each transmissive object's front looks through, its back cap and side walls seen from
        inside (the farthest slice at each pixel), into the captured `backdrop`, and rebuilds its mipmaps. Glass
        then refracts its own far walls and their tint, not just the photo. Leaves the far-walls target bound."""
        transmissive = [(obj, glyph) for obj, glyph in drawn if transmits(obj.material) and obj.transform.depth > 1e-4]
        if not (self.glass_far_walls and transmissive):
            return
        width, height = backdrop.width, backdrop.height
        if self.far_target is None or (self.far_target.width, self.far_target.height) != (width, height):
            if self.far_target is not None:
                self.far_target.delete()
            self.far_target = RenderTarget(width, height)
        _blit(backdrop.fbo, self.far_target.fbo, width, height)  # walls go over a copy: the shader reads backdrop
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.far_target.fbo)
        GL.glViewport(0, 0, width, height)
        GL.glClearDepth(0.0)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glClearDepth(1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_GREATER)  # the farthest slice: where the view leaves the text
        program = self.beauty
        for obj, glyph in transmissive:
            layers = self.effective_layers(scene, obj.transform, height)
            self._bind_common(program, scene, obj.transform, glyph, proj, IDENTITY_GL, lighting)
            program.set_int("u_layers", layers)
            self._bind_material(program, scene, obj.material, lighting, environment, backdrop.texture)
            program.set_int("u_far_walls", 1)
            self.quad.draw(layers)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDisable(GL.GL_DEPTH_TEST)
        _blit(self.far_target.fbo, backdrop.fbo, width, height)
        backdrop.texture.bind(0)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.far_target.fbo)

    def _bind_material(self, program, scene, m, lighting, environment, backdrop) -> None:
        if self.noise is not None:
            self.noise.bind(3)
            program.set_int("u_noise", 3)
        program.set_vec3("u_light_dir", lighting.light_dir)
        program.set_vec3("u_light_color", lighting.light_color)
        program.set_vec3("u_ambient", lighting.ambient)
        program.set_float("u_side_darkening", m.side_darkening)
        program.set_float("u_bevel_strength", m.bevel_strength)
        has_backdrop = backdrop is not None
        _set_surface(program, "u_face", m.kind, vars(m), m.face_color, has_backdrop, 1.0)
        if m.side_kind in PRESETS:
            side = PRESETS[m.side_kind]
            _set_surface(program, "u_side", m.side_kind, side, side["side_color"], has_backdrop, SIDE_GLOW)
        else:
            _set_surface(program, "u_side", m.kind, vars(m), m.side_color, has_backdrop, SIDE_GLOW)
        program.set_float("u_outline_width", m.outline_width)
        program.set_vec3("u_outline_color", srgb_to_linear(m.outline_color))
        program.set_float("u_outline_metallic", m.outline_metallic)
        if backdrop is not None:
            backdrop.bind(2)
            program.set_int("u_backdrop", 2)
            program.set_float("u_backdrop_lod", backdrop.max_lod)
        program.set_int("u_has_environment", int(environment is not None))
        if environment is not None:
            environment.bind(1)
            program.set_int("u_environment", 1)
            program.set_float("u_environment_lod", environment.max_lod)
            program.set_float("u_environment_strength", scene.light.reflection_strength)

    def draw_shadow_caster(
        self, scene: Scene, obj: TextObject, glyph: Glyph, proj: np.ndarray, lighting: LightingState, darken,
    ) -> None:
        """Front and back slices plus the traced side walls form a closed caster for any light angle.

        `darken`: how much this object's full-strength shadow darkens each color channel."""
        flatten = math3d.to_gl(lighting.shadow_matrix)
        caps = min(2, self._layer_count(obj.transform))

        program = self.cap_shadow
        self._bind_common(program, scene, obj.transform, glyph, proj, flatten, lighting)
        program.set_int("u_layers", caps)
        program.set_int("u_use_mask", 1)
        program.set_float("u_falloff", scene.shadow.falloff)
        program.set_vec3("u_darken", darken)
        self.quad.draw(caps)

        if caps > 1:
            program = self.wall_shadow
            self._bind_common(program, scene, obj.transform, glyph, proj, flatten, lighting)
            program.set_int("u_use_mask", 0)
            program.set_float("u_falloff", scene.shadow.falloff)
            program.set_vec3("u_darken", darken)
            glyph.walls.draw()

    @staticmethod
    def effective_layers(scene: Scene, t: TransformParams, viewport_height: int) -> int:
        """The user's layer count, raised if needed so slices never separate into visible bands."""
        if t.depth <= 1e-4:
            return 1
        focal_px = (viewport_height / 2) / math.tan(math.radians(scene.camera.fov) / 2)
        depth_px = t.depth * t.size * focal_px / ground_model.text_distance(scene.ground, t, scene.wall)
        needed = math.ceil(depth_px * LAYERS_PER_PIXEL)
        return min(MAX_LAYERS, max(int(t.layers), needed))

    @staticmethod
    def _layer_count(t: TransformParams) -> int:
        return max(1, int(t.layers)) if t.depth > 1e-4 else 1

    def _bind_common(self, program, scene, t, glyph, proj, flatten_gl, lighting) -> None:
        model = ground_model.text_model_matrix(scene.ground, t, scene.wall)
        program.use()
        glyph.mask.bind(0)
        program.set_int("u_mask", 0)
        program.set_mat4("u_model", math3d.to_gl(model))
        program.set_mat4("u_proj", math3d.to_gl(proj))
        program.set_mat4("u_flatten", flatten_gl)
        program.set_vec2("u_origin", glyph.raster.origin)
        program.set_vec2("u_size", glyph.raster.size)
        program.set_float("u_depth", t.depth)
        program.set_vec4("u_ground_plane", lighting.ground_plane)
        program.set_float("u_world_size", t.size)

    def delete(self) -> None:
        self.beauty.delete()
        self.depth.delete()
        self.cap_shadow.delete()
        self.wall_shadow.delete()
        if self.far_target is not None:
            self.far_target.delete()
        if self.noise is not None:
            self.noise.delete()


def _blit(source_fbo: int, target_fbo: int, width: int, height: int) -> None:
    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, source_fbo)
    GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, target_fbo)
    GL.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)


def _set_surface(
    program: ShaderProgram, name: str, kind: str, values, color, has_backdrop: bool, glow: float
) -> None:
    """Sets one `Surface` uniform of LAYER_STACK_FS from a kind and its material values (a MaterialParams' fields
    or a preset). `color` is the plain albedo: the face color for the face, the side color for the walls. A
    transmissive kind needs the backdrop to show through. `glow` scales the emission (dimmer on the sides)."""
    program.set_vec3(f"{name}.color", srgb_to_linear(color))
    program.set_vec3(f"{name}.pattern_color", srgb_to_linear(values["face_color"]))
    program.set_vec3(f"{name}.secondary_color", srgb_to_linear(values["secondary_color"]))
    program.set_float(f"{name}.metallic", values["metallic"])
    program.set_float(f"{name}.roughness", values["roughness"])
    program.set_int(f"{name}.pattern", PATTERNS.get(kind, 0))
    program.set_float(f"{name}.texture_scale", values["texture_scale"])
    program.set_float(f"{name}.texture_strength", values["texture_strength"])
    program.set_float(f"{name}.bump_strength", values["bump_strength"])
    program.set_float(f"{name}.grain_angle", math.radians(values["grain_angle"]))
    program.set_float(f"{name}.varnish", values["varnish"])
    program.set_int(f"{name}.transmissive", int(kind in TRANSMISSION and has_backdrop))
    program.set_float(f"{name}.absorption", values["absorption"])
    emission = np.asarray(srgb_to_linear(values["emission_color"])) * values["emission_strength"] * glow
    program.set_vec3(f"{name}.emission", emission)
