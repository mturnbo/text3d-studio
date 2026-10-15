"""Thin wrappers over raw OpenGL 4.1 core objects. A GL context must be current for every call."""

import ctypes

import numpy as np
from OpenGL import GL


class ShaderProgram:
    def __init__(self, vertex_src: str, fragment_src: str):
        vs = self._compile(vertex_src, GL.GL_VERTEX_SHADER)
        fs = self._compile(fragment_src, GL.GL_FRAGMENT_SHADER)
        self.handle = GL.glCreateProgram()
        GL.glAttachShader(self.handle, vs)
        GL.glAttachShader(self.handle, fs)
        GL.glLinkProgram(self.handle)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        if not GL.glGetProgramiv(self.handle, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(self.handle).decode())
        self._locations: dict[str, int] = {}

    @staticmethod
    def _compile(source: str, kind) -> int:
        shader = GL.glCreateShader(kind)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(shader).decode())
        return shader

    def use(self) -> None:
        GL.glUseProgram(self.handle)

    def _loc(self, name: str) -> int:
        if name not in self._locations:
            self._locations[name] = GL.glGetUniformLocation(self.handle, name)
        return self._locations[name]

    def set_int(self, name: str, value: int) -> None:
        GL.glUniform1i(self._loc(name), int(value))

    def set_float(self, name: str, value: float) -> None:
        GL.glUniform1f(self._loc(name), float(value))

    def set_vec2(self, name: str, value) -> None:
        GL.glUniform2f(self._loc(name), *map(float, value))

    def set_vec3(self, name: str, value) -> None:
        GL.glUniform3f(self._loc(name), *map(float, value))

    def set_vec4(self, name: str, value) -> None:
        GL.glUniform4f(self._loc(name), *map(float, value))

    def set_mat4(self, name: str, column_major: np.ndarray) -> None:
        GL.glUniformMatrix4fv(self._loc(name), 1, GL.GL_FALSE, column_major)

    def delete(self) -> None:
        GL.glDeleteProgram(self.handle)


class UnitQuad:
    """A VAO holding a 0..1 quad as a 4-vertex triangle strip (attribute 0)."""

    def __init__(self):
        vertices = np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype=np.float32)
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)

    def draw(self, instances: int = 1) -> None:
        GL.glBindVertexArray(self.vao)
        if instances == 1:
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        else:
            GL.glDrawArraysInstanced(GL.GL_TRIANGLE_STRIP, 0, 4, instances)
        GL.glBindVertexArray(0)

    def delete(self) -> None:
        GL.glDeleteBuffers(1, [self.vbo])
        GL.glDeleteVertexArrays(1, [self.vao])


class TriangleBuffer:
    """A VAO of float32 vec3 triangle vertices (attribute 0), replaceable at runtime."""

    def __init__(self):
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.count = 0
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)

    def upload(self, vertices: np.ndarray) -> None:
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self.count = len(vertices)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, max(vertices.nbytes, 4), vertices if self.count else None, GL.GL_DYNAMIC_DRAW)

    def draw(self) -> None:
        if self.count == 0:
            return
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.count)
        GL.glBindVertexArray(0)

    def delete(self) -> None:
        GL.glDeleteBuffers(1, [self.vbo])
        GL.glDeleteVertexArrays(1, [self.vao])


class Texture2D:
    def __init__(self, internal_format=GL.GL_RGBA8, pixel_format=GL.GL_RGBA, mipmaps: bool = False):
        self.handle = GL.glGenTextures(1)
        self.internal_format = internal_format
        self.pixel_format = pixel_format
        self.mipmaps = mipmaps
        self.width = self.height = 0
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.handle)
        min_filter = GL.GL_LINEAR_MIPMAP_LINEAR if mipmaps else GL.GL_LINEAR
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, min_filter)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

    def upload(self, pixels: np.ndarray) -> None:
        """Uploads an (H, W, 4) uint8 array. Row 0 lands at texture v=0."""
        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        self.height, self.width = pixels.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.handle)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, self.internal_format, self.width, self.height, 0,
            self.pixel_format, GL.GL_UNSIGNED_BYTE, pixels,
        )
        if self.mipmaps:
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

    def allocate(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.handle)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, self.internal_format, width, height, 0,
            self.pixel_format, GL.GL_UNSIGNED_BYTE, None,
        )

    def bind(self, unit: int = 0) -> None:
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.handle)

    def delete(self) -> None:
        GL.glDeleteTextures(1, [self.handle])


class ColorTarget:
    """Framebuffer with a single sampleable color texture and no depth (used for shadow passes)."""

    def __init__(self, internal_format=GL.GL_R8, pixel_format=GL.GL_RED):
        self.texture = Texture2D(internal_format, pixel_format)
        self.fbo = GL.glGenFramebuffers(1)
        self.width = self.height = 0

    def resize(self, width: int, height: int) -> None:
        if (width, height) == (self.width, self.height):
            return
        self.width, self.height = width, height
        self.texture.allocate(width, height)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.texture.handle, 0
        )
        _check_complete()

    def delete(self) -> None:
        GL.glDeleteFramebuffers(1, [self.fbo])
        self.texture.delete()


class RenderTarget:
    """Offscreen RGBA8 + depth framebuffer, optionally multisampled, for full-resolution export."""

    def __init__(self, width: int, height: int, samples: int = 0):
        self.width, self.height, self.samples = width, height, samples
        self.fbo = GL.glGenFramebuffers(1)
        self.color_rb, self.depth_rb = GL.glGenRenderbuffers(2)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        for rb, fmt, attachment in (
            (self.color_rb, GL.GL_RGBA8, GL.GL_COLOR_ATTACHMENT0),
            (self.depth_rb, GL.GL_DEPTH_COMPONENT24, GL.GL_DEPTH_ATTACHMENT),
        ):
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
            if samples > 0:
                GL.glRenderbufferStorageMultisample(GL.GL_RENDERBUFFER, samples, fmt, width, height)
            else:
                GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, fmt, width, height)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, attachment, GL.GL_RENDERBUFFER, rb)
        _check_complete()

    def delete(self) -> None:
        GL.glDeleteFramebuffers(1, [self.fbo])
        GL.glDeleteRenderbuffers(2, [self.color_rb, self.depth_rb])


def _check_complete() -> None:
    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"Framebuffer incomplete (status 0x{int(status):x})")
