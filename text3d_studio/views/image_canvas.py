"""ImageCanvas: QOpenGLWidget that hosts the Compositor and translates mouse input into scene edits."""

import time
from collections import deque

import numpy as np
from OpenGL import GL
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from ..render.compositor import Compositor
from ..render.exporter import render_document
from ..text_raster import rasterize_text
from ..viewmodel import SceneViewModel

TEXT_DEBOUNCE_MS = 60
ROTATE_DEG_PER_PX = 0.4
WHEEL_SCALE_STEP = 1.06


class ImageCanvas(QOpenGLWidget):
    fps_changed = pyqtSignal(float)
    error = pyqtSignal(str)

    def __init__(self, viewmodel: SceneViewModel, parent=None):
        super().__init__(parent)
        self.vm = viewmodel
        self.compositor: Compositor | None = None
        self._glyph_dirty = True
        self._background_dirty = True
        self._drag_origin: QPointF | None = None
        self._frame_times: deque[float] = deque(maxlen=30)

        self._text_timer = QTimer(self, singleShot=True, interval=TEXT_DEBOUNCE_MS)
        self._text_timer.timeout.connect(self._mark_glyph_dirty)
        self.vm.scene_changed.connect(lambda *_: self.update())
        self.vm.text_changed.connect(self._text_timer.start)
        self.vm.background_changed.connect(self._mark_background_dirty)

        self.setMinimumSize(480, 320)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    # ----- GL lifecycle -------------------------------------------------------------------------

    def initializeGL(self) -> None:
        self.compositor = Compositor()
        self.context().aboutToBeDestroyed.connect(self._release_gl)

    def paintGL(self) -> None:
        self._sync_resources()
        dpr = self.devicePixelRatioF()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        GL.glViewport(0, 0, int(self.width() * dpr), int(self.height() * dpr))
        GL.glClearColor(0.11, 0.11, 0.12, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.compositor.render(self.vm.scene, self.defaultFramebufferObject(), self._image_rect(dpr))
        self._track_fps()

    def _sync_resources(self) -> None:
        if self._background_dirty:
            self._background_dirty = False
            try:
                self.compositor.set_background(self.vm.document.background)
            except ValueError as exc:
                self.error.emit(str(exc))
        if self._glyph_dirty:
            self._glyph_dirty = False
            self.compositor.set_glyph(rasterize_text(self.vm.scene.text))

    def _release_gl(self) -> None:
        self.makeCurrent()
        self.compositor.delete()
        self.compositor = None
        self.doneCurrent()

    def _mark_glyph_dirty(self) -> None:
        self._glyph_dirty = True
        self.update()

    def _mark_background_dirty(self) -> None:
        self._background_dirty = True
        self.update()

    def _track_fps(self) -> None:
        self._frame_times.append(time.perf_counter())
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                self.fps_changed.emit((len(self._frame_times) - 1) / span)

    # ----- Layout -------------------------------------------------------------------------------

    def _image_rect(self, scale: float = 1.0) -> tuple[int, int, int, int]:
        """Largest rect with the document's aspect ratio centered in the widget (GL coords)."""
        img_w, img_h = self.vm.document.native_size
        avail_w, avail_h = self.width() * scale, self.height() * scale
        fit = min(avail_w / img_w, avail_h / img_h)
        w, h = max(1, int(img_w * fit)), max(1, int(img_h * fit))
        return (int((avail_w - w) / 2), int((avail_h - h) / 2), w, h)

    # ----- Export -------------------------------------------------------------------------------

    def render_export(self) -> np.ndarray:
        """Renders the composition at the background's native resolution using this GL context."""
        self.makeCurrent()
        try:
            self._sync_resources()
            return render_document(self.compositor, self.vm.document)
        finally:
            self.doneCurrent()

    # ----- Mouse interaction --------------------------------------------------------------------
    # Left drag: move X/Y   Right drag: rotate X/Y   Shift+drag: rotate Z
    # Wheel: scale          Ctrl/Cmd+wheel: push/pull along Z

    def mousePressEvent(self, event) -> None:
        self._drag_origin = event.position()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_origin = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._frame_times.clear()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_origin is None:
            return
        delta = event.position() - self._drag_origin
        self._drag_origin = event.position()
        buttons, mods = event.buttons(), event.modifiers()

        if mods & Qt.KeyboardModifier.ShiftModifier:
            self.vm.nudge("transform", "rot_z", -delta.x() * ROTATE_DEG_PER_PX)
        elif buttons & Qt.MouseButton.RightButton:
            self.vm.nudge("transform", "rot_y", delta.x() * ROTATE_DEG_PER_PX)
            self.vm.nudge("transform", "rot_x", delta.y() * ROTATE_DEG_PER_PX)
        elif buttons & Qt.MouseButton.LeftButton:
            units = self._world_units_per_pixel()
            self.vm.nudge("transform", "x", delta.x() * units)
            self.vm.nudge("transform", "y", -delta.y() * units)

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.vm.nudge("transform", "z", steps * 0.5)
        else:
            self.vm.set("transform", "size", self.vm.scene.transform.size * WHEEL_SCALE_STEP**steps)

    def _world_units_per_pixel(self) -> float:
        _, _, _, rect_h = self._image_rect()
        half_height = abs(self.vm.scene.transform.z) * np.tan(np.radians(self.vm.scene.camera.fov) / 2)
        return 2.0 * half_height / rect_h
