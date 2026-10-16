"""ImageCanvas: QOpenGLWidget that hosts the Compositor and translates mouse input into scene edits."""

import os
import time
from collections import deque

import numpy as np
from OpenGL import GL
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout

from .. import ground as ground_model
from .. import math3d
from ..image_io import SUPPORTED_EXTENSIONS
from ..render.compositor import Compositor
from ..render.exporter import render_document
from ..render.grid_renderer import GridRenderer
from ..text_raster import rasterize_text
from ..viewmodel import SceneViewModel

TEXT_DEBOUNCE_MS = 60
ROTATE_DEG_PER_PX = 0.4
WHEEL_SCALE_STEP = 1.06
TEXT_TOOL = "text"
GRID_TOOL = "grid"
FIT_TOOL = "fit"
HANDLE_GRAB_PX = 12.0  # how close to a fit corner a press must land to grab it
HORIZON_GRAB_PX = 8.0  # how close to the horizon a press must land to grab it
HORIZON_END_FRACTION = 0.15  # presses on the outer 15% of the horizon roll it instead of tilting
MIN_ROLL_SPAN_PX = 20.0  # roll needs the cursor this far from the pivot to define a direction
MIN_CELL, MAX_CELL = 0.1, 10.0
GRID_ROTATE_DEG_PER_PX = 0.3
FPS_IDLE_RESET_S = 0.25  # a gap this long means interaction stopped; start a fresh measurement


def dropped_image_path(mime_data) -> str | None:
    """Returns the path when the drag holds exactly one local file with a supported image extension."""
    urls = mime_data.urls() if mime_data.hasUrls() else []
    if len(urls) != 1 or not urls[0].isLocalFile():
        return None
    path = urls[0].toLocalFile()
    if not os.path.isfile(path) or not path.lower().endswith(SUPPORTED_EXTENSIONS):
        return None
    return path


class EmptyStateOverlay(QFrame):
    """Centered prompt shown over the canvas until a background image is loaded."""

    open_requested = pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("emptyState")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(
            "#emptyState { background: rgba(20, 20, 22, 210); border-radius: 12px; }"
            "QLabel { color: #e8e8e8; }"
        )
        shortcut = QKeySequence(QKeySequence.StandardKey.Open).toString(QKeySequence.SequenceFormat.NativeText)
        title = QLabel("No image loaded")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.message = QLabel(f"Open a photo ({shortcut}) or drop one here to place 3D text on it")
        self.button = QPushButton("Open Image…")
        self.button.clicked.connect(lambda: self.open_requested.emit())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(10)
        for widget in (title, self.message, self.button):
            layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignHCenter)


class ImageCanvas(QOpenGLWidget):
    fps_changed = pyqtSignal(float)
    error = pyqtSignal(str)
    open_requested = pyqtSignal()
    file_dropped = pyqtSignal(str)
    tool_changed = pyqtSignal(str)
    fit_status = pyqtSignal(str)  # why the traced corners can't be fitted, or "" when they can

    def __init__(self, viewmodel: SceneViewModel, parent=None):
        super().__init__(parent)
        self.vm = viewmodel
        self.compositor: Compositor | None = None
        self.grid_renderer: GridRenderer | None = None
        self._glyph_dirty = True
        self._background_dirty = True
        self._drag_origin: QPointF | None = None
        self._drag_ground_start: tuple[float, float] | None = None  # ground point grabbed on press
        self._drag_text_start: tuple[float, float] | None = None
        self.tool = TEXT_TOOL  # editor state, not part of the scene
        self._grid_drag: str | None = None  # "tilt", "roll", "shift" or "rotate" while dragging in the grid tool
        self._roll_pivot: tuple[float, float] | None = None
        self._offset_start: tuple[float, float] | None = None
        self.fit_corners: list[tuple[float, float]] | None = None  # fractions of the image rect
        self.fit_valid = True
        self._fit_handle: int | None = None
        self._frame_times: deque[float] = deque(maxlen=30)

        self._text_timer = QTimer(self, singleShot=True, interval=TEXT_DEBOUNCE_MS)
        self._text_timer.timeout.connect(self._mark_glyph_dirty)
        self.vm.scene_changed.connect(lambda *_: self.update())
        self.vm.text_changed.connect(self._text_timer.start)
        self.vm.background_changed.connect(self._mark_background_dirty)

        self.empty_state = EmptyStateOverlay(self)
        self.empty_state.open_requested.connect(self.open_requested)
        self.vm.background_changed.connect(self._update_empty_state)
        self._update_empty_state()

        self.setMinimumSize(480, 320)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)  # hover feedback for the grid tool's handles

    # ----- GL lifecycle -------------------------------------------------------------------------

    def initializeGL(self) -> None:
        self.compositor = Compositor()
        self.grid_renderer = GridRenderer(self.compositor.quad)
        self.context().aboutToBeDestroyed.connect(self._release_gl)

    def paintGL(self) -> None:
        self._sync_resources()
        dpr = self.devicePixelRatioF()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.defaultFramebufferObject())
        GL.glViewport(0, 0, int(self.width() * dpr), int(self.height() * dpr))
        GL.glClearColor(0.11, 0.11, 0.12, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        rect = self._image_rect(dpr)
        self.compositor.render(self.vm.scene, self.defaultFramebufferObject(), rect)
        if self.vm.scene.grid.visible:
            # Editor overlay: drawn here, after the composite, so exports never contain it.
            self.grid_renderer.render(
                self.vm.scene, self.compositor.text.raster, self.defaultFramebufferObject(), rect, dpr
            )
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
        self.grid_renderer.delete()
        self.grid_renderer = None
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
        now = time.perf_counter()
        if self._frame_times and now - self._frame_times[-1] > FPS_IDLE_RESET_S:
            self._frame_times.clear()
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                self.fps_changed.emit((len(self._frame_times) - 1) / span)

    # ----- Layout -------------------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._center_empty_state()

    def _update_empty_state(self) -> None:
        self.empty_state.setVisible(self.vm.document.background is None)
        self._center_empty_state()

    def _center_empty_state(self) -> None:
        self.empty_state.adjustSize()
        x = (self.width() - self.empty_state.width()) // 2
        y = (self.height() - self.empty_state.height()) // 2
        self.empty_state.move(max(0, x), max(0, y))

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

    # ----- Drag and drop ------------------------------------------------------------------------

    # Ignoring the drag makes the OS show its not-allowed cursor.

    def dragEnterEvent(self, event) -> None:
        if dropped_image_path(event.mimeData()) is None:
            event.ignore()
        else:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        path = dropped_image_path(event.mimeData())
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.file_dropped.emit(path)

    # ----- Tools --------------------------------------------------------------------------------

    def set_tool(self, tool: str) -> None:
        if tool == self.tool:
            return
        self.tool = tool
        if tool in (GRID_TOOL, FIT_TOOL):
            self.vm.set("grid", "visible", True)  # adjusting an invisible grid makes no sense
        if tool == FIT_TOOL:
            self.fit_corners = self._initial_fit_corners()
            self.fit_valid = True
            self.fit_status.emit("")
        self.setCursor(Qt.CursorShape.OpenHandCursor if tool == TEXT_TOOL else Qt.CursorShape.ArrowCursor)
        self.tool_changed.emit(tool)

    # ----- Mouse interaction --------------------------------------------------------------------
    # Left drag: slide the text along the ground   Right drag: yaw
    # Wheel: scale                                 Ctrl/Cmd+wheel: move nearer/farther along the ground

    def mousePressEvent(self, event) -> None:
        if self.tool == GRID_TOOL:
            self._grid_press(event)
            return
        if self.tool == FIT_TOOL:
            self._fit_handle = self.fit_handle_at(event.position())
            return
        self._drag_origin = event.position()
        self._drag_ground_start = self.ground_point_at(event.position())
        t = self.vm.scene.transform
        self._drag_text_start = (t.x, t.z)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_origin = None
        self._drag_ground_start = None
        self._grid_drag = None
        self._fit_handle = None
        if self.tool == TEXT_TOOL:
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self.tool == FIT_TOOL:
            if self._fit_handle is None:
                over_handle = self.fit_handle_at(event.position()) is not None
                self.setCursor(Qt.CursorShape.PointingHandCursor if over_handle else Qt.CursorShape.ArrowCursor)
            else:
                self._move_fit_corner(self._fit_handle, event.position())
            return
        if self.tool == GRID_TOOL:
            if self._grid_drag is None:
                self._update_grid_cursor(event.position())
            else:
                self._grid_move(event)
            return
        if self._drag_origin is None:
            return
        delta = event.position() - self._drag_origin
        self._drag_origin = event.position()

        if event.buttons() & Qt.MouseButton.RightButton:
            self.vm.nudge("transform", "rot_y", delta.x() * ROTATE_DEG_PER_PX)
        elif event.buttons() & Qt.MouseButton.LeftButton:
            self._slide_text_to(event.position())

    def _slide_text_to(self, position: QPointF) -> None:
        """Moves the text so the ground point grabbed on press stays under the cursor."""
        current = self.ground_point_at(position)
        if self._drag_ground_start is None or current is None:
            return  # grabbed or dragged above the horizon: there is no ground there
        start_x, start_z = self._drag_text_start
        self.vm.set("transform", "x", start_x + current[0] - self._drag_ground_start[0])
        self.vm.set("transform", "z", start_z + current[1] - self._drag_ground_start[1])

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        if self.tool == FIT_TOOL:
            return
        if self.tool == GRID_TOOL:
            cell = self.vm.scene.grid.cell_size * WHEEL_SCALE_STEP**steps
            self.vm.set("grid", "cell_size", float(np.clip(cell, MIN_CELL, MAX_CELL)))
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.vm.nudge("transform", "z", steps * 0.5)
        else:
            self.vm.set("transform", "size", self.vm.scene.transform.size * WHEEL_SCALE_STEP**steps)

    # ----- Grid tool --------------------------------------------------------------------------
    # Drag the horizon's middle: tilt   Drag its outer ends: roll around its center
    # Drag the ground: slide the lines   Right drag: rotate the grid   Wheel: cell size

    def _grid_press(self, event) -> None:
        self._drag_origin = event.position()
        if event.button() == Qt.MouseButton.RightButton:
            self._grid_drag = "rotate"
            return
        handle = self.horizon_handle_at(event.position())
        if handle == "roll":
            self._roll_pivot = self._horizon_center()
            if self._roll_pivot is None:
                handle = "tilt"  # a vertical horizon has no center row to pivot on
        if handle is None and self.ground_point_at(event.position()) is not None:
            handle = "shift"
            self._drag_ground_start = self.ground_point_at(event.position())
            grid = self.vm.scene.grid
            self._offset_start = (grid.offset_x, grid.offset_z)
        self._grid_drag = handle

    def _grid_move(self, event) -> None:
        pixel = self._image_pixel(event.position())
        _, _, rect_w, rect_h = self._image_rect()
        ground, fov = self.vm.scene.ground, self.vm.scene.camera.fov
        if self._grid_drag == "tilt":
            pitch = math3d.pitch_for_horizon_through(pixel, ground.roll, fov, (rect_w, rect_h))
            self.vm.set("ground", "pitch", float(np.clip(pitch, -ground_model.MAX_PITCH, ground_model.MAX_PITCH)))
        elif self._grid_drag == "roll":
            if np.hypot(pixel[0] - self._roll_pivot[0], pixel[1] - self._roll_pivot[1]) < MIN_ROLL_SPAN_PX:
                return
            pitch, roll = math3d.tilt_for_horizon_through(self._roll_pivot, pixel, fov, (rect_w, rect_h))
            self.vm.set("ground", "roll", float(np.clip(roll, -ground_model.MAX_ROLL, ground_model.MAX_ROLL)))
            self.vm.set("ground", "pitch", float(np.clip(pitch, -ground_model.MAX_PITCH, ground_model.MAX_PITCH)))
        elif self._grid_drag == "shift":
            current = self.ground_point_at(event.position())
            if current is None:
                return
            self.vm.set("grid", "offset_x", self._offset_start[0] + current[0] - self._drag_ground_start[0])
            self.vm.set("grid", "offset_z", self._offset_start[1] + current[1] - self._drag_ground_start[1])
        elif self._grid_drag == "rotate":
            delta = event.position().x() - self._drag_origin.x()
            self._drag_origin = event.position()
            self.vm.nudge("ground", "yaw", delta * GRID_ROTATE_DEG_PER_PX)

    def _update_grid_cursor(self, position: QPointF) -> None:
        cursors = {"tilt": Qt.CursorShape.SizeVerCursor, "roll": Qt.CursorShape.SizeBDiagCursor}
        handle = self.horizon_handle_at(position)
        if handle is None and self.ground_point_at(position) is not None:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(cursors.get(handle, Qt.CursorShape.ArrowCursor))

    def horizon_handle_at(self, position: QPointF) -> str | None:
        """"tilt" or "roll" when a widget position is on the horizon (its middle or its outer ends)."""
        line = self._horizon_line()
        if line is None:
            return None
        u, v = self._image_pixel(position)
        if abs(line @ np.array([u, v, 1.0])) > HORIZON_GRAB_PX:
            return None
        _, _, rect_w, _ = self._image_rect()
        at_end = u < rect_w * HORIZON_END_FRACTION or u > rect_w * (1.0 - HORIZON_END_FRACTION)
        return "roll" if at_end else "tilt"

    def _horizon_line(self) -> np.ndarray | None:
        _, _, rect_w, rect_h = self._image_rect()
        up = ground_model.ground_up(self.vm.scene.ground)
        return math3d.horizon_line(up, self.vm.scene.camera.fov, (rect_w, rect_h))

    def _horizon_center(self) -> tuple[float, float] | None:
        """The horizon's point in the image's center column, in image pixels."""
        line = self._horizon_line()
        if line is None or abs(line[1]) < 1e-6:
            return None
        _, _, rect_w, _ = self._image_rect()
        u = rect_w / 2.0
        return (u, -(line[0] * u + line[2]) / line[1])

    def _image_pixel(self, position: QPointF) -> tuple[float, float]:
        rect_x, rect_y, _, _ = self._image_rect()
        return (position.x() - rect_x, position.y() - rect_y)

    # ----- Fit Rectangle tool -------------------------------------------------------------------
    # Drag the four corners onto a rectangle in the photo; the ground is solved on every move.

    def fit_corners_px(self) -> list[tuple[float, float]]:
        _, _, rect_w, rect_h = self._image_rect()
        return [(fx * rect_w, fy * rect_h) for fx, fy in self.fit_corners]

    def fit_handle_at(self, position: QPointF) -> int | None:
        if self.fit_corners is None:
            return None
        u, v = self._image_pixel(position)
        distances = [np.hypot(u - cu, v - cv) for cu, cv in self.fit_corners_px()]
        nearest = int(np.argmin(distances))
        return nearest if distances[nearest] <= HANDLE_GRAB_PX else None

    def _move_fit_corner(self, index: int, position: QPointF) -> None:
        _, _, rect_w, rect_h = self._image_rect()
        u, v = self._image_pixel(position)
        self.fit_corners[index] = (u / rect_w, v / rect_h)
        self._solve_fit()
        self.update()

    def _solve_fit(self) -> None:
        """Fits the ground to the traced corners; an impossible shape leaves the scene untouched."""
        _, _, rect_w, rect_h = self._image_rect()
        scene = self.vm.scene
        grid = scene.grid
        try:
            fit = ground_model.fit_ground_to_rectangle(
                self.fit_corners_px(), (grid.fit_width, grid.fit_depth), scene.camera.fov,
                (rect_w, rect_h), grid.cell_size, grid.fit_estimate_fov,
            )
        except ValueError as exc:
            self.fit_valid = False
            self.fit_status.emit(str(exc))
            return
        self.fit_valid = True
        self.vm.set("camera", "fov", fit.fov)
        self.vm.set_ground_pose(pitch=fit.pitch, yaw=fit.yaw, roll=fit.roll, height=fit.height)
        self.vm.set("grid", "offset_x", fit.offset_x)
        self.vm.set("grid", "offset_z", fit.offset_z)
        self.fit_status.emit("")

    def _initial_fit_corners(self) -> list[tuple[float, float]]:
        """A fit-sized rectangle on the current ground near the bottom middle, so nothing jumps until a drag.

        Falls back to a plain screen-space quad when that spot shows no ground.
        """
        _, _, rect_w, rect_h = self._image_rect()
        scene = self.vm.scene
        center = self._ground_point_at_pixel((rect_w / 2.0, rect_h * 0.75))
        if center is not None:
            half_w, half_d = scene.grid.fit_width / 2.0, scene.grid.fit_depth / 2.0
            x, z = center
            ground_corners = [(x - half_w, z + half_d), (x + half_w, z + half_d), (x + half_w, z - half_d), (x - half_w, z - half_d)]
            pixels = [
                math3d.project_to_image(ground_model.ground_to_world(scene.ground, gx, gz), scene.camera.fov, (rect_w, rect_h))
                for gx, gz in ground_corners
            ]
            if all(p is not None for p in pixels):
                return [(float(p[0]) / rect_w, float(p[1]) / rect_h) for p in pixels]
        return [(0.3, 0.9), (0.7, 0.9), (0.65, 0.65), (0.35, 0.65)]

    def ground_point_at(self, position: QPointF) -> tuple[float, float] | None:
        """Ground (x, z) under a widget position, or None when that pixel shows sky or lies off the image."""
        return self._ground_point_at_pixel(self._image_pixel(position))

    def _ground_point_at_pixel(self, pixel) -> tuple[float, float] | None:
        _, _, rect_w, rect_h = self._image_rect()
        if not (0 <= pixel[0] <= rect_w and 0 <= pixel[1] <= rect_h):
            return None
        scene = self.vm.scene
        ray = math3d.image_ray(pixel, scene.camera.fov, (rect_w, rect_h))
        hit = math3d.intersect_ray_plane((0.0, 0.0, 0.0), ray, ground_model.ground_plane(scene.ground))
        if hit is None:
            return None
        return ground_model.world_to_ground(scene.ground, hit)
