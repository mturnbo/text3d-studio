"""ImageCanvas: QOpenGLWidget that hosts the Compositor and translates mouse input into scene edits."""

import dataclasses
import os
import time
from collections import deque

import numpy as np
from OpenGL import GL
from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QPainter, QPen, QPolygonF
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget

from .. import ground as ground_model
from .. import math3d
from ..lighting import solve_sun
from ..image_io import SUPPORTED_EXTENSIONS
from ..render.compositor import Compositor
from ..render.exporter import render_document
from ..render.grid_renderer import GridRenderer
from ..model import TextObject, TextParams
from ..project_io import PROJECT_EXTENSION
from ..text_raster import GlyphRaster, rasterize_text
from ..viewmodel import SceneViewModel

TEXT_DEBOUNCE_MS = 60
MAX_TEXT_DISTANCE = 150.0  # a plain drag won't push the text further than this (the Position Z range)
ROTATE_DEG_PER_PX = 0.4
WHEEL_SCALE_STEP = 1.06
TEXT_TOOL = "text"
GRID_TOOL = "grid"
FIT_TOOL = "fit"
WALL_TOOL = "wall"
SUN_TOOL = "sun"
SUN_POINT_LABELS = ("Top", "Base", "Shadow tip")
SUN_PROMPTS = (
    "Match Sun: click the top of a vertical object (a post, a person, a bottle)",
    "Match Sun: click its base on the floor",
    "Match Sun: click the tip of its shadow",
)
HANDLE_GRAB_PX = 12.0  # how close to a fit corner a press must land to grab it
FIT_START_WIDTH = 0.3  # the traced rectangle starts about this fraction of the image wide
FIT_VALID_COLOR = QColor(255, 179, 51)
FIT_INVALID_COLOR = QColor(235, 64, 52)
HORIZON_GRAB_PX = 8.0  # how close to the horizon a press must land to grab it
HORIZON_END_FRACTION = 0.15  # presses on the outer 15% of the horizon roll it instead of tilting
MIN_ROLL_SPAN_PX = 20.0  # roll needs the cursor this far from the pivot to define a direction
MIN_CELL, MAX_CELL = 0.1, 10.0
GRID_ROTATE_DEG_PER_PX = 0.3
FPS_IDLE_RESET_S = 0.25  # a gap this long means interaction stopped; start a fresh measurement
WALL_GRID_HALF_WIDTH = 40.0  # the wall preview's grid reaches this far along the wall each way, in world units
WALL_GRID_HEIGHT = 12.0
EYEDROPPER_RADIUS = 2  # the eyedropper averages a (2r + 1) x (2r + 1) block of photo pixels
WALL_GRID_MIN_SPACING = 0.5  # coarser than the floor grid's smallest cells, so the preview stays light


def dropped_image_path(mime_data) -> str | None:
    """Returns the path when the drag holds exactly one local file that is a supported image or a project."""
    urls = mime_data.urls() if mime_data.hasUrls() else []
    if len(urls) != 1 or not urls[0].isLocalFile():
        return None
    path = urls[0].toLocalFile()
    if not os.path.isfile(path) or not path.lower().endswith((*SUPPORTED_EXTENSIONS, PROJECT_EXTENSION)):
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
        self._shortcut = shortcut
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.message = QLabel()
        self.show_missing_photo(None)
        self.button = QPushButton("Open Image…")
        self.button.clicked.connect(lambda: self.open_requested.emit())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(10)
        for widget in (self.title, self.message, self.button):
            layout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignHCenter)

    def show_missing_photo(self, path: str | None) -> None:
        """Names the photo an opened project refers to but couldn't find; None restores the usual prompt."""
        if path is None:
            self.title.setText("No image loaded")
            self.message.setText(f"Open a photo ({self._shortcut}) or drop one here to place 3D text on it")
        else:
            self.title.setText("Photo not found")
            self.message.setText(
                f"{os.path.basename(path)} couldn't be found. Open an image ({self._shortcut}) or drop one here"
            )
        self.adjustSize()


class FitHandlesOverlay(QWidget):
    """Draws the Fit Rectangle handles over the GL canvas.

    Plain widget painting rather than QPainter inside paintGL: in the core-profile context the GL paint
    engine garbles text, and keeping it out of paintGL leaves the renderer's GL state alone.
    """

    def __init__(self, canvas: "ImageCanvas"):
        super().__init__(canvas)
        self.canvas = canvas
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

    def paintEvent(self, event) -> None:
        if self.canvas.fit_corners is None:
            return
        rect_x, rect_y, _, _ = self.canvas._image_rect()
        points = [QPointF(rect_x + u, rect_y + v) for u, v in self.canvas.fit_corners_px()]
        center = sum(points, QPointF()) / 4.0
        color = FIT_VALID_COLOR if self.canvas.fit_valid else FIT_INVALID_COLOR
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(color, 2))
        painter.drawPolygon(QPolygonF(points))
        painter.setPen(QPen(color, 4))  # edge 1-2 sets the grid's x direction
        painter.drawLine(points[0], points[1])
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        for number, point in enumerate(points, start=1):
            painter.setPen(QPen(color, 2))
            painter.setBrush(QColor(255, 255, 255, 230))
            painter.drawEllipse(point, 7, 7)
            outward = point - center
            length = max((outward.x() ** 2 + outward.y() ** 2) ** 0.5, 1e-6)
            label = point + outward * (18.0 / length) - QPointF(4, -5)
            painter.drawText(label, str(number))
        painter.end()


class ToolOverlay(QWidget):
    """Preview-only drawing for the Set Wall and Match Sun tools, painted over the GL canvas like FitHandlesOverlay,
    so exports never contain it."""

    def __init__(self, canvas: "ImageCanvas"):
        super().__init__(canvas)
        self.canvas = canvas
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(QRectF(*self.canvas._image_rect()))  # never over the bars beside the photo
        if self.canvas.tool == WALL_TOOL:
            self._paint_wall(painter)
        if self.canvas.tool == SUN_TOOL:
            self._paint_sun_points(painter)
        painter.end()

    def _paint_sun_points(self, painter: QPainter) -> None:
        points = [QPointF(*p) for p in self.canvas.sun_points_px()]
        color = FIT_VALID_COLOR if self.canvas.sun_valid else FIT_INVALID_COLOR
        painter.setPen(QPen(color, 2))
        for a, b in zip(points, points[1:]):  # the object, then its shadow
            painter.drawLine(a, b)
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        for point, label in zip(points, SUN_POINT_LABELS):
            painter.setBrush(QColor(255, 255, 255, 230))
            painter.drawEllipse(point, 6, 6)
            painter.drawText(point + QPointF(10, -8), label)

    def _paint_wall(self, painter: QPainter) -> None:
        grid_lines, base_line = self.canvas.wall_preview_px()
        grid = self.canvas.vm.scene.grid
        color = QColor.fromRgbF(*grid.color, 0.35 * grid.opacity)
        painter.setPen(QPen(color, 1))
        for a, b in grid_lines:
            painter.drawLine(QPointF(*a), QPointF(*b))
        if base_line is not None:
            painter.setPen(QPen(QColor.fromRgbF(*grid.accent_color), 3))
            painter.drawLine(QPointF(*base_line[0]), QPointF(*base_line[1]))


class PinchFilter(QObject):
    """Turns trackpad pinches (macOS native zoom gestures) into scale factors."""

    pinched = pyqtSignal(float)

    def eventFilter(self, watched, event) -> bool:
        if (event.type() == QEvent.Type.NativeGesture
                and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture):
            self.pinched.emit(1.0 + event.value())  # value() is the change in magnification since the last event
            return True
        return False


class ImageCanvas(QOpenGLWidget):
    fps_changed = pyqtSignal(float)
    error = pyqtSignal(str)
    open_requested = pyqtSignal()
    file_dropped = pyqtSignal(str)
    tool_changed = pyqtSignal(str)
    fit_status = pyqtSignal(str)  # why the traced corners can't be fitted, or "" when they can
    tool_status = pyqtSignal(str)  # what the Match Sun tool wants next, its result, or why it failed

    def __init__(self, viewmodel: SceneViewModel, parent=None):
        super().__init__(parent)
        self.vm = viewmodel
        self.compositor: Compositor | None = None
        self.grid_renderer: GridRenderer | None = None
        self._glyph_dirty = True
        # Each object's raster with the text it was made from, so only objects whose text changed re-rasterize.
        self._rasters: dict[int, tuple[TextParams, GlyphRaster | None]] = {}
        self._unsent: set[int] = set()  # rasterized but not yet uploaded to the compositor
        self._background_dirty = True
        self._drag_origin: QPointF | None = None  # previous cursor position (yaw drags are incremental)
        self._drag_press: QPointF | None = None
        self._drag_mode: str | None = None  # "move", "slide" or "yaw", chosen by the button and modifiers on press
        self._drag_ground_start: tuple[float, float] | None = None  # ground point grabbed on press
        self._drag_transform = None  # the text's placement on press
        self._drag_wall_start: tuple[float, float] | None = None  # wall point grabbed on press (wall-mounted text)
        self._editing = False  # a press began an undo step (vm.begin_edit) that the release will end
        self.tool = TEXT_TOOL  # editor state, not part of the scene
        self._grid_drag: str | None = None  # "tilt", "roll", "shift" or "rotate" while dragging in the grid tool
        self._roll_pivot: tuple[float, float] | None = None
        self._offset_start: tuple[float, float] | None = None
        self.fit_corners: list[tuple[float, float]] | None = None  # fractions of the image rect
        self._pose_base = None  # placement snapshot when a ground-changing drag began; the text re-places from it
        # Fit Rectangle re-places the text from the start of the whole tracing session, since corners are
        # dragged one at a time and the shape is only meaningful once all four are on the rectangle.
        self._fit_settled = None  # placement snapshot right after the last fit, to notice outside edits
        self.fit_overlay = FitHandlesOverlay(self)
        self.fit_valid = True
        self._fit_handle: int | None = None
        self._wall_start: tuple[float, float] | None = None  # ground point where a Set Wall drag began
        self.sun_points: list[tuple[float, float]] = []  # Match Sun clicks (top, base, tip), fractions of the image rect
        self.sun_valid = True
        self.eyedropper: tuple[str, str] | None = None  # (section, field) the next click picks from the photo
        self.tool_overlay = ToolOverlay(self)
        self._frame_times: deque[float] = deque(maxlen=30)

        self._text_timer = QTimer(self, singleShot=True, interval=TEXT_DEBOUNCE_MS)
        self._text_timer.timeout.connect(self._mark_glyph_dirty)
        self.vm.scene_changed.connect(lambda *_: self.update())
        self.vm.scene_changed.connect(lambda *_: self.tool_overlay.update())
        self.vm.text_changed.connect(self._text_timer.start)
        self.vm.objects_changed.connect(self._mark_glyph_dirty)  # new objects show at once, without the debounce
        self.vm.selection_changed.connect(lambda *_: self.update())
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
        self.pinch_filter = PinchFilter(self)
        self.pinch_filter.pinched.connect(self._scale_by)
        self.installEventFilter(self.pinch_filter)

    # ----- GL lifecycle -------------------------------------------------------------------------

    def initializeGL(self) -> None:
        self.compositor = Compositor()
        self.grid_renderer = GridRenderer(self.compositor.quad)
        self._unsent = set(self._rasters)  # a new context starts without glyphs
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
            obj = self.vm.selected_object
            self.grid_renderer.render(
                self.vm.scene, obj, self.compositor.glyph_raster(obj.id), self.defaultFramebufferObject(), rect, dpr
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
            self._update_rasters()
        self.compositor.retain_glyphs(self._rasters)
        for object_id in self._unsent:
            self.compositor.set_glyph(object_id, self._rasters[object_id][1])
        self._unsent.clear()

    def _update_rasters(self) -> None:
        """Re-rasterizes the objects whose text changed since their last raster, and forgets deleted ones."""
        objects = self.vm.scene.objects
        ids = {obj.id for obj in objects}
        for object_id in [i for i in self._rasters if i not in ids]:
            del self._rasters[object_id]
            self._unsent.discard(object_id)
        for obj in objects:
            cached = self._rasters.get(obj.id)
            if cached is None or cached[0] != obj.text:
                self._rasterize(obj)

    def _rasterize(self, obj: TextObject) -> GlyphRaster | None:
        raster = rasterize_text(obj.text)
        self._rasters[obj.id] = (dataclasses.replace(obj.text), raster)
        self._unsent.add(obj.id)
        return raster

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
        self.fit_overlay.setGeometry(self.rect())
        self.tool_overlay.setGeometry(self.rect())
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
            self._capture_pose_base()
            self._fit_settled = self._pose_snapshot()
            self.fit_status.emit("")
        self.fit_overlay.setVisible(tool == FIT_TOOL)
        self.fit_overlay.update()
        self.eyedropper = None
        if tool == SUN_TOOL:
            self.sun_points = []
            self.sun_valid = True
            self.tool_status.emit(SUN_PROMPTS[0])
        self.tool_overlay.setVisible(tool in (WALL_TOOL, SUN_TOOL))
        self.tool_overlay.update()
        self._set_tool_cursor()
        self.tool_changed.emit(tool)

    def _set_tool_cursor(self) -> None:
        self.setCursor(Qt.CursorShape.OpenHandCursor if self.tool == TEXT_TOOL else Qt.CursorShape.ArrowCursor)

    # ----- Mouse interaction --------------------------------------------------------------------
    # Press: select the object under the cursor; every gesture acts on the selected object
    # Drag: move the text on screen, keeping its on-screen size    Shift+drag: lock to horizontal/vertical
    # Option/Alt+drag: slide along the ground in perspective        Cmd/Ctrl+drag or right drag: yaw
    # Wheel or pinch: scale                                         Cmd/Ctrl+wheel: nearer/farther

    def mousePressEvent(self, event) -> None:
        if self.eyedropper is not None:
            self._pick_color(event.position())
            return
        if self.tool == GRID_TOOL:
            self._grid_press(event)
        elif self.tool == FIT_TOOL:
            self._fit_handle = self.fit_handle_at(event.position())
            if self._pose_snapshot() != self._fit_settled:
                self._capture_pose_base()  # something else moved the ground or text since the last fit
        elif self.tool == WALL_TOOL:
            self._wall_start = self.ground_point_at(event.position())
        elif self.tool == SUN_TOOL:
            self._add_sun_point(event.position())
        else:
            self._text_press(event)
        # The whole press→release drag is one undo step. It begins after the press's selection, which isn't one,
        # so undoing the drag shows the object it moved selected.
        if not self._editing:
            self._editing = True
            self.vm.begin_edit()

    def _text_press(self, event) -> None:
        # A press on an object selects it; elsewhere the selection stays, so the drag still moves it. With one
        # object there is nothing to choose, which also spares a raster for a canvas that never painted.
        if len(self.vm.scene.objects) > 1:
            index = self.object_at(event.position())
            if index is not None:
                self.vm.select(index)
        self._drag_origin = self._drag_press = event.position()
        self._drag_mode = self._text_drag_mode(event)
        self._drag_ground_start = self.ground_point_at(event.position())
        self._drag_wall_start = self.wall_point_at(event.position())
        self._drag_transform = dataclasses.replace(self.vm.selected_object.transform)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    @staticmethod
    def _text_drag_mode(event) -> str:
        # Qt reports macOS's Command key as ControlModifier and Option as AltModifier.
        modifiers = event.modifiers()
        if event.button() == Qt.MouseButton.RightButton or modifiers & Qt.KeyboardModifier.ControlModifier:
            return "yaw"
        if modifiers & Qt.KeyboardModifier.AltModifier:
            return "slide"
        return "move"

    def mouseReleaseEvent(self, event) -> None:
        self._drag_origin = None
        self._drag_press = None
        self._drag_mode = None
        self._drag_ground_start = None
        self._drag_wall_start = None
        self._grid_drag = None
        self._fit_handle = None
        self._wall_start = None
        if self.tool == TEXT_TOOL and self.eyedropper is None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        # Ends with the last button up. A release with no press here (a dialog closed on the press) ends nothing.
        if self._editing and not event.buttons():
            self._editing = False
            self.vm.end_edit()

    def mouseMoveEvent(self, event) -> None:
        if self.eyedropper is not None:
            return  # keep the eyedropper's cursor
        if self.tool == FIT_TOOL:
            if self._fit_handle is None:
                over_handle = self.fit_handle_at(event.position()) is not None
                self.setCursor(Qt.CursorShape.PointingHandCursor if over_handle else Qt.CursorShape.ArrowCursor)
            else:
                self._move_fit_corner(self._fit_handle, event.position())
            return
        if self.tool == WALL_TOOL:
            if self._wall_start is not None:
                self._drag_wall_to(event.position())
            return
        if self.tool == SUN_TOOL:
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

        if self._drag_mode == "yaw":
            self.vm.nudge("transform", "rot_y", delta.x() * ROTATE_DEG_PER_PX)
            return
        position = event.position()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            position = self._locked_to_axis(position)
        on_wall = self._drag_transform.mount == "wall"
        if self._drag_mode == "slide" and on_wall:
            self._slide_wall_text_to(position)
        elif self._drag_mode == "slide":
            self._slide_text_to(position)
        elif on_wall:
            self._move_wall_text_on_screen(position)
        else:
            self._move_text_on_screen(position)

    def object_at(self, position: QPointF) -> int | None:
        """Index of the nearest object whose ink is under a widget position, or None.

        Tests the cursor ray against each object's front face and its glyph mask, so a press in a letter's
        counter (the hole of an O) or between letters goes through to whatever is behind.
        """
        pixel = self._image_pixel(position)
        _, _, rect_w, rect_h = self._image_rect()
        if not (0 <= pixel[0] <= rect_w and 0 <= pixel[1] <= rect_h):
            return None
        scene = self.vm.scene
        ray = math3d.image_ray(pixel, scene.camera.fov, (rect_w, rect_h))
        nearest, nearest_distance = None, np.inf
        for index, obj in enumerate(scene.objects):
            hit = ground_model.text_face_hit(scene.ground, obj.transform, ray, scene.wall)
            if hit is None or hit[0] >= nearest_distance:
                continue
            raster = self._raster_of(obj)
            if raster is not None and raster.covers(hit[1], hit[2]):
                nearest, nearest_distance = index, hit[0]
        return nearest

    def _raster_of(self, obj: TextObject) -> GlyphRaster | None:
        """The raster on screen for `obj` (it can lag the text by the debounce), made now if it has none yet."""
        cached = self._rasters.get(obj.id)
        return cached[1] if cached is not None else self._rasterize(obj)

    def _locked_to_axis(self, position: QPointF) -> QPointF:
        """The cursor position with the smaller of its horizontal and vertical motion since the press removed."""
        moved = position - self._drag_press
        if abs(moved.x()) >= abs(moved.y()):
            return QPointF(position.x(), self._drag_press.y())
        return QPointF(self._drag_press.x(), position.y())

    def _slide_text_to(self, position: QPointF) -> None:
        """Moves the text so the ground point grabbed on press stays under the cursor."""
        current = self.ground_point_at(position)
        if self._drag_ground_start is None or current is None:
            return  # grabbed or dragged above the horizon: there is no ground there
        start = self._drag_transform
        self.vm.set("transform", "x", start.x + current[0] - self._drag_ground_start[0])
        self.vm.set("transform", "z", start.z + current[1] - self._drag_ground_start[1])

    def _move_text_on_screen(self, position: QPointF) -> None:
        """Moves the text by the cursor's on-screen motion, keeping its on-screen size.

        The text's anchor moves by the same pixels as the cursor and lands on the ground under that pixel; the
        size scales with the anchor's change in distance, so dragging up moves the text up rather than shrinking
        it into the distance. Stops where the anchor would leave the ground or go past MAX_TEXT_DISTANCE.
        """
        if self._drag_ground_start is None:
            return  # pressed on the sky
        scene = self.vm.scene
        start = self._drag_transform
        _, _, rect_w, rect_h = self._image_rect()
        anchor = ground_model.text_position(scene.ground, start)
        anchor_px = math3d.project_to_image(anchor, scene.camera.fov, (rect_w, rect_h))
        if anchor_px is None:
            return
        moved = position - self._drag_press
        ray = math3d.image_ray(
            (anchor_px[0] + moved.x(), anchor_px[1] + moved.y()), scene.camera.fov, (rect_w, rect_h)
        )
        hit = math3d.intersect_ray_plane((0.0, 0.0, 0.0), ray, ground_model.ground_plane(scene.ground))
        if hit is None or np.linalg.norm(hit) > MAX_TEXT_DISTANCE:
            return
        x, z = ground_model.world_to_ground(scene.ground, hit)
        self.vm.set("transform", "x", x)
        self.vm.set("transform", "z", z)
        self.vm.set("transform", "size", start.size * float(np.linalg.norm(hit) / np.linalg.norm(anchor)))

    def _slide_wall_text_to(self, position: QPointF) -> None:
        """Moves wall text so the wall point grabbed on press stays under the cursor (never below the floor)."""
        current = self.wall_point_at(position)
        if self._drag_wall_start is None or current is None:
            return
        start = self._drag_transform
        self.vm.set("transform", "wall_x", start.wall_x + current[0] - self._drag_wall_start[0])
        self.vm.set("transform", "wall_y", max(start.wall_y + current[1] - self._drag_wall_start[1], 0.0))

    def _move_wall_text_on_screen(self, position: QPointF) -> None:
        """Like _move_text_on_screen, on the wall: the anchor follows the cursor's pixels onto the wall, and the size
        scales with its distance. Below the floor it stays on the wall's base line."""
        if self._drag_wall_start is None:
            return  # pressed where the wall can't be seen
        scene = self.vm.scene
        start = self._drag_transform
        _, _, rect_w, rect_h = self._image_rect()
        anchor = ground_model.text_position(scene.ground, start, scene.wall)
        anchor_px = math3d.project_to_image(anchor, scene.camera.fov, (rect_w, rect_h))
        if anchor_px is None:
            return
        moved = position - self._drag_press
        ray = math3d.image_ray(
            (anchor_px[0] + moved.x(), anchor_px[1] + moved.y()), scene.camera.fov, (rect_w, rect_h)
        )
        hit = math3d.intersect_ray_plane((0.0, 0.0, 0.0), ray, ground_model.wall_plane(scene.ground, scene.wall))
        if hit is None or np.linalg.norm(hit) > MAX_TEXT_DISTANCE:
            return
        wall_x, wall_y = ground_model.world_to_wall(scene.ground, scene.wall, hit)
        wall_y = max(wall_y, 0.0)
        point = ground_model.wall_to_world(scene.ground, scene.wall, wall_x, wall_y)
        self.vm.set("transform", "wall_x", wall_x)
        self.vm.set("transform", "wall_y", wall_y)
        self.vm.set("transform", "size", start.size * float(np.linalg.norm(point) / np.linalg.norm(anchor)))

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        if self.tool == TEXT_TOOL and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.vm.nudge("transform", "z", steps * 0.5)
        else:
            self._scale_by(WHEEL_SCALE_STEP**steps)

    def _scale_by(self, factor: float) -> None:
        """Scales the text in the text tool and the grid cells in the grid tool."""
        if self.tool == GRID_TOOL:
            cell = self.vm.scene.grid.cell_size * factor
            self.vm.set("grid", "cell_size", float(np.clip(cell, MIN_CELL, MAX_CELL)))
        elif self.tool == TEXT_TOOL:
            self.vm.set("transform", "size", self.vm.selected_object.transform.size * factor)

    # ----- Grid tool --------------------------------------------------------------------------
    # Drag the horizon's middle: tilt   Drag its outer ends: roll around its center
    # Drag the ground: slide the lines   Right drag: rotate the grid   Wheel: cell size

    def _capture_pose_base(self) -> None:
        self._pose_base = self._pose_snapshot()

    def _pose_snapshot(self):
        return self.vm.placement_snapshot()

    def _grid_press(self, event) -> None:
        self._drag_origin = event.position()
        self._capture_pose_base()
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
            pitch = float(np.clip(pitch, -ground_model.MAX_PITCH, ground_model.MAX_PITCH))
            self.vm.set_ground_pose(anchor=self._pose_base, pitch=pitch)
        elif self._grid_drag == "roll":
            if np.hypot(pixel[0] - self._roll_pivot[0], pixel[1] - self._roll_pivot[1]) < MIN_ROLL_SPAN_PX:
                return
            pitch, roll = math3d.tilt_for_horizon_through(self._roll_pivot, pixel, fov, (rect_w, rect_h))
            self.vm.set_ground_pose(
                anchor=self._pose_base,
                roll=float(np.clip(roll, -ground_model.MAX_ROLL, ground_model.MAX_ROLL)),
                pitch=float(np.clip(pitch, -ground_model.MAX_PITCH, ground_model.MAX_PITCH)),
            )
        elif self._grid_drag == "shift":
            current = self.ground_point_at(event.position())
            if current is None:
                return
            self.vm.set("grid", "offset_x", self._offset_start[0] + current[0] - self._drag_ground_start[0])
            self.vm.set("grid", "offset_z", self._offset_start[1] + current[1] - self._drag_ground_start[1])
        elif self._grid_drag == "rotate":
            delta = event.position().x() - self._drag_origin.x()
            base_yaw = self._pose_base[0].yaw
            self.vm.set_ground_pose(anchor=self._pose_base, yaw=base_yaw + delta * GRID_ROTATE_DEG_PER_PX)

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
        self.fit_overlay.update()
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
            self._fit_settled = self._pose_snapshot()
            self.fit_status.emit(str(exc))
            return
        self.fit_valid = True
        self.vm.set("camera", "fov", fit.fov)
        self.vm.set_ground_pose(
            anchor=self._pose_base, pitch=fit.pitch, yaw=fit.yaw, roll=fit.roll, height=fit.height
        )
        self.vm.set("grid", "offset_x", fit.offset_x)
        self.vm.set("grid", "offset_z", fit.offset_z)
        self._fit_settled = self._pose_snapshot()
        self.fit_status.emit("")

    def _initial_fit_corners(self) -> list[tuple[float, float]]:
        """A rectangle lying on the current ground near the bottom middle, about FIT_START_WIDTH of the image wide.

        It has the fit's proportions but is sized for easy grabbing; the first drag reinterprets it at the
        fit's real size, which sets the ground's scale. Falls back to a plain screen-space quad when that
        spot shows no ground.
        """
        _, _, rect_w, rect_h = self._image_rect()
        scene = self.vm.scene
        center = self._ground_point_at_pixel((rect_w / 2.0, rect_h * 0.8))
        if center is not None:
            distance = float(np.linalg.norm(ground_model.ground_to_world(scene.ground, *center)))
            width = FIT_START_WIDTH * rect_w * distance / math3d.focal_length_px(scene.camera.fov, rect_h)
            half_w = width / 2.0
            half_d = half_w * scene.grid.fit_depth / scene.grid.fit_width
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

    def wall_point_at(self, position: QPointF) -> tuple[float, float] | None:
        """Wall coordinates (along, height) under a widget position, or None off the image or where the wall's plane
        isn't in view. Heights below the floor are returned as they are."""
        pixel = self._image_pixel(position)
        _, _, rect_w, rect_h = self._image_rect()
        if not (0 <= pixel[0] <= rect_w and 0 <= pixel[1] <= rect_h):
            return None
        scene = self.vm.scene
        ray = math3d.image_ray(pixel, scene.camera.fov, (rect_w, rect_h))
        hit = math3d.intersect_ray_plane((0.0, 0.0, 0.0), ray, ground_model.wall_plane(scene.ground, scene.wall))
        if hit is None:
            return None
        return ground_model.world_to_wall(scene.ground, scene.wall, hit)

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

    # ----- Set Wall tool --------------------------------------------------------------------------
    # Drag along the line where the wall meets the floor; the wall's base line follows the drag.

    def _drag_wall_to(self, position: QPointF) -> None:
        current = self.ground_point_at(position)
        if current is None:
            return  # dragged into the sky: there is no floor there
        wall = ground_model.wall_through(self._wall_start, current)
        if wall is None:
            return
        self.vm.set("wall", "angle", wall.angle)
        self.vm.set("wall", "offset", wall.offset)

    def wall_preview_px(self):
        """The wall preview in widget pixels: ([grid line end points], base line end points or None)."""
        rect_x, rect_y, rect_w, rect_h = self._image_rect()
        scene = self.vm.scene
        spacing = max(scene.grid.cell_size, WALL_GRID_MIN_SPACING)
        lines = ground_model.wall_grid_lines(scene.ground, scene.wall, spacing, WALL_GRID_HALF_WIDTH, WALL_GRID_HEIGHT)
        base = (
            ground_model.wall_to_world(scene.ground, scene.wall, -WALL_GRID_HALF_WIDTH, 0.0),
            ground_model.wall_to_world(scene.ground, scene.wall, WALL_GRID_HALF_WIDTH, 0.0),
        )
        projected = []
        for a, b in [base, *lines]:
            segment = math3d.project_segment(a, b, scene.camera.fov, (rect_w, rect_h))
            if segment is not None:
                segment = tuple((p[0] + rect_x, p[1] + rect_y) for p in segment)
            projected.append(segment)
        return [p for p in projected[1:] if p is not None], projected[0]

    # ----- Match Sun tool -------------------------------------------------------------------------
    # Click the top of a vertical object, its base on the floor and the tip of its shadow; the light is solved.

    def sun_points_px(self) -> list[tuple[float, float]]:
        """The Match Sun clicks in widget pixels."""
        rect_x, rect_y, rect_w, rect_h = self._image_rect()
        return [(rect_x + fx * rect_w, rect_y + fy * rect_h) for fx, fy in self.sun_points]

    def _add_sun_point(self, position: QPointF) -> None:
        _, _, rect_w, rect_h = self._image_rect()
        u, v = self._image_pixel(position)
        if not (0 <= u <= rect_w and 0 <= v <= rect_h):
            return
        if len(self.sun_points) == len(SUN_PROMPTS):
            self.sun_points = []  # a fourth click starts over
            self.sun_valid = True
        self.sun_points.append((u / rect_w, v / rect_h))
        self.tool_overlay.update()
        if len(self.sun_points) < len(SUN_PROMPTS):
            self.tool_status.emit(SUN_PROMPTS[len(self.sun_points)])
            return
        top, base, tip = [(fx * rect_w, fy * rect_h) for fx, fy in self.sun_points]
        scene = self.vm.scene
        try:
            azimuth, elevation = solve_sun(scene.ground, scene.camera.fov, (rect_w, rect_h), top, base, tip)
        except ValueError as exc:
            self.sun_valid = False
            self.tool_status.emit(f"Can't match the sun: {exc}. Click again to start over")
            return
        self.vm.begin_edit()  # azimuth and elevation together are one undo step
        self.vm.set("light", "azimuth", azimuth)
        self.vm.set("light", "elevation", elevation)
        self.vm.end_edit()
        self.tool_status.emit(f"Sun matched: azimuth {azimuth:.0f}°, elevation {elevation:.0f}°. Click again to start over")

    # ----- Eyedropper -----------------------------------------------------------------------------
    # After a panel's Pick button, the next click on the photo sets that colour. Escape cancels.

    def start_eyedropper(self, section: str, field: str) -> None:
        self.eyedropper = (section, field)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()  # so Escape reaches the canvas
        self.tool_status.emit("Click the photo to pick the colour · Esc cancels")

    def cancel_eyedropper(self) -> None:
        self.eyedropper = None
        self._set_tool_cursor()
        self.tool_status.emit("")

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self.eyedropper is not None:
            self.cancel_eyedropper()
            return
        super().keyPressEvent(event)

    def _pick_color(self, position: QPointF) -> None:
        """Sets the eyedropper's colour to the photo's average around the clicked pixel."""
        background = self.vm.document.background
        u, v = self._image_pixel(position)
        _, _, rect_w, rect_h = self._image_rect()
        if background is None or not (0 <= u < rect_w and 0 <= v < rect_h):
            self.tool_status.emit("Click on the photo to pick the colour · Esc cancels")
            return
        height, width = background.shape[:2]
        column, row = min(int(u / rect_w * width), width - 1), min(int(v / rect_h * height), height - 1)
        r = EYEDROPPER_RADIUS
        block = background[max(row - r, 0):row + r + 1, max(column - r, 0):column + r + 1, :3]
        color = tuple(float(c) for c in block.reshape(-1, 3).mean(axis=0) / 255.0)
        section, field = self.eyedropper
        self.vm.set(section, field, color)
        self.cancel_eyedropper()
