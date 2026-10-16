"""MainWindow: menus, layout and file dialogs. Holds no rendering or image-processing logic."""

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QSplitter

from ..image_io import SUPPORTED_LOAD, load_image, save_image
from ..preferences import Preferences
from ..viewmodel import SceneViewModel
from .control_panel import ControlPanel
from .image_canvas import FIT_TOOL, GRID_TOOL, TEXT_TOOL, ImageCanvas

EXPORT_FILTERS = "PNG image (*.png);;JPEG image (*.jpg *.jpeg)"

TOOL_HINTS = {
    TEXT_TOOL: "Left-drag: slide on ground · Right-drag: yaw · Wheel: scale · Ctrl/Cmd+wheel: distance",
    GRID_TOOL: "Drag horizon: tilt · Drag horizon ends: roll · Drag ground: shift lines · "
    "Right-drag: rotate grid · Wheel: cell size",
    FIT_TOOL: "Drag the four corners onto a rectangle in the photo (a tile, rug, table top) · "
    "Its real size (Grid › Fit width/depth) sets the scale",
}


class MainWindow(QMainWindow):
    def __init__(self, viewmodel: SceneViewModel, preferences: Preferences):
        super().__init__()
        self.vm = viewmodel
        self.preferences = preferences
        self.setWindowTitle("Text3D Studio")
        self.resize(1440, 900)

        self.canvas = ImageCanvas(viewmodel)
        self.panel = ControlPanel(viewmodel)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1080, 360])
        self.setCentralWidget(splitter)

        self.size_label = QLabel()
        self.fps_label = QLabel()
        self.hint_label = QLabel(TOOL_HINTS[TEXT_TOOL])
        self.statusBar().addPermanentWidget(self.size_label)
        self.statusBar().addPermanentWidget(self.hint_label)
        self.statusBar().addPermanentWidget(self.fps_label)

        self.canvas.fps_changed.connect(lambda fps: self.fps_label.setText(f"{fps:5.1f} FPS"))
        self.canvas.error.connect(lambda msg: QMessageBox.warning(self, "Rendering problem", msg))
        self.canvas.open_requested.connect(lambda: self.open_image_dialog())
        self.canvas.file_dropped.connect(self.load_image_path)
        self.vm.background_changed.connect(self._update_size_label)
        self._update_size_label()
        self._build_actions()

    def _build_actions(self) -> None:
        # Wrapped in a lambda: QAction.triggered passes a `checked` bool the handler must not receive.
        self.open_action = self._make_action(
            "&Open Image…", QKeySequence.StandardKey.Open, lambda: self.open_image_dialog()
        )
        self.export_action = self._make_action("&Export…", QKeySequence("Ctrl+E"), self.export_image)
        quit_action = self._make_action("&Quit", QKeySequence.StandardKey.Quit, self.close)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        self.recent_menu = file_menu.addMenu("Open &Recent")
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)
        self._rebuild_recent_menu()

        # Single-key shortcut: text fields still receive "G" because they claim printable keys first.
        self.grid_action = QAction("Show &Grid", self)
        self.grid_action.setCheckable(True)
        self.grid_action.setShortcut(QKeySequence("G"))
        self.grid_action.setChecked(self.vm.scene.grid.visible)
        self.grid_action.toggled.connect(lambda checked: self.vm.set("grid", "visible", checked))
        self.vm.scene_changed.connect(self._sync_grid_action)
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.grid_action)

        self.tool_actions = {}
        tool_group = QActionGroup(self)
        tools_menu = self.menuBar().addMenu("&Tools")
        tools = ((TEXT_TOOL, "Move &Text", "V"), (GRID_TOOL, "Adjust &Grid", "P"), (FIT_TOOL, "Fit &Rectangle", "R"))
        for tool, text, key in tools:
            action = QAction(text, self)
            action.setCheckable(True)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(lambda _checked=False, t=tool: self.canvas.set_tool(t))
            tool_group.addAction(action)
            tools_menu.addAction(action)
            self.tool_actions[tool] = action
        self.tool_actions[self.canvas.tool].setChecked(True)
        self.canvas.tool_changed.connect(self._on_tool_changed)
        self.canvas.fit_status.connect(self._show_fit_status)

        self.toolbar = self.addToolBar("Main")
        self.toolbar.setObjectName("mainToolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.export_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.grid_action)
        self.toolbar.addSeparator()
        for action in self.tool_actions.values():
            self.toolbar.addAction(action)

    def _on_tool_changed(self, tool: str) -> None:
        self.tool_actions[tool].setChecked(True)
        self.hint_label.setText(TOOL_HINTS[tool])

    def _show_fit_status(self, message: str) -> None:
        if message:
            self.statusBar().showMessage(f"Can't fit: {message}")
        else:
            self.statusBar().clearMessage()

    def _sync_grid_action(self, section: str, field: str) -> None:
        if (section, field) == ("grid", "visible"):
            self.grid_action.setChecked(self.vm.scene.grid.visible)

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        paths = self.preferences.recent_files()
        if not paths:
            self.recent_menu.addAction("No Recent Images").setEnabled(False)
        for path in paths:
            action = self.recent_menu.addAction(_menu_label(path))
            action.setData(path)
            action.setStatusTip(path)
            action.triggered.connect(lambda _checked=False, p=path: self.open_recent(p))
        self.recent_menu.addSeparator()
        clear_action = self.recent_menu.addAction("Clear Menu")
        clear_action.setEnabled(bool(paths))
        clear_action.triggered.connect(self.clear_recent_files)

    def open_recent(self, path: str) -> None:
        if not os.path.isfile(path):
            self.preferences.remove_recent_file(path)
            self._rebuild_recent_menu()
            QMessageBox.information(
                self, "Image not found", f"{path}\n\nThis file no longer exists and was removed from Open Recent."
            )
            return
        self.load_image_path(path)

    def clear_recent_files(self) -> None:
        self.preferences.clear_recent_files()
        self._rebuild_recent_menu()

    def _make_action(self, text, shortcut, handler) -> QAction:
        action = QAction(text, self)
        action.setShortcut(shortcut)
        action.triggered.connect(handler)
        return action

    def open_image_dialog(self) -> None:
        start_dir = self.preferences.last_directory()
        path, _ = QFileDialog.getOpenFileName(self, "Open background image", start_dir, SUPPORTED_LOAD)
        if path:
            self.load_image_path(path)

    def load_image_path(self, path: str) -> None:
        had_background = self.vm.document.background is not None
        previous_size = self.vm.document.native_size
        try:
            pixels = _load_with_busy_cursor(path)
        except ValueError as exc:
            QMessageBox.warning(self, "Could not open image", str(exc))
            return
        self.vm.set_background(pixels, path)
        self.preferences.set_last_directory(os.path.dirname(os.path.abspath(path)))
        self.preferences.add_recent_file(path)
        self._rebuild_recent_menu()
        self.setWindowTitle(f"Text3D Studio — {os.path.basename(path)}")
        self.statusBar().showMessage(self._load_message(path, previous_size if had_background else None), 8000)

    def _load_message(self, path: str, previous_size: tuple[int, int] | None) -> str:
        """Export size follows the image, so call out when it changed; the text keeps its position."""
        width, height = self.vm.document.native_size
        name = os.path.basename(path)
        if previous_size is None or previous_size == (width, height):
            return f"Loaded {name} · export size {width}×{height}"
        return f"Loaded {name} · export size is now {width}×{height} (was {previous_size[0]}×{previous_size[1]})"

    def export_image(self) -> None:
        source = self.vm.document.background_path
        suggested = (os.path.splitext(source)[0] + "_3d.png") if source else "composition.png"
        path, _ = QFileDialog.getSaveFileName(self, "Export composition", suggested, EXPORT_FILTERS)
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".png"
        self.statusBar().showMessage("Rendering at native resolution…")
        try:
            save_image(self.canvas.render_export(), path)
        except (ValueError, OSError, RuntimeError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        width, height = self.vm.document.native_size
        self.statusBar().showMessage(f"Exported {width}×{height} → {path}", 8000)

    def _update_size_label(self) -> None:
        width, height = self.vm.document.native_size
        has_image = self.vm.document.background is not None
        self.size_label.setText(f"{width}×{height}" if has_image else f"No image · canvas {width}×{height}")


def _load_with_busy_cursor(path: str):
    """Decoding a large photo blocks the UI for a moment; show a wait cursor meanwhile."""
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        return load_image(path)
    finally:
        QApplication.restoreOverrideCursor()


def _menu_label(path: str) -> str:
    """"photo.jpg — ~/Pictures", with "&" escaped so Qt doesn't treat it as a mnemonic."""
    folder = os.path.dirname(path)
    home = os.path.expanduser("~")
    if folder == home or folder.startswith(home + os.sep):
        folder = "~" + folder[len(home):]
    return f"{os.path.basename(path)} — {folder}".replace("&", "&&")
