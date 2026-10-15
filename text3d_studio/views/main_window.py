"""MainWindow: menus, layout and file dialogs. Holds no rendering or image-processing logic."""

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox, QSplitter

from ..image_io import SUPPORTED_LOAD, load_image, save_image
from ..preferences import Preferences
from ..viewmodel import SceneViewModel
from .control_panel import ControlPanel
from .image_canvas import ImageCanvas

EXPORT_FILTERS = "PNG image (*.png);;JPEG image (*.jpg *.jpeg)"


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
        hint = QLabel("Left-drag: move · Right-drag: rotate · Shift-drag: roll · Wheel: scale · Ctrl/Cmd+wheel: depth")
        self.statusBar().addPermanentWidget(self.size_label)
        self.statusBar().addPermanentWidget(hint)
        self.statusBar().addPermanentWidget(self.fps_label)

        self.canvas.fps_changed.connect(lambda fps: self.fps_label.setText(f"{fps:5.1f} FPS"))
        self.canvas.error.connect(lambda msg: QMessageBox.warning(self, "Rendering problem", msg))
        self.canvas.open_requested.connect(lambda: self.open_image())
        self.canvas.file_dropped.connect(self.open_image)
        self.vm.background_changed.connect(self._update_size_label)
        self._update_size_label()
        self._build_actions()

    def _build_actions(self) -> None:
        self.open_action = self._make_action("&Open Image…", QKeySequence.StandardKey.Open, self.open_image)
        self.export_action = self._make_action("&Export…", QKeySequence("Ctrl+E"), self.export_image)
        quit_action = self._make_action("&Quit", QKeySequence.StandardKey.Quit, self.close)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        self.recent_menu = file_menu.addMenu("Open &Recent")
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)
        self._rebuild_recent_menu()

        self.toolbar = self.addToolBar("Main")
        self.toolbar.setObjectName("mainToolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.export_action)

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
        self.open_image(path)

    def clear_recent_files(self) -> None:
        self.preferences.clear_recent_files()
        self._rebuild_recent_menu()

    def _make_action(self, text, shortcut, handler) -> QAction:
        action = QAction(text, self)
        action.setShortcut(shortcut)
        action.triggered.connect(handler)
        return action

    def open_image(self, path: str | None = None) -> None:
        if not path:
            start_dir = self.preferences.last_directory()
            path, _ = QFileDialog.getOpenFileName(self, "Open background image", start_dir, SUPPORTED_LOAD)
        if not path:
            return
        try:
            self.vm.set_background(load_image(path), path)
        except ValueError as exc:
            QMessageBox.warning(self, "Could not open image", str(exc))
            return
        self.preferences.set_last_directory(os.path.dirname(os.path.abspath(path)))
        self.preferences.add_recent_file(path)
        self._rebuild_recent_menu()
        self.setWindowTitle(f"Text3D Studio — {os.path.basename(path)}")

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


def _menu_label(path: str) -> str:
    """"photo.jpg — ~/Pictures", with "&" escaped so Qt doesn't treat it as a mnemonic."""
    folder = os.path.dirname(path)
    home = os.path.expanduser("~")
    if folder == home or folder.startswith(home + os.sep):
        folder = "~" + folder[len(home):]
    return f"{os.path.basename(path)} — {folder}".replace("&", "&&")
