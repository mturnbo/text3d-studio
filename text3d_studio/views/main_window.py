"""MainWindow: menus, layout and file dialogs. Holds no rendering or image-processing logic."""

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox, QSplitter

from ..image_io import SUPPORTED_LOAD, load_image, save_image
from ..viewmodel import SceneViewModel
from .control_panel import ControlPanel
from .image_canvas import ImageCanvas

EXPORT_FILTERS = "PNG image (*.png);;JPEG image (*.jpg *.jpeg)"


class MainWindow(QMainWindow):
    def __init__(self, viewmodel: SceneViewModel):
        super().__init__()
        self.vm = viewmodel
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
        self.vm.background_changed.connect(self._update_size_label)
        self._update_size_label()
        self._build_menu()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, shortcut, handler in (
            ("&Open Image…", QKeySequence.StandardKey.Open, self.open_image),
            ("&Export…", QKeySequence("Ctrl+E"), self.export_image),
            ("&Quit", QKeySequence.StandardKey.Quit, self.close),
        ):
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            file_menu.addAction(action)

    def open_image(self, path: str | None = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open background image", "", SUPPORTED_LOAD)
        if not path:
            return
        try:
            self.vm.set_background(load_image(path), path)
        except ValueError as exc:
            QMessageBox.warning(self, "Could not open image", str(exc))
            return
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
