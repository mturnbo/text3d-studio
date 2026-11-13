"""MainWindow: menus, layout and file dialogs. Holds no rendering or image-processing logic."""

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton, QSpinBox, QSplitter, QVBoxLayout,
)

from .. import presets, project_io
from ..image_io import SUPPORTED_LOAD, load_image, save_image
from ..preferences import Preferences
from ..render.exporter import batch_jobs, long_edge_scale, scaled_size
from ..viewmodel import SceneViewModel
from .control_panel import ControlPanel
from .image_canvas import FIT_TOOL, GRID_TOOL, MASK_TOOL, SUN_TOOL, TEXT_TOOL, WALL_TOOL, ImageCanvas

EXPORT_FILTERS = "PNG image (*.png);;JPEG image (*.jpg *.jpeg)"
TRANSPARENT_FILTER = "PNG image (*.png)"  # JPEG has no transparency

TOOL_HINTS = {
    TEXT_TOOL: "Drag: move · Shift: lock axis · Option/Alt: slide in perspective · Cmd/Ctrl or right-drag: yaw · "
    "Wheel/pinch: scale",
    GRID_TOOL: "Drag horizon: tilt · Drag horizon ends: roll · Drag ground: shift lines · "
    "Right-drag: rotate grid · Wheel: cell size",
    FIT_TOOL: "Drag the four corners onto a rectangle in the photo (a tile, rug, table top) · "
    "Its real size (Grid › Fit width/depth) sets the scale",
    WALL_TOOL: "Drag along the line where the wall meets the floor",
    SUN_TOOL: "Click the top of a vertical object, its base, then the tip of its shadow",
    MASK_TOOL: "Paint over parts of the photo that stand in front of the text · Option/Alt: erase · "
    "Brush size and feather in Mask",
}


class MainWindow(QMainWindow):
    def __init__(self, viewmodel: SceneViewModel, preferences: Preferences):
        super().__init__()
        self.vm = viewmodel
        self.preferences = preferences
        self.project_path: str | None = None  # the open project file, or None for an unsaved scene
        self.project_embeds_photo = False  # Save Project keeps the photo embedded when the project was saved that way
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
        self.canvas.file_dropped.connect(self.open_path)
        self.vm.background_changed.connect(self._update_size_label)
        self._update_size_label()
        self._build_actions()
        self._build_project_actions()
        self.vm.dirty_changed.connect(self.setWindowModified)
        self._update_title()
        self._build_edit_menu()
        self._build_looks_menu()

    def _build_actions(self) -> None:
        # Wrapped in a lambda: QAction.triggered passes a `checked` bool the handler must not receive.
        self.open_action = self._make_action(
            "&Open Image…", QKeySequence.StandardKey.Open, lambda: self.open_image_dialog()
        )
        self.export_action = self._make_action("&Export…", QKeySequence("Ctrl+E"), self.export_image)
        self.export_transparent_action = self._make_action(
            "Export &Transparent PNG…", QKeySequence("Ctrl+Shift+E"), lambda: self.export_transparent_image()
        )
        self.export_transparent_action.setStatusTip("Export only the text and its shadow, over transparency")
        self.batch_export_action = self._make_action(
            "&Batch Export…", QKeySequence("Ctrl+Alt+E"), lambda: self.batch_export()
        )
        quit_action = self._make_action("&Quit", QKeySequence.StandardKey.Quit, self.close)

        file_menu = self.file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        self.recent_menu = file_menu.addMenu("Open &Recent")
        file_menu.addAction(self.export_action)
        file_menu.addAction(self.export_transparent_action)
        file_menu.addAction(self.batch_export_action)
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
        view_menu = self.view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.grid_action)
        self.original_action = QAction("Show &Original Photo", self)
        self.original_action.setCheckable(True)
        self.original_action.setStatusTip("Show the photo without the text (or hold \\ over the canvas)")
        self.original_action.toggled.connect(self.canvas.set_show_original)
        view_menu.addAction(self.original_action)
        view_menu.addSeparator()
        # Wrapped in lambdas: QAction.triggered passes a `checked` bool.
        self.zoom_in_action = self._make_action("Zoom &In", QKeySequence("Ctrl+="), lambda: self.canvas.zoom_in())
        self.zoom_in_action.setShortcuts([QKeySequence("Ctrl+="), QKeySequence(QKeySequence.StandardKey.ZoomIn)])
        self.zoom_out_action = self._make_action("Zoom &Out", QKeySequence("Ctrl+-"), lambda: self.canvas.zoom_out())
        self.zoom_fit_action = self._make_action("&Fit", QKeySequence("Ctrl+0"), lambda: self.canvas.zoom_to_fit())
        self.zoom_actual_action = self._make_action(
            "&Actual Size", QKeySequence("Ctrl+1"), lambda: self.canvas.zoom_actual_size()
        )
        self.zoom_actual_action.setStatusTip("One photo pixel per screen pixel · hold Space and drag to pan")
        for action in (self.zoom_in_action, self.zoom_out_action, self.zoom_fit_action, self.zoom_actual_action):
            view_menu.addAction(action)
        self.zoom_label = QLabel()
        self.statusBar().addPermanentWidget(self.zoom_label)
        self.canvas.zoom_changed.connect(self._update_zoom_label)
        self._update_zoom_label()

        self.tool_actions = {}
        tool_group = QActionGroup(self)
        tools_menu = self.menuBar().addMenu("&Tools")
        tools = (
            (TEXT_TOOL, "Move &Text", "V"), (GRID_TOOL, "Adjust &Grid", "P"), (FIT_TOOL, "Fit &Rectangle", "R"),
            (WALL_TOOL, "Set &Wall", "W"), (SUN_TOOL, "Match &Sun", "L"), (MASK_TOOL, "&Mask", "M"),
        )
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
        self.canvas.tool_status.connect(lambda message: self.statusBar().showMessage(message))
        self.panel.eyedropper_requested.connect(self.canvas.start_eyedropper)

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

    def _build_project_actions(self) -> None:
        self.open_project_action = self._make_action(
            "Open &Project…", QKeySequence("Ctrl+Shift+O"), lambda: self.open_project_dialog()
        )
        self.save_project_action = self._make_action(
            "&Save Project", QKeySequence.StandardKey.Save, lambda: self.save_project()
        )
        self.save_project_as_action = self._make_action(
            "Save Project &As…", QKeySequence("Ctrl+Shift+S"), lambda: self.save_project_as()
        )
        self.save_project_with_photo_action = self._make_action(
            "Save Project &With Photo…", QKeySequence(), lambda: self.save_project_as(embed_photo=True)
        )
        self.save_project_with_photo_action.setStatusTip("Save a copy of the photo inside the project file")
        before = self.export_action
        self.file_menu.insertAction(before, self.open_project_action)
        self.file_menu.insertSeparator(before)
        for action in (self.save_project_action, self.save_project_as_action, self.save_project_with_photo_action):
            self.file_menu.insertAction(before, action)
        self.file_menu.insertSeparator(before)

    def open_project_dialog(self) -> None:
        start_dir = self.preferences.last_directory()
        path, _ = QFileDialog.getOpenFileName(self, "Open project", start_dir, project_io.PROJECT_FILTER)
        if path:
            self.open_project_path(path)

    def open_project_path(self, path: str) -> None:
        if not self.maybe_save_changes():
            return
        try:
            loaded = project_io.load_project(path)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Could not open project", str(exc))
            return
        self.vm.load_document(loaded.document)
        self.project_path = path
        self.project_embeds_photo = loaded.photo_embedded
        self._remember_file(path)
        self.canvas.empty_state.show_missing_photo(loaded.missing_photo)
        self._update_title()
        if loaded.warnings:
            self.statusBar().showMessage(" · ".join(loaded.warnings))
        else:
            self.statusBar().showMessage(f"Opened {os.path.basename(path)}", 8000)

    def save_project(self) -> bool:
        """Saves to the open project file, or asks for one. Returns False when cancelled or failed."""
        if self.project_path is None:
            return self.save_project_as()
        return self._write_project(self.project_path, self.project_embeds_photo)

    def save_project_as(self, embed_photo: bool = False) -> bool:
        path, _ = QFileDialog.getSaveFileName(self, "Save project", self._suggested_project_path(),
                                              project_io.PROJECT_FILTER)
        if not path:
            return False
        if not os.path.splitext(path)[1]:
            path += project_io.PROJECT_EXTENSION
        return self._write_project(path, embed_photo)

    def _suggested_project_path(self) -> str:
        if self.project_path:
            return self.project_path
        photo = self.vm.document.background_path
        if photo:
            return os.path.splitext(photo)[0] + project_io.PROJECT_EXTENSION
        return os.path.join(self.preferences.last_directory(), "Untitled" + project_io.PROJECT_EXTENSION)

    def _write_project(self, path: str, embed_photo: bool) -> bool:
        document = self.vm.document
        try:
            project_io.save_project(path, document, embed_photo=embed_photo)
        except (ValueError, OSError) as exc:
            QMessageBox.critical(self, "Could not save project", str(exc))
            return False
        self.project_path = path
        self.project_embeds_photo = embed_photo or (document.background is not None and document.background_path is None)
        self.vm.mark_saved()
        self._remember_file(path)
        self._update_title()
        self.statusBar().showMessage(f"Saved {os.path.basename(path)}", 8000)
        return True

    def maybe_save_changes(self) -> bool:
        """Offers to save unsaved changes. Returns False when the user cancels (or the save doesn't happen)."""
        if not self.vm.dirty:
            return True
        name = os.path.basename(self.project_path) if self.project_path else "this project"
        buttons = QMessageBox.StandardButton
        answer = QMessageBox.question(
            self, "Unsaved changes", f"Do you want to save the changes to {name}?",
            buttons.Save | buttons.Discard | buttons.Cancel, buttons.Save,
        )
        if answer == buttons.Save:
            return self.save_project()
        return answer == buttons.Discard

    def closeEvent(self, event) -> None:
        if self.maybe_save_changes():
            event.accept()
        else:
            event.ignore()

    def _remember_file(self, path: str) -> None:
        self.preferences.set_last_directory(os.path.dirname(os.path.abspath(path)))
        self.preferences.add_recent_file(path)
        self._rebuild_recent_menu()

    def _update_title(self) -> None:
        """The project's name, else the photo's; "[*]" is where Qt shows the unsaved-changes marker."""
        name = self.project_path or self.vm.document.background_path
        self.setWindowTitle(f"Text3D Studio — {os.path.basename(name)}[*]" if name else "Text3D Studio[*]")
        self.setWindowModified(self.vm.dirty)

    def _build_edit_menu(self) -> None:
        # Wrapped in lambdas: QAction.triggered passes a `checked` bool.
        self.undo_action = self._make_action("&Undo", QKeySequence.StandardKey.Undo, lambda: self.vm.undo())
        self.redo_action = self._make_action("&Redo", QKeySequence.StandardKey.Redo, lambda: self.vm.redo())
        edit_menu = QMenu("&Edit", self)
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        view_menu = next(a for a in self.menuBar().actions() if a.text() == "&View")
        self.menuBar().insertMenu(view_menu, edit_menu)  # File, Edit, View, Tools
        self.vm.history_changed.connect(self._update_undo_actions)
        self._update_undo_actions()

    def _update_undo_actions(self) -> None:
        self.undo_action.setEnabled(self.vm.can_undo)
        self.redo_action.setEnabled(self.vm.can_redo)

    def _build_looks_menu(self) -> None:
        # User looks are read from disk when the menu opens, so a startup never touches the looks folder.
        self.looks_menu = self.menuBar().addMenu("&Looks")
        self.looks_menu.aboutToShow.connect(self.refresh_looks_menu)
        self.user_looks: list[presets.Look] = []
        self._rebuild_looks_menu()

    def refresh_looks_menu(self) -> None:
        self.user_looks, warnings = presets.load_user_looks(self.preferences.looks_directory())
        self._rebuild_looks_menu()
        if warnings:
            self.statusBar().showMessage(" · ".join(warnings))

    def _rebuild_looks_menu(self) -> None:
        menu = self.looks_menu
        menu.clear()
        for look in (*presets.BUILTIN_LOOKS, None, *self.user_looks):
            if look is None:
                menu.addSeparator()
                continue
            action = menu.addAction(look.name.replace("&", "&&"))
            action.triggered.connect(lambda _checked=False, l=look: self.apply_look(l))
        menu.addSeparator()
        menu.addAction("&Save Look…").triggered.connect(lambda: self.save_look())
        self.delete_look_menu = menu.addMenu("&Delete Look")
        self.delete_look_menu.setEnabled(bool(self.user_looks))
        for look in self.user_looks:
            action = self.delete_look_menu.addAction(look.name.replace("&", "&&"))
            action.triggered.connect(lambda _checked=False, l=look: self.delete_look(l))

    def apply_look(self, look: presets.Look) -> None:
        presets.apply_look(self.vm, look)
        self.statusBar().showMessage(f"Applied the look “{look.name}”", 8000)

    def save_look(self) -> bool:
        """Asks for a name and saves the selected object's material with the light and shadow as a user look."""
        name, ok = QInputDialog.getText(self, "Save Look", "Name of the look:")
        if not ok:
            return False
        folder = self.preferences.looks_directory()
        look = presets.capture_look(self.vm, name)
        try:
            try:
                presets.save_user_look(folder, look)
            except FileExistsError:
                buttons = QMessageBox.StandardButton
                answer = QMessageBox.question(
                    self, "Replace look", f"A look called “{look.name}” already exists. Replace it?",
                    buttons.Yes | buttons.No, buttons.No,
                )
                if answer != buttons.Yes:
                    return False
                presets.save_user_look(folder, look, replace=True)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Could not save look", str(exc))
            return False
        self.refresh_looks_menu()
        self.statusBar().showMessage(f"Saved the look “{look.name}”", 8000)
        return True

    def delete_look(self, look: presets.Look) -> bool:
        buttons = QMessageBox.StandardButton
        answer = QMessageBox.question(
            self, "Delete look", f"Delete the look “{look.name}”? This can't be undone.",
            buttons.Yes | buttons.No, buttons.No,
        )
        if answer != buttons.Yes:
            return False
        try:
            presets.delete_user_look(look)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Could not delete look", str(exc))
            return False
        self.refresh_looks_menu()
        self.statusBar().showMessage(f"Deleted the look “{look.name}”", 8000)
        return True

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
            self.recent_menu.addAction("No Recent Files").setEnabled(False)
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
                self, "File not found", f"{path}\n\nThis file no longer exists and was removed from Open Recent."
            )
            return
        self.open_path(path)

    def open_path(self, path: str) -> None:
        """Opens a project or an image, by extension: used by drops and Open Recent."""
        if path.lower().endswith(project_io.PROJECT_EXTENSION):
            self.open_project_path(path)
        else:
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
        # No save prompt: a new photo replaces only the photo. The scene stays, and the change itself becomes
        # an unsaved change of the open project.
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
        self._update_title()
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

    def export_transparent_image(self) -> None:
        """Exports the text and shadow alone as a straight-alpha PNG. Glass still shows the photo it refracts."""
        source = self.vm.document.background_path
        suggested = (os.path.splitext(source)[0] + "_3d_transparent.png") if source else "composition_transparent.png"
        path, _ = QFileDialog.getSaveFileName(self, "Export transparent PNG", suggested, TRANSPARENT_FILTER)
        if not path:
            return
        extension = os.path.splitext(path)[1].lower()
        if not extension:
            path += ".png"
        elif extension != ".png":
            QMessageBox.warning(
                self, "PNG only", "A transparent export can only be saved as PNG: JPEG and other formats have no "
                "transparency. Choose a name ending in .png."
            )
            return
        self.statusBar().showMessage("Rendering at native resolution…")
        try:
            save_image(self.canvas.render_export(transparent=True), path)
        except (ValueError, OSError, RuntimeError) as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        width, height = self.vm.document.native_size
        self.statusBar().showMessage(f"Exported transparent {width}×{height} → {path}", 8000)

    def batch_export(self) -> None:
        source = self.vm.document.background_path
        folder = os.path.dirname(source) if source and os.path.isdir(os.path.dirname(source)) else ""
        dialog = BatchExportDialog(self.vm.document.native_size, folder or self.preferences.last_directory(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.run_batch_export(dialog.folder(), dialog.scales(), dialog.formats())

    def run_batch_export(self, folder: str, scales, formats) -> list[str]:
        """Writes every size in every format into `folder`; returns the files written (stops at the first failure)."""
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Batch export", f"The folder {folder or '(none)'} doesn't exist.")
            return []
        source = self.vm.document.background_path
        stem = os.path.splitext(os.path.basename(source))[0] if source else "composition"
        jobs = batch_jobs(stem, self.vm.document.native_size, scales, formats, folder)
        written, renders = [], {}
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for job in jobs:
                self.statusBar().showMessage(f"Rendering {job.size[0]}×{job.size[1]}…")
                key = (job.size, job.transparent)
                if key not in renders:
                    renders[key] = self.canvas.render_export(transparent=job.transparent, scale=job.scale)
                save_image(renders[key], job.path)
                written.append(job.path)
        except (ValueError, OSError, RuntimeError) as exc:
            QMessageBox.critical(self, "Batch export failed", f"{exc}\n\n{len(written)} of {len(jobs)} files were written.")
            return written
        finally:
            QApplication.restoreOverrideCursor()
        self.preferences.set_last_directory(folder)
        self.statusBar().showMessage(f"Exported {len(written)} files → {folder}", 8000)
        return written

    def _update_zoom_label(self) -> None:
        percent = f"{self.canvas.zoom_factor() * 100:.0f}%"
        self.zoom_label.setText(f"Fit ({percent})" if self.canvas.is_fitted else percent)

    def _update_size_label(self) -> None:
        width, height = self.vm.document.native_size
        has_image = self.vm.document.background is not None
        self.size_label.setText(f"{width}×{height}" if has_image else f"No image · canvas {width}×{height}")


class BatchExportDialog(QDialog):
    """Sizes, formats and a folder for exporting several files in one go."""

    def __init__(self, native_size: tuple[int, int], folder: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Export")
        self.native_size = native_size
        width, height = native_size
        half = scaled_size(native_size, 0.5)
        self.native_check = QCheckBox(f"Native ({width}×{height})")
        self.native_check.setChecked(True)
        self.half_check = QCheckBox(f"50% ({half[0]}×{half[1]})")
        self.custom_check = QCheckBox("Long edge")
        self.custom_spin = QSpinBox()
        self.custom_spin.setRange(16, 16384)
        self.custom_spin.setValue(min(1920, max(native_size)))
        self.custom_spin.setSuffix(" px")
        self.png_check = QCheckBox("PNG")
        self.png_check.setChecked(True)
        self.jpg_check = QCheckBox("JPG")
        self.transparent_check = QCheckBox("Transparent PNG (text and shadow only)")
        self.folder_edit = QLineEdit(folder)
        browse = QPushButton("Choose…")
        browse.clicked.connect(lambda: self._choose_folder())
        note = QLabel(
            "Files are named <photo>_3d_<width>x<height>.png or .jpg; transparent ones end in _transparent.png. "
            "Existing files are never replaced: a number is added to the new file's name instead."
        )
        note.setWordWrap(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        custom = QHBoxLayout()
        custom.addWidget(self.custom_check)
        custom.addWidget(self.custom_spin)
        sizes = QVBoxLayout()
        for widget in (self.native_check, self.half_check):
            sizes.addWidget(widget)
        sizes.addLayout(custom)
        formats = QVBoxLayout()
        for widget in (self.png_check, self.jpg_check, self.transparent_check):
            formats.addWidget(widget)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(browse)
        form = QFormLayout()
        form.addRow("Sizes", sizes)
        form.addRow("Formats", formats)
        form.addRow("Folder", folder_row)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(self.buttons)

        for check in (self.native_check, self.half_check, self.custom_check, self.png_check, self.jpg_check,
                      self.transparent_check):
            check.toggled.connect(self._update_ok)
        self.folder_edit.textChanged.connect(self._update_ok)
        self._update_ok()

    def scales(self) -> list[float]:
        scales = []
        if self.native_check.isChecked():
            scales.append(1.0)
        if self.half_check.isChecked():
            scales.append(0.5)
        if self.custom_check.isChecked():
            scales.append(long_edge_scale(self.native_size, self.custom_spin.value()))
        return scales

    def formats(self) -> list[str]:
        checks = (("png", self.png_check), ("jpg", self.jpg_check), ("transparent", self.transparent_check))
        return [name for name, check in checks if check.isChecked()]

    def folder(self) -> str:
        return self.folder_edit.text().strip()

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export into", self.folder())
        if folder:
            self.folder_edit.setText(folder)

    def _update_ok(self) -> None:
        ok = bool(self.scales() and self.formats() and self.folder())
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)


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
