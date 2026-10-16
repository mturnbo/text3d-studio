import os

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QMimeData, QPoint, QPointF, QSettings, Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence, QMouseEvent

from text3d_studio import ground as ground_model
from text3d_studio import math3d
from text3d_studio.image_io import save_image
from text3d_studio.model import Document
from text3d_studio.preferences import Preferences
from text3d_studio.viewmodel import SceneViewModel
from text3d_studio.views.image_canvas import ImageCanvas, dropped_image_path
from text3d_studio.views.main_window import MainWindow


@pytest.fixture
def vm(qapp):
    return SceneViewModel(Document())


@pytest.fixture
def preferences(tmp_path):
    return Preferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))


@pytest.fixture
def window(vm, preferences):
    win = MainWindow(vm, preferences)
    yield win
    win.close()


def background():
    pixels = np.zeros((40, 60, 4), np.uint8)
    pixels[..., 3] = 255
    return pixels


def test_empty_state_visible_without_background(vm):
    canvas = ImageCanvas(vm)
    assert canvas.empty_state.isVisibleTo(canvas)


def test_empty_state_hides_once_background_loads(vm):
    canvas = ImageCanvas(vm)
    vm.set_background(background(), "photo.png")
    assert not canvas.empty_state.isVisibleTo(canvas)


def test_empty_state_returns_when_background_is_cleared(vm):
    canvas = ImageCanvas(vm)
    vm.set_background(background(), "photo.png")
    vm.set_background(None)
    assert canvas.empty_state.isVisibleTo(canvas)


def test_empty_state_message_shows_native_open_shortcut(vm):
    canvas = ImageCanvas(vm)
    shortcut = QKeySequence(QKeySequence.StandardKey.Open).toString(QKeySequence.SequenceFormat.NativeText)
    assert shortcut and shortcut in canvas.empty_state.message.text()


def test_empty_state_button_emits_open_requested_once(vm):
    canvas = ImageCanvas(vm)
    emitted = []
    canvas.open_requested.connect(lambda: emitted.append(True))
    canvas.empty_state.button.click()
    assert emitted == [True]


def test_empty_state_button_opens_the_dialog(window):
    calls = []
    window.open_image_dialog = lambda: calls.append(True)
    window.canvas.empty_state.button.click()
    assert calls == [True]


def test_empty_state_is_centered_on_canvas(window, qapp):
    window.show()
    qapp.processEvents()
    canvas, overlay = window.canvas, window.canvas.empty_state
    assert abs(overlay.geometry().center().x() - canvas.width() // 2) <= 1
    assert abs(overlay.geometry().center().y() - canvas.height() // 2) <= 1


def test_toolbar_shares_open_and_export_actions_with_file_menu(window):
    buttons = [a for a in window.toolbar.actions() if not a.isSeparator()]
    assert buttons == [window.open_action, window.export_action, window.grid_action]
    file_menu = window.menuBar().actions()[0].menu()
    assert window.open_action in file_menu.actions()
    assert window.export_action in file_menu.actions()


def test_open_action_keeps_standard_shortcut(window):
    assert window.open_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Open)


# ----- Drag and drop ----------------------------------------------------------------------------


def image_file(tmp_path, name="photo.png"):
    path = tmp_path / name
    save_image(background(), str(path))
    return str(path)


def mime_with(*urls):
    mime = QMimeData()
    mime.setUrls([QUrl(u) if "://" in u else QUrl.fromLocalFile(u) for u in urls])
    return mime


def drag_enter(canvas, mime):
    event = QDragEnterEvent(
        QPoint(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    canvas.dragEnterEvent(event)
    return event.isAccepted()


def drop(canvas, mime):
    event = QDropEvent(
        QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    canvas.dropEvent(event)
    return event.isAccepted()


def test_canvas_accepts_drops(vm):
    assert ImageCanvas(vm).acceptDrops()


@pytest.mark.parametrize("name", ["photo.png", "photo.JPG", "scan.tiff", "pic.webp"])
def test_single_supported_local_file_is_accepted(vm, tmp_path, name):
    path = image_file(tmp_path, name)
    canvas = ImageCanvas(vm)
    assert dropped_image_path(mime_with(path)) == path
    assert drag_enter(canvas, mime_with(path))


def test_drop_emits_file_dropped_with_local_path(vm, tmp_path):
    path = image_file(tmp_path)
    canvas = ImageCanvas(vm)
    dropped = []
    canvas.file_dropped.connect(dropped.append)
    assert drop(canvas, mime_with(path))
    assert dropped == [path]


def test_unsupported_extension_is_rejected(vm, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    canvas = ImageCanvas(vm)
    assert dropped_image_path(mime_with(str(path))) is None
    assert not drag_enter(canvas, mime_with(str(path)))


def test_multiple_files_are_rejected(vm, tmp_path):
    first, second = image_file(tmp_path, "a.png"), image_file(tmp_path, "b.png")
    assert not drag_enter(ImageCanvas(vm), mime_with(first, second))


def test_remote_url_is_rejected(vm):
    assert not drag_enter(ImageCanvas(vm), mime_with("https://example.com/photo.png"))


def test_missing_file_is_rejected(vm, tmp_path):
    assert not drag_enter(ImageCanvas(vm), mime_with(str(tmp_path / "gone.png")))


def test_directory_with_image_extension_is_rejected(vm, tmp_path):
    folder = tmp_path / "album.png"
    folder.mkdir()
    assert not drag_enter(ImageCanvas(vm), mime_with(str(folder)))


def test_non_file_drag_is_rejected(vm):
    mime = QMimeData()
    mime.setText("just some text")
    assert not drag_enter(ImageCanvas(vm), mime)


def test_rejected_drop_emits_nothing(vm, tmp_path):
    canvas = ImageCanvas(vm)
    dropped = []
    canvas.file_dropped.connect(dropped.append)
    assert not drop(canvas, mime_with("https://example.com/photo.png"))
    assert dropped == []


def test_dropping_onto_window_loads_the_background(window, tmp_path):
    path = image_file(tmp_path)
    drop(window.canvas, mime_with(path))
    assert window.vm.document.background_path == path
    assert window.vm.document.native_size == (60, 40)
    assert not window.canvas.empty_state.isVisibleTo(window.canvas)


def test_empty_state_mentions_dropping(vm):
    assert "drop" in ImageCanvas(vm).empty_state.message.text().lower()


# ----- Last folder and recent files -------------------------------------------------------------


def recent_entries(window):
    return [a for a in window.recent_menu.actions() if a.data()]


def test_open_dialog_starts_in_last_folder(window, preferences, tmp_path, monkeypatch):
    preferences.set_last_directory(str(tmp_path))
    seen = []
    monkeypatch.setattr(
        "text3d_studio.views.main_window.QFileDialog.getOpenFileName",
        lambda parent, caption, directory, filters: seen.append(directory) or ("", ""),
    )
    window.open_image_dialog()
    assert seen == [str(tmp_path)]


def test_opening_an_image_remembers_folder_and_recent_entry(window, preferences, tmp_path):
    path = image_file(tmp_path)
    window.load_image_path(path)
    assert preferences.last_directory() == str(tmp_path)
    assert [a.data() for a in recent_entries(window)] == [path]


def test_dropped_image_is_added_to_recent(window, preferences, tmp_path):
    path = image_file(tmp_path)
    drop(window.canvas, mime_with(path))
    assert preferences.recent_files() == [path]


def test_failed_load_is_not_added_to_recent(window, preferences, tmp_path, monkeypatch):
    broken = tmp_path / "broken.png"
    broken.write_text("not an image")
    warnings = []
    monkeypatch.setattr("text3d_studio.views.main_window.QMessageBox.warning", lambda *args: warnings.append(args))
    window.load_image_path(str(broken))
    assert len(warnings) == 1
    assert preferences.recent_files() == []


def test_empty_recent_menu_shows_disabled_placeholder_and_clear(window):
    texts = {a.text(): a.isEnabled() for a in window.recent_menu.actions() if not a.isSeparator()}
    assert texts == {"No Recent Images": False, "Clear Menu": False}


def test_recent_menu_lists_newest_first(window, tmp_path):
    first, second = image_file(tmp_path, "first.png"), image_file(tmp_path, "second.png")
    window.load_image_path(first)
    window.load_image_path(second)
    assert [a.data() for a in recent_entries(window)] == [second, first]


def test_recent_menu_restores_saved_entries_on_startup(vm, preferences, tmp_path):
    path = image_file(tmp_path)
    preferences.add_recent_file(path)
    win = MainWindow(vm, preferences)
    assert [a.data() for a in recent_entries(win)] == [path]


def test_choosing_a_recent_entry_loads_it(window, preferences, tmp_path):
    path = image_file(tmp_path)
    preferences.add_recent_file(path)
    window._rebuild_recent_menu()
    recent_entries(window)[0].trigger()
    assert window.vm.document.background_path == path


def test_missing_recent_file_is_removed_with_message(window, preferences, tmp_path, monkeypatch):
    missing = str(tmp_path / "deleted.png")
    preferences.add_recent_file(missing)
    window._rebuild_recent_menu()
    messages = []
    monkeypatch.setattr("text3d_studio.views.main_window.QMessageBox.information", lambda *args: messages.append(args))
    recent_entries(window)[0].trigger()
    assert len(messages) == 1 and missing in messages[0][2]
    assert preferences.recent_files() == []
    assert recent_entries(window) == []
    assert window.vm.document.background is None


def test_clear_menu_empties_recent_files(window, preferences, tmp_path):
    window.load_image_path(image_file(tmp_path))
    clear = next(a for a in window.recent_menu.actions() if a.text() == "Clear Menu")
    assert clear.isEnabled()
    clear.trigger()
    assert preferences.recent_files() == []
    assert recent_entries(window) == []


def test_recent_label_shortens_home_and_escapes_ampersands():
    from text3d_studio.views.main_window import _menu_label

    home = os.path.expanduser("~")
    assert _menu_label(os.path.join(home, "Pictures", "R&D.png")) == "R&&D.png — ~/Pictures"
    assert _menu_label("/Volumes/Card/photo.jpg") == "photo.jpg — /Volumes/Card"


# ----- Open handling, size notice and busy cursor -----------------------------------------------


def sized_image(tmp_path, name, width, height):
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[..., 3] = 255
    path = tmp_path / name
    save_image(pixels, str(path))
    return str(path)


def fake_dialog(monkeypatch, result):
    calls = []

    def get_open_file_name(*args):
        calls.append(args)
        return (result, "")

    monkeypatch.setattr("text3d_studio.views.main_window.QFileDialog.getOpenFileName", get_open_file_name)
    return calls


def test_open_action_shows_dialog_and_loads_chosen_file(window, tmp_path, monkeypatch):
    path = image_file(tmp_path)
    calls = fake_dialog(monkeypatch, path)
    window.open_action.trigger()
    assert len(calls) == 1
    assert window.vm.document.background_path == path


def test_cancelled_dialog_loads_nothing(window, preferences, monkeypatch):
    fake_dialog(monkeypatch, "")
    window.open_action.trigger()
    assert window.vm.document.background is None
    assert preferences.recent_files() == []


def test_first_load_reports_export_size(window, tmp_path):
    window.load_image_path(sized_image(tmp_path, "a.png", 300, 200))
    assert window.statusBar().currentMessage() == "Loaded a.png · export size 300×200"


def test_loading_a_different_size_reports_the_change(window, tmp_path):
    window.load_image_path(sized_image(tmp_path, "a.png", 300, 200))
    window.load_image_path(sized_image(tmp_path, "b.png", 120, 90))
    assert window.statusBar().currentMessage() == "Loaded b.png · export size is now 120×90 (was 300×200)"


def test_loading_the_same_size_does_not_report_a_change(window, tmp_path):
    window.load_image_path(sized_image(tmp_path, "a.png", 300, 200))
    window.load_image_path(sized_image(tmp_path, "b.png", 300, 200))
    assert window.statusBar().currentMessage() == "Loaded b.png · export size 300×200"


def test_loading_a_new_image_keeps_the_text_transform(window, tmp_path):
    window.vm.set("transform", "x", 2.5)
    window.vm.set("transform", "rot_y", 33.0)
    before = dict(vars(window.vm.scene.transform))
    window.load_image_path(sized_image(tmp_path, "a.png", 300, 200))
    window.load_image_path(sized_image(tmp_path, "b.png", 120, 900))
    assert vars(window.vm.scene.transform) == before


def test_busy_cursor_shown_during_decode_and_restored(window, tmp_path, monkeypatch, qapp):
    import text3d_studio.views.main_window as main_window

    path = image_file(tmp_path)
    seen = []
    real_load = main_window.load_image

    def spy_load(p):
        seen.append(qapp.overrideCursor().shape())
        return real_load(p)

    monkeypatch.setattr(main_window, "load_image", spy_load)
    window.load_image_path(path)
    assert seen == [Qt.CursorShape.WaitCursor]
    assert qapp.overrideCursor() is None


def test_busy_cursor_restored_when_decode_fails(window, tmp_path, monkeypatch, qapp):
    broken = tmp_path / "broken.png"
    broken.write_text("not an image")
    monkeypatch.setattr("text3d_studio.views.main_window.QMessageBox.warning", lambda *args: None)
    window.load_image_path(str(broken))
    assert qapp.overrideCursor() is None
    assert window.vm.document.background is None


# ----- Ground model controls --------------------------------------------------------------------


def test_panel_has_ground_controls_and_no_legacy_transform_controls(vm):
    from text3d_studio.views.control_panel import ControlPanel

    keys = set(ControlPanel(vm)._setters)
    assert {("ground", f) for f in ("pitch", "roll", "yaw", "height")} <= keys
    assert {("transform", f) for f in ("x", "z", "rot_y", "size", "depth", "layers")} <= keys
    assert not {("transform", f) for f in ("y", "rot_x", "rot_z")} & keys


def test_ground_slider_and_model_stay_in_sync(vm):
    from text3d_studio.views.control_panel import ControlPanel, SliderRow

    panel = ControlPanel(vm)
    row = panel.findChild(SliderRow, "ground.pitch")
    row.spin.setValue(12.0)
    assert vm.scene.ground.pitch == 12.0
    vm.set("ground", "pitch", -7.5)
    assert row.spin.value() == -7.5



def sized_canvas(vm, width=800, height=500):
    canvas = ImageCanvas(vm)
    canvas.resize(width, height)
    return canvas


def mouse(canvas, kind, pos, button, buttons=None):
    event = QMouseEvent(
        kind, QPointF(*pos), QPointF(*pos), button,
        button if buttons is None else buttons, Qt.KeyboardModifier.NoModifier,
    )
    {
        QEvent.Type.MouseButtonPress: canvas.mousePressEvent,
        QEvent.Type.MouseMove: canvas.mouseMoveEvent,
        QEvent.Type.MouseButtonRelease: canvas.mouseReleaseEvent,
    }[kind](event)


def drag(canvas, start, end, button=Qt.MouseButton.LeftButton):
    mouse(canvas, QEvent.Type.MouseButtonPress, start, button)
    mouse(canvas, QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, button)
    mouse(canvas, QEvent.Type.MouseButtonRelease, end, button, Qt.MouseButton.NoButton)


def text_anchor_on_screen(canvas):
    rect_x, rect_y, rect_w, rect_h = canvas._image_rect()
    scene = canvas.vm.scene
    pixel = math3d.project_to_image(
        ground_model.text_position(scene.ground, scene.transform), scene.camera.fov, (rect_w, rect_h)
    )
    return (pixel[0] + rect_x, pixel[1] + rect_y)


@pytest.mark.parametrize("target", [(520, 420), (250, 330), (400, 290)])
def test_dragging_text_keeps_the_grabbed_ground_point_under_the_cursor(vm, target):
    vm.set("ground", "pitch", 15.0)
    canvas = sized_canvas(vm)
    drag(canvas, text_anchor_on_screen(canvas), target)
    np.testing.assert_allclose(text_anchor_on_screen(canvas), target, atol=1e-6)


def test_dragging_keeps_the_text_on_the_ground(vm):
    canvas = sized_canvas(vm)
    drag(canvas, text_anchor_on_screen(canvas), (600, 450))
    position = ground_model.text_position(vm.scene.ground, vm.scene.transform)
    plane = ground_model.ground_plane(vm.scene.ground)
    assert plane[:3] @ position + plane[3] == pytest.approx(0, abs=1e-9)


def test_drag_starting_in_the_sky_does_nothing(vm):
    canvas = sized_canvas(vm)
    before = (vm.scene.transform.x, vm.scene.transform.z)
    drag(canvas, (400, 50), (450, 300))
    assert (vm.scene.transform.x, vm.scene.transform.z) == before


def test_dragging_into_the_sky_stops_at_the_last_ground_position(vm):
    canvas = sized_canvas(vm)
    start = text_anchor_on_screen(canvas)
    mouse(canvas, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    mouse(canvas, QEvent.Type.MouseMove, (start[0] + 40, start[1]), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    settled = (vm.scene.transform.x, vm.scene.transform.z)
    mouse(canvas, QEvent.Type.MouseMove, (start[0] + 40, 20), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    assert (vm.scene.transform.x, vm.scene.transform.z) == settled


def test_right_drag_only_changes_yaw(vm):
    canvas = sized_canvas(vm)
    before_ground = dict(vars(vm.scene.ground))
    before = (vm.scene.transform.x, vm.scene.transform.z, vm.scene.transform.rot_y)
    drag(canvas, (400, 300), (450, 340), Qt.MouseButton.RightButton)
    assert (vm.scene.transform.x, vm.scene.transform.z) == before[:2]
    assert vm.scene.transform.rot_y == pytest.approx(before[2] + 50 * 0.4)
    assert vars(vm.scene.ground) == before_ground



# ----- Grid toggle --------------------------------------------------------------------------------


def test_grid_action_toggles_the_grid(window):
    assert not window.grid_action.isChecked()
    window.grid_action.trigger()
    assert window.vm.scene.grid.visible
    window.grid_action.trigger()
    assert not window.vm.scene.grid.visible


def test_grid_action_follows_the_model(window):
    window.vm.set("grid", "visible", True)
    assert window.grid_action.isChecked()


def test_grid_action_has_g_shortcut_and_view_menu_entry(window):
    assert window.grid_action.shortcut() == QKeySequence("G")
    view_menu = next(a.menu() for a in window.menuBar().actions() if a.text() == "&View")
    assert window.grid_action in view_menu.actions()


def test_grid_panel_controls_are_bound(vm):
    from text3d_studio.views.control_panel import ControlPanel, SliderRow

    panel = ControlPanel(vm)
    assert {("grid", f) for f in ("visible", "cell_size", "opacity", "color", "accent_color")} <= set(panel._setters)
    panel.findChild(SliderRow, "grid.cell_size").spin.setValue(2.5)
    assert vm.scene.grid.cell_size == 2.5
