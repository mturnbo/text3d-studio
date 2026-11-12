import copy
import dataclasses
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
    tools = list(window.tool_actions.values())
    assert buttons == [window.open_action, window.export_action, window.grid_action, *tools]
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
    before = dict(vars(window.vm.scene.objects[0].transform))
    window.load_image_path(sized_image(tmp_path, "a.png", 300, 200))
    window.load_image_path(sized_image(tmp_path, "b.png", 120, 900))
    assert vars(window.vm.scene.objects[0].transform) == before


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


def mouse(canvas, kind, pos, button, buttons=None, modifiers=Qt.KeyboardModifier.NoModifier):
    event = QMouseEvent(
        kind, QPointF(*pos), QPointF(*pos), button,
        button if buttons is None else buttons, modifiers,
    )
    {
        QEvent.Type.MouseButtonPress: canvas.mousePressEvent,
        QEvent.Type.MouseMove: canvas.mouseMoveEvent,
        QEvent.Type.MouseButtonRelease: canvas.mouseReleaseEvent,
    }[kind](event)


def drag(canvas, start, end, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.NoModifier):
    mouse(canvas, QEvent.Type.MouseButtonPress, start, button, modifiers=modifiers)
    mouse(canvas, QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, button, modifiers)
    mouse(canvas, QEvent.Type.MouseButtonRelease, end, button, Qt.MouseButton.NoButton, modifiers)


OPTION = Qt.KeyboardModifier.AltModifier
COMMAND = Qt.KeyboardModifier.ControlModifier  # Qt's name for macOS's Command key


def apparent_size(vm):
    """On-screen text size up to a constant: world size over distance from the camera."""
    return vm.scene.objects[0].transform.size / np.linalg.norm(ground_model.text_position(vm.scene.ground, vm.scene.objects[0].transform))


def text_anchor_on_screen(canvas, index=None):
    """Where an object's anchor shows in widget pixels: the selected object's unless `index` names another."""
    rect_x, rect_y, rect_w, rect_h = canvas._image_rect()
    scene = canvas.vm.scene
    obj = canvas.vm.selected_object if index is None else scene.objects[index]
    pixel = math3d.project_to_image(
        ground_model.text_position(scene.ground, obj.transform), scene.camera.fov, (rect_w, rect_h)
    )
    return (pixel[0] + rect_x, pixel[1] + rect_y)


@pytest.mark.parametrize("target", [(520, 420), (250, 330), (400, 290)])
def test_dragging_text_keeps_the_grabbed_ground_point_under_the_cursor(vm, target):
    vm.set("ground", "pitch", 15.0)
    canvas = sized_canvas(vm)
    drag(canvas, text_anchor_on_screen(canvas), target)
    np.testing.assert_allclose(text_anchor_on_screen(canvas), target, atol=1e-6)


@pytest.mark.parametrize("target", [(420, 275), (400, 480), (150, 300)])
def test_plain_drag_moves_the_text_without_changing_its_on_screen_size(vm, target):
    canvas = sized_canvas(vm)
    before = apparent_size(vm)
    drag(canvas, text_anchor_on_screen(canvas), target)
    assert apparent_size(vm) == pytest.approx(before)
    assert vm.scene.objects[0].transform.size != pytest.approx(3.0)  # it had to change size to keep its apparent size


def test_plain_drag_moves_the_text_by_the_cursor_motion_wherever_it_is_grabbed(vm):
    canvas = sized_canvas(vm)
    anchor = text_anchor_on_screen(canvas)
    grab = (anchor[0] + 60, anchor[1] - 20)
    drag(canvas, grab, (grab[0] - 100, grab[1] - 40))
    np.testing.assert_allclose(text_anchor_on_screen(canvas), (anchor[0] - 100, anchor[1] - 40), atol=1e-6)


def test_option_drag_slides_along_the_ground_in_perspective(vm):
    canvas = sized_canvas(vm)
    start = text_anchor_on_screen(canvas)
    drag(canvas, start, (start[0], start[1] - 40), modifiers=OPTION)
    np.testing.assert_allclose(text_anchor_on_screen(canvas), (start[0], start[1] - 40), atol=1e-6)
    assert vm.scene.objects[0].transform.size == 3.0  # further away, so it looks smaller
    assert vm.scene.objects[0].transform.z < -12.0


@pytest.mark.parametrize("modifiers", [Qt.KeyboardModifier.ShiftModifier, Qt.KeyboardModifier.ShiftModifier | OPTION])
@pytest.mark.parametrize("motion, kept_axis", [((80, 9), 1), ((-7, -50), 0)])
def test_shift_drag_locks_to_the_dominant_axis(vm, modifiers, motion, kept_axis):
    canvas = sized_canvas(vm)
    start = text_anchor_on_screen(canvas)
    drag(canvas, start, (start[0] + motion[0], start[1] + motion[1]), modifiers=modifiers)
    assert text_anchor_on_screen(canvas)[kept_axis] == pytest.approx(start[kept_axis])
    moved_axis = 1 - kept_axis
    assert text_anchor_on_screen(canvas)[moved_axis] == pytest.approx(start[moved_axis] + motion[moved_axis])


def test_command_drag_only_changes_yaw(vm):
    canvas = sized_canvas(vm)
    before = (vm.scene.objects[0].transform.x, vm.scene.objects[0].transform.z, vm.scene.objects[0].transform.size, vm.scene.objects[0].transform.rot_y)
    drag(canvas, (400, 300), (450, 340), modifiers=COMMAND)
    assert (vm.scene.objects[0].transform.x, vm.scene.objects[0].transform.z, vm.scene.objects[0].transform.size) == before[:3]
    assert vm.scene.objects[0].transform.rot_y == pytest.approx(before[3] + 50 * 0.4)


def test_plain_drag_stops_before_pushing_the_text_too_far_away(vm):
    from text3d_studio.views.image_canvas import MAX_TEXT_DISTANCE

    vm.set("ground", "pitch", 15.0)  # puts the horizon on screen
    canvas = sized_canvas(vm)
    start = text_anchor_on_screen(canvas)
    for y in np.linspace(start[1], 0, 60):
        mouse(canvas, QEvent.Type.MouseMove if y != start[1] else QEvent.Type.MouseButtonPress, (start[0], y),
              Qt.MouseButton.NoButton if y != start[1] else Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    distance = np.linalg.norm(ground_model.text_position(vm.scene.ground, vm.scene.objects[0].transform))
    assert 20 < distance <= MAX_TEXT_DISTANCE


def pinch(canvas, value):
    from PyQt6.QtGui import QNativeGestureEvent, QPointingDevice
    from PyQt6.QtWidgets import QApplication

    event = QNativeGestureEvent(
        Qt.NativeGestureType.ZoomNativeGesture, QPointingDevice.primaryPointingDevice(), 2,
        QPointF(400, 300), QPointF(400, 300), QPointF(400, 300), value, QPointF(), 0,
    )
    return QApplication.sendEvent(canvas, event)


def test_pinch_scales_text_in_text_tool_cells_in_grid_tool_and_nothing_when_fitting(vm):
    from text3d_studio.views.image_canvas import FIT_TOOL, GRID_TOOL

    canvas = sized_canvas(vm)
    assert pinch(canvas, 0.25)
    assert vm.scene.objects[0].transform.size == pytest.approx(3.0 * 1.25)
    canvas.set_tool(GRID_TOOL)
    pinch(canvas, -0.5)
    assert vm.scene.grid.cell_size == pytest.approx(0.5)
    assert vm.scene.objects[0].transform.size == pytest.approx(3.0 * 1.25)
    canvas.set_tool(FIT_TOOL)
    before = (vm.scene.objects[0].transform.size, vm.scene.grid.cell_size)
    pinch(canvas, 0.3)
    assert (vm.scene.objects[0].transform.size, vm.scene.grid.cell_size) == before


def test_dragging_keeps_the_text_on_the_ground(vm):
    canvas = sized_canvas(vm)
    drag(canvas, text_anchor_on_screen(canvas), (600, 450))
    position = ground_model.text_position(vm.scene.ground, vm.scene.objects[0].transform)
    plane = ground_model.ground_plane(vm.scene.ground)
    assert plane[:3] @ position + plane[3] == pytest.approx(0, abs=1e-9)


def test_drag_starting_in_the_sky_does_nothing(vm):
    canvas = sized_canvas(vm)
    before = (vm.scene.objects[0].transform.x, vm.scene.objects[0].transform.z)
    drag(canvas, (400, 50), (450, 300))
    assert (vm.scene.objects[0].transform.x, vm.scene.objects[0].transform.z) == before


def test_dragging_into_the_sky_stops_at_the_last_ground_position(vm):
    canvas = sized_canvas(vm)
    start = text_anchor_on_screen(canvas)
    mouse(canvas, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    mouse(canvas, QEvent.Type.MouseMove, (start[0] + 40, start[1]), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    settled = (vm.scene.objects[0].transform.x, vm.scene.objects[0].transform.z)
    mouse(canvas, QEvent.Type.MouseMove, (start[0] + 40, 20), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    assert (vm.scene.objects[0].transform.x, vm.scene.objects[0].transform.z) == settled


def test_right_drag_only_changes_yaw(vm):
    canvas = sized_canvas(vm)
    before_ground = dict(vars(vm.scene.ground))
    before = (vm.scene.objects[0].transform.x, vm.scene.objects[0].transform.z, vm.scene.objects[0].transform.rot_y)
    drag(canvas, (400, 300), (450, 340), Qt.MouseButton.RightButton)
    assert (vm.scene.objects[0].transform.x, vm.scene.objects[0].transform.z) == before[:2]
    assert vm.scene.objects[0].transform.rot_y == pytest.approx(before[2] + 50 * 0.4)
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
    fields = ("visible", "cell_size", "opacity", "color", "accent_color", "fit_width", "fit_depth", "fit_estimate_fov")
    assert {("grid", f) for f in fields} <= set(panel._setters)
    panel.findChild(SliderRow, "grid.cell_size").spin.setValue(2.5)
    assert vm.scene.grid.cell_size == 2.5


def activate(window, qapp):
    """Shows the window and makes it Qt's active window, so focus works even while another app holds the
    OS focus (macOS won't let a test steal it)."""
    from PyQt6.QtWidgets import QApplication

    window.show()
    qapp.processEvents()  # first let the OS's activation events for this and earlier windows arrive
    QApplication.setActiveWindow(window)
    qapp.processEvents()


def test_g_key_on_canvas_toggles_grid(window, qapp):
    from PyQt6.QtTest import QTest

    activate(window, qapp)
    window.canvas.setFocus()
    qapp.processEvents()
    QTest.keyClick(window.canvas, Qt.Key.Key_G)
    assert window.vm.scene.grid.visible


@pytest.mark.parametrize("field", ["transform.size", "ground.pitch"])
def test_g_toggles_the_grid_while_a_number_field_has_focus(window, qapp, field):
    from PyQt6.QtTest import QTest

    from text3d_studio.views.control_panel import SliderRow

    activate(window, qapp)
    spin = window.panel.findChild(SliderRow, field).spin
    spin.setFocus()
    qapp.processEvents()
    assert spin.hasFocus()
    QTest.keyClick(spin, Qt.Key.Key_G)
    assert window.vm.scene.grid.visible
    QTest.keyClick(spin, Qt.Key.Key_G)
    assert not window.vm.scene.grid.visible


def test_number_fields_still_take_digits_and_tool_letters_switch_tools(window, qapp):
    from PyQt6.QtTest import QTest

    from text3d_studio.views.control_panel import SliderRow
    from text3d_studio.views.image_canvas import GRID_TOOL

    activate(window, qapp)
    spin = window.panel.findChild(SliderRow, "transform.size").spin
    spin.setFocus()
    spin.selectAll()
    QTest.keyClicks(spin, "5")
    QTest.keyClick(spin, Qt.Key.Key_Return)
    assert window.vm.scene.objects[0].transform.size == 5.0
    QTest.keyClick(spin, Qt.Key.Key_P)
    assert window.canvas.tool == GRID_TOOL


def test_typing_g_in_the_text_field_types_instead_of_toggling(window, qapp):
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLineEdit

    window.show()
    edit = window.panel.findChild(QLineEdit)
    edit.setFocus()
    edit.end(False)
    qapp.processEvents()
    QTest.keyClick(edit, Qt.Key.Key_G)
    assert edit.text() == "HELLOg"
    assert not window.vm.scene.grid.visible



# ----- Tools ----------------------------------------------------------------------------------------


def test_text_tool_is_active_by_default(window):
    from text3d_studio.views.image_canvas import TEXT_TOOL

    assert window.canvas.tool == TEXT_TOOL
    assert window.tool_actions[TEXT_TOOL].isChecked()


def test_choosing_grid_tool_shows_grid_and_updates_hint(window):
    from text3d_studio.views.image_canvas import GRID_TOOL, TEXT_TOOL
    from text3d_studio.views.main_window import TOOL_HINTS

    window.tool_actions[GRID_TOOL].trigger()
    assert window.canvas.tool == GRID_TOOL
    assert window.vm.scene.grid.visible
    assert not window.tool_actions[TEXT_TOOL].isChecked()
    assert window.hint_label.text() == TOOL_HINTS[GRID_TOOL]


def test_tool_shortcuts(window):
    from text3d_studio.views.image_canvas import GRID_TOOL, TEXT_TOOL

    assert window.tool_actions[TEXT_TOOL].shortcut() == QKeySequence("V")
    assert window.tool_actions[GRID_TOOL].shortcut() == QKeySequence("P")


def test_switching_back_to_text_tool_keeps_grid_visible(window):
    from text3d_studio.views.image_canvas import GRID_TOOL, TEXT_TOOL

    window.canvas.set_tool(GRID_TOOL)
    window.canvas.set_tool(TEXT_TOOL)
    assert window.tool_actions[TEXT_TOOL].isChecked()
    assert window.vm.scene.grid.visible


def test_dragging_over_the_text_in_grid_tool_moves_grid_lines_not_text(vm):
    from text3d_studio.views.image_canvas import GRID_TOOL

    canvas = sized_canvas(vm)
    canvas.set_tool(GRID_TOOL)
    before = dict(vars(vm.scene.objects[0].transform))
    drag(canvas, text_anchor_on_screen(canvas), (600, 450))
    assert vars(vm.scene.objects[0].transform) == before
    assert (vm.scene.grid.offset_x, vm.scene.grid.offset_z) != (0.0, 0.0)



# ----- Grid tool: horizon -------------------------------------------------------------------------


def grid_canvas(vm, **ground):
    from text3d_studio.views.image_canvas import GRID_TOOL

    for name, value in ground.items():
        vm.set("ground", name, value)
    canvas = sized_canvas(vm)
    canvas.set_tool(GRID_TOOL)
    return canvas


def horizon_distance(canvas, widget_pos):
    line = canvas._horizon_line()
    u, v = canvas._image_pixel(QPointF(*widget_pos))
    return abs(line @ np.array([u, v, 1.0]))


def test_dragging_the_horizon_middle_tilts_ground_to_follow_cursor(vm):
    canvas = grid_canvas(vm)
    drag(canvas, (400, 250), (400, 190))  # level ground: horizon starts mid-height
    assert vm.scene.ground.pitch > 0  # raising the horizon means looking down
    assert horizon_distance(canvas, (400, 190)) < 1e-6
    assert vm.scene.ground.roll == 0


def test_dragging_a_horizon_end_rolls_around_the_center(vm):
    canvas = grid_canvas(vm)
    drag(canvas, (40, 250), (40, 300))
    assert vm.scene.ground.roll != 0
    assert horizon_distance(canvas, (40, 300)) < 1e-6
    assert horizon_distance(canvas, (400, 250)) < 1e-6  # the center stays put


def test_horizon_drags_keep_the_text_in_place_on_screen(vm):
    canvas = grid_canvas(vm)
    before = text_anchor_on_screen(canvas)
    drag(canvas, (400, 250), (400, 215))  # tilt: horizon now at row 215
    drag(canvas, (760, 215), (760, 240))  # roll from the right end
    assert vm.scene.ground.pitch != 0 and vm.scene.ground.roll != 0
    np.testing.assert_allclose(text_anchor_on_screen(canvas), before, atol=1e-6)


def test_horizon_drags_keep_every_object_in_place_on_screen(vm):
    vm.add_object()
    vm.set("transform", "x", -2.0)
    canvas = grid_canvas(vm)
    before = [text_anchor_on_screen(canvas, i) for i in range(2)]
    drag(canvas, (400, 250), (400, 215))
    drag(canvas, (760, 215), (760, 240))
    drag(canvas, (400, 120), (460, 120), Qt.MouseButton.RightButton)  # grid yaw
    assert vm.scene.ground.pitch != 0 and vm.scene.ground.roll != 0 and vm.scene.ground.yaw != 0
    for i in range(2):
        np.testing.assert_allclose(text_anchor_on_screen(canvas, i), before[i], atol=1e-6)


def test_tilt_is_clamped_to_the_slider_range(vm):
    canvas = grid_canvas(vm)
    drag(canvas, (400, 250), (400, -50000))
    assert vm.scene.ground.pitch <= 89.0


def test_press_away_from_the_horizon_leaves_ground_alone(vm):
    canvas = grid_canvas(vm)
    before = dict(vars(vm.scene.ground))
    drag(canvas, (400, 120), (400, 60))
    assert vars(vm.scene.ground) == before


@pytest.mark.parametrize(
    "pos, shape",
    [((400, 250), Qt.CursorShape.SizeVerCursor), ((30, 251), Qt.CursorShape.SizeBDiagCursor), ((400, 100), Qt.CursorShape.ArrowCursor)],
)
def test_hovering_shows_which_horizon_handle_is_under_the_cursor(vm, pos, shape):
    canvas = grid_canvas(vm)
    mouse(canvas, QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, Qt.MouseButton.NoButton)
    assert canvas.cursor().shape() == shape



# ----- Grid tool: shift, rotate, scale ------------------------------------------------------------


@pytest.mark.parametrize("target", [(520, 420), (300, 330)])
def test_dragging_the_ground_slides_grid_lines_with_the_cursor(vm, target):
    canvas = grid_canvas(vm)
    start = (450, 380)
    grabbed = canvas.ground_point_at(QPointF(*start))
    drag(canvas, start, target)
    now_under_cursor = canvas.ground_point_at(QPointF(*target))
    grid = vm.scene.grid
    # The grid point that was grabbed (in the grid's own, offset coordinates) is now under the cursor.
    np.testing.assert_allclose(
        (now_under_cursor[0] - grid.offset_x, now_under_cursor[1] - grid.offset_z), grabbed, atol=1e-9
    )


def test_ground_drag_does_not_change_ground_pose(vm):
    canvas = grid_canvas(vm, pitch=5.0)
    before = dict(vars(vm.scene.ground))
    drag(canvas, (450, 380), (300, 420))
    assert vars(vm.scene.ground) == before


def test_right_drag_rotates_grid_and_keeps_text_in_place(vm):
    canvas = grid_canvas(vm)
    before = text_anchor_on_screen(canvas)
    drag(canvas, (400, 300), (480, 300), Qt.MouseButton.RightButton)
    assert vm.scene.ground.yaw == pytest.approx(80 * 0.3)
    np.testing.assert_allclose(text_anchor_on_screen(canvas), before, atol=1e-6)


def wheel(canvas, steps):
    from PyQt6.QtCore import QPoint
    from PyQt6.QtGui import QWheelEvent

    event = QWheelEvent(
        QPointF(400, 300), QPointF(400, 300), QPoint(), QPoint(0, int(120 * steps)),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False,
    )
    canvas.wheelEvent(event)


def test_wheel_scales_cells_in_grid_tool_and_text_in_text_tool(vm):
    from text3d_studio.views.image_canvas import TEXT_TOOL

    canvas = grid_canvas(vm)
    wheel(canvas, 2)
    assert vm.scene.grid.cell_size == pytest.approx(1.06**2)
    assert vm.scene.objects[0].transform.size == 3.0
    canvas.set_tool(TEXT_TOOL)
    wheel(canvas, 1)
    assert vm.scene.objects[0].transform.size == pytest.approx(3.0 * 1.06)


def test_cell_size_is_clamped(vm):
    canvas = grid_canvas(vm)
    wheel(canvas, 200)
    assert vm.scene.grid.cell_size == 10.0
    wheel(canvas, -400)
    assert vm.scene.grid.cell_size == 0.1


def test_hovering_the_ground_in_grid_tool_shows_move_cursor(vm):
    canvas = grid_canvas(vm)
    mouse(canvas, QEvent.Type.MouseMove, (450, 400), Qt.MouseButton.NoButton, Qt.MouseButton.NoButton)
    assert canvas.cursor().shape() == Qt.CursorShape.SizeAllCursor


# ----- Fit Rectangle tool -------------------------------------------------------------------------


def fit_canvas(vm):
    from text3d_studio.views.image_canvas import FIT_TOOL

    canvas = sized_canvas(vm)
    canvas.set_tool(FIT_TOOL)
    return canvas


def true_rectangle_px(canvas, truth, corners=((-1.0, -7.0), (1.0, -7.0), (1.0, -10.0), (-1.0, -10.0)), fov=None):
    """Widget positions of a 2 x 3 rectangle lying on ground `truth`, seen through `fov` (default: the scene's)."""
    rect_x, rect_y, rect_w, rect_h = canvas._image_rect()
    fov = canvas.vm.scene.camera.fov if fov is None else fov
    pixels = [math3d.project_to_image(ground_model.ground_to_world(truth, x, z), fov, (rect_w, rect_h)) for x, z in corners]
    return [(rect_x + u, rect_y + v) for u, v in pixels]


def drag_fit_corner(canvas, index, target, steps=5):
    rect_x, rect_y, _, _ = canvas._image_rect()
    u, v = canvas.fit_corners_px()[index]
    start = (rect_x + u, rect_y + v)
    mouse(canvas, QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    for k in range(1, steps + 1):
        point = (start[0] + (target[0] - start[0]) * k / steps, start[1] + (target[1] - start[1]) * k / steps)
        mouse(canvas, QEvent.Type.MouseMove, point, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    mouse(canvas, QEvent.Type.MouseButtonRelease, target, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)


def test_tracing_a_rectangle_recovers_the_ground_and_keeps_the_text_in_place(vm):
    from text3d_studio.model import GroundParams

    vm.set("grid", "fit_width", 2.0)
    vm.set("grid", "fit_depth", 3.0)
    canvas = fit_canvas(vm)
    anchor_before = text_anchor_on_screen(canvas)
    truth = GroundParams(pitch=14.0, yaw=30.0, roll=-3.0, height=1.8)
    for index, target in enumerate(true_rectangle_px(canvas, truth)):
        drag_fit_corner(canvas, index, target)
    g = vm.scene.ground
    np.testing.assert_allclose((g.pitch, g.yaw, g.roll, g.height), (14.0, 30.0, -3.0, 1.8), atol=1e-6)
    assert canvas.fit_valid
    np.testing.assert_allclose(text_anchor_on_screen(canvas), anchor_before, atol=1e-6)



def test_fit_tool_shortcut_and_handles_only_in_fit_tool(window):
    from text3d_studio.views.image_canvas import FIT_TOOL, TEXT_TOOL

    assert window.tool_actions[FIT_TOOL].shortcut() == QKeySequence("R")
    window.tool_actions[FIT_TOOL].trigger()
    assert window.canvas.fit_overlay.isVisibleTo(window.canvas)
    window.tool_actions[TEXT_TOOL].trigger()
    assert not window.canvas.fit_overlay.isVisibleTo(window.canvas)


def test_starting_handles_describe_the_current_ground(vm):
    vm.set_ground_pose(pitch=9.0, yaw=20.0, roll=2.0)
    canvas = fit_canvas(vm)
    _, _, rect_w, rect_h = canvas._image_rect()
    fit = ground_model.fit_ground_to_rectangle(canvas.fit_corners_px(), (1.0, 1.0), vm.scene.camera.fov, (rect_w, rect_h), 1.0)
    np.testing.assert_allclose((fit.pitch, fit.yaw, fit.roll), (9.0, 20.0, 2.0), atol=1e-6)
    xs = [u for u, _ in canvas.fit_corners_px()]
    assert (max(xs) - min(xs)) > 0.25 * rect_w  # big enough to grab


def test_clicking_a_handle_without_dragging_changes_nothing(vm):
    canvas = fit_canvas(vm)
    before = (dict(vars(vm.scene.ground)), dict(vars(vm.scene.objects[0].transform)))
    rect_x, rect_y, _, _ = canvas._image_rect()
    u, v = canvas.fit_corners_px()[0]
    mouse(canvas, QEvent.Type.MouseButtonPress, (rect_x + u, rect_y + v), Qt.MouseButton.LeftButton)
    mouse(canvas, QEvent.Type.MouseButtonRelease, (rect_x + u, rect_y + v), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    assert (dict(vars(vm.scene.ground)), dict(vars(vm.scene.objects[0].transform))) == before


def test_crossed_corners_turn_invalid_and_leave_the_scene_alone(vm):
    canvas = fit_canvas(vm)
    messages = []
    canvas.fit_status.connect(messages.append)
    before = dict(vars(vm.scene.ground))
    rect_x, rect_y, _, _ = canvas._image_rect()
    corner_2 = canvas.fit_corners_px()[1]
    # Straight to a crossed shape (corner 4 past corner 2): the scene keeps its last valid fit.
    drag_fit_corner(canvas, 3, (rect_x + corner_2[0] + 60, rect_y + corner_2[1] + 20), steps=1)
    assert not canvas.fit_valid
    assert "convex" in messages[-1]
    assert vars(vm.scene.ground) == before


def test_tracing_can_estimate_the_field_of_view(vm):
    from text3d_studio.model import GroundParams

    vm.set("grid", "fit_width", 2.0)
    vm.set("grid", "fit_depth", 3.0)
    vm.set("grid", "fit_estimate_fov", True)
    canvas = fit_canvas(vm)
    truth = GroundParams(pitch=12.0, yaw=35.0, roll=2.0, height=1.6)
    for index, target in enumerate(true_rectangle_px(canvas, truth, fov=55.0)):
        drag_fit_corner(canvas, index, target)
    assert vm.scene.camera.fov == pytest.approx(55.0, abs=1e-6)
    np.testing.assert_allclose((vm.scene.ground.pitch, vm.scene.ground.height), (12.0, 1.6), atol=1e-6)


def test_moving_the_text_between_corner_drags_is_respected(vm):
    from text3d_studio.model import GroundParams

    vm.set("grid", "fit_width", 2.0)
    vm.set("grid", "fit_depth", 3.0)
    canvas = fit_canvas(vm)
    targets = true_rectangle_px(canvas, GroundParams(pitch=14.0, yaw=30.0, roll=-3.0, height=1.8))
    drag_fit_corner(canvas, 0, targets[0])
    drag_fit_corner(canvas, 1, targets[1])
    vm.set("transform", "x", vm.scene.objects[0].transform.x + 1.0)  # the user nudges the text mid-session
    anchor = text_anchor_on_screen(canvas)
    drag_fit_corner(canvas, 2, targets[2])
    drag_fit_corner(canvas, 3, targets[3])
    np.testing.assert_allclose(text_anchor_on_screen(canvas), anchor, atol=1e-6)


def test_horizon_drag_across_the_text_and_back_returns_it(vm):
    canvas = grid_canvas(vm)
    before = (vm.scene.objects[0].transform.x, vm.scene.objects[0].transform.z, vm.scene.objects[0].transform.size)
    mouse(canvas, QEvent.Type.MouseButtonPress, (400, 250), Qt.MouseButton.LeftButton)
    for row in (300, 380, 460, 380, 300, 250):  # the horizon sweeps down past the text, then back
        mouse(canvas, QEvent.Type.MouseMove, (400, row), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    mouse(canvas, QEvent.Type.MouseButtonRelease, (400, 250), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    np.testing.assert_allclose((vm.scene.objects[0].transform.x, vm.scene.objects[0].transform.z, vm.scene.objects[0].transform.size), before, atol=1e-9)


# ----- Materials ----------------------------------------------------------------------------------


def material_combo(panel):
    from PyQt6.QtWidgets import QComboBox

    return panel.findChild(QComboBox, "material.kind")


def test_material_dropdown_lists_every_kind(vm):
    from text3d_studio.materials import KINDS
    from text3d_studio.views.control_panel import ControlPanel

    combo = material_combo(ControlPanel(vm))
    assert [combo.itemData(i) for i in range(combo.count())] == list(KINDS)
    assert [combo.itemText(i) for i in range(combo.count())] == ["Metal", "Plastic", "Stone", "Wood", "Glass"]
    assert combo.currentData() == "metal"


def test_choosing_a_material_applies_its_preset_and_updates_sliders(vm):
    from text3d_studio.views.control_panel import ControlPanel, SliderRow

    panel = ControlPanel(vm)
    combo = material_combo(panel)
    combo.activated.emit(combo.findData("stone"))
    assert vm.scene.objects[0].material.kind == "stone"
    assert panel.findChild(SliderRow, "material.roughness").spin.value() == 0.85


def test_dropdown_follows_the_model(vm):
    from text3d_studio.views.control_panel import ControlPanel

    panel = ControlPanel(vm)
    vm.apply_material("glass")
    assert material_combo(panel).currentData() == "glass"


def material_row_visible(panel, field):
    from PyQt6.QtWidgets import QWidget

    return panel.findChild(QWidget, f"material.{field}").isVisibleTo(panel)


@pytest.mark.parametrize("kind", ["metal", "plastic", "stone", "wood", "glass"])
def test_each_material_shows_its_controls_and_hides_metallic(vm, kind):
    from text3d_studio.views.control_panel import MATERIAL_CONTROLS, ControlPanel

    panel = ControlPanel(vm)
    vm.apply_material(kind)
    assert not material_row_visible(panel, "metallic")
    for field in MATERIAL_CONTROLS[kind]:
        assert material_row_visible(panel, field), field


def test_glass_relabels_its_colors_and_others_restore_them(vm):
    from PyQt6.QtWidgets import QWidget

    from text3d_studio.views.control_panel import ControlPanel

    panel = ControlPanel(vm)
    tint = panel.findChild(QWidget, "material.face_color")
    form = tint.parentWidget().layout()
    vm.apply_material("glass")
    assert form.labelForField(tint).text() == "Tint"
    vm.apply_material("plastic")
    assert form.labelForField(tint).text() == "Face color"


def test_every_material_control_table_covers_all_kinds_with_real_fields():
    import dataclasses

    from text3d_studio.materials import KINDS
    from text3d_studio.model import MaterialParams
    from text3d_studio.views.control_panel import MATERIAL_CONTROLS, MATERIAL_LABELS

    fields = {f.name for f in dataclasses.fields(MaterialParams)}
    assert set(MATERIAL_CONTROLS) == set(KINDS)
    for kind, shown in MATERIAL_CONTROLS.items():
        assert set(shown) <= fields - {"kind", "metallic"}, kind
        assert set(MATERIAL_LABELS.get(kind, {})) <= set(shown), kind


@pytest.mark.parametrize("kind", ["metal", "plastic", "glass"])
def test_untextured_materials_hide_texture_controls(vm, kind):
    from text3d_studio.views.control_panel import ControlPanel

    panel = ControlPanel(vm)
    vm.apply_material(kind)
    for field in ("secondary_color", "texture_scale", "texture_strength", "grain_angle", "bump_strength", "varnish"):
        assert not material_row_visible(panel, field), field


def test_glass_shows_frost_and_tint_depth_and_other_kinds_hide_them(vm):
    from PyQt6.QtWidgets import QWidget

    from text3d_studio.views.control_panel import ControlPanel

    panel = ControlPanel(vm)
    vm.apply_material("glass")
    assert material_row_visible(panel, "absorption")
    roughness = panel.findChild(QWidget, "material.roughness")
    assert roughness.parentWidget().layout().labelForField(roughness).text() == "Frost"
    for kind in ("metal", "plastic", "stone", "wood"):
        vm.apply_material(kind)
        assert not material_row_visible(panel, "absorption"), kind
        assert roughness.parentWidget().layout().labelForField(roughness).text() == "Roughness", kind


def test_wood_shows_grain_controls_and_hides_side_color(vm):
    from PyQt6.QtWidgets import QWidget

    from text3d_studio.views.control_panel import ControlPanel

    panel = ControlPanel(vm)
    vm.apply_material("wood")
    assert material_row_visible(panel, "grain_angle") and material_row_visible(panel, "varnish")
    assert not material_row_visible(panel, "side_color")
    secondary = panel.findChild(QWidget, "material.secondary_color")
    assert secondary.parentWidget().layout().labelForField(secondary).text() == "Dark wood"
    vm.apply_material("stone")
    assert secondary.parentWidget().layout().labelForField(secondary).text() == "Vein color"
    assert not material_row_visible(panel, "grain_angle")


# ----- Panel scrolling ----------------------------------------------------------------------------------


def wheel_over(widget, steps=-1):
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtWidgets import QApplication

    center = QPointF(widget.width() / 2, widget.height() / 2)
    event = QWheelEvent(
        center, widget.mapToGlobal(center), QPoint(), QPoint(0, int(120 * steps)),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False,
    )
    QApplication.sendEvent(widget, event)


def scrollable_panel(window, qapp):
    window.resize(1200, 500)  # short enough that the panel scrolls
    activate(window, qapp)
    return window.panel


@pytest.mark.parametrize("control", ["slider", "spin", "material"])
def test_wheel_over_an_unfocused_control_scrolls_the_panel_instead(window, qapp, control):
    from PyQt6.QtWidgets import QComboBox

    from text3d_studio.views.control_panel import SliderRow

    panel = scrollable_panel(window, qapp)
    row = panel.findChild(SliderRow, "transform.size")
    widget = {"slider": row.slider, "spin": row.spin, "material": panel.findChild(QComboBox, "material.kind")}[control]
    panel.ensureWidgetVisible(widget)
    window.canvas.setFocus()
    scrolled = panel.verticalScrollBar().value()
    before = (window.vm.scene.objects[0].transform.size, window.vm.scene.objects[0].material.kind)
    wheel_over(widget)
    assert (window.vm.scene.objects[0].transform.size, window.vm.scene.objects[0].material.kind) == before
    assert panel.verticalScrollBar().value() > scrolled


@pytest.mark.parametrize("control", ["slider", "spin"])
def test_wheel_over_a_focused_control_changes_its_value(window, qapp, control):
    from text3d_studio.views.control_panel import SliderRow

    panel = scrollable_panel(window, qapp)
    row = panel.findChild(SliderRow, "transform.size")
    widget = row.slider if control == "slider" else row.spin
    widget.setFocus()
    qapp.processEvents()
    wheel_over(widget, steps=1)
    assert window.vm.scene.objects[0].transform.size != 3.0


def test_clicking_a_slider_gives_it_focus(window, qapp):
    from PyQt6.QtTest import QTest

    from text3d_studio.views.control_panel import SliderRow

    panel = scrollable_panel(window, qapp)
    slider = panel.findChild(SliderRow, "transform.size").slider
    QTest.mouseClick(slider, Qt.MouseButton.LeftButton)
    assert slider.hasFocus()


# ----- Multiple text objects ----------------------------------------------------------------------


def test_canvas_rasterizes_only_objects_whose_text_changed(vm):
    vm.add_object()
    canvas = sized_canvas(vm)
    canvas._update_rasters()
    first, second = (canvas._rasters[o.id][1] for o in vm.scene.objects)
    vm.set("text", "content", "NEW")  # the selected (second) object
    vm.set("transform", "size", 4.0)  # not a raster field
    canvas._update_rasters()
    assert canvas._rasters[vm.scene.objects[0].id][1] is first
    assert canvas._rasters[vm.scene.objects[1].id][1] is not second
    deleted = vm.selected_object.id
    vm.delete_object()
    canvas._update_rasters()
    assert set(canvas._rasters) == {vm.scene.objects[0].id}
    assert deleted not in canvas._unsent


def test_new_objects_rasterize_at_once_but_typing_waits_for_the_debounce(vm):
    canvas = sized_canvas(vm)
    canvas._glyph_dirty = False
    vm.set("text", "content", "TYPING")
    assert not canvas._glyph_dirty and canvas._text_timer.isActive()
    vm.add_object()
    assert canvas._glyph_dirty


def place(vm, index, content, **transform):
    vm.select(index)
    vm.set("text", "content", content)
    for name, value in {"rot_y": 0.0, **transform}.items():
        vm.set("transform", name, value)


def two_objects(vm, first="OO", second="XX"):
    """A sized canvas with two objects side by side, facing the camera; the first is selected."""
    vm.add_object()
    place(vm, 0, first, x=-3.5, z=-12.0, size=3.0)
    place(vm, 1, second, x=3.5, z=-12.0, size=3.0)
    vm.select(0)
    return sized_canvas(vm)


def on_face(canvas, index, local):
    """Widget position of a point on an object's front face, given in em units."""
    scene = canvas.vm.scene
    model = ground_model.text_model_matrix(scene.ground, scene.objects[index].transform)
    world = model @ np.array([local[0], local[1], 0.0, 1.0])
    rect_x, rect_y, rect_w, rect_h = canvas._image_rect()
    pixel = math3d.project_to_image(world[:3], scene.camera.fov, (rect_w, rect_h))
    return (pixel[0] + rect_x, pixel[1] + rect_y)


def ink_point(canvas, index):
    """An em-unit point well inside an object's ink: the solid texel nearest the raster's middle."""
    raster = canvas._raster_of(canvas.vm.scene.objects[index])
    alpha = raster.rgba[..., 3]
    solid = alpha == 255
    for axis in (0, 1):
        for shift in (-6, 6):
            solid &= np.roll(alpha, shift, axis) == 255
    rows, cols = np.nonzero(solid)
    height, width = alpha.shape
    nearest = np.argmin((rows - height / 2) ** 2 + (cols - width / 2) ** 2)
    u, v = (cols[nearest] + 0.5) / width, (rows[nearest] + 0.5) / height
    return (raster.origin[0] + u * raster.size[0], raster.origin[1] + (1 - v) * raster.size[1])


def counter_of_o(canvas, index):
    """The middle of the hole of an object whose text is a single O."""
    raster = canvas._raster_of(canvas.vm.scene.objects[index])
    pad = -raster.origin[1]
    return (0.0, (raster.size[1] - 2 * pad) / 2)


def click(canvas, pos, button=Qt.MouseButton.LeftButton):
    mouse(canvas, QEvent.Type.MouseButtonPress, pos, button)
    mouse(canvas, QEvent.Type.MouseButtonRelease, pos, button, Qt.MouseButton.NoButton)


def transforms(vm):
    return [dataclasses.replace(o.transform) for o in vm.scene.objects]


def test_clicking_an_object_selects_it_without_moving_anything(vm):
    canvas = two_objects(vm)
    before = transforms(vm)
    for index in (1, 0, 1):
        click(canvas, on_face(canvas, index, ink_point(canvas, index)))
        assert vm.selected == index
    assert transforms(vm) == before


def test_dragging_an_object_selects_and_moves_only_it(vm):
    canvas = two_objects(vm)
    before = transforms(vm)
    start = on_face(canvas, 1, ink_point(canvas, 1))
    drag(canvas, start, (start[0] - 40, start[1] + 30))
    assert vm.selected == 1
    assert vm.scene.objects[0].transform == before[0]
    assert vm.scene.objects[1].transform != before[1]


def test_a_press_on_empty_ground_keeps_the_selection_and_drags_it(vm):
    canvas = two_objects(vm)
    vm.select(1)
    before = transforms(vm)
    assert canvas.object_at(QPointF(400, 470)) is None
    drag(canvas, (400, 470), (430, 450))
    assert vm.selected == 1
    assert vm.scene.objects[0].transform == before[0]
    assert vm.scene.objects[1].transform != before[1]


def test_a_press_in_a_letters_hole_does_not_pick_that_object(vm):
    canvas = two_objects(vm, first="O")
    vm.select(1)
    hole = on_face(canvas, 0, counter_of_o(canvas, 0))
    assert canvas.object_at(QPointF(*hole)) is None
    click(canvas, hole)
    assert vm.selected == 1


def test_a_press_through_a_hole_picks_the_object_behind(vm):
    canvas = two_objects(vm, first="O", second="MMMM")
    place(vm, 1, "MMMM", x=0.0, z=-40.0, size=12.0)  # far behind the O, showing through its counter
    hole = on_face(canvas, 0, counter_of_o(canvas, 0))
    ring = on_face(canvas, 0, ink_point(canvas, 0))
    vm.select(0)
    click(canvas, hole)
    assert vm.selected == 1
    click(canvas, ring)
    assert vm.selected == 0


def test_the_nearest_object_wins_where_they_overlap(vm):
    canvas = two_objects(vm, first="I", second="I")
    place(vm, 0, "I", x=0.0, z=-20.0, size=6.0)  # twice as far and twice as big: behind the other I
    place(vm, 1, "I", x=0.0, z=-10.0, size=3.0)
    point = QPointF(*on_face(canvas, 1, ink_point(canvas, 1)))
    assert canvas.object_at(point) == 1
    vm.set("transform", "x", 40.0)  # move the near I aside: the far one was under it all along
    assert canvas.object_at(point) == 0
    vm.select(0)
    vm.set("transform", "z", -8.0)  # bring the far one in front of where the near one stood
    vm.select(1)
    vm.set("transform", "x", 0.0)
    assert canvas.object_at(point) == 0


def test_a_press_off_the_image_picks_nothing(vm):
    canvas = two_objects(vm)
    canvas.vm.set_background(np.zeros((100, 400, 4), np.uint8))  # wide: letterboxed above and below
    assert canvas.object_at(QPointF(400, 5)) is None


def test_wheel_and_yaw_act_only_on_the_selected_object(vm):
    canvas = two_objects(vm)
    vm.select(1)
    before = transforms(vm)
    wheel(canvas, 1)
    assert vm.scene.objects[1].transform.size == pytest.approx(before[1].size * 1.06)
    start = on_face(canvas, 0, ink_point(canvas, 0))
    drag(canvas, start, (start[0] + 50, start[1]), Qt.MouseButton.RightButton)  # picks the first, then yaws it
    assert vm.selected == 0
    assert vm.scene.objects[0].transform.rot_y == pytest.approx(50 * 0.4)
    assert vm.scene.objects[0].transform.size == before[0].size
    assert vm.scene.objects[1].transform.rot_y == before[1].rot_y


def test_export_renders_every_object(window, qapp):
    window.vm.set_background(np.full((200, 320, 4), 255, np.uint8), "photo.png")
    window.vm.set("shadow", "enabled", False)
    window.show()
    qapp.processEvents()
    one = window.canvas.render_export()
    window.vm.add_object()
    window.vm.set("text", "content", "MORE")
    two = window.canvas.render_export()
    assert set(window.canvas.compositor.glyphs) == {o.id for o in window.vm.scene.objects}
    covered = lambda pixels: np.any(pixels[..., :3] != 255, axis=-1)
    assert covered(two).sum() > covered(one).sum() + 200
    assert (covered(two) & covered(one)).sum() >= 0.99 * covered(one).sum()  # the first object is still there
    window.vm.delete_object()
    np.testing.assert_array_equal(window.canvas.render_export(), one)


# ----- Objects list in the panel ------------------------------------------------------------------


def object_rows(panel):
    return [panel.object_list.item(i).text() for i in range(panel.object_list.count())]


def test_objects_list_starts_with_the_one_object_and_delete_disabled(vm):
    from text3d_studio.views.control_panel import ControlPanel

    panel = ControlPanel(vm)
    assert object_rows(panel) == ["HELLO"]
    assert panel.object_list.currentRow() == 0
    assert panel.add_button.isEnabled() and panel.duplicate_button.isEnabled()
    assert not panel.delete_button.isEnabled()


def test_add_duplicate_and_delete_buttons_edit_the_objects(vm):
    from text3d_studio.views.control_panel import ControlPanel

    panel = ControlPanel(vm)
    vm.set("text", "content", "FIRST")
    panel.add_button.click()
    assert object_rows(panel) == ["FIRST", "HELLO"] and panel.object_list.currentRow() == 1
    assert panel.delete_button.isEnabled()
    panel.duplicate_button.click()
    assert object_rows(panel) == ["FIRST", "HELLO", "HELLO"] and vm.selected == 2
    panel.object_list.setCurrentRow(0)
    panel.delete_button.click()
    assert object_rows(panel) == ["HELLO", "HELLO"] and vm.selected == 0
    panel.delete_button.click()
    assert object_rows(panel) == ["HELLO"]
    assert not panel.delete_button.isEnabled()
    panel.delete_button.click()  # disabled: does nothing
    assert len(vm.scene.objects) == 1


def test_choosing_a_row_selects_the_object_and_refreshes_every_control(vm):
    from PyQt6.QtWidgets import QLineEdit

    from text3d_studio.views.control_panel import ControlPanel, SliderRow

    panel = ControlPanel(vm)
    vm.add_object()
    vm.set("text", "content", "WOOD")
    vm.set("transform", "depth", 1.5)
    vm.apply_material("wood")
    text = panel.findChild(QLineEdit, "text.content")
    assert (text.text(), material_combo(panel).currentData()) == ("WOOD", "wood")
    panel.object_list.setCurrentRow(0)
    assert vm.selected == 0
    assert text.text() == "HELLO"
    assert panel.findChild(SliderRow, "transform.depth").spin.value() == 0.35
    assert material_combo(panel).currentData() == "metal"
    assert not material_row_visible(panel, "grain_angle") and material_row_visible(panel, "side_color")
    panel.object_list.setCurrentRow(1)
    assert material_row_visible(panel, "grain_angle") and not material_row_visible(panel, "side_color")
    assert panel.findChild(SliderRow, "transform.depth").spin.value() == 1.5


def test_controls_edit_the_object_chosen_in_the_list(vm):
    from text3d_studio.views.control_panel import ControlPanel, SliderRow

    panel = ControlPanel(vm)
    vm.add_object()
    panel.object_list.setCurrentRow(0)
    panel.findChild(SliderRow, "transform.layers").spin.setValue(200)
    assert [o.transform.layers for o in vm.scene.objects] == [200, 96]


def test_typing_renames_the_selected_row(vm):
    from PyQt6.QtWidgets import QLineEdit

    from text3d_studio.views.control_panel import ControlPanel

    panel = ControlPanel(vm)
    vm.add_object()
    panel.findChild(QLineEdit, "text.content").setText("SIGN")
    assert object_rows(panel) == ["HELLO", "SIGN"]
    vm.set("text", "content", "   ")
    assert object_rows(panel) == ["HELLO", "(empty)"]


def test_picking_on_the_canvas_updates_the_list(window):
    from text3d_studio.views.control_panel import SliderRow

    vm, panel, canvas = window.vm, window.panel, window.canvas
    vm.add_object()
    place(vm, 0, "OO", x=-3.5, z=-12.0, size=3.0)
    place(vm, 1, "XX", x=3.5, z=-12.0, size=3.0)
    canvas.resize(800, 500)
    click(canvas, on_face(canvas, 0, ink_point(canvas, 0)))
    assert vm.selected == 0 and panel.object_list.currentRow() == 0
    assert panel.findChild(SliderRow, "transform.x").spin.value() == -3.5


# ----- Undo -------------------------------------------------------------------------------------------


def undo_steps(vm):
    count = 0
    while vm.undo():
        count += 1
    return count


def long_drag(canvas, start, end, button=Qt.MouseButton.LeftButton, steps=10):
    mouse(canvas, QEvent.Type.MouseButtonPress, start, button)
    for k in range(1, steps + 1):
        point = (start[0] + (end[0] - start[0]) * k / steps, start[1] + (end[1] - start[1]) * k / steps)
        mouse(canvas, QEvent.Type.MouseMove, point, Qt.MouseButton.NoButton, button)
    mouse(canvas, QEvent.Type.MouseButtonRelease, end, button, Qt.MouseButton.NoButton)


@pytest.mark.parametrize("button", [Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton])
def test_one_text_drag_is_one_undo_step(vm, button):
    vm.set("ground", "pitch", 15.0)
    vm.clear_history()
    canvas = sized_canvas(vm)
    before = copy.deepcopy(vm.scene)
    start = text_anchor_on_screen(canvas)
    long_drag(canvas, start, (start[0] + 120, start[1] + 40), button)
    assert vm.scene != before
    assert undo_steps(vm) == 1
    assert vm.scene == before


@pytest.mark.parametrize("start, end", [((400, 300), (400, 340)), ((400, 300), (480, 300))], ids=["shift", "rotate"])
def test_one_grid_drag_is_one_undo_step(vm, start, end):
    canvas = grid_canvas(vm, pitch=10.0)
    vm.clear_history()
    before = copy.deepcopy(vm.scene)
    button = Qt.MouseButton.LeftButton if end[1] != start[1] else Qt.MouseButton.RightButton
    long_drag(canvas, start, end, button)
    assert vm.scene != before
    assert undo_steps(vm) == 1
    assert vm.scene == before


def test_one_horizon_drag_that_re_places_the_text_is_one_undo_step(vm):
    canvas = grid_canvas(vm, pitch=10.0)
    vm.clear_history()
    before = copy.deepcopy(vm.scene)
    line = canvas._horizon_line()
    start = (400, canvas._image_rect()[1] - (line[0] * (400 - canvas._image_rect()[0]) + line[2]) / line[1])
    assert canvas.horizon_handle_at(QPointF(*start)) == "tilt"
    long_drag(canvas, start, (start[0], start[1] - 40))
    assert vm.scene.ground.pitch != before.ground.pitch
    assert vm.scene.objects[0].transform != before.objects[0].transform
    assert undo_steps(vm) == 1
    assert vm.scene == before


def test_each_fit_corner_drag_is_one_undo_step(vm):
    from text3d_studio.model import GroundParams

    canvas = fit_canvas(vm)
    vm.clear_history()
    before = copy.deepcopy(vm.scene)
    truth = GroundParams(pitch=14.0, yaw=30.0, roll=-3.0, height=1.8)
    for index, target in enumerate(true_rectangle_px(canvas, truth)):
        drag_fit_corner(canvas, index, target)
    assert undo_steps(vm) == 4
    assert vm.scene == before


def test_a_click_without_motion_records_nothing(vm):
    canvas = sized_canvas(vm)
    click(canvas, text_anchor_on_screen(canvas))
    assert not vm.can_undo


def test_clicking_another_object_then_dragging_it_undoes_the_move_but_keeps_it_selected(vm):
    canvas = two_objects(vm)
    vm.clear_history()
    before = transforms(vm)
    start = on_face(canvas, 1, ink_point(canvas, 1))
    long_drag(canvas, start, (start[0], start[1] + 30))
    assert vm.selected == 1
    vm.undo()
    assert transforms(vm) == before
    assert vm.selected == 1
    assert not vm.can_undo


def test_a_second_button_during_a_drag_stays_in_the_same_step(vm):
    canvas = sized_canvas(vm)
    start = text_anchor_on_screen(canvas)
    left, right = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton
    both = left | right
    mouse(canvas, QEvent.Type.MouseButtonPress, start, left)
    mouse(canvas, QEvent.Type.MouseMove, (start[0] + 30, start[1]), Qt.MouseButton.NoButton, left)
    mouse(canvas, QEvent.Type.MouseButtonPress, (start[0] + 30, start[1]), right, both)
    mouse(canvas, QEvent.Type.MouseButtonRelease, (start[0] + 30, start[1]), right, left)
    assert not vm.can_undo  # the left button is still down
    mouse(canvas, QEvent.Type.MouseMove, (start[0] + 60, start[1]), Qt.MouseButton.NoButton, left)
    mouse(canvas, QEvent.Type.MouseButtonRelease, (start[0] + 60, start[1]), left, Qt.MouseButton.NoButton)
    assert undo_steps(vm) == 1


def test_a_stray_release_without_a_press_does_not_break_undo(vm):
    canvas = sized_canvas(vm)
    mouse(canvas, QEvent.Type.MouseButtonRelease, (400, 300), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    vm.set("light", "azimuth", 10.0)
    assert undo_steps(vm) == 1


def test_wheel_scaling_burst_is_one_undo_step(qapp):
    now = [0.0]
    vm = SceneViewModel(Document(), clock=lambda: now[0])
    canvas = sized_canvas(vm)
    for _ in range(8):
        wheel(canvas, 1)
        now[0] += 0.05
    assert vm.scene.objects[0].transform.size != 3.0
    vm.undo()
    assert vm.scene.objects[0].transform.size == 3.0
    assert not vm.can_undo


def slider_handle_center(slider):
    from PyQt6.QtWidgets import QStyle, QStyleOptionSlider

    option = QStyleOptionSlider()
    slider.initStyleOption(option)
    handle = slider.style().subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, slider)
    return handle.center()


def drag_slider(slider, dx, steps=10):
    from PyQt6.QtWidgets import QApplication

    start = QPointF(slider_handle_center(slider))
    left = Qt.MouseButton.LeftButton

    def send(kind, pos, button, buttons):
        QApplication.sendEvent(slider, QMouseEvent(kind, pos, slider.mapToGlobal(pos), button, buttons,
                                                   Qt.KeyboardModifier.NoModifier))

    send(QEvent.Type.MouseButtonPress, start, left, left)
    for k in range(1, steps + 1):
        send(QEvent.Type.MouseMove, start + QPointF(dx * k / steps, 0), Qt.MouseButton.NoButton, left)
    send(QEvent.Type.MouseButtonRelease, start + QPointF(dx, 0), left, Qt.MouseButton.NoButton)


def test_one_slider_drag_is_one_undo_step(window, qapp):
    from text3d_studio.views.control_panel import SliderRow

    activate(window, qapp)
    row = window.panel.findChild(SliderRow, "light.azimuth")
    window.panel.ensureWidgetVisible(row)
    qapp.processEvents()
    values = []
    window.vm.scene_changed.connect(lambda s, f: values.append(window.vm.scene.light.azimuth) if s == "light" else None)
    drag_slider(row.slider, 80)
    assert len(set(values)) > 2  # the drag passed through several values
    assert undo_steps(window.vm) == 1
    assert window.vm.scene.light.azimuth == 35.0
    assert row.spin.value() == 35.0



def test_a_slider_click_is_one_undo_step(window, qapp):
    from PyQt6.QtTest import QTest

    from text3d_studio.views.control_panel import SliderRow

    activate(window, qapp)
    row = window.panel.findChild(SliderRow, "light.azimuth")
    window.panel.ensureWidgetVisible(row)
    qapp.processEvents()
    QTest.mouseClick(row.slider, Qt.MouseButton.LeftButton, pos=QPoint(row.slider.width() - 4, row.slider.height() // 2))
    assert window.vm.scene.light.azimuth != 35.0
    assert undo_steps(window.vm) == 1
    assert window.vm.scene.light.azimuth == 35.0
