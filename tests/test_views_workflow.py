"""Window workflow: looks, export variants, before/after and zoom. Builds MainWindows, so it sorts after test_render."""

import dataclasses
import os

import numpy as np
import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QMessageBox

from text3d_studio import presets
from text3d_studio.model import Document
from text3d_studio.preferences import Preferences
from text3d_studio.viewmodel import SceneViewModel
from text3d_studio.views.main_window import MainWindow

MAIN = "text3d_studio.views.main_window"


@pytest.fixture
def looks_dir(tmp_path):
    return str(tmp_path / "looks")


@pytest.fixture
def preferences(tmp_path, looks_dir):
    return Preferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat), looks_dir)


@pytest.fixture
def answers(monkeypatch):
    """Answers every QMessageBox.question with `answer[0]` and records the questions."""
    asked = []
    answer = [QMessageBox.StandardButton.Discard]
    monkeypatch.setattr(f"{MAIN}.QMessageBox.question", lambda *args, **kwargs: asked.append(args) or answer[0])
    return asked, answer


@pytest.fixture
def window(qapp, preferences, answers):
    win = MainWindow(SceneViewModel(Document()), preferences)
    yield win
    win.close()


def shown(monkeypatch, kind="warning"):
    calls = []
    monkeypatch.setattr(f"{MAIN}.QMessageBox.{kind}", lambda *args: calls.append(args))
    return calls


def name_prompt(monkeypatch, name, ok=True):
    monkeypatch.setattr(f"{MAIN}.QInputDialog.getText", lambda *args, **kwargs: (name, ok))


# ----- Looks --------------------------------------------------------------------------------------


def menu_texts(menu):
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def test_preferences_default_looks_folder_is_in_app_data(tmp_path):
    prefs = Preferences(QSettings(str(tmp_path / "p.ini"), QSettings.Format.IniFormat))
    assert prefs.looks_directory().endswith(os.path.join("", "looks"))
    assert Preferences(prefs.settings, str(tmp_path)).looks_directory() == str(tmp_path)


def test_looks_menu_lists_builtins_and_actions(window):
    window.refresh_looks_menu()
    texts = menu_texts(window.looks_menu)
    assert texts[: len(presets.BUILTIN_LOOKS)] == [look.name for look in presets.BUILTIN_LOOKS]
    assert "&Save Look…" in texts and "&Delete Look" in texts
    assert not window.delete_look_menu.isEnabled()  # no user looks yet, and built-ins can't be deleted


def test_choosing_a_look_applies_it_as_one_undo_step(window):
    before = dataclasses.replace(window.vm.scene.light)
    action = next(a for a in window.looks_menu.actions() if a.text() == "Chrome at noon")
    action.trigger()
    assert window.vm.scene.light.elevation == 70.0
    assert window.vm.scene.objects[0].material.roughness == 0.08
    assert "Chrome at noon" in window.statusBar().currentMessage()
    window.undo_action.trigger()
    assert window.vm.scene.light == before


def test_save_look_prompts_for_a_name_and_lists_it(window, monkeypatch, looks_dir):
    window.vm.set("light", "elevation", 21.0)
    name_prompt(monkeypatch, "Evening")
    assert window.save_look()
    looks, _ = presets.load_user_looks(looks_dir)
    assert [(l.name, l.light["elevation"]) for l in looks] == [("Evening", 21.0)]
    assert "Evening" in menu_texts(window.looks_menu)
    assert menu_texts(window.delete_look_menu) == ["Evening"]


def test_cancelling_the_name_prompt_saves_nothing(window, monkeypatch, looks_dir):
    name_prompt(monkeypatch, "Never", ok=False)
    assert not window.save_look()
    assert not os.path.exists(looks_dir)


@pytest.mark.parametrize("name", ["", "Oak"])
def test_an_empty_or_builtin_name_is_refused_with_a_message(window, monkeypatch, looks_dir, name):
    warnings = shown(monkeypatch)
    name_prompt(monkeypatch, name)
    assert not window.save_look()
    assert len(warnings) == 1
    assert presets.load_user_looks(looks_dir)[0] == []


def test_saving_over_a_user_look_asks_before_replacing(window, monkeypatch, answers, looks_dir):
    asked, answer = answers
    name_prompt(monkeypatch, "Mine")
    window.save_look()
    window.vm.set("shadow", "opacity", 0.11)
    answer[0] = QMessageBox.StandardButton.No
    assert not window.save_look()
    assert len(asked) == 1
    assert presets.load_user_looks(looks_dir)[0][0].shadow["opacity"] != 0.11
    answer[0] = QMessageBox.StandardButton.Yes
    assert window.save_look()
    looks, _ = presets.load_user_looks(looks_dir)
    assert [(l.name, l.shadow["opacity"]) for l in looks] == [("Mine", 0.11)]


def test_delete_look_asks_and_removes_only_user_looks(window, monkeypatch, answers, looks_dir):
    _, answer = answers
    name_prompt(monkeypatch, "Mine")
    window.save_look()
    look = window.user_looks[0]
    answer[0] = QMessageBox.StandardButton.No
    assert not window.delete_look(look)
    assert os.path.isfile(look.path)
    answer[0] = QMessageBox.StandardButton.Yes
    window.delete_look_menu.actions()[0].trigger()
    assert not os.path.exists(look.path)
    assert window.user_looks == [] and not window.delete_look_menu.isEnabled()


def test_deleting_a_builtin_is_refused(window, monkeypatch, answers):
    _, answer = answers
    answer[0] = QMessageBox.StandardButton.Yes
    warnings = shown(monkeypatch)
    assert not window.delete_look(presets.BUILTIN_LOOKS[0])
    assert len(warnings) == 1


def test_a_bad_look_file_is_skipped_and_reported_in_the_status_bar(window, looks_dir):
    os.makedirs(looks_dir)
    with open(os.path.join(looks_dir, "broken.json"), "w") as file:
        file.write("{")
    window.looks_menu.aboutToShow.emit()
    assert window.user_looks == []
    assert "broken.json" in window.statusBar().currentMessage()


# ----- Transparent export -------------------------------------------------------------------------


def save_dialog(monkeypatch, result):
    calls = []
    monkeypatch.setattr(f"{MAIN}.QFileDialog.getSaveFileName", lambda *args: calls.append(args) or (result, ""))
    return calls


def photo(window, width=160, height=100):
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[..., :3] = (200, 180, 150)
    pixels[..., 3] = 255
    window.vm.set_background(pixels, "/photos/street.jpg")


def test_transparent_export_writes_a_png_with_alpha(window, monkeypatch, tmp_path, qapp):
    from text3d_studio.image_io import load_image

    photo(window)
    window.show()
    qapp.processEvents()
    path = str(tmp_path / "out")
    calls = save_dialog(monkeypatch, path)
    window.export_transparent_action.trigger()
    assert calls[0][2] == "/photos/street_3d_transparent.png"
    assert calls[0][3] == "PNG image (*.png)"  # only PNG is offered
    pixels = load_image(path + ".png")
    assert pixels.shape == (100, 160, 4)
    assert pixels[0, 0, 3] == 0 and pixels[..., 3].max() == 255
    assert "transparent" in window.statusBar().currentMessage()


@pytest.mark.parametrize("name", ["out.jpg", "out.JPEG", "out.bmp"])
def test_transparent_export_refuses_formats_without_alpha(window, monkeypatch, tmp_path, name):
    photo(window)
    save_dialog(monkeypatch, str(tmp_path / name))
    warnings = shown(monkeypatch)
    window.export_transparent_image()
    assert len(warnings) == 1 and "PNG" in warnings[0][1]
    assert not (tmp_path / name).exists()


def test_cancelled_transparent_export_writes_nothing(window, monkeypatch, tmp_path):
    save_dialog(monkeypatch, "")
    window.export_transparent_image()
    assert not [name for name in os.listdir(tmp_path) if name.endswith(".png")]


# ----- Batch export -------------------------------------------------------------------------------


def test_batch_export_writes_every_size_and_format(window, tmp_path, qapp):
    from text3d_studio.image_io import load_image

    photo(window, 160, 100)
    window.show()
    qapp.processEvents()
    out = tmp_path / "out"
    out.mkdir()
    written = window.run_batch_export(str(out), [1.0, 0.5, 40 / 160], ["png", "jpg", "transparent"])
    names = sorted(os.listdir(out))
    assert names == sorted([
        "street_3d_160x100.png", "street_3d_160x100.jpg", "street_3d_160x100_transparent.png",
        "street_3d_80x50.png", "street_3d_80x50.jpg", "street_3d_80x50_transparent.png",
        "street_3d_40x25.png", "street_3d_40x25.jpg", "street_3d_40x25_transparent.png",
    ])
    assert len(written) == 9
    for name in names:
        width, height = (int(n) for n in name.split("_")[2].split(".")[0].split("x"))
        pixels = load_image(str(out / name))
        assert pixels.shape[:2] == (height, width), name
        assert (pixels[0, 0, 3] == 0) == ("transparent" in name), name
    assert "9 files" in window.statusBar().currentMessage()


def test_batch_export_keeps_existing_files(window, tmp_path, qapp):
    photo(window, 160, 100)
    window.show()
    qapp.processEvents()
    existing = tmp_path / "street_3d_160x100.png"
    existing.write_bytes(b"keep me")
    written = window.run_batch_export(str(tmp_path), [1.0], ["png"])
    assert written == [str(tmp_path / "street_3d_160x100-2.png")]
    assert existing.read_bytes() == b"keep me"


def test_batch_export_into_a_missing_folder_is_refused(window, monkeypatch, tmp_path):
    warnings = shown(monkeypatch)
    assert window.run_batch_export(str(tmp_path / "missing"), [1.0], ["png"]) == []
    assert len(warnings) == 1


def test_batch_export_reports_a_failure_and_what_was_written(window, monkeypatch, tmp_path, qapp):
    photo(window, 160, 100)
    window.show()
    qapp.processEvents()
    errors = shown(monkeypatch, "critical")
    written = window.run_batch_export(str(tmp_path), [1.0, 1000.0], ["png"])  # 160000 px wide is past the GPU's limit
    assert written == [str(tmp_path / "street_3d_160x100.png")]
    assert len(errors) == 1 and "1 of 2" in errors[0][2]
    assert sorted(n for n in os.listdir(tmp_path) if n.endswith(".png")) == ["street_3d_160x100.png"]


def batch_dialog(window):
    from text3d_studio.views.main_window import BatchExportDialog

    return BatchExportDialog((3000, 2000), "/photos", window)


def test_batch_dialog_defaults_to_native_png_in_the_given_folder(window):
    dialog = batch_dialog(window)
    assert dialog.scales() == [1.0] and dialog.formats() == ["png"] and dialog.folder() == "/photos"
    assert "1500×1000" in dialog.half_check.text()


def test_batch_dialog_collects_sizes_and_formats(window):
    dialog = batch_dialog(window)
    dialog.half_check.setChecked(True)
    dialog.custom_check.setChecked(True)
    dialog.custom_spin.setValue(1200)
    dialog.jpg_check.setChecked(True)
    dialog.transparent_check.setChecked(True)
    assert dialog.scales() == [1.0, 0.5, 0.4]
    assert dialog.formats() == ["png", "jpg", "transparent"]


def test_batch_dialog_needs_a_size_a_format_and_a_folder(window):
    from PyQt6.QtWidgets import QDialogButtonBox

    dialog = batch_dialog(window)
    ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert ok.isEnabled()
    dialog.png_check.setChecked(False)
    assert not ok.isEnabled()
    dialog.transparent_check.setChecked(True)
    assert ok.isEnabled()
    dialog.native_check.setChecked(False)
    assert not ok.isEnabled()
    dialog.half_check.setChecked(True)
    dialog.folder_edit.setText("  ")
    assert not ok.isEnabled()


def test_batch_dialog_choose_button_picks_a_folder(window, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QPushButton

    monkeypatch.setattr(f"{MAIN}.QFileDialog.getExistingDirectory", lambda *args: str(tmp_path))
    dialog = batch_dialog(window)
    next(b for b in dialog.findChildren(QPushButton) if b.text() == "Choose…").click()
    assert dialog.folder() == str(tmp_path)


def test_batch_export_action_runs_what_the_dialog_chose(window, monkeypatch, tmp_path, qapp):
    from PyQt6.QtWidgets import QDialog

    photo(window, 160, 100)
    window.show()
    qapp.processEvents()

    def accept(dialog):
        dialog.folder_edit.setText(str(tmp_path))
        dialog.half_check.setChecked(True)
        dialog.native_check.setChecked(False)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(f"{MAIN}.BatchExportDialog.exec", accept)
    window.batch_export_action.trigger()
    assert "street_3d_80x50.png" in os.listdir(tmp_path)


def test_cancelled_batch_dialog_exports_nothing(window, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QDialog

    photo(window, 160, 100)
    monkeypatch.setattr(f"{MAIN}.BatchExportDialog.exec", lambda dialog: QDialog.DialogCode.Rejected)
    window.batch_export()
    assert not [name for name in os.listdir(tmp_path) if name.endswith((".png", ".jpg"))]


# ----- Before/after -------------------------------------------------------------------------------


def activate(window, qapp):
    """Shows the window and makes it Qt's active window (see AGENTS.md, Focus in tests)."""
    from PyQt6.QtWidgets import QApplication

    window.show()
    qapp.processEvents()
    QApplication.setActiveWindow(window)
    qapp.processEvents()


def focused_canvas(window, qapp):
    activate(window, qapp)
    window.canvas.setFocus()
    qapp.processEvents()
    return window.canvas


def preview(window, qapp):
    """The canvas's framebuffer as an (H, W, 4) array in device pixels."""
    from text3d_studio.image_io import qimage_to_rgba

    qapp.processEvents()
    return qimage_to_rgba(window.canvas.grabFramebuffer())


def changed_from_photo(pixels):
    return np.any(np.abs(pixels[..., :3].astype(int) - (200, 180, 150)) > 3, axis=-1)


def test_holding_backslash_shows_the_photo_alone(window, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    photo(window, 320, 200)
    window.vm.set("shadow", "enabled", False)
    canvas = focused_canvas(window, qapp)
    with_text = preview(window, qapp)
    QTest.keyPress(canvas, Qt.Key.Key_Backslash)
    assert canvas.showing_original
    photo_only = preview(window, qapp)
    QTest.keyRelease(canvas, Qt.Key.Key_Backslash)
    assert not canvas.showing_original
    after = preview(window, qapp)
    x, y, w, h = canvas._image_rect(canvas.devicePixelRatioF())
    image = lambda p: p[y + 2: y + h - 2, x + 2: x + w - 2]
    assert changed_from_photo(image(with_text)).sum() > 500
    assert not changed_from_photo(image(photo_only)).any()
    assert np.array_equal(after, with_text)


def test_auto_repeat_keeps_showing_the_photo_until_the_real_release(window, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    canvas = focused_canvas(window, qapp)
    QTest.keyPress(canvas, Qt.Key.Key_Backslash)
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QKeyEvent

    repeat_release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Backslash, Qt.KeyboardModifier.NoModifier, "\\", True)
    repeat_press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backslash, Qt.KeyboardModifier.NoModifier, "\\", True)
    qapp.sendEvent(canvas, repeat_release)
    qapp.sendEvent(canvas, repeat_press)
    assert canvas.showing_original
    QTest.keyRelease(canvas, Qt.Key.Key_Backslash)
    assert not canvas.showing_original


def test_losing_focus_while_holding_backslash_stops_showing_the_photo(window, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QLineEdit

    canvas = focused_canvas(window, qapp)
    QTest.keyPress(canvas, Qt.Key.Key_Backslash)
    window.panel.findChild(QLineEdit).setFocus()
    qapp.processEvents()
    assert not canvas.showing_original


def test_view_menu_toggle_shows_the_photo_and_holding_does_not_end_it(window, qapp):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    canvas = focused_canvas(window, qapp)
    assert window.original_action in window.view_menu.actions()
    window.original_action.setChecked(True)
    assert canvas.showing_original
    QTest.keyPress(canvas, Qt.Key.Key_Backslash)
    QTest.keyRelease(canvas, Qt.Key.Key_Backslash)
    assert canvas.showing_original
    window.original_action.setChecked(False)
    assert not canvas.showing_original


def test_showing_the_photo_hides_the_grid_and_fit_handles(window, qapp):
    from text3d_studio.views.image_canvas import FIT_TOOL

    photo(window, 320, 200)
    window.vm.set("text", "content", "")
    activate(window, qapp)
    window.canvas.set_tool(FIT_TOOL)
    assert window.vm.scene.grid.visible
    grid = preview(window, qapp)
    window.original_action.setChecked(True)
    photo_only = preview(window, qapp)
    x, y, w, h = window.canvas._image_rect(window.canvas.devicePixelRatioF())
    assert changed_from_photo(grid[y: y + h, x: x + w]).sum() > 500
    assert not changed_from_photo(photo_only[y + 2: y + h - 2, x + 2: x + w - 2]).any()
    assert not window.canvas.fit_overlay.isVisibleTo(window.canvas)
    window.original_action.setChecked(False)
    assert window.canvas.fit_overlay.isVisibleTo(window.canvas)


def test_exports_ignore_the_before_after_view(window, qapp):
    photo(window, 160, 100)
    activate(window, qapp)
    normal = window.canvas.render_export()
    window.original_action.setChecked(True)
    assert np.array_equal(window.canvas.render_export(), normal)
    assert window.canvas.render_export(transparent=True)[..., 3].max() == 255


# ----- Zoom and pan -------------------------------------------------------------------------------


def zoom_canvas(qapp, width=800, height=500, image=(1600, 1000)):
    from text3d_studio.views.image_canvas import ImageCanvas

    vm = SceneViewModel(Document())
    pixels = np.zeros((image[1], image[0], 4), np.uint8)
    pixels[..., :3] = (200, 180, 150)
    pixels[..., 3] = 255
    vm.set_background(pixels, "photo.png")
    canvas = ImageCanvas(vm)
    canvas.resize(width, height)
    return canvas


def mouse(canvas, kind, pos, button, buttons=None, modifiers=None):
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    modifiers = Qt.KeyboardModifier.NoModifier if modifiers is None else modifiers
    event = QMouseEvent(kind, QPointF(*pos), QPointF(*pos), button, button if buttons is None else buttons, modifiers)
    {
        QEvent.Type.MouseButtonPress: canvas.mousePressEvent,
        QEvent.Type.MouseMove: canvas.mouseMoveEvent,
        QEvent.Type.MouseButtonRelease: canvas.mouseReleaseEvent,
    }[kind](event)


def drag(canvas, start, end, modifiers=None):
    from PyQt6.QtCore import QEvent, Qt

    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    mouse(canvas, QEvent.Type.MouseButtonPress, start, left, modifiers=modifiers)
    mouse(canvas, QEvent.Type.MouseMove, end, none, left, modifiers)
    mouse(canvas, QEvent.Type.MouseButtonRelease, end, left, none, modifiers)


def key(canvas, qapp, kind, which):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent

    event_type = QEvent.Type.KeyPress if kind == "press" else QEvent.Type.KeyRelease
    qapp.sendEvent(canvas, QKeyEvent(event_type, which, Qt.KeyboardModifier.NoModifier))


def test_canvas_starts_fitted(qapp):
    canvas = zoom_canvas(qapp)
    dpr = canvas.devicePixelRatioF()
    assert canvas.is_fitted
    assert canvas.zoom_factor() == pytest.approx(0.5 * dpr)
    assert canvas._image_rect() == (0, 0, 800, 500)


def test_zoom_in_and_out_step_through_the_levels(qapp):
    from text3d_studio.views.image_canvas import ZOOM_LEVELS

    canvas = zoom_canvas(qapp, image=(400, 250))  # small enough that every level fits under the size cap
    fit = canvas.zoom_factor()
    canvas.zoom_in()
    assert canvas.zoom_factor() == min(z for z in ZOOM_LEVELS if z > fit * 1.001)
    canvas.zoom_out()
    canvas.zoom_out()
    assert canvas.zoom_factor() == max(z for z in ZOOM_LEVELS if z < fit / 1.001)
    for _ in range(len(ZOOM_LEVELS) + 2):
        canvas.zoom_in()
    assert canvas.zoom_factor() == ZOOM_LEVELS[-1]
    for _ in range(len(ZOOM_LEVELS) + 2):
        canvas.zoom_out()
    assert canvas.zoom_factor() == ZOOM_LEVELS[0]
    canvas.zoom_to_fit()
    assert canvas.is_fitted and canvas.zoom_factor() == fit


def test_zoom_stops_before_the_image_gets_too_large_to_render(qapp):
    from text3d_studio.views.image_canvas import MAX_ZOOMED_SIDE_PX

    canvas = zoom_canvas(qapp, image=(4000, 2500))
    for _ in range(20):
        canvas.zoom_in()
    assert canvas.zoom_factor() == pytest.approx(MAX_ZOOMED_SIDE_PX / 4000)
    canvas.zoom_actual_size()
    assert canvas.zoom_factor() == 1.0  # 4000 px fits under the cap


def test_actual_size_shows_one_image_pixel_per_device_pixel(qapp):
    canvas = zoom_canvas(qapp, image=(1600, 1000))
    canvas.zoom_actual_size()
    assert canvas._image_rect(canvas.devicePixelRatioF())[2:] == (1600, 1000)
    dpr = canvas.devicePixelRatioF()
    _, _, w, h = canvas._image_rect()
    assert (w, h) == (round(1600 / dpr), round(1000 / dpr))


def test_actual_size_preview_matches_the_photo_pixel_for_pixel(window, qapp):
    from text3d_studio.image_io import qimage_to_rgba

    stripes = np.zeros((60, 90, 4), np.uint8)
    stripes[..., 3] = 255
    stripes[:, ::2, :3] = 255  # one-pixel columns: any resampling would blur them to grey
    stripes[::3, :, 1] = 128
    window.vm.set_background(stripes, "/photos/stripes.png")
    window.vm.set("text", "content", "")
    window.show()
    qapp.processEvents()
    canvas = window.canvas
    canvas.zoom_actual_size()
    qapp.processEvents()
    grabbed = qimage_to_rgba(canvas.grabFramebuffer())
    x, y, w, h = canvas._image_rect(canvas.devicePixelRatioF())
    assert (w, h) == (90, 60)
    assert np.array_equal(grabbed[y: y + h, x: x + w, :3], stripes[..., :3])


def test_zooming_keeps_the_point_at_the_center(qapp):
    canvas = zoom_canvas(qapp)
    canvas.zoom_actual_size()
    canvas.pan_by(-100, 40)
    before = canvas._image_pixel(canvas.rect().center().toPointF())
    x, y, w, h = canvas._image_rect()
    fraction = (before[0] / w, before[1] / h)
    canvas.zoom_in()
    after = canvas._image_pixel(canvas.rect().center().toPointF())
    _, _, w, h = canvas._image_rect()
    assert after[0] / w == pytest.approx(fraction[0], abs=0.003)
    assert after[1] / h == pytest.approx(fraction[1], abs=0.003)


def test_space_drag_pans_and_is_clamped_to_the_image(qapp):
    from PyQt6.QtCore import Qt

    canvas = zoom_canvas(qapp, image=(3200, 2000))  # larger than the canvas at 100% on any screen
    canvas.zoom_actual_size()
    x0, y0, w, h = canvas._image_rect()
    key(canvas, qapp, "press", Qt.Key.Key_Space)
    drag(canvas, (400, 250), (430, 260))
    x1, y1, _, _ = canvas._image_rect()
    assert (x1 - x0, y1 - y0) == (30, 10)
    drag(canvas, (400, 250), (5000, 5000))
    x2, y2, _, _ = canvas._image_rect()
    assert (x2, y2) == (0, 0)  # the image's top left corner reaches the canvas's and stops
    drag(canvas, (400, 250), (-9000, -9000))
    x3, y3, _, _ = canvas._image_rect()
    assert (x3 + w, y3 + h) == pytest.approx((800, 500), abs=1)
    key(canvas, qapp, "release", Qt.Key.Key_Space)


def test_panning_does_nothing_while_the_image_fits(qapp):
    from PyQt6.QtCore import Qt

    canvas = zoom_canvas(qapp)
    before = canvas._image_rect()
    transform = dataclasses.replace(canvas.vm.selected_object.transform)
    key(canvas, qapp, "press", Qt.Key.Key_Space)
    drag(canvas, (400, 300), (300, 200))
    assert canvas._image_rect() == before
    assert canvas.vm.selected_object.transform == transform  # Space + drag never moves the text


def test_panning_is_not_an_undo_step_and_moves_no_text(qapp):
    from PyQt6.QtCore import Qt

    canvas = zoom_canvas(qapp)
    canvas.zoom_actual_size()
    transform = dataclasses.replace(canvas.vm.selected_object.transform)
    key(canvas, qapp, "press", Qt.Key.Key_Space)
    drag(canvas, (400, 300), (300, 200))
    key(canvas, qapp, "release", Qt.Key.Key_Space)
    assert canvas.vm.selected_object.transform == transform
    assert not canvas.vm.can_undo
    drag(canvas, (400, 300), (420, 300))  # without Space, the drag moves the text again
    assert canvas.vm.selected_object.transform != transform


def text_anchor(canvas):
    from text3d_studio import ground as ground_model
    from text3d_studio import math3d

    rect_x, rect_y, rect_w, rect_h = canvas._image_rect()
    scene = canvas.vm.scene
    pixel = math3d.project_to_image(
        ground_model.text_position(scene.ground, canvas.vm.selected_object.transform), scene.camera.fov, (rect_w, rect_h)
    )
    return (pixel[0] + rect_x, pixel[1] + rect_y)


def test_dragging_the_text_while_zoomed_follows_the_cursor(qapp):
    canvas = zoom_canvas(qapp)
    canvas.vm.set("ground", "pitch", 12.0)
    canvas.zoom_in()
    canvas.zoom_in()
    start = text_anchor(canvas)
    assert 0 < start[0] < 800 and 0 < start[1] < 500  # the text is on screen
    drag(canvas, start, (start[0] + 60, start[1] - 25))
    after = text_anchor(canvas)
    assert after == pytest.approx((start[0] + 60, start[1] - 25), abs=0.6)


def test_option_slide_keeps_the_grabbed_ground_point_under_the_cursor_when_zoomed(qapp):
    from PyQt6.QtCore import Qt

    canvas = zoom_canvas(qapp)
    canvas.vm.set("ground", "pitch", 12.0)
    canvas.zoom_actual_size()
    canvas.pan_by(0, -120)
    start, end = (400.0, 330.0), (440.0, 300.0)
    grabbed, target = canvas.ground_point_at(QPointF_(start)), canvas.ground_point_at(QPointF_(end))
    assert grabbed is not None and target is not None
    before = dataclasses.replace(canvas.vm.selected_object.transform)
    drag(canvas, start, end, Qt.KeyboardModifier.AltModifier)
    after = canvas.vm.selected_object.transform
    assert (after.x - before.x, after.z - before.z) == pytest.approx(
        (target[0] - grabbed[0], target[1] - grabbed[1]), abs=1e-9
    )
    assert canvas.ground_point_at(QPointF_(start)) == pytest.approx(grabbed)  # the zoomed mapping is unchanged


def QPointF_(pos):
    from PyQt6.QtCore import QPointF

    return QPointF(*pos)


def test_picking_selects_the_object_under_the_cursor_when_zoomed(qapp):
    from PyQt6.QtCore import QEvent, Qt

    canvas = zoom_canvas(qapp)
    vm = canvas.vm
    vm.add_object()
    for index, x in ((0, -3.5), (1, 3.5)):
        vm.select(index)
        vm.set("text", "content", "HH")
        vm.set("transform", "rot_y", 0.0)
        vm.set("transform", "x", x)
        vm.set("transform", "z", -12.0)
        vm.set("transform", "size", 3.0)
    vm.select(1)
    canvas.zoom_in()
    canvas.zoom_in()
    canvas.pan_by(400, 0)  # bring the left object into view
    hit = None
    for x in range(0, 800, 4):
        for y in range(0, 500, 4):
            if canvas.object_at(QPointF_((x, y))) == 0:
                hit = (x, y)
                break
        if hit:
            break
    assert hit is not None
    mouse(canvas, QEvent.Type.MouseButtonPress, hit, Qt.MouseButton.LeftButton)
    mouse(canvas, QEvent.Type.MouseButtonRelease, hit, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)
    assert vm.selected == 0


def test_export_ignores_the_zoom(window, qapp):
    photo(window, 160, 100)
    window.show()
    qapp.processEvents()
    fitted = window.canvas.render_export()
    window.zoom_in_action.trigger()
    window.zoom_in_action.trigger()
    window.canvas.pan_by(-30, 10)
    qapp.processEvents()
    assert not window.canvas.is_fitted
    assert np.array_equal(window.canvas.render_export(), fitted)


def test_view_menu_zoom_actions_and_shortcuts(window):
    from PyQt6.QtGui import QKeySequence

    actions = window.view_menu.actions()
    expected = {
        window.zoom_in_action: "Ctrl+=", window.zoom_out_action: "Ctrl+-",
        window.zoom_fit_action: "Ctrl+0", window.zoom_actual_action: "Ctrl+1",
    }
    for action, shortcut in expected.items():
        assert action in actions
        assert action.shortcut() == QKeySequence(shortcut)


def test_zoom_label_follows_the_zoom(window, qapp):
    photo(window, 160, 100)
    window.zoom_actual_action.trigger()
    assert window.zoom_label.text() == "100%"
    window.zoom_fit_action.trigger()
    assert window.zoom_label.text().startswith("Fit (")


def test_a_new_photo_starts_fitted(window):
    photo(window, 160, 100)
    window.canvas.zoom_actual_size()
    photo(window, 320, 200)
    assert window.canvas.is_fitted


def test_a_panned_preview_shows_the_matching_part_of_the_photo(window, qapp):
    from text3d_studio.image_io import qimage_to_rgba

    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, (1800, 2600, 4), dtype=np.uint8)  # bigger than the canvas at 100%
    noise[..., 3] = 255
    window.vm.set_background(noise, "/photos/noise.png")
    window.vm.set("text", "content", "")
    window.show()
    qapp.processEvents()
    canvas = window.canvas
    canvas.zoom_actual_size()
    canvas.pan_by(-150, -60)
    qapp.processEvents()
    grabbed = qimage_to_rgba(canvas.grabFramebuffer())
    x, y, w, h = canvas._image_rect(canvas.devicePixelRatioF())
    assert x < 0 and y < 0  # the image overflows the canvas on every side
    fb_h, fb_w = grabbed.shape[:2]
    visible = grabbed[max(0, y): min(fb_h, y + h), max(0, x): min(fb_w, x + w), :3]
    expected = noise[max(0, -y): max(0, -y) + visible.shape[0], max(0, -x): max(0, -x) + visible.shape[1], :3]
    assert visible.shape[0] > 100 and visible.shape[1] > 100
    assert np.array_equal(visible, expected)


def test_fit_handles_follow_the_zoom(qapp):
    from text3d_studio.views.image_canvas import FIT_TOOL

    canvas = zoom_canvas(qapp)
    canvas.vm.set("ground", "pitch", 12.0)
    canvas.set_tool(FIT_TOOL)
    canvas.zoom_in()
    rect_x, rect_y, rect_w, rect_h = canvas._image_rect()
    corners = [(rect_x + u, rect_y + v) for u, v in canvas.fit_corners_px()]
    on_screen = [i for i, (x, y) in enumerate(corners) if 0 < x < 800 and 0 < y < 500]
    assert on_screen
    index = on_screen[0]
    assert canvas.fit_handle_at(QPointF_(corners[index])) == index
    target = (corners[index][0] + 15, corners[index][1] - 10)
    drag(canvas, corners[index], target)
    fx, fy = canvas.fit_corners[index]
    assert (rect_x + fx * rect_w, rect_y + fy * rect_h) == pytest.approx(target, abs=1e-6)


@pytest.mark.parametrize("tool_name", ["WALL_TOOL", "SUN_TOOL", "FIT_TOOL"])
def test_showing_the_photo_hides_every_tool_overlay_even_when_switching_tools(window, qapp, tool_name):
    from text3d_studio.views import image_canvas

    tool = getattr(image_canvas, tool_name)
    canvas = window.canvas
    overlay = canvas.fit_overlay if tool == image_canvas.FIT_TOOL else canvas.tool_overlay
    photo(window, 320, 200)
    activate(window, qapp)
    canvas.set_tool(tool)
    assert overlay.isVisibleTo(canvas)
    window.original_action.setChecked(True)
    assert not canvas.fit_overlay.isVisibleTo(canvas) and not canvas.tool_overlay.isVisibleTo(canvas)
    canvas.set_tool(image_canvas.TEXT_TOOL)
    canvas.set_tool(tool)  # picking a tool while the photo shows must not bring its overlay back
    assert not overlay.isVisibleTo(canvas)
    window.original_action.setChecked(False)
    assert overlay.isVisibleTo(canvas)
