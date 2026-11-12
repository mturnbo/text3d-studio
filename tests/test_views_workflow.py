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
