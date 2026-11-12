import json
import os

import numpy as np
import pytest
from PyQt6.QtCore import QMimeData, QPoint, QPointF, QSettings, Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence
from PyQt6.QtWidgets import QMessageBox

from text3d_studio.image_io import save_image
from text3d_studio.model import Document
from text3d_studio.preferences import Preferences
from text3d_studio.project_io import load_project
from text3d_studio.viewmodel import SceneViewModel
from text3d_studio.views.control_panel import SliderRow
from text3d_studio.views.image_canvas import dropped_image_path
from text3d_studio.views.main_window import MainWindow

MAIN = "text3d_studio.views.main_window"


@pytest.fixture
def preferences(tmp_path):
    return Preferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))


@pytest.fixture
def prompts(monkeypatch):
    """Answers every unsaved-changes prompt with `answer[0]` (Discard unless a test changes it) and records them."""
    asked = []
    answer = [QMessageBox.StandardButton.Discard]
    monkeypatch.setattr(f"{MAIN}.QMessageBox.question", lambda *args, **kwargs: asked.append(args) or answer[0])
    return asked, answer


@pytest.fixture
def window(qapp, preferences, prompts):
    win = MainWindow(SceneViewModel(Document()), preferences)
    yield win
    win.close()


def image_file(folder, name="photo.png"):
    pixels = np.zeros((40, 60, 4), np.uint8)
    pixels[..., 3] = 255
    pixels[:, :30, 0] = 200
    path = str(folder / name)
    save_image(pixels, path)
    return path


def save_dialog(monkeypatch, result):
    calls = []
    monkeypatch.setattr(f"{MAIN}.QFileDialog.getSaveFileName", lambda *args: calls.append(args) or (result, ""))
    return calls


def open_dialog(monkeypatch, result):
    calls = []
    monkeypatch.setattr(f"{MAIN}.QFileDialog.getOpenFileName", lambda *args: calls.append(args) or (result, ""))
    return calls


def errors(monkeypatch, kind="warning"):
    shown = []
    monkeypatch.setattr(f"{MAIN}.QMessageBox.{kind}", lambda *args: shown.append(args))
    return shown


def edited_window(window, tmp_path):
    """A photo, a second object and a light change: something worth saving."""
    window.load_image_path(image_file(tmp_path))
    window.vm.add_object()
    window.vm.set("text", "content", "SECOND")
    window.vm.set("light", "azimuth", 123.0)
    return window


def slider_value(window, name):
    return window.panel.findChild(SliderRow, name).spin.value()


# ----- Menu -------------------------------------------------------------------------------------


def test_file_menu_has_project_actions_with_shortcuts(window):
    texts = [a.text() for a in window.file_menu.actions() if not a.isSeparator()]
    for text in ("Open &Project…", "&Save Project", "Save Project &As…", "Save Project &With Photo…"):
        assert text in texts
    assert texts.index("Open &Project…") < texts.index("&Save Project") < texts.index("&Export…")
    assert window.open_project_action.shortcut() == QKeySequence("Ctrl+Shift+O")
    assert window.save_project_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Save)
    assert window.save_project_as_action.shortcut() == QKeySequence("Ctrl+Shift+S")


# ----- Saving -----------------------------------------------------------------------------------


def test_save_without_a_project_asks_for_a_path(window, tmp_path, monkeypatch):
    edited_window(window, tmp_path)
    target = str(tmp_path / "scene.t3d")
    calls = save_dialog(monkeypatch, target)
    window.save_project_action.trigger()
    assert len(calls) == 1
    assert calls[0][2] == str(tmp_path / "photo.t3d")  # suggested next to the photo
    assert window.project_path == target
    assert load_project(target).document.scene == window.vm.scene
    assert not window.vm.dirty


def test_save_with_a_project_writes_without_asking(window, tmp_path, monkeypatch):
    edited_window(window, tmp_path)
    save_dialog(monkeypatch, str(tmp_path / "scene.t3d"))
    window.save_project()
    calls = save_dialog(monkeypatch, "")
    window.vm.set("camera", "fov", 22.0)
    assert window.save_project() is True
    assert calls == []
    assert load_project(str(tmp_path / "scene.t3d")).document.scene.camera.fov == 22.0


def test_save_as_adds_the_extension(window, tmp_path, monkeypatch):
    save_dialog(monkeypatch, str(tmp_path / "noext"))
    window.save_project_as_action.trigger()
    assert window.project_path == str(tmp_path / "noext.t3d")
    assert os.path.isfile(tmp_path / "noext.t3d")


def test_save_as_suggests_untitled_without_a_photo(window, tmp_path, monkeypatch, preferences):
    preferences.set_last_directory(str(tmp_path))
    calls = save_dialog(monkeypatch, "")
    window.save_project_as()
    assert calls[0][2] == str(tmp_path / "Untitled.t3d")


def test_cancelled_save_as_writes_nothing_and_stays_dirty(window, tmp_path, monkeypatch):
    edited_window(window, tmp_path)
    save_dialog(monkeypatch, "")
    assert window.save_project() is False
    assert window.project_path is None and window.vm.dirty
    assert not any(name.endswith(".t3d") for name in os.listdir(tmp_path))


def test_plain_save_links_the_photo(window, tmp_path, monkeypatch):
    edited_window(window, tmp_path)
    save_dialog(monkeypatch, str(tmp_path / "scene.t3d"))
    window.save_project_as_action.trigger()
    assert json.loads((tmp_path / "scene.t3d").read_text())["photo"]["relative_path"] == "photo.png"


def test_save_with_photo_embeds_it_and_later_saves_keep_embedding(window, tmp_path, monkeypatch):
    edited_window(window, tmp_path)
    save_dialog(monkeypatch, str(tmp_path / "scene.t3d"))
    window.save_project_with_photo_action.trigger()
    assert "embedded_png" in json.loads((tmp_path / "scene.t3d").read_text())["photo"]
    window.vm.set("camera", "fov", 22.0)
    window.save_project_action.trigger()
    assert "embedded_png" in json.loads((tmp_path / "scene.t3d").read_text())["photo"]
    save_dialog(monkeypatch, str(tmp_path / "linked.t3d"))
    window.save_project_as_action.trigger()
    assert "relative_path" in json.loads((tmp_path / "linked.t3d").read_text())["photo"]


def test_failed_save_shows_an_error_and_stays_dirty(window, tmp_path, monkeypatch):
    edited_window(window, tmp_path)
    shown = errors(monkeypatch, "critical")
    save_dialog(monkeypatch, str(tmp_path / "missing-folder" / "scene.t3d"))
    assert window.save_project() is False
    assert len(shown) == 1
    assert window.project_path is None and window.vm.dirty


def test_saved_project_is_added_to_recent(window, tmp_path, monkeypatch, preferences):
    save_dialog(monkeypatch, str(tmp_path / "scene.t3d"))
    window.save_project()
    assert preferences.recent_files()[0] == str(tmp_path / "scene.t3d")


# ----- Title ------------------------------------------------------------------------------------


def test_title_follows_project_and_modified_marker(window, tmp_path, monkeypatch):
    assert window.windowTitle() == "Text3D Studio[*]" and not window.isWindowModified()
    window.load_image_path(image_file(tmp_path))
    assert window.windowTitle() == "Text3D Studio — photo.png[*]"
    assert window.isWindowModified()
    save_dialog(monkeypatch, str(tmp_path / "scene.t3d"))
    window.save_project()
    assert window.windowTitle() == "Text3D Studio — scene.t3d[*]"
    assert not window.isWindowModified()
    window.vm.set("light", "azimuth", 5.0)
    assert window.isWindowModified()


# ----- Opening ----------------------------------------------------------------------------------


def saved_project(tmp_path, monkeypatch, preferences, embed=False):
    """Saves an edited scene from a throwaway window and returns its path."""
    source = edited_window(MainWindow(SceneViewModel(Document()), preferences), tmp_path)
    save_dialog(monkeypatch, str(tmp_path / "scene.t3d"))
    source.save_project_as(embed_photo=embed)
    source.vm.mark_saved()
    return str(tmp_path / "scene.t3d")


def test_open_project_restores_scene_photo_and_controls(window, tmp_path, monkeypatch, preferences):
    path = saved_project(tmp_path, monkeypatch, preferences)
    calls = open_dialog(monkeypatch, path)
    window.open_project_action.trigger()
    assert len(calls) == 1
    vm = window.vm
    assert [o.text.content for o in vm.scene.objects] == ["HELLO", "SECOND"]
    assert vm.selected == 0
    assert vm.document.background_path == str(tmp_path / "photo.png")
    assert vm.document.native_size == (60, 40)
    assert window.panel.object_list.count() == 2
    assert slider_value(window, "light.azimuth") == 123.0
    assert window.project_path == path
    assert window.windowTitle() == "Text3D Studio — scene.t3d[*]" and not window.isWindowModified()
    assert not window.canvas.empty_state.isVisibleTo(window.canvas)
    assert "Opened scene.t3d" in window.statusBar().currentMessage()


def test_cancelled_open_dialog_changes_nothing(window, monkeypatch):
    open_dialog(monkeypatch, "")
    window.open_project_action.trigger()
    assert window.project_path is None


def test_opening_an_embedded_project_keeps_embedding_on_save(window, tmp_path, monkeypatch, preferences):
    path = saved_project(tmp_path, monkeypatch, preferences, embed=True)
    os.remove(tmp_path / "photo.png")
    window.open_project_path(path)
    assert window.vm.document.background is not None and window.project_embeds_photo
    window.vm.set("camera", "fov", 30.0)
    window.save_project()
    assert load_project(path).photo_embedded


def test_bad_project_shows_a_warning_and_keeps_the_document(window, tmp_path, monkeypatch):
    edited_window(window, tmp_path)
    broken = tmp_path / "broken.t3d"
    broken.write_text("{ not json")
    shown = errors(monkeypatch)
    scene = window.vm.scene
    window.open_project_path(str(broken))
    assert len(shown) == 1 and "not a valid project" in shown[0][2]
    assert window.vm.scene is scene and window.project_path is None


def test_missing_photo_warns_in_status_bar_and_overlay(window, tmp_path, monkeypatch, preferences):
    path = saved_project(tmp_path, monkeypatch, preferences)
    os.remove(tmp_path / "photo.png")
    window.open_project_path(path)
    assert window.vm.document.background is None
    assert "Photo not found" in window.statusBar().currentMessage()
    overlay = window.canvas.empty_state
    assert overlay.isVisibleTo(window.canvas)
    assert overlay.title.text() == "Photo not found"
    assert "photo.png" in overlay.message.text() and "Open an image" in overlay.message.text()
    assert [o.text.content for o in window.vm.scene.objects] == ["HELLO", "SECOND"]


def test_overlay_returns_to_the_usual_prompt_for_a_project_without_photo(window, tmp_path, monkeypatch, preferences):
    path = saved_project(tmp_path, monkeypatch, preferences)
    os.remove(tmp_path / "photo.png")
    window.open_project_path(path)
    save_dialog(monkeypatch, str(tmp_path / "bare.t3d"))
    window.vm.load_document(Document())
    window.save_project_as()
    window.open_project_path(str(tmp_path / "bare.t3d"))
    assert window.canvas.empty_state.title.text() == "No image loaded"


# ----- Unsaved-changes prompt -------------------------------------------------------------------

SAVE, DISCARD, CANCEL = (QMessageBox.StandardButton.Save, QMessageBox.StandardButton.Discard,
                         QMessageBox.StandardButton.Cancel)


def saved_window(window, tmp_path, monkeypatch):
    """An edited window saved to scene.t3d, then edited again (dirty)."""
    edited_window(window, tmp_path)
    save_dialog(monkeypatch, str(tmp_path / "scene.t3d"))
    window.save_project()
    window.vm.set("camera", "fov", 21.0)
    return str(tmp_path / "scene.t3d")


def test_clean_window_closes_without_asking(window, prompts):
    assert window.close() is True
    assert prompts[0] == []


def test_close_cancel_keeps_the_window_open(window, tmp_path, prompts):
    edited_window(window, tmp_path)
    prompts[1][0] = CANCEL
    assert window.close() is False
    assert len(prompts[0]) == 1 and "Unsaved changes" in prompts[0][0]
    assert window.vm.dirty


def test_close_discard_closes_without_saving(window, tmp_path, monkeypatch, prompts):
    path = saved_window(window, tmp_path, monkeypatch)
    assert window.close() is True
    assert load_project(path).document.scene.camera.fov != 21.0


def test_close_save_writes_the_project_first(window, tmp_path, monkeypatch, prompts):
    path = saved_window(window, tmp_path, monkeypatch)
    prompts[1][0] = SAVE
    assert window.close() is True
    assert load_project(path).document.scene.camera.fov == 21.0
    assert "scene.t3d" in prompts[0][0][2]


def test_close_save_cancelled_in_the_dialog_keeps_the_window(window, tmp_path, monkeypatch, prompts):
    edited_window(window, tmp_path)
    prompts[1][0] = SAVE
    save_dialog(monkeypatch, "")
    assert window.close() is False
    assert window.vm.dirty


def test_opening_a_project_over_changes_can_be_cancelled(window, tmp_path, monkeypatch, prompts):
    path = saved_window(window, tmp_path, monkeypatch)
    scene = window.vm.scene
    prompts[1][0] = CANCEL
    window.open_project_path(path)
    assert window.vm.scene is scene and window.vm.dirty


def test_opening_a_project_over_changes_discards_them(window, tmp_path, monkeypatch, prompts):
    path = saved_window(window, tmp_path, monkeypatch)
    window.open_project_path(path)
    assert len(prompts[0]) == 1
    assert window.vm.scene.camera.fov != 21.0 and not window.vm.dirty


def test_opening_a_project_over_changes_can_save_them_first(window, tmp_path, monkeypatch, prompts):
    path = saved_window(window, tmp_path, monkeypatch)
    prompts[1][0] = SAVE
    window.open_project_path(path)
    assert window.vm.scene.camera.fov == 21.0  # saved, then reopened
    assert not window.vm.dirty


def test_opening_a_project_into_a_clean_window_does_not_ask(window, tmp_path, monkeypatch, preferences, prompts):
    path = saved_project(tmp_path, monkeypatch, preferences)
    window.open_project_path(path)
    assert prompts[0] == []


def test_opening_an_image_over_changes_can_be_cancelled(window, tmp_path, monkeypatch, prompts):
    saved_window(window, tmp_path, monkeypatch)
    prompts[1][0] = CANCEL
    window.load_image_path(image_file(tmp_path, "other.png"))
    assert window.vm.document.background_path == str(tmp_path / "photo.png")


def test_opening_an_image_over_changes_can_discard_or_save(window, tmp_path, monkeypatch, prompts):
    path = saved_window(window, tmp_path, monkeypatch)
    window.load_image_path(image_file(tmp_path, "other.png"))
    assert window.vm.document.background_path == str(tmp_path / "other.png")
    assert load_project(path).document.scene.camera.fov != 21.0
    window.vm.set("camera", "fov", 33.0)
    prompts[1][0] = SAVE
    window.load_image_path(image_file(tmp_path, "third.png"))
    assert load_project(path).document.scene.camera.fov == 33.0
    assert window.vm.document.background_path == str(tmp_path / "third.png")


def test_opening_an_image_into_a_clean_window_does_not_ask(window, tmp_path, prompts):
    window.load_image_path(image_file(tmp_path))
    assert prompts[0] == []


# ----- Drag and drop, Open Recent ---------------------------------------------------------------


def mime_with(path):
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(path)])
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


def test_canvas_accepts_a_dragged_project(window, tmp_path, monkeypatch, preferences):
    path = saved_project(tmp_path, monkeypatch, preferences)
    assert dropped_image_path(mime_with(path)) == path
    assert drag_enter(window.canvas, mime_with(path))


def test_project_extension_check_ignores_case(window, tmp_path):
    path = tmp_path / "SCENE.T3D"
    path.write_text("{}")
    assert drag_enter(window.canvas, mime_with(str(path)))


def test_canvas_rejects_other_files_and_missing_projects(window, tmp_path):
    other = tmp_path / "notes.json"
    other.write_text("{}")
    assert not drag_enter(window.canvas, mime_with(str(other)))
    assert not drag_enter(window.canvas, mime_with(str(tmp_path / "gone.t3d")))


def test_dropping_a_project_opens_it(window, tmp_path, monkeypatch, preferences):
    path = saved_project(tmp_path, monkeypatch, preferences)
    assert drop(window.canvas, mime_with(path))
    assert window.project_path == path
    assert [o.text.content for o in window.vm.scene.objects] == ["HELLO", "SECOND"]
    assert preferences.recent_files()[0] == path


def test_dropping_a_project_over_changes_asks_first(window, tmp_path, monkeypatch, preferences, prompts):
    path = saved_project(tmp_path, monkeypatch, preferences)
    window.vm.set("camera", "fov", 21.0)
    prompts[1][0] = QMessageBox.StandardButton.Cancel
    drop(window.canvas, mime_with(path))
    assert len(prompts[0]) == 1 and window.project_path is None


def test_dropping_a_broken_project_warns(window, tmp_path, monkeypatch):
    path = tmp_path / "broken.t3d"
    path.write_text("nope")
    shown = errors(monkeypatch)
    drop(window.canvas, mime_with(str(path)))
    assert len(shown) == 1 and window.project_path is None


def recent_entries(window):
    return [a for a in window.recent_menu.actions() if a.data()]


def test_recent_menu_holds_images_and_projects(window, tmp_path, monkeypatch):
    photo = image_file(tmp_path)
    window.load_image_path(photo)
    save_dialog(monkeypatch, str(tmp_path / "scene.t3d"))
    window.save_project()
    assert [a.data() for a in recent_entries(window)] == [str(tmp_path / "scene.t3d"), photo]


def test_choosing_a_recent_project_opens_it(window, tmp_path, monkeypatch, preferences):
    path = saved_project(tmp_path, monkeypatch, preferences)
    window._rebuild_recent_menu()
    entry = next(a for a in recent_entries(window) if a.data() == path)
    entry.trigger()
    assert window.project_path == path
    assert window.windowTitle() == "Text3D Studio — scene.t3d[*]"


def test_choosing_a_recent_image_still_loads_it(window, tmp_path, preferences):
    photo = image_file(tmp_path)
    preferences.add_recent_file(photo)
    window._rebuild_recent_menu()
    recent_entries(window)[0].trigger()
    assert window.vm.document.background_path == photo and window.project_path is None


def test_missing_recent_project_is_removed_with_message(window, tmp_path, monkeypatch, preferences):
    missing = str(tmp_path / "gone.t3d")
    preferences.add_recent_file(missing)
    window._rebuild_recent_menu()
    shown = errors(monkeypatch, "information")
    recent_entries(window)[0].trigger()
    assert len(shown) == 1 and missing in shown[0][2]
    assert preferences.recent_files() == []


def test_opening_a_project_disables_undo(window, tmp_path, monkeypatch, preferences):
    path = saved_project(tmp_path, monkeypatch, preferences)
    window.vm.set("camera", "fov", 21.0)
    assert window.undo_action.isEnabled()
    window.open_project_path(path)
    assert not window.undo_action.isEnabled() and not window.redo_action.isEnabled()
    window.undo_action.trigger()
    assert window.vm.scene.camera.fov != 21.0
