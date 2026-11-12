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
