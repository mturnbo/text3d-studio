import os

import pytest
from PyQt6.QtCore import QSettings

from text3d_studio.preferences import MAX_RECENT_FILES, Preferences


def open_preferences(tmp_path):
    return Preferences(QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))


@pytest.fixture
def prefs(tmp_path):
    return open_preferences(tmp_path)


def test_defaults_are_empty(prefs):
    assert prefs.recent_files() == []
    assert prefs.last_directory() == ""


def test_newest_file_comes_first(prefs):
    prefs.add_recent_file("/photos/a.png")
    prefs.add_recent_file("/photos/b.png")
    assert prefs.recent_files() == ["/photos/b.png", "/photos/a.png"]


def test_reopening_a_file_moves_it_to_front_without_duplicating(prefs):
    for name in ("a", "b", "c", "a"):
        prefs.add_recent_file(f"/photos/{name}.png")
    assert prefs.recent_files() == ["/photos/a.png", "/photos/c.png", "/photos/b.png"]


def test_relative_paths_are_stored_absolute_and_deduplicated(prefs, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    prefs.add_recent_file("photo.png")
    prefs.add_recent_file(str(tmp_path / "photo.png"))
    assert prefs.recent_files() == [str(tmp_path / "photo.png")]


def test_list_is_capped_and_drops_the_oldest(prefs):
    for i in range(MAX_RECENT_FILES + 2):
        prefs.add_recent_file(f"/photos/{i}.png")
    recent = prefs.recent_files()
    assert len(recent) == MAX_RECENT_FILES
    assert recent[0] == f"/photos/{MAX_RECENT_FILES + 1}.png"
    assert "/photos/0.png" not in recent and "/photos/1.png" not in recent


def test_remove_and_clear(prefs):
    prefs.add_recent_file("/photos/a.png")
    prefs.add_recent_file("/photos/b.png")
    prefs.remove_recent_file("/photos/a.png")
    prefs.remove_recent_file("/photos/never-added.png")
    assert prefs.recent_files() == ["/photos/b.png"]
    prefs.clear_recent_files()
    assert prefs.recent_files() == []


@pytest.mark.parametrize("paths", [["/photos/only.png"], ["/my photos/été & co.png", "/photos/b.jpg"]])
def test_recent_files_persist_across_instances(tmp_path, paths):
    first = open_preferences(tmp_path)
    for path in reversed(paths):
        first.add_recent_file(path)
    assert open_preferences(tmp_path).recent_files() == paths


def test_last_directory_persists(tmp_path):
    folder = tmp_path / "shoot"
    folder.mkdir()
    open_preferences(tmp_path).set_last_directory(str(folder))
    assert open_preferences(tmp_path).last_directory() == str(folder)


def test_deleted_last_directory_is_forgotten(prefs, tmp_path):
    folder = tmp_path / "gone"
    folder.mkdir()
    prefs.set_last_directory(str(folder))
    os.rmdir(folder)
    assert prefs.last_directory() == ""
