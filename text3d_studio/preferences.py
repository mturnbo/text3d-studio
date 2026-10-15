"""Preferences: persistent per-user state (last folder, recent files) backed by QSettings."""

import os

from PyQt6.QtCore import QSettings

MAX_RECENT_FILES = 8


class Preferences:
    def __init__(self, settings: QSettings):
        self.settings = settings

    def last_directory(self) -> str:
        """The folder of the last opened image, or "" if unset or since deleted."""
        directory = self.settings.value("last_directory", "", type=str)
        return directory if directory and os.path.isdir(directory) else ""

    def set_last_directory(self, directory: str) -> None:
        self.settings.setValue("last_directory", os.path.abspath(directory))
        self.settings.sync()

    def recent_files(self) -> list[str]:
        """Most recently opened first."""
        return list(self.settings.value("recent_files", [], type=list))

    def add_recent_file(self, path: str) -> None:
        path = os.path.abspath(path)
        others = [p for p in self.recent_files() if p != path]
        self._store_recent([path, *others][:MAX_RECENT_FILES])

    def remove_recent_file(self, path: str) -> None:
        self._store_recent([p for p in self.recent_files() if p != path])

    def clear_recent_files(self) -> None:
        self._store_recent([])

    def _store_recent(self, paths: list[str]) -> None:
        self.settings.setValue("recent_files", paths)
        self.settings.sync()
