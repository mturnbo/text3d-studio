"""Preferences: persistent per-user state (last folder, recent files) backed by QSettings, and where looks live."""

import os

from PyQt6.QtCore import QSettings, QStandardPaths

MAX_RECENT_FILES = 8


class Preferences:
    def __init__(self, settings: QSettings, looks_directory: str | None = None):
        self.settings = settings
        self._looks_directory = looks_directory

    def looks_directory(self) -> str:
        """The folder of the user's saved looks: the one given, else "looks" in the app's data folder."""
        if self._looks_directory is not None:
            return self._looks_directory
        data = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
        return os.path.join(data, "looks")

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
