"""ViewModel: the single mutation point for the Document, exposing Qt signals to views."""

import dataclasses

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from .model import Document

# Fields that change the rasterized glyph mask (expensive, CPU) rather than just uniforms (cheap, GPU).
RASTER_FIELDS = {"content", "font_family", "bold", "italic", "bevel_width"}


class SceneViewModel(QObject):
    scene_changed = pyqtSignal(str, str)  # (section, field)
    text_changed = pyqtSignal()
    background_changed = pyqtSignal()

    def __init__(self, document: Document | None = None):
        super().__init__()
        self.document = document or Document()

    @property
    def scene(self):
        return self.document.scene

    def get(self, section: str, name: str):
        return getattr(self._section(section, name), name)

    def set(self, section: str, name: str, value) -> None:
        target = self._section(section, name)
        if getattr(target, name) == value:
            return
        setattr(target, name, value)
        if section == "text" and name in RASTER_FIELDS:
            self.text_changed.emit()
        self.scene_changed.emit(section, name)

    def nudge(self, section: str, name: str, delta: float) -> None:
        self.set(section, name, self.get(section, name) + delta)

    def set_background(self, pixels: np.ndarray, path: str | None = None) -> None:
        self.document.background = pixels
        self.document.background_path = path
        self.background_changed.emit()

    def _section(self, section: str, name: str):
        target = getattr(self.document.scene, section, None)
        if target is None or not dataclasses.is_dataclass(target):
            raise KeyError(f"Unknown scene section: {section!r}")
        if name not in {f.name for f in dataclasses.fields(target)}:
            raise KeyError(f"Unknown field {name!r} in section {section!r}")
        return target
