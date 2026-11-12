"""ViewModel: the single mutation point for the Document, exposing Qt signals to views."""

import copy
import dataclasses
import time

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from . import ground as ground_model
from .history import History
from .materials import KINDS, PRESETS
from .model import Document, Scene, TextObject, TransformParams

# Fields that change the rasterized glyph mask (expensive, CPU) rather than just uniforms (cheap, GPU).
RASTER_FIELDS = {"content", "font_family", "bold", "italic", "bevel_width", "align", "letter_spacing",
                 "line_height", "arc"}
OBJECT_SECTIONS = ("text", "transform", "material")  # per-object sections; the rest are shared by the scene
SHARED_SECTIONS = tuple(f.name for f in dataclasses.fields(Scene) if f.name != "objects")


class SceneViewModel(QObject):
    scene_changed = pyqtSignal(str, str)  # (section, field)
    text_changed = pyqtSignal()
    background_changed = pyqtSignal()
    objects_changed = pyqtSignal()  # an object was added or deleted
    selection_changed = pyqtSignal(int)  # the selected object's index; its text, transform and material changed
    dirty_changed = pyqtSignal(bool)  # the document gained or lost unsaved changes
    document_loaded = pyqtSignal()  # load_document replaced the whole document
    history_changed = pyqtSignal()  # can_undo or can_redo may have changed

    def __init__(self, document: Document | None = None, clock=time.monotonic):
        super().__init__()
        self.document = document or Document()
        self.selected = 0  # editor state: the object that "text", "transform" and "material" edits address
        # Undo states are (deep copy of the scene, selected index). Every scene-changing entry point records one.
        self.history = History(clock=clock)
        self._edit_depth = 0
        self._edit_start = None  # the state when the outermost edit began
        # Ids are never reused, so nothing keyed by a deleted object's id (a cached raster) can match a new one.
        self._next_object_id = max(o.id for o in self.scene.objects) + 1
        self.dirty = False  # unsaved changes since the last save or load
        self.scene_changed.connect(self._mark_dirty)
        self.objects_changed.connect(self._mark_dirty)
        self.background_changed.connect(self._mark_dirty)

    @property
    def scene(self):
        return self.document.scene

    @property
    def selected_object(self) -> TextObject:
        return self.scene.objects[self.selected]

    def get(self, section: str, name: str):
        return getattr(self._section(section, name), name)

    def set(self, section: str, name: str, value) -> None:
        target = self._section(section, name)
        if getattr(target, name) == value:
            return
        self.begin_edit()
        base = self.placement_snapshot() if section == "ground" else None
        self._assign(target, section, name, value)
        if base is not None:
            self._place_objects_from(*base)
        self.end_edit(merge_key=self._merge_key(section, name))

    def set_ground_pose(self, anchor=None, **values) -> None:
        """Sets several ground fields together and re-places the text once, against the final pose.

        Setting them one by one could pass through a pose whose horizon sits above the text and lose it.
        `anchor` is an optional placement_snapshot() to re-place the objects from instead of the current state:
        drags pass the state from when they started, so a pose mid-drag that can't hold the text doesn't
        strand it for the rest of the drag.
        """
        self.begin_edit()
        old_ground = dataclasses.replace(self.scene.ground)
        base = anchor if anchor is not None else self.placement_snapshot()
        for name, value in values.items():
            target = self._section("ground", name)
            if getattr(target, name) != value:
                self._assign(target, "ground", name, value)
        if self.scene.ground != old_ground:
            self._place_objects_from(*base)
        self.end_edit()

    def placement_snapshot(self):
        """(ground, [each object's transform]): copies of everything a ground edit re-places the objects from."""
        return (dataclasses.replace(self.scene.ground), [dataclasses.replace(o.transform) for o in self.scene.objects])

    def _assign(self, target, section: str, name: str, value) -> None:
        setattr(target, name, value)
        if section == "text" and name in RASTER_FIELDS:
            self.text_changed.emit()
        self.scene_changed.emit(section, name)

    def _place_objects_from(self, base_ground, base_transforms) -> None:
        """Aligning the ground to the photo shouldn't fling the text around: keep every object where it appeared
        on `base_ground`. When that's impossible on the new ground, fall back to the object's base placement."""
        for obj, base in zip(self.scene.objects, base_transforms, strict=True):
            placement = ground_model.reanchor_text(base_ground, self.scene.ground, base)
            if placement is None:
                placement = (base.x, base.z, base.rot_y, base.size)
            for name, value in zip(("x", "z", "rot_y", "size"), placement):
                if getattr(obj.transform, name) != value:
                    self._assign(obj.transform, "transform", name, value)

    def apply_material(self, kind: str) -> None:
        """Switches the material kind and loads its preset values (replacing any adjustments)."""
        if kind not in KINDS:
            raise ValueError(f"Unknown material: {kind!r}")
        self.begin_edit()
        self.set("material", "kind", kind)
        for name, value in PRESETS[kind].items():
            self.set("material", name, value)
        self.end_edit()

    def select(self, index: int) -> None:
        if not 0 <= index < len(self.scene.objects):
            raise IndexError(f"No text object at index {index}")
        if index != self.selected:
            self.selected = index
            self.selection_changed.emit(index)

    def add_object(self) -> None:
        """Adds a default text object in front of the selected one and selects it."""
        base = ground_model.placement_in_front(self.scene.ground, self.selected_object.transform)
        transform = TransformParams(x=base.x, z=base.z, rot_y=base.rot_y, size=base.size)
        self.begin_edit()
        self._insert(TextObject(id=self._next_id(), transform=transform))
        self.end_edit()

    def duplicate_object(self) -> None:
        """Adds a copy of the selected object in front of it and selects the copy."""
        source = self.selected_object
        self.begin_edit()
        self._insert(TextObject(
            id=self._next_id(),
            text=dataclasses.replace(source.text),
            transform=ground_model.placement_in_front(self.scene.ground, source.transform),
            material=dataclasses.replace(source.material),
        ))
        self.end_edit()

    def delete_object(self) -> bool:
        """Deletes the selected object. Refuses (returns False) when it is the last one."""
        if len(self.scene.objects) == 1:
            return False
        self.begin_edit()
        del self.scene.objects[self.selected]
        self.selected = min(self.selected, len(self.scene.objects) - 1)
        self.objects_changed.emit()
        self.selection_changed.emit(self.selected)
        self.end_edit()
        return True

    def _insert(self, obj: TextObject) -> None:
        self.scene.objects.append(obj)
        self.objects_changed.emit()
        self.select(len(self.scene.objects) - 1)

    def _next_id(self) -> int:
        self._next_object_id += 1
        return self._next_object_id - 1

    def nudge(self, section: str, name: str, delta: float) -> None:
        self.set(section, name, self.get(section, name) + delta)

    def set_background(self, pixels: np.ndarray, path: str | None = None) -> None:
        """Not undoable: a new photo starts a fresh history, since undoing across photos would confuse."""
        self.document.background = pixels
        self.document.background_path = path
        self.background_changed.emit()
        self.clear_history()

    # ----- Undo -------------------------------------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        return self.history.can_undo

    @property
    def can_redo(self) -> bool:
        return self.history.can_redo

    def begin_edit(self) -> None:
        """Starts grouping changes into one undo step (a drag, say). Edits nest; the outermost end records."""
        if self._edit_depth == 0:
            self._edit_start = self._state()
        self._edit_depth += 1

    def end_edit(self, merge_key=None) -> None:
        """Ends an edit. The outermost one records a step if the scene changed since its begin_edit().

        `merge_key` lets consecutive quick changes to the same field merge into one step (see History).
        """
        self._edit_depth -= 1
        if self._edit_depth > 0:
            return
        start, self._edit_start = self._edit_start, None
        if start[0] != self.scene:
            self.history.record(start, merge_key)
            self.history_changed.emit()

    def undo(self) -> bool:
        """Restores the state before the last step. False when there is nothing to undo or an edit is under way."""
        if self._edit_depth > 0:
            return False
        state = self.history.undo(self._state())
        return self._restore(state)

    def redo(self) -> bool:
        if self._edit_depth > 0:
            return False
        state = self.history.redo(self._state())
        return self._restore(state)

    def clear_history(self) -> None:
        self.history.clear()
        self.history_changed.emit()

    def _merge_key(self, section: str, name: str):
        if section in OBJECT_SECTIONS:
            return (section, name, self.selected_object.id)
        return (section, name)

    def _state(self):
        return (copy.deepcopy(self.scene), self.selected)

    def _restore(self, state) -> bool:
        """Puts a recorded state back in place (the Scene object stays the same) and refreshes every view."""
        if state is None:
            return False
        scene, selected = state
        old = copy.copy(self.scene)
        for field in dataclasses.fields(Scene):
            setattr(self.scene, field.name, getattr(scene, field.name))
        self.selected = selected
        self.objects_changed.emit()  # any object may have come back, gone or changed
        self.selection_changed.emit(self.selected)  # refreshes every per-object control
        for section in SHARED_SECTIONS:
            before, after = getattr(old, section), getattr(self.scene, section)
            for field in dataclasses.fields(after):
                if getattr(before, field.name) != getattr(after, field.name):
                    self.scene_changed.emit(section, field.name)
        if [o.text for o in old.objects] != [o.text for o in self.scene.objects]:
            self.text_changed.emit()
        self.history_changed.emit()
        return True

    def load_document(self, document: Document) -> None:
        """Replaces the document wholesale (opening a project), refreshes every view and leaves it clean.

        Selection resets to the first object and the undo history is cleared. Other listeners that keep
        per-document state reset it on `document_loaded`, emitted last.
        """
        self.document = document
        self.selected = 0
        self._next_object_id = max(o.id for o in self.scene.objects) + 1
        self.background_changed.emit()
        self.objects_changed.emit()  # the canvas re-rasterizes every object whose text differs from its cached raster
        self.selection_changed.emit(0)  # per-object controls show the first object
        for section in SHARED_SECTIONS:
            for f in dataclasses.fields(getattr(self.scene, section)):
                self.scene_changed.emit(section, f.name)
        self.clear_history()  # opening a project can't be undone
        self._set_dirty(False)
        self.document_loaded.emit()

    def mark_saved(self) -> None:
        self._set_dirty(False)

    def _mark_dirty(self, *_args) -> None:
        self._set_dirty(True)

    def _set_dirty(self, dirty: bool) -> None:
        if dirty != self.dirty:
            self.dirty = dirty
            self.dirty_changed.emit(dirty)

    def _section(self, section: str, name: str):
        owner = self.selected_object if section in OBJECT_SECTIONS else self.scene
        target = getattr(owner, section, None)
        if target is None or not dataclasses.is_dataclass(target):
            raise KeyError(f"Unknown scene section: {section!r}")
        if name not in {f.name for f in dataclasses.fields(target)}:
            raise KeyError(f"Unknown field {name!r} in section {section!r}")
        return target
