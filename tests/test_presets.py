import copy
import dataclasses
import json
import os

import pytest

from text3d_studio import presets
from text3d_studio.materials import KINDS
from text3d_studio.model import Document, LightParams, MaterialParams, ShadowParams
from text3d_studio.presets import (
    BUILTIN_LOOKS,
    apply_look,
    capture_look,
    delete_user_look,
    load_user_looks,
    save_user_look,
)
from text3d_studio.viewmodel import SceneViewModel


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def vm(qapp):
    return SceneViewModel(Document(), clock=FakeClock())


def builtin(name):
    return next(look for look in BUILTIN_LOOKS if look.name == name)


def test_builtin_looks_are_complete_and_distinct():
    names = [look.name for look in BUILTIN_LOOKS]
    assert {"Gold at sunset", "Chrome at noon", "Frosted glass, overcast", "Marble", "Oak"} <= set(names)
    assert len(set(names)) == len(names)
    for look in BUILTIN_LOOKS:
        assert look.builtin and look.path is None
        assert set(look.material) == {f.name for f in dataclasses.fields(MaterialParams)}
        assert set(look.light) == {f.name for f in dataclasses.fields(LightParams)}
        assert set(look.shadow) == {f.name for f in dataclasses.fields(ShadowParams)}
        assert look.material["kind"] in KINDS
    assert len({json.dumps([l.material, l.light, l.shadow]) for l in BUILTIN_LOOKS}) == len(BUILTIN_LOOKS)


def test_applying_a_look_sets_the_selected_material_light_and_shadow_only(vm):
    vm.add_object()
    vm.set("text", "content", "SECOND")
    vm.select(0)
    before = copy.deepcopy(vm.scene)
    look = builtin("Gold at sunset")
    apply_look(vm, look)
    scene = vm.scene
    assert dataclasses.asdict(scene.objects[0].material) == look.material
    assert dataclasses.asdict(scene.light) == look.light
    assert dataclasses.asdict(scene.shadow) == look.shadow
    assert scene.objects[1] == before.objects[1]  # the other object keeps its material
    assert scene.objects[0].text == before.objects[0].text
    assert scene.objects[0].transform == before.objects[0].transform
    for section in ("ground", "grid", "camera"):
        assert getattr(scene, section) == getattr(before, section)


def test_applying_a_look_is_one_undo_step(vm):
    before = copy.deepcopy(vm.scene)
    apply_look(vm, builtin("Frosted glass, overcast"))
    assert vm.scene.objects[0].material.kind == "glass"
    assert vm.undo()
    assert vm.scene == before
    assert not vm.can_undo
    assert vm.redo()
    assert vm.scene.objects[0].material.kind == "glass"


def test_applying_a_look_with_unknown_or_missing_fields_sets_only_known_ones(vm):
    look = presets.Look("Partial", material={"roughness": 0.9, "sparkle": 3}, light={"elevation": 5.0}, shadow={})
    before = copy.deepcopy(vm.scene)
    apply_look(vm, look)
    assert vm.scene.objects[0].material.roughness == 0.9
    assert vm.scene.objects[0].material.kind == before.objects[0].material.kind
    assert vm.scene.light.elevation == 5.0
    assert vm.scene.light.azimuth == before.light.azimuth
    assert vm.scene.shadow == before.shadow


def test_a_saved_look_loads_back_identically(vm, tmp_path):
    vm.apply_material("wood")
    vm.set("light", "color", (0.9, 0.5, 0.25))
    vm.set("shadow", "softness", 0.42)
    saved = save_user_look(str(tmp_path), capture_look(vm, "  Warm wood  "))
    assert saved.name == "Warm wood" and os.path.isfile(saved.path)
    looks, warnings = load_user_looks(str(tmp_path))
    assert warnings == []
    assert len(looks) == 1
    loaded = looks[0]
    assert (loaded.name, loaded.material, loaded.light, loaded.shadow) == (
        saved.name, saved.material, saved.light, saved.shadow
    )
    assert isinstance(loaded.light["color"], tuple)

    other = SceneViewModel(Document())
    apply_look(other, loaded)
    assert other.scene.objects[0].material == vm.scene.objects[0].material
    assert other.scene.light == vm.scene.light
    assert other.scene.shadow == vm.scene.shadow


def test_missing_folder_has_no_looks(tmp_path):
    assert load_user_looks(str(tmp_path / "nowhere")) == ([], [])


def test_saving_creates_the_folder(vm, tmp_path):
    folder = tmp_path / "a" / "looks"
    save_user_look(str(folder), capture_look(vm, "Mine"))
    assert [look.name for look in load_user_looks(str(folder))[0]] == ["Mine"]


@pytest.mark.parametrize("content, reason", [
    ("{not json", "not valid JSON"),
    (json.dumps({"format": "something-else", "version": 1}), "not a Text3D Studio look"),
    (json.dumps({"format": presets.FORMAT_NAME, "version": 99, "name": "x"}), "unsupported look version"),
    (json.dumps({"format": presets.FORMAT_NAME, "version": 1, "name": " "}), "no name"),
    (json.dumps({"format": presets.FORMAT_NAME, "version": 1, "name": "x", "material": {}, "light": {}}), "ShadowParams"),
    (json.dumps({"format": presets.FORMAT_NAME, "version": 1, "name": "x", "material": {"kind": "cheese"},
                 "light": {}, "shadow": {}}), "unknown material"),
    (json.dumps({"format": presets.FORMAT_NAME, "version": 1, "name": "x", "material": {"side_kind": "cheese"},
                 "light": {}, "shadow": {}}), "unknown side material"),
    (json.dumps({"format": presets.FORMAT_NAME, "version": 1, "name": "x", "material": {"roughness": "high"},
                 "light": {}, "shadow": {}}), "roughness is not a number"),
    (json.dumps({"format": presets.FORMAT_NAME, "version": 1, "name": "x", "material": {},
                 "light": {"color": [1, 0]}, "shadow": {}}), "color is not a colour"),
    (json.dumps({"format": presets.FORMAT_NAME, "version": 1, "name": "x", "material": {},
                 "light": {}, "shadow": {"enabled": 1}}), "enabled is not true or false"),
])
def test_bad_look_files_are_skipped_with_a_warning(vm, tmp_path, content, reason):
    save_user_look(str(tmp_path), capture_look(vm, "Good"))
    (tmp_path / "bad.json").write_text(content, encoding="utf-8")
    looks, warnings = load_user_looks(str(tmp_path))
    assert [look.name for look in looks] == ["Good"]
    assert len(warnings) == 1
    assert "bad.json" in warnings[0] and reason in warnings[0]


def test_unreadable_bytes_and_other_files_are_handled(vm, tmp_path):
    (tmp_path / "binary.json").write_bytes(b"\xff\xfe\x00garbage")
    (tmp_path / "notes.txt").write_text("not a look")
    looks, warnings = load_user_looks(str(tmp_path))
    assert looks == []
    assert len(warnings) == 1 and "binary.json" in warnings[0]


def test_unknown_fields_from_a_newer_version_are_ignored(vm, tmp_path):
    saved = save_user_look(str(tmp_path), capture_look(vm, "Future"))
    data = json.loads(open(saved.path).read())
    data["material"]["sparkle"] = 1.0
    data["extra_section"] = {}
    open(saved.path, "w").write(json.dumps(data))
    looks, warnings = load_user_looks(str(tmp_path))
    assert warnings == [] and "sparkle" not in looks[0].material


def test_a_name_already_used_needs_replace(vm, tmp_path):
    first = save_user_look(str(tmp_path), capture_look(vm, "Mine"))
    vm.set("light", "elevation", 77.0)
    with pytest.raises(FileExistsError):
        save_user_look(str(tmp_path), capture_look(vm, "MINE"))  # names compare case-insensitively
    assert load_user_looks(str(tmp_path))[0][0].light["elevation"] == first.light["elevation"]
    replaced = save_user_look(str(tmp_path), capture_look(vm, "MINE"), replace=True)
    assert replaced.path == first.path
    looks, _ = load_user_looks(str(tmp_path))
    assert [(l.name, l.light["elevation"]) for l in looks] == [("MINE", 77.0)]


@pytest.mark.parametrize("name", ["Marble", "chrome AT noon"])
def test_a_builtin_name_cannot_be_saved(vm, tmp_path, name):
    with pytest.raises(ValueError, match="built-in"):
        save_user_look(str(tmp_path), capture_look(vm, name))
    assert not os.listdir(tmp_path)


def test_an_empty_name_cannot_be_saved(vm, tmp_path):
    with pytest.raises(ValueError):
        save_user_look(str(tmp_path), capture_look(vm, "   "))


def test_names_that_are_not_safe_file_names_still_save_separately(vm, tmp_path):
    save_user_look(str(tmp_path), capture_look(vm, "a/b"))
    save_user_look(str(tmp_path), capture_look(vm, "a_b"))
    save_user_look(str(tmp_path), capture_look(vm, "../.."))
    looks, warnings = load_user_looks(str(tmp_path))
    assert warnings == []
    assert sorted(l.name for l in looks) == ["../..", "a/b", "a_b"]
    assert all(os.path.dirname(l.path) == str(tmp_path) for l in looks)


def test_a_file_claiming_a_builtin_or_duplicate_name_is_skipped(vm, tmp_path):
    save_user_look(str(tmp_path), capture_look(vm, "Mine"))
    for filename, name in (("x1.json", "Oak"), ("x2.json", "mine")):
        data = dataclasses.asdict(capture_look(vm, name))
        data.update(format=presets.FORMAT_NAME, version=1)
        (tmp_path / filename).write_text(json.dumps({k: v for k, v in data.items() if k not in ("builtin", "path")}))
    looks, warnings = load_user_looks(str(tmp_path))
    assert [l.name for l in looks] == ["Mine"]
    assert len(warnings) == 2


def test_deleting_a_user_look_removes_its_file(vm, tmp_path):
    saved = save_user_look(str(tmp_path), capture_look(vm, "Gone"))
    delete_user_look(saved)
    assert not os.path.exists(saved.path)
    assert load_user_looks(str(tmp_path)) == ([], [])


def test_builtin_looks_cannot_be_deleted():
    with pytest.raises(ValueError, match="built-in"):
        delete_user_look(BUILTIN_LOOKS[0])
