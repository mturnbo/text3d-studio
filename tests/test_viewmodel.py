import dataclasses

import numpy as np
import pytest

from text3d_studio.model import Document
from text3d_studio.viewmodel import SceneViewModel


@pytest.fixture
def vm(qapp):
    return SceneViewModel(Document())


def record(signal):
    events = []
    signal.connect(lambda *args: events.append(args))
    return events


def test_set_updates_model_and_emits_scene_changed(vm):
    events = record(vm.scene_changed)
    vm.set("light", "azimuth", 120.0)
    assert vm.scene.light.azimuth == 120.0
    assert events == [("light", "azimuth")]


def test_setting_same_value_emits_nothing(vm):
    events = record(vm.scene_changed)
    vm.set("transform", "x", vm.scene.objects[0].transform.x)
    assert events == []


def test_raster_fields_emit_text_changed_but_transform_does_not(vm):
    text_events = record(vm.text_changed)
    vm.set("text", "content", "World")
    vm.set("text", "bevel_width", 9.0)
    vm.set("transform", "size", 5.0)
    assert len(text_events) == 2


def test_nudge_adds_delta(vm):
    vm.set("transform", "rot_y", 10.0)
    vm.nudge("transform", "rot_y", -2.5)
    assert vm.scene.objects[0].transform.rot_y == 7.5


@pytest.mark.parametrize("section, field", [("nope", "x"), ("transform", "nope"), ("background", "x")])
def test_unknown_section_or_field_raises(vm, section, field):
    with pytest.raises(KeyError):
        vm.set(section, field, 1)


def test_set_background_updates_native_size_and_emits(vm):
    events = record(vm.background_changed)
    vm.set_background(np.zeros((300, 500, 4), np.uint8), "/tmp/bg.png")
    assert vm.document.native_size == (500, 300)
    assert vm.document.background_path == "/tmp/bg.png"
    assert len(events) == 1


def test_default_native_size_without_background():
    assert Document().native_size == (1600, 1000)


def test_default_scene_holds_one_default_text_object():
    from text3d_studio.model import MaterialParams, Scene, TextObject, TextParams, TransformParams

    scene = Scene()
    assert scene.objects == [TextObject(id=0, text=TextParams(), transform=TransformParams(), material=MaterialParams())]
    assert not any(hasattr(scene, name) for name in ("text", "transform", "material"))  # they live on objects


def test_scenes_and_objects_never_share_mutable_state():
    from text3d_studio.model import Scene, TextObject

    first, second = Scene(), Scene()
    first.objects.append(TextObject(id=1))
    first.objects[0].transform.x = 4.0
    assert len(second.objects) == 1
    assert second.objects[0].transform.x == 0.0
    a, b = TextObject(), TextObject()
    a.text.content = "A"
    assert b.text.content == "HELLO"


def test_grid_settings_are_editable_and_hidden_by_default(vm):
    events = record(vm.scene_changed)
    assert vm.scene.grid.visible is False
    vm.set("grid", "visible", True)
    vm.set("grid", "cell_size", 0.5)
    assert events == [("grid", "visible"), ("grid", "cell_size")]


def test_ground_edits_keep_the_text_on_the_same_camera_ray(vm):
    from text3d_studio import ground

    before = ground.text_position(vm.scene.ground, vm.scene.objects[0].transform)
    vm.set("ground", "pitch", 8.0)
    after = ground.text_position(vm.scene.ground, vm.scene.objects[0].transform)
    np.testing.assert_allclose(after / np.linalg.norm(after), before / np.linalg.norm(before), atol=1e-9)


def test_ground_edits_report_the_moved_text_fields(vm):
    events = record(vm.scene_changed)
    vm.set("ground", "yaw", 15.0)
    assert ("ground", "yaw") in events
    assert {("transform", "x"), ("transform", "z"), ("transform", "rot_y")} <= set(events)


def test_text_stays_put_when_the_ground_tips_past_it(vm):
    vm.set("transform", "z", -60.0)
    before = (vm.scene.objects[0].transform.x, vm.scene.objects[0].transform.z)
    vm.set("ground", "pitch", -10.0)
    assert (vm.scene.objects[0].transform.x, vm.scene.objects[0].transform.z) == before


def test_set_ground_pose_reanchors_once_against_the_final_pose(vm):
    from text3d_studio import ground
    from text3d_studio.model import GroundParams

    old = dataclasses.replace(vm.scene.ground)
    expected = ground.reanchor_text(old, GroundParams(pitch=9.0, yaw=25.0, roll=-3.0, height=2.2), vm.scene.objects[0].transform)
    vm.set_ground_pose(pitch=9.0, yaw=25.0, roll=-3.0, height=2.2)
    g, t = vm.scene.ground, vm.scene.objects[0].transform
    assert (g.pitch, g.yaw, g.roll, g.height) == (9.0, 25.0, -3.0, 2.2)
    np.testing.assert_allclose((t.x, t.z, t.rot_y, t.size), expected, atol=1e-12)


def test_set_ground_pose_keeps_text_that_one_by_one_edits_would_lose(qapp):
    from text3d_studio import ground

    def start():
        model = SceneViewModel(Document())
        model.set("ground", "pitch", -1.0)  # horizon just below center
        model.set("transform", "z", -150.0)  # far away, just below that horizon
        return model

    # Rolling first (pitch still -1) drops the horizon at the center column below the anchor, so the text
    # can't be re-placed on that intermediate pose. The final pose (pitch 10, roll 55) reaches it again.
    one_step, one_by_one = start(), start()
    old = dataclasses.replace(one_step.scene.ground)
    assert ground.reanchor_text(old, dataclasses.replace(old, roll=55.0), one_step.scene.objects[0].transform) is None
    one_step.set_ground_pose(roll=55.0, pitch=10.0)
    one_by_one.set("ground", "roll", 55.0)
    one_by_one.set("ground", "pitch", 10.0)

    expected = ground.reanchor_text(old, one_step.scene.ground, start().scene.objects[0].transform)
    t = one_step.scene.objects[0].transform
    np.testing.assert_allclose((t.x, t.z, t.rot_y, t.size), expected, atol=1e-12)
    u = one_by_one.scene.objects[0].transform
    assert not np.allclose((u.x, u.z, u.size), (t.x, t.z, t.size))


@pytest.mark.parametrize("kind", ["plastic", "stone", "wood", "glass", "metal"])
def test_apply_material_loads_the_preset(vm, kind):
    from text3d_studio.materials import PRESETS

    vm.apply_material(kind)
    material = vm.scene.objects[0].material
    assert material.kind == kind
    assert {name: getattr(material, name) for name in PRESETS[kind]} == PRESETS[kind]


def test_apply_material_replaces_adjustments_and_reports_each_field(vm):
    vm.set("material", "roughness", 0.9)
    events = record(vm.scene_changed)
    vm.apply_material("plastic")
    assert vm.scene.objects[0].material.roughness == 0.3
    assert ("material", "kind") in events and ("material", "roughness") in events


def test_reapplying_the_same_material_resets_it(vm):
    vm.apply_material("wood")
    vm.set("material", "face_color", (0.1, 0.2, 0.3))
    vm.apply_material("wood")
    from text3d_studio.materials import PRESETS

    assert vm.scene.objects[0].material.face_color == PRESETS["wood"]["face_color"]


def test_unknown_material_is_rejected(vm):
    with pytest.raises(ValueError):
        vm.apply_material("cheese")
    assert vm.scene.objects[0].material.kind == "metal"


# ----- Multiple text objects ----------------------------------------------------------------------


def test_text_transform_and_material_edits_address_only_the_selected_object(vm):
    vm.add_object()
    assert vm.selected == 1
    vm.set("text", "content", "SECOND")
    vm.set("transform", "rot_y", 33.0)
    vm.apply_material("wood")
    first, second = vm.scene.objects
    assert (second.text.content, second.transform.rot_y, second.material.kind) == ("SECOND", 33.0, "wood")
    assert (first.text.content, first.transform.rot_y, first.material.kind) == ("HELLO", -20.0, "metal")
    vm.select(0)
    assert vm.get("text", "content") == "HELLO"
    vm.nudge("transform", "rot_y", 5.0)
    assert (first.transform.rot_y, second.transform.rot_y) == (-15.0, 33.0)


def test_shared_sections_are_the_same_whichever_object_is_selected(vm):
    vm.add_object()
    vm.set("light", "azimuth", 77.0)
    vm.select(0)
    assert vm.get("light", "azimuth") == 77.0


def test_add_object_appends_a_default_object_in_front_of_the_selected_one_and_selects_it(vm):
    from text3d_studio import ground
    from text3d_studio.model import MaterialParams, TextParams

    objects, selections = record(vm.objects_changed), record(vm.selection_changed)
    vm.set("text", "content", "FIRST")
    vm.apply_material("glass")
    vm.add_object()
    first, new = vm.scene.objects
    assert (new.text, new.material) == (TextParams(), MaterialParams())  # a default object, not a copy
    assert new.id != first.id
    assert (len(objects), selections) == (1, [(1,)])
    assert new.transform == ground.placement_in_front(vm.scene.ground, first.transform)


def test_duplicate_copies_the_selected_object_with_a_new_id(vm):
    from text3d_studio import ground

    vm.set("text", "content", "COPY ME")
    vm.set("transform", "depth", 0.9)
    vm.apply_material("stone")
    vm.set("material", "roughness", 0.123)
    vm.duplicate_object()
    source, copy = vm.scene.objects
    assert vm.selected == 1
    assert copy.id != source.id
    assert copy.text == source.text and copy.material == source.material
    assert copy.transform == ground.placement_in_front(vm.scene.ground, source.transform)
    assert copy.transform.depth == 0.9
    vm.set("text", "content", "CHANGED")  # a copy, not shared sections
    vm.set("material", "roughness", 0.9)
    assert (source.text.content, source.material.roughness) == ("COPY ME", 0.123)


def test_a_new_object_shows_below_the_one_it_was_placed_in_front_of(vm):
    """The camera's ray to the original's base passes above the new object's top, so they can't overlap."""
    from text3d_studio import ground

    for pitch, size in ((0.0, 3.0), (20.0, 8.0), (-5.0, 0.5)):
        vm.set("ground", "pitch", pitch)
        vm.set("transform", "size", size)
        old = vm.selected_object.transform
        new = ground.placement_in_front(vm.scene.ground, old)
        up = ground.ground_up(vm.scene.ground)
        old_base = ground.text_position(vm.scene.ground, old)
        new_top = ground.text_position(vm.scene.ground, new) + up * new.size * ground.INK_HEIGHT_EM
        # The new top sits below the plane through the camera, the original's base and the ground's x axis.
        normal = np.cross(old_base, ground.ground_rotation(vm.scene.ground)[:3, 0])
        normal *= np.sign(normal @ up)
        assert new_top @ normal < 0
        assert new.size < old.size and np.linalg.norm(ground.text_position(vm.scene.ground, new)) < np.linalg.norm(old_base)


def test_delete_refuses_to_remove_the_last_object(vm):
    objects, selections = record(vm.objects_changed), record(vm.selection_changed)
    assert vm.delete_object() is False
    assert len(vm.scene.objects) == 1
    assert (objects, selections) == ([], [])


def test_delete_removes_the_selected_object_and_clamps_the_selection(vm):
    vm.add_object()
    vm.add_object()
    ids = [o.id for o in vm.scene.objects]
    objects, selections = record(vm.objects_changed), record(vm.selection_changed)
    assert vm.selected == 2
    assert vm.delete_object() is True
    assert [o.id for o in vm.scene.objects] == ids[:2]
    assert vm.selected == 1  # the last object went, so the selection moves to the new last one
    vm.select(0)
    vm.delete_object()
    assert [o.id for o in vm.scene.objects] == [ids[1]]
    assert vm.selected == 0
    assert len(objects) == 2
    assert selections == [(1,), (0,), (0,)]  # deleting always reports a change: the selected object is new


def test_ids_of_deleted_objects_are_never_reused(vm):
    vm.add_object()
    deleted = vm.selected_object.id
    vm.delete_object()
    vm.add_object()
    assert vm.selected_object.id != deleted
    assert len({o.id for o in vm.scene.objects}) == 2


def test_new_ids_continue_after_those_already_in_the_document(qapp):
    from text3d_studio.model import Scene, TextObject

    model = SceneViewModel(Document(scene=Scene(objects=[TextObject(id=7), TextObject(id=3)])))
    model.add_object()
    assert model.selected_object.id == 8


def test_select_emits_only_on_change_and_rejects_missing_indices(vm):
    vm.add_object()
    events = record(vm.selection_changed)
    vm.select(1)
    vm.select(0)
    assert events == [(0,)]
    for index in (-1, 2):
        with pytest.raises(IndexError):
            vm.select(index)
    assert vm.selected == 0
