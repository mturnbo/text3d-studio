import copy
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


def test_ground_edits_keep_every_object_on_its_camera_ray(vm):
    from text3d_studio import ground

    vm.add_object()
    vm.set("transform", "x", 3.0)
    before = [ground.text_position(vm.scene.ground, o.transform) for o in vm.scene.objects]
    vm.set("ground", "pitch", 8.0)
    vm.set_ground_pose(roll=4.0, yaw=12.0, height=2.0)
    for obj, old in zip(vm.scene.objects, before):
        new = ground.text_position(vm.scene.ground, obj.transform)
        np.testing.assert_allclose(new / np.linalg.norm(new), old / np.linalg.norm(old), atol=1e-9)


def test_set_ground_pose_re_places_every_object_from_the_anchor(vm):
    from text3d_studio import ground

    vm.add_object()
    anchor = vm.placement_snapshot()
    vm.set_ground_pose(pitch=30.0)
    vm.set_ground_pose(anchor=anchor, pitch=6.0)  # as if dragged straight from the anchor's pose
    for obj, base in zip(vm.scene.objects, anchor[1]):
        expected = ground.reanchor_text(anchor[0], vm.scene.ground, base)
        t = obj.transform
        np.testing.assert_allclose((t.x, t.z, t.rot_y, t.size), expected, atol=1e-12)


def test_an_object_the_new_ground_cannot_hold_keeps_its_place_while_others_move(vm):
    vm.set("transform", "z", -60.0)  # far away: a downward tilt puts it above the horizon
    vm.add_object()
    vm.set("transform", "z", -4.0)
    far_before = dataclasses.replace(vm.scene.objects[0].transform)
    near_before = dataclasses.replace(vm.scene.objects[1].transform)
    vm.set("ground", "pitch", -10.0)
    assert vm.scene.objects[0].transform == far_before
    assert vm.scene.objects[1].transform != near_before


def test_placement_snapshot_is_a_copy(vm):
    vm.add_object()
    ground, transforms = vm.placement_snapshot()
    vm.set("ground", "pitch", 5.0)
    assert ground.pitch == 0.0
    assert all(t.z != o.transform.z for t, o in zip(transforms, vm.scene.objects))


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


# ----- Undo / redo ------------------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def uvm(qapp, clock):
    return SceneViewModel(Document(), clock=clock)


def scene_copy(vm):
    return copy.deepcopy(vm.scene)


def undo_count(vm):
    count = 0
    while vm.undo():
        count += 1
    return count


def test_fresh_viewmodel_has_nothing_to_undo_and_undo_redo_return_false(uvm):
    assert not uvm.can_undo and not uvm.can_redo
    before = scene_copy(uvm)
    assert uvm.undo() is False
    assert uvm.redo() is False
    assert uvm.scene == before


@pytest.mark.parametrize("edit", [
    lambda vm: vm.set("light", "azimuth", 120.0),
    lambda vm: vm.set("text", "content", "WORLD"),
    lambda vm: vm.nudge("transform", "x", 1.5),
    lambda vm: vm.set("ground", "pitch", 12.0),
    lambda vm: vm.set_ground_pose(pitch=10.0, height=2.5),
    lambda vm: vm.apply_material("wood"),
    lambda vm: vm.add_object(),
    lambda vm: vm.duplicate_object(),
    lambda vm: (vm.add_object(), vm.clear_history(), vm.delete_object()),
], ids=["set", "set-text", "nudge", "set-ground", "set_ground_pose", "apply_material", "add", "duplicate", "delete"])
def test_every_entry_point_is_undoable_in_one_step_and_redoable(uvm, edit):
    edit(uvm)
    before_redo = scene_copy(uvm)
    assert uvm.can_undo
    assert uvm.undo()
    assert not uvm.can_undo
    assert uvm.scene != before_redo
    assert uvm.redo()
    assert uvm.scene == before_redo
    assert not uvm.can_redo


def test_undo_restores_the_exact_previous_scene(uvm):
    original = scene_copy(uvm)
    uvm.set("light", "azimuth", 120.0)
    uvm.set("shadow", "opacity", 0.1)
    uvm.undo()
    uvm.undo()
    assert uvm.scene == original


def test_setting_an_unchanged_value_or_selecting_records_nothing(uvm):
    uvm.set("light", "azimuth", uvm.scene.light.azimuth)
    uvm.add_object()
    uvm.clear_history()
    uvm.select(0)
    assert not uvm.can_undo


def test_refused_delete_and_invalid_edits_record_nothing(uvm):
    assert uvm.delete_object() is False
    with pytest.raises(KeyError):
        uvm.set("light", "nope", 1.0)
    with pytest.raises(ValueError):
        uvm.apply_material("cheese")
    assert not uvm.can_undo


def test_a_group_of_fifty_sets_is_one_step(uvm):
    original = scene_copy(uvm)
    uvm.begin_edit()
    for i in range(50):
        uvm.set("transform", "x", float(i + 1))
    uvm.end_edit()
    assert undo_count(uvm) == 1
    assert uvm.scene == original


def test_a_group_that_changes_nothing_records_nothing(uvm):
    events = record(uvm.history_changed)
    uvm.begin_edit()
    uvm.set("transform", "x", 4.0)
    uvm.set("transform", "x", 0.0)  # back where it started
    uvm.end_edit()
    uvm.begin_edit()
    uvm.end_edit()
    assert not uvm.can_undo
    assert events == []


def test_nested_groups_record_once_when_the_outermost_ends(uvm):
    uvm.begin_edit()
    uvm.set("transform", "x", 1.0)
    uvm.begin_edit()
    uvm.set("transform", "z", -5.0)
    uvm.end_edit()
    assert not uvm.can_undo  # the inner end doesn't record
    uvm.set("light", "azimuth", 10.0)
    uvm.end_edit()
    assert undo_count(uvm) == 1


def test_undo_is_refused_during_a_group(uvm):
    uvm.set("light", "azimuth", 10.0)
    uvm.begin_edit()
    uvm.set("transform", "x", 1.0)
    assert uvm.undo() is False
    uvm.end_edit()
    assert undo_count(uvm) == 2


def test_quick_changes_to_one_field_merge_into_one_step(uvm, clock):
    original = scene_copy(uvm)
    for i in range(10):
        uvm.nudge("transform", "size", 0.1)
        clock.now += 0.5
    assert undo_count(uvm) == 1
    assert uvm.scene == original


def test_typing_in_the_text_field_merges(uvm, clock):
    for text in ("H", "HE", "HEL", "HELP"):
        uvm.set("text", "content", text)
        clock.now += 0.2
    uvm.undo()
    assert uvm.scene.objects[0].text.content == "HELLO"
    assert not uvm.can_undo


def test_changes_to_different_fields_do_not_merge(uvm):
    uvm.set("transform", "size", 4.0)
    uvm.set("transform", "x", 1.0)
    uvm.set("transform", "size", 5.0)
    assert undo_count(uvm) == 3


def test_the_same_field_of_another_object_does_not_merge(uvm):
    uvm.add_object()
    uvm.clear_history()
    uvm.set("transform", "size", 4.0)
    uvm.select(0)
    uvm.set("transform", "size", 5.0)
    assert undo_count(uvm) == 2


def test_a_pause_longer_than_the_merge_window_starts_a_new_step(uvm, clock):
    uvm.set("transform", "size", 4.0)
    clock.now += 0.61
    uvm.set("transform", "size", 5.0)
    uvm.undo()
    assert uvm.scene.objects[0].transform.size == 4.0
    assert undo_count(uvm) == 1


def test_a_group_ends_a_burst(uvm):
    uvm.set("transform", "size", 4.0)
    uvm.begin_edit()
    uvm.set("transform", "x", 2.0)
    uvm.end_edit()
    uvm.set("transform", "size", 5.0)
    assert undo_count(uvm) == 3


def test_a_new_change_after_undo_clears_redo(uvm):
    uvm.set("light", "azimuth", 10.0)
    uvm.undo()
    assert uvm.can_redo
    uvm.set("light", "elevation", 20.0)
    assert not uvm.can_redo
    assert uvm.redo() is False


def test_undoing_add_removes_the_object_and_restores_the_selection(uvm):
    uvm.add_object()
    uvm.add_object()
    uvm.select(0)
    uvm.duplicate_object()
    assert len(uvm.scene.objects) == 4 and uvm.selected == 3
    uvm.undo()
    assert len(uvm.scene.objects) == 3
    assert uvm.selected == 0
    uvm.redo()
    assert len(uvm.scene.objects) == 4 and uvm.selected == 3


def test_undoing_delete_brings_back_the_object_with_its_id_and_selection(uvm):
    uvm.add_object()
    uvm.set("text", "content", "SECOND")
    uvm.select(0)
    uvm.set("text", "content", "FIRST")
    ids = [o.id for o in uvm.scene.objects]
    uvm.delete_object()
    assert [o.text.content for o in uvm.scene.objects] == ["SECOND"]
    uvm.undo()
    assert [o.id for o in uvm.scene.objects] == ids
    assert [o.text.content for o in uvm.scene.objects] == ["FIRST", "SECOND"]
    assert uvm.selected == 0


def test_ids_stay_unique_after_undoing_an_add_and_adding_again(uvm):
    uvm.add_object()
    added = uvm.scene.objects[1].id
    uvm.undo()
    uvm.add_object()
    assert uvm.scene.objects[1].id != added


def test_ground_edit_that_replaces_every_object_undoes_in_one_step(uvm):
    uvm.add_object()
    uvm.clear_history()
    original = scene_copy(uvm)
    uvm.set("ground", "pitch", 15.0)
    assert [o.transform for o in uvm.scene.objects] != [o.transform for o in original.objects]
    uvm.undo()
    assert uvm.scene == original
    assert not uvm.can_undo


def test_set_ground_pose_with_anchor_undoes_in_one_step(uvm):
    original = scene_copy(uvm)
    anchor = uvm.placement_snapshot()
    uvm.set_ground_pose(anchor=anchor, pitch=8.0, roll=3.0, height=2.0)
    uvm.undo()
    assert uvm.scene == original
    assert not uvm.can_undo


def test_set_background_clears_history(uvm):
    uvm.set("light", "azimuth", 10.0)
    uvm.undo()
    uvm.set("light", "azimuth", 20.0)
    uvm.undo()
    assert uvm.can_redo
    events = record(uvm.history_changed)
    uvm.set_background(np.zeros((4, 4, 4), np.uint8), "photo.png")
    assert not uvm.can_undo and not uvm.can_redo
    assert events


def test_clear_history_emits_and_empties(uvm):
    uvm.set("light", "azimuth", 10.0)
    events = record(uvm.history_changed)
    uvm.clear_history()
    assert not uvm.can_undo
    assert len(events) == 1


def test_history_changed_fires_on_record_undo_and_redo(uvm):
    events = record(uvm.history_changed)
    uvm.set("light", "azimuth", 10.0)
    uvm.undo()
    uvm.redo()
    assert len(events) == 3


def test_restoring_does_not_record_a_step(uvm):
    uvm.set("light", "azimuth", 10.0)
    uvm.undo()
    assert not uvm.can_undo
    assert uvm.can_redo


def test_history_is_bounded_to_one_hundred_steps(uvm, clock):
    for i in range(120):
        uvm.set("light", "azimuth", float(i + 1))
        clock.now += 1.0
    assert undo_count(uvm) == 100
    assert uvm.scene.light.azimuth == 20.0


def test_undo_emits_changes_for_shared_fields_and_refreshes_objects(uvm):
    uvm.set("light", "azimuth", 10.0)
    uvm.set("text", "content", "WORLD")
    uvm.undo()
    scene_events, text_events = record(uvm.scene_changed), record(uvm.text_changed)
    object_events, selection_events = record(uvm.objects_changed), record(uvm.selection_changed)
    uvm.undo()
    assert scene_events == [("light", "azimuth")]
    assert text_events == []
    assert object_events == [()]
    assert selection_events == [(0,)]


def test_undoing_a_text_change_emits_text_changed(uvm):
    uvm.set("text", "content", "WORLD")
    events = record(uvm.text_changed)
    uvm.undo()
    assert len(events) == 1


def test_undo_keeps_the_scene_object_in_place(uvm):
    scene = uvm.scene
    uvm.add_object()
    uvm.undo()
    assert uvm.scene is scene
    assert uvm.document.scene is scene


def test_later_edits_do_not_alter_recorded_states(uvm):
    uvm.set("transform", "x", 1.0)
    uvm.set("transform", "z", -5.0)  # mutates the live objects the first state must not share
    uvm.undo()
    uvm.undo()
    assert uvm.scene.objects[0].transform == Document().scene.objects[0].transform
