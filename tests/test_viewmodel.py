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
    vm.set("transform", "x", vm.scene.transform.x)
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
    assert vm.scene.transform.rot_y == 7.5


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


def test_grid_settings_are_editable_and_hidden_by_default(vm):
    events = record(vm.scene_changed)
    assert vm.scene.grid.visible is False
    vm.set("grid", "visible", True)
    vm.set("grid", "cell_size", 0.5)
    assert events == [("grid", "visible"), ("grid", "cell_size")]


def test_ground_edits_keep_the_text_on_the_same_camera_ray(vm):
    from text3d_studio import ground

    before = ground.text_position(vm.scene.ground, vm.scene.transform)
    vm.set("ground", "pitch", 8.0)
    after = ground.text_position(vm.scene.ground, vm.scene.transform)
    np.testing.assert_allclose(after / np.linalg.norm(after), before / np.linalg.norm(before), atol=1e-9)


def test_ground_edits_report_the_moved_text_fields(vm):
    events = record(vm.scene_changed)
    vm.set("ground", "yaw", 15.0)
    assert ("ground", "yaw") in events
    assert {("transform", "x"), ("transform", "z"), ("transform", "rot_y")} <= set(events)


def test_text_stays_put_when_the_ground_tips_past_it(vm):
    vm.set("transform", "z", -60.0)
    before = (vm.scene.transform.x, vm.scene.transform.z)
    vm.set("ground", "pitch", -10.0)
    assert (vm.scene.transform.x, vm.scene.transform.z) == before


def test_set_ground_pose_reanchors_once_against_the_final_pose(vm):
    from text3d_studio import ground
    from text3d_studio.model import GroundParams

    old = dataclasses.replace(vm.scene.ground)
    expected = ground.reanchor_text(old, GroundParams(pitch=9.0, yaw=25.0, roll=-3.0, height=2.2), vm.scene.transform)
    vm.set_ground_pose(pitch=9.0, yaw=25.0, roll=-3.0, height=2.2)
    g, t = vm.scene.ground, vm.scene.transform
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
    assert ground.reanchor_text(old, dataclasses.replace(old, roll=55.0), one_step.scene.transform) is None
    one_step.set_ground_pose(roll=55.0, pitch=10.0)
    one_by_one.set("ground", "roll", 55.0)
    one_by_one.set("ground", "pitch", 10.0)

    expected = ground.reanchor_text(old, one_step.scene.ground, start().scene.transform)
    t = one_step.scene.transform
    np.testing.assert_allclose((t.x, t.z, t.rot_y, t.size), expected, atol=1e-12)
    u = one_by_one.scene.transform
    assert not np.allclose((u.x, u.z, u.size), (t.x, t.z, t.size))


@pytest.mark.parametrize("kind", ["plastic", "stone", "wood", "glass", "metal"])
def test_apply_material_loads_the_preset(vm, kind):
    from text3d_studio.materials import PRESETS

    vm.apply_material(kind)
    material = vm.scene.material
    assert material.kind == kind
    assert {name: getattr(material, name) for name in PRESETS[kind]} == PRESETS[kind]


def test_apply_material_replaces_adjustments_and_reports_each_field(vm):
    vm.set("material", "roughness", 0.9)
    events = record(vm.scene_changed)
    vm.apply_material("plastic")
    assert vm.scene.material.roughness == 0.3
    assert ("material", "kind") in events and ("material", "roughness") in events


def test_reapplying_the_same_material_resets_it(vm):
    vm.apply_material("wood")
    vm.set("material", "face_color", (0.1, 0.2, 0.3))
    vm.apply_material("wood")
    assert vm.scene.material.face_color == (0.72, 0.50, 0.30)


def test_unknown_material_is_rejected(vm):
    with pytest.raises(ValueError):
        vm.apply_material("cheese")
    assert vm.scene.material.kind == "metal"
