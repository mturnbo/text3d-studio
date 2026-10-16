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
