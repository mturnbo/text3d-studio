import dataclasses

import pytest

from text3d_studio.materials import KINDS, PRESETS, TRANSMISSION
from text3d_studio.model import MaterialParams

MATERIAL_FIELDS = {f.name for f in dataclasses.fields(MaterialParams)}


def test_every_kind_has_a_complete_preset():
    assert set(PRESETS) == set(KINDS)
    for kind in KINDS:
        assert set(PRESETS[kind]) == MATERIAL_FIELDS - {"kind"}, kind


def test_default_material_is_the_metal_preset():
    default = MaterialParams()
    assert default.kind == "metal"
    assert {k: getattr(default, k) for k in PRESETS["metal"]} == PRESETS["metal"]


@pytest.mark.parametrize("kind", KINDS)
def test_preset_values_are_in_range(kind):
    preset = PRESETS[kind]
    assert 0.0 <= preset["metallic"] <= 1.0
    assert 0.02 <= preset["roughness"] <= 1.0
    assert 0.0 <= preset["side_darkening"] <= 1.0
    for color in (preset["face_color"], preset["side_color"], preset["secondary_color"]):
        assert len(color) == 3 and all(0.0 <= c <= 1.0 for c in color)
    assert 0.02 <= preset["texture_scale"] <= 10.0
    for name in ("texture_strength", "bump_strength", "varnish"):
        assert 0.0 <= preset[name] <= 1.0, name
    assert -90.0 <= preset["grain_angle"] <= 90.0
    assert 0.0 <= preset["absorption"] <= 5.0


def test_only_stone_and_wood_carry_a_texture():
    assert [kind for kind in KINDS if PRESETS[kind]["texture_strength"] > 0] == ["stone", "wood"]


def test_only_metal_is_metallic():
    assert [kind for kind in KINDS if PRESETS[kind]["metallic"] > 0] == ["metal"]


def test_only_glass_transmits_and_absorbs():
    assert set(TRANSMISSION) == {"glass"}
    assert all(0.0 < amount < 1.0 for amount in TRANSMISSION.values())
    assert [kind for kind in KINDS if PRESETS[kind]["absorption"] > 0] == ["glass"]


def test_side_kinds_are_same_or_any_kind_and_presets_keep_their_own_sides():
    from text3d_studio.materials import SIDE_KINDS

    assert SIDE_KINDS == ("same",) + KINDS
    assert all(PRESETS[kind]["side_kind"] == "same" for kind in KINDS)


@pytest.mark.parametrize("kind, side_kind, expected", [
    ("metal", "same", False), ("glass", "same", True), ("metal", "glass", True), ("glass", "stone", True),
    ("plastic", "wood", False),
])
def test_a_material_transmits_when_its_face_or_its_sides_are_glass(kind, side_kind, expected):
    from text3d_studio.materials import transmits

    assert transmits(MaterialParams(kind=kind, side_kind=side_kind)) is expected


@pytest.mark.parametrize("value, expected", [
    ("stone", "stone"), ("same", "same"), ("glass", "glass"), ("cheese", "same"), ("", "same"), (3, "same"),
    (None, "same"), (["stone"], "same"),
])
def test_sanitizing_keeps_known_side_kinds_only(value, expected):
    from text3d_studio.materials import sanitized_material_params

    assert sanitized_material_params(MaterialParams(side_kind=value)).side_kind == expected
