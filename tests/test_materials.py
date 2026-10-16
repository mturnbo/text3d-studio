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
