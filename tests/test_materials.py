import dataclasses

import pytest

from text3d_studio.materials import KINDS, PRESETS
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
    for color in (preset["face_color"], preset["side_color"]):
        assert len(color) == 3 and all(0.0 <= c <= 1.0 for c in color)


def test_only_metal_is_metallic():
    assert [kind for kind in KINDS if PRESETS[kind]["metallic"] > 0] == ["metal"]
