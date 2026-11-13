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
    assert preset["outline_width"] == 0.0  # no kind starts with an outline
    assert 0.0 <= preset["outline_metallic"] <= 1.0
    assert len(preset["outline_color"]) == 3 and all(0.0 <= c <= 1.0 for c in preset["outline_color"])
    assert len(preset["emission_color"]) == 3 and all(0.0 <= c <= 1.0 for c in preset["emission_color"])
    assert 0.0 <= preset["emission_strength"] <= 5.0


def test_only_stone_and_wood_carry_a_texture():
    assert [kind for kind in KINDS if PRESETS[kind]["texture_strength"] > 0] == ["stone", "wood"]


def test_only_metal_is_metallic():
    assert [kind for kind in KINDS if PRESETS[kind]["metallic"] > 0] == ["metal"]


def test_only_glass_transmits_and_absorbs():
    assert set(TRANSMISSION) == {"glass"}
    assert all(0.0 < amount < 1.0 for amount in TRANSMISSION.values())
    assert [kind for kind in KINDS if PRESETS[kind]["absorption"] > 0] == ["glass"]


def test_only_neon_glows_and_its_tube_is_dark():
    assert [kind for kind in KINDS if PRESETS[kind]["emission_strength"] > 0] == ["neon"]
    assert max(PRESETS["neon"]["face_color"]) < 0.5  # with the glow off it is a dark plastic


def test_sanitizing_clamps_the_glow():
    from text3d_studio.materials import sanitized_material_params

    for value, expected in ((2.5, 2.5), (50.0, 5.0), (-1.0, 0.0), (float("nan"), 0.0), ("bright", 0.0), (10**400, 5.0)):
        params = sanitized_material_params(MaterialParams(emission_strength=value, emission_color=(1.5, 0.5, -1)))
        assert params.emission_strength == expected
        assert params.emission_color == (1.0, 0.5, 0.0)


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


@pytest.mark.parametrize("value, expected", [
    (0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (5.0, 1.0), (-2.0, 0.0), (1, 1), (float("nan"), 0.0), (float("inf"), 0.0),
    ("wide", 0.0), (True, 0.0), (None, 0.0), (10**400, 1.0), (-10**400, 0.0),
])
def test_sanitizing_clamps_the_outline_width_and_metallic(value, expected):
    from text3d_studio.materials import sanitized_material_params

    params = sanitized_material_params(MaterialParams(outline_width=value, outline_metallic=value))
    assert params.outline_width == expected and params.outline_metallic == expected


@pytest.mark.parametrize("value, expected", [
    ((0.2, 0.4, 0.6), (0.2, 0.4, 0.6)), ((2.0, -1.0, 0.5), (1.0, 0.0, 0.5)), ((1, 0, 0), (1.0, 0.0, 0.0)),
    ((0.1, 0.2), None), ((0.1, float("nan"), 0.3), None), (("red", 0.0, 0.0), None), (0.5, None), ("red", None),
    ((10**400, 0, -10**400), (1.0, 0.0, 0.0)),
])
def test_sanitizing_repairs_the_outline_color(value, expected):
    from text3d_studio.materials import sanitized_material_params

    params = sanitized_material_params(MaterialParams(outline_color=value))
    assert params.outline_color == (expected if expected is not None else MaterialParams().outline_color)


def test_sanitizing_leaves_valid_materials_unchanged():
    from text3d_studio.materials import sanitized_material_params

    for kind in KINDS:
        params = MaterialParams(kind=kind, **{**PRESETS[kind], "outline_width": 0.4, "side_kind": "stone"})
        assert sanitized_material_params(params) == params
