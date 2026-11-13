"""Material kinds and the preset values each one starts from.

A preset covers every MaterialParams field except `kind`. Picking a kind applies its preset; the user can
then adjust the values without leaving the kind.
"""

import dataclasses
import math

KINDS = ("metal", "plastic", "stone", "wood", "glass")

# What the side walls and back cap are made of: "same" as the face, or another kind's preset look.
SIDE_KINDS = ("same",) + KINDS

# Ranges of the shading options sanitized_material_params clamps (the panel's sliders span the same), and
# their colours.
SHADING_RANGES = {"outline_width": (0.0, 1.0), "outline_metallic": (0.0, 1.0)}
SHADING_COLORS = ("outline_color",)

# Fraction of light a kind lets through. Transmissive kinds show what is behind them and cast a lighter,
# tinted shadow; every other kind is opaque.
TRANSMISSION = {"glass": 0.65}

PRESETS = {
    "metal": dict(
        face_color=(0.92, 0.78, 0.45), side_color=(0.70, 0.52, 0.25), secondary_color=(0.70, 0.52, 0.25),
        metallic=1.0, roughness=0.35, side_darkening=0.45, bevel_strength=1.6,
        texture_scale=1.0, texture_strength=0.0, bump_strength=0.0, grain_angle=0.0, varnish=0.0, absorption=0.0,
        side_kind="same", outline_width=0.0, outline_color=(0.08, 0.08, 0.08), outline_metallic=0.0,
    ),
    "plastic": dict(
        face_color=(0.80, 0.10, 0.10), side_color=(0.62, 0.07, 0.07), secondary_color=(0.62, 0.07, 0.07),
        metallic=0.0, roughness=0.3, side_darkening=0.35, bevel_strength=1.6,
        texture_scale=1.0, texture_strength=0.0, bump_strength=0.0, grain_angle=0.0, varnish=0.0, absorption=0.0,
        side_kind="same", outline_width=0.0, outline_color=(0.08, 0.08, 0.08), outline_metallic=0.0,
    ),
    "stone": dict(
        face_color=(0.66, 0.63, 0.58), side_color=(0.66, 0.63, 0.58), secondary_color=(0.32, 0.30, 0.29),
        metallic=0.0, roughness=0.85, side_darkening=0.3, bevel_strength=0.8,
        texture_scale=0.6, texture_strength=0.6, bump_strength=0.45, grain_angle=0.0, varnish=0.0, absorption=0.0,
        side_kind="same", outline_width=0.0, outline_color=(0.08, 0.08, 0.08), outline_metallic=0.0,
    ),
    "wood": dict(
        face_color=(0.78, 0.57, 0.35), side_color=(0.78, 0.57, 0.35), secondary_color=(0.45, 0.27, 0.13),
        metallic=0.0, roughness=0.55, side_darkening=0.3, bevel_strength=1.0,
        texture_scale=0.12, texture_strength=0.7, bump_strength=0.1, grain_angle=0.0, varnish=0.35, absorption=0.0,
        side_kind="same", outline_width=0.0, outline_color=(0.08, 0.08, 0.08), outline_metallic=0.0,
    ),
    "glass": dict(
        face_color=(0.85, 0.92, 0.95), side_color=(0.75, 0.85, 0.90), secondary_color=(0.75, 0.85, 0.90),
        metallic=0.0, roughness=0.05, side_darkening=0.2, bevel_strength=1.6,
        texture_scale=1.0, texture_strength=0.0, bump_strength=0.0, grain_angle=0.0, varnish=0.0, absorption=3.0,
        side_kind="same", outline_width=0.0, outline_color=(0.08, 0.08, 0.08), outline_metallic=0.0,
    ),
}


def transmits(material) -> bool:
    """Whether any part of the material (its face, or its sides) is transmissive and needs the glass backdrop."""
    return material.kind in TRANSMISSION or material.side_kind in TRANSMISSION


def sanitized_material_params(params):
    """The params with the shading options a project file could set badly brought back: an unknown side kind
    falls back to "same", a number or colour that isn't one takes its default, and values are clamped to range."""
    defaults = type(params)()
    values = {"side_kind": params.side_kind if params.side_kind in SIDE_KINDS else "same"}
    for name, (low, high) in SHADING_RANGES.items():
        value = getattr(params, name)
        values[name] = min(max(value, low), high) if _is_finite_number(value) else getattr(defaults, name)
    for name in SHADING_COLORS:
        color = getattr(params, name)
        valid = isinstance(color, tuple) and len(color) == 3 and all(_is_finite_number(c) for c in color)
        values[name] = tuple(min(max(float(c), 0.0), 1.0) for c in color) if valid else getattr(defaults, name)
    return dataclasses.replace(params, **values)


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
