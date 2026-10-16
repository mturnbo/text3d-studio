"""Material kinds and the preset values each one starts from.

A preset covers every MaterialParams field except `kind`. Picking a kind applies its preset; the user can
then adjust the values without leaving the kind.
"""

KINDS = ("metal", "plastic", "stone", "wood", "glass")

PRESETS = {
    "metal": dict(
        face_color=(0.92, 0.78, 0.45), side_color=(0.70, 0.52, 0.25), secondary_color=(0.70, 0.52, 0.25),
        metallic=1.0, roughness=0.35, side_darkening=0.45, bevel_strength=1.6,
        texture_scale=1.0, texture_strength=0.0, bump_strength=0.0, grain_angle=0.0, varnish=0.0,
    ),
    "plastic": dict(
        face_color=(0.80, 0.10, 0.10), side_color=(0.62, 0.07, 0.07), secondary_color=(0.62, 0.07, 0.07),
        metallic=0.0, roughness=0.3, side_darkening=0.35, bevel_strength=1.6,
        texture_scale=1.0, texture_strength=0.0, bump_strength=0.0, grain_angle=0.0, varnish=0.0,
    ),
    "stone": dict(
        face_color=(0.66, 0.63, 0.58), side_color=(0.66, 0.63, 0.58), secondary_color=(0.32, 0.30, 0.29),
        metallic=0.0, roughness=0.85, side_darkening=0.3, bevel_strength=0.8,
        texture_scale=0.6, texture_strength=0.6, bump_strength=0.45, grain_angle=0.0, varnish=0.0,
    ),
    "wood": dict(
        face_color=(0.78, 0.57, 0.35), side_color=(0.78, 0.57, 0.35), secondary_color=(0.45, 0.27, 0.13),
        metallic=0.0, roughness=0.55, side_darkening=0.3, bevel_strength=1.0,
        texture_scale=0.12, texture_strength=0.7, bump_strength=0.1, grain_angle=0.0, varnish=0.35,
    ),
    "glass": dict(
        face_color=(0.85, 0.92, 0.95), side_color=(0.75, 0.85, 0.90), secondary_color=(0.75, 0.85, 0.90),
        metallic=0.0, roughness=0.05, side_darkening=0.2, bevel_strength=1.6,
        texture_scale=1.0, texture_strength=0.0, bump_strength=0.0, grain_angle=0.0, varnish=0.0,
    ),
}
