"""Material kinds and the preset values each one starts from.

A preset covers every MaterialParams field except `kind`. Picking a kind applies its preset; the user can
then adjust the values without leaving the kind.
"""

KINDS = ("metal", "plastic", "stone", "wood", "glass")

PRESETS = {
    "metal": dict(
        face_color=(0.92, 0.78, 0.45), side_color=(0.70, 0.52, 0.25),
        metallic=1.0, roughness=0.35, side_darkening=0.45, bevel_strength=1.6,
    ),
    "plastic": dict(
        face_color=(0.80, 0.10, 0.10), side_color=(0.62, 0.07, 0.07),
        metallic=0.0, roughness=0.3, side_darkening=0.35, bevel_strength=1.6,
    ),
    "stone": dict(
        face_color=(0.62, 0.60, 0.56), side_color=(0.55, 0.53, 0.50),
        metallic=0.0, roughness=0.85, side_darkening=0.3, bevel_strength=0.8,
    ),
    "wood": dict(
        face_color=(0.72, 0.50, 0.30), side_color=(0.55, 0.36, 0.20),
        metallic=0.0, roughness=0.55, side_darkening=0.3, bevel_strength=1.0,
    ),
    "glass": dict(
        face_color=(0.85, 0.92, 0.95), side_color=(0.75, 0.85, 0.90),
        metallic=0.0, roughness=0.05, side_darkening=0.2, bevel_strength=1.6,
    ),
}
