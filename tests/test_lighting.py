import numpy as np
import pytest

from text3d_studio import ground, math3d
from text3d_studio.lighting import MIN_SHADOW_ELEVATION, LightingEngine, solve_sun, srgb_to_linear
from text3d_studio.model import GroundParams, LightParams


def test_light_direction_is_unit_and_scaled_colors():
    light = LightParams(intensity=2.0, color=(1, 0.5, 0.25), ambient_color=(1, 1, 1), ambient_intensity=0.3)
    state = LightingEngine().compute(light, GroundParams())
    assert np.linalg.norm(state.light_dir) == pytest.approx(1.0)
    np.testing.assert_allclose(state.light_color, 2.0 * srgb_to_linear((1, 0.5, 0.25)))
    np.testing.assert_allclose(state.ambient, [0.3, 0.3, 0.3])  # white is 1.0 in both spaces


def test_light_follows_ground_tilt_but_not_grid_yaw():
    light = LightParams(azimuth=40, elevation=30)
    engine = LightingEngine()
    flat = engine.compute(light, GroundParams())
    yawed = engine.compute(light, GroundParams(yaw=90))
    tilted = engine.compute(light, GroundParams(pitch=25))
    np.testing.assert_allclose(flat.light_dir, yawed.light_dir)
    expected = math3d.rotation_x(25)[:3, :3] @ flat.light_dir
    np.testing.assert_allclose(tilted.light_dir, expected, atol=1e-12)


def test_shadow_falls_on_the_ground_plane():
    g = GroundParams(pitch=10, roll=-5, yaw=30, height=2.2)
    state = LightingEngine().compute(LightParams(), g)
    np.testing.assert_allclose(state.ground_plane, ground.ground_plane(g))
    flattened = state.shadow_matrix @ np.array([1.0, 0.5, -9.0, 1.0])
    point = flattened[:3] / flattened[3]
    assert state.ground_plane[:3] @ point + state.ground_plane[3] == pytest.approx(0.0, abs=1e-9)


def test_grazing_light_shadow_stays_finite():
    state = LightingEngine().compute(LightParams(elevation=0.0), GroundParams())
    point = state.shadow_matrix @ np.array([0.0, 1.0, -12.0, 1.0])
    assert abs(point[3]) > 1e-3
    assert np.all(np.isfinite(point[:3] / point[3]))
    assert MIN_SHADOW_ELEVATION > 0



def test_srgb_to_linear_matches_the_standard_curve():
    np.testing.assert_allclose(srgb_to_linear([0.0, 1.0]), [0.0, 1.0])
    assert srgb_to_linear([0.5])[0] == pytest.approx(0.21404, abs=1e-5)
    assert srgb_to_linear([0.04])[0] == pytest.approx(0.04 / 12.92)  # linear toe below the knee


# ----- Match Sun ----------------------------------------------------------------------------------

IMAGE = (1200, 800)


def synthetic_clicks(g, fov, azimuth, elevation, base_xz=(0.8, -9.0), height=1.2, rounded=False):
    """Pixels of a vertical post's top and base and its shadow's tip under a sun at (azimuth, elevation)."""
    base = ground.ground_to_world(g, *base_xz)
    top = base + height * ground.ground_up(g)
    toward_light = ground.light_frame(g) @ math3d.direction_from_angles(azimuth, 0.0)
    tip = base - toward_light * height / np.tan(np.radians(elevation))
    pixels = [math3d.project_to_image(p, fov, IMAGE) for p in (top, base, tip)]
    return [np.round(p) for p in pixels] if rounded else pixels


@pytest.mark.parametrize("g", [GroundParams(), GroundParams(pitch=18, roll=-5, yaw=40, height=1.7)])
@pytest.mark.parametrize("azimuth, elevation", [(35, 40), (-120, 25), (170, 60), (80, 12)])
def test_sun_is_solved_back_from_a_projected_post_and_its_shadow(g, azimuth, elevation):
    solved = solve_sun(g, 45.0, IMAGE, *synthetic_clicks(g, 45.0, azimuth, elevation))
    assert solved == pytest.approx((azimuth, elevation), abs=1e-6)


def test_sun_solve_tolerates_clicks_rounded_to_whole_pixels():
    g = GroundParams(pitch=15, height=1.6)
    azimuth, elevation = solve_sun(g, 40.0, IMAGE, *synthetic_clicks(g, 40.0, -60, 35, rounded=True))
    assert azimuth == pytest.approx(-60, abs=1.5) and elevation == pytest.approx(35, abs=1.5)


def test_solved_sun_lights_the_scene_from_where_the_shadow_points_away_from():
    g = GroundParams(pitch=10)
    top, base, tip = synthetic_clicks(g, 40.0, 50, 30)
    azimuth, elevation = solve_sun(g, 40.0, IMAGE, top, base, tip)
    light_dir = LightingEngine().compute(LightParams(azimuth=azimuth, elevation=elevation), g).light_dir
    base_w = math3d.intersect_ray_plane((0, 0, 0), math3d.image_ray(base, 40.0, IMAGE), ground.ground_plane(g))
    tip_w = math3d.intersect_ray_plane((0, 0, 0), math3d.image_ray(tip, 40.0, IMAGE), ground.ground_plane(g))
    assert light_dir @ (tip_w - base_w) < 0


def test_sun_solve_uses_the_closest_approach_when_the_top_click_is_off_the_vertical():
    g = GroundParams(pitch=12)
    top, base, tip = synthetic_clicks(g, 40.0, 30, 45)
    nudged = (top[0] + 6.0, top[1])  # a little to the side of the post's top
    azimuth, elevation = solve_sun(g, 40.0, IMAGE, nudged, base, tip)
    assert azimuth == pytest.approx(30, abs=1e-6)  # the azimuth only depends on the base and the tip
    assert elevation == pytest.approx(45, abs=1.0)


@pytest.mark.parametrize(
    "case, message",
    [
        ("no_shadow", "no length"),
        ("top_below_base", "above the base"),
        ("base_in_sky", "base must be on the floor"),
        ("tip_in_sky", "tip must be on the floor"),
    ],
)
def test_sun_solve_explains_clicks_that_cannot_describe_a_shadow(case, message):
    g = GroundParams(pitch=10)
    top, base, tip = synthetic_clicks(g, 40.0, 30, 40)
    sky = (600.0, 20.0)
    clicks = {
        "no_shadow": (top, base, base),
        "top_below_base": ((base[0], base[1] + 40.0), base, tip),
        "base_in_sky": (top, sky, tip),
        "tip_in_sky": (top, base, sky),
    }[case]
    with pytest.raises(ValueError, match=message):
        solve_sun(g, 40.0, IMAGE, *clicks)


def test_sun_solve_refuses_a_height_seen_straight_down_the_post():
    g = GroundParams(pitch=89.0)  # looking (almost) straight down: the post's line nearly runs along the view ray
    base = ground.ground_to_world(g, 0.0, 0.0)
    pixel = math3d.project_to_image(base, 40.0, IMAGE)
    if pixel is None:
        pytest.skip("the point below the camera is out of view")
    with pytest.raises(ValueError):
        solve_sun(g, 40.0, IMAGE, pixel, pixel, (pixel[0] + 50.0, pixel[1]))
