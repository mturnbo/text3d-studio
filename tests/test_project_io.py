import dataclasses
import json
import os
import shutil
import typing

import numpy as np
import pytest

from text3d_studio import model
from text3d_studio.image_io import save_image
from text3d_studio.model import Document, Scene, TextObject, TextParams, TransformParams
from text3d_studio.project_io import COLOR_FIELDS, FORMAT_VERSION, is_allowed_photo_path, load_project, save_project


def photo_pixels(height=30, width=50):
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    return pixels


def photo_file(folder, name="photo.png"):
    path = str(folder / name)
    save_image(photo_pixels(), path)
    return path


def busy_scene() -> Scene:
    """Every section moved off its defaults, with a gap in the ids as a deletion leaves."""
    scene = Scene(objects=[
        TextObject(id=0, text=TextParams(content="ONE", italic=True), transform=TransformParams(x=1.5, z=-8.0)),
        TextObject(id=3, text=TextParams(content="TWO", font_family="Courier"), transform=TransformParams(size=2.0)),
    ])
    scene.objects[1].material = dataclasses.replace(scene.objects[1].material, kind="glass", face_color=(0.1, 0.2, 0.3))
    scene.ground = dataclasses.replace(scene.ground, pitch=12.5, roll=-3.0, height=2.0)
    scene.grid = dataclasses.replace(scene.grid, visible=True, accent_color=(0.5, 0.25, 1.0))
    scene.light = dataclasses.replace(scene.light, azimuth=80.0, color=(1.0, 0.5, 0.25), reflect_photo=False)
    scene.shadow = dataclasses.replace(scene.shadow, enabled=False, color=(0.3, 0.3, 0.4), contact_strength=0.7,
                                       contact_size=0.3)
    scene.camera = dataclasses.replace(scene.camera, fov=55.0)
    scene.objects[1].transform = dataclasses.replace(scene.objects[1].transform, mount="wall", wall_x=-2.5, wall_y=1.25)
    scene.wall = dataclasses.replace(scene.wall, angle=-35.0, offset=7.5, enabled=True)
    scene.mask = dataclasses.replace(scene.mask, brush_size=0.12, feather=0.25)
    scene.effects = dataclasses.replace(scene.effects, reflection_strength=0.4, reflection_fade=1.5, bloom_strength=1.2,
                                        bloom_threshold=0.7, bloom_radius=0.5, grain_amount=0.3,
                                        grain_size=2.0, blur_radius=0.25, grade_brightness=0.1, grade_contrast=1.2,
                                        grade_saturation=0.8, grade_warmth=-0.3)
    return scene


def write_json(path, data):
    path.write_text(json.dumps(data))
    return str(path)


def project_dict(**overrides):
    return {"format": "text3d-studio-project", "version": 1, "scene": {}, "photo": None, **overrides}


# ----- Round trips ------------------------------------------------------------------------------


def test_multi_object_scene_round_trips_every_section(qapp, tmp_path):
    document = Document(scene=busy_scene())
    path = str(tmp_path / "scene.t3d")
    save_project(path, document)
    loaded = load_project(path)
    assert loaded.warnings == []
    assert loaded.document.scene == document.scene
    for section in ("ground", "grid", "light", "shadow", "camera", "wall"):
        assert getattr(loaded.document.scene, section) == getattr(document.scene, section)
    assert [o.id for o in loaded.document.scene.objects] == [0, 3]
    assert loaded.document.background is None


def test_text_layout_round_trips(qapp, tmp_path):
    text = TextParams(content="OPEN\n\nLATE\n", align="right", letter_spacing=-0.05, line_height=1.4,
                      arc=-72.5)
    path = str(tmp_path / "layout.t3d")
    save_project(path, Document(scene=Scene(objects=[TextObject(text=text)])))
    assert load_project(path).document.scene.objects[0].text == text


def test_colours_come_back_as_tuples(qapp, tmp_path):
    path = str(tmp_path / "scene.t3d")
    save_project(path, Document(scene=busy_scene()))
    scene = load_project(path).document.scene
    assert scene.grid.accent_color == (0.5, 0.25, 1.0) and isinstance(scene.grid.accent_color, tuple)
    assert isinstance(scene.objects[1].material.face_color, tuple)
    assert isinstance(scene.shadow.color, tuple)


def test_every_colour_field_is_restored_as_a_tuple():
    for cls in (model.GridParams, model.MaterialParams, model.LightParams, model.ShadowParams, model.TextParams,
                model.TransformParams, model.GroundParams, model.CameraParams, model.WallParams):
        hints = typing.get_type_hints(cls)
        colours = {f.name for f in dataclasses.fields(cls) if hints[f.name] == model.Color}
        assert colours == set(COLOR_FIELDS.get(cls, ())), cls.__name__


def test_file_is_versioned_json(qapp, tmp_path):
    path = tmp_path / "scene.t3d"
    save_project(str(path), Document())
    data = json.loads(path.read_text())
    assert data["format"] == "text3d-studio-project"
    assert data["version"] == FORMAT_VERSION == 1
    assert data["photo"] is None
    assert data["scene"]["objects"][0]["text"]["content"] == "HELLO"


def test_saving_overwrites_an_existing_project(qapp, tmp_path):
    path = str(tmp_path / "scene.t3d")
    save_project(path, Document())
    save_project(path, Document(scene=busy_scene()))
    assert load_project(path).document.scene == busy_scene()
    assert os.listdir(tmp_path) == ["scene.t3d"]


# ----- Photo ------------------------------------------------------------------------------------


def test_linked_photo_is_stored_relative_and_absolute(qapp, tmp_path):
    (tmp_path / "images").mkdir()
    photo = photo_file(tmp_path / "images")
    path = tmp_path / "scene.t3d"
    save_project(str(path), Document(background=photo_pixels(), background_path=photo))
    entry = json.loads(path.read_text())["photo"]
    assert entry == {"relative_path": os.path.join("images", "photo.png"), "absolute_path": photo}


def test_relative_photo_survives_moving_the_folder(qapp, tmp_path):
    folder = tmp_path / "before"
    folder.mkdir()
    photo = photo_file(folder)
    save_project(str(folder / "scene.t3d"), Document(background=photo_pixels(), background_path=photo))
    moved = tmp_path / "after"
    shutil.move(str(folder), str(moved))
    loaded = load_project(str(moved / "scene.t3d"))
    assert loaded.warnings == [] and loaded.missing_photo is None
    assert loaded.document.background_path == str(moved / "photo.png")
    assert np.array_equal(loaded.document.background, photo_pixels())
    assert not loaded.photo_embedded


def test_absolute_path_is_used_when_the_project_moves_alone(qapp, tmp_path):
    photo = photo_file(tmp_path)
    (tmp_path / "projects").mkdir()
    save_project(str(tmp_path / "scene.t3d"), Document(background=photo_pixels(), background_path=photo))
    shutil.move(str(tmp_path / "scene.t3d"), str(tmp_path / "projects" / "scene.t3d"))
    loaded = load_project(str(tmp_path / "projects" / "scene.t3d"))
    assert loaded.warnings == []
    assert loaded.document.background_path == photo


def test_embedded_photo_round_trips_pixel_exact(qapp, tmp_path):
    pixels = photo_pixels()
    pixels[0, :, 3] = np.arange(50, dtype=np.uint8) * 5  # translucent row: alpha must survive too
    photo = photo_file(tmp_path)
    path = str(tmp_path / "scene.t3d")
    save_project(path, Document(background=pixels, background_path=photo), embed_photo=True)
    os.remove(photo)  # the project no longer needs it
    loaded = load_project(path)
    assert loaded.warnings == []
    assert loaded.photo_embedded
    assert loaded.document.background_path is None
    assert np.array_equal(loaded.document.background, pixels)
    assert "absolute_path" not in json.loads(open(path).read())["photo"]


def test_background_without_a_path_is_embedded(qapp, tmp_path):
    path = str(tmp_path / "scene.t3d")
    save_project(path, Document(background=photo_pixels()))
    loaded = load_project(path)
    assert loaded.photo_embedded
    assert np.array_equal(loaded.document.background, photo_pixels())


def test_missing_photo_is_a_warning_and_leaves_no_background(qapp, tmp_path):
    photo = photo_file(tmp_path)
    path = str(tmp_path / "scene.t3d")
    save_project(path, Document(scene=busy_scene(), background=photo_pixels(), background_path=photo))
    os.remove(photo)
    loaded = load_project(path)
    assert loaded.document.background is None and loaded.document.background_path is None
    assert loaded.missing_photo == photo
    assert len(loaded.warnings) == 1 and photo in loaded.warnings[0]
    assert loaded.document.scene == busy_scene()


def test_unreadable_photo_is_a_warning(qapp, tmp_path):
    photo = tmp_path / "photo.png"
    photo.write_text("not an image")
    path = write_json(tmp_path / "scene.t3d", project_dict(photo={"absolute_path": str(photo)}))
    loaded = load_project(path)
    assert loaded.document.background is None
    assert loaded.missing_photo == str(photo)


def test_damaged_embedded_photo_is_a_warning(qapp, tmp_path):
    path = write_json(tmp_path / "scene.t3d", project_dict(photo={"embedded_png": "!!not base64!!"}))
    loaded = load_project(path)
    assert loaded.document.background is None
    assert len(loaded.warnings) == 1


# ----- Failures ---------------------------------------------------------------------------------


@pytest.mark.parametrize("content", ["", "{", '{"format": "text3d-studio-project", "version": 1, "sce'])
def test_corrupt_or_truncated_json_raises(qapp, tmp_path, content):
    path = tmp_path / "broken.t3d"
    path.write_text(content)
    with pytest.raises(ValueError, match="not a valid project"):
        load_project(str(path))


def test_binary_file_raises(qapp, tmp_path):
    path = tmp_path / "photo.t3d"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x00")
    with pytest.raises(ValueError):
        load_project(str(path))


@pytest.mark.parametrize("data", [[1, 2], {"scene": {}}, {"format": "something-else", "version": 1}])
def test_wrong_format_raises(qapp, tmp_path, data):
    with pytest.raises(ValueError, match="not a Text3D Studio project"):
        load_project(write_json(tmp_path / "other.t3d", data))


def test_newer_version_raises(qapp, tmp_path):
    path = write_json(tmp_path / "future.t3d", project_dict(version=FORMAT_VERSION + 1))
    with pytest.raises(ValueError, match="newer"):
        load_project(path)


@pytest.mark.parametrize("version", [None, 0, "1", 1.5])
def test_invalid_version_raises(qapp, tmp_path, version):
    with pytest.raises(ValueError, match="version"):
        load_project(write_json(tmp_path / "odd.t3d", project_dict(version=version)))


def test_damaged_scene_structure_raises(qapp, tmp_path):
    path = write_json(tmp_path / "bad.t3d", project_dict(scene={"objects": [5], "light": "bright"}))
    with pytest.raises(ValueError, match="damaged scene"):
        load_project(path)


def test_missing_project_file_raises_oserror(qapp, tmp_path):
    with pytest.raises(OSError):
        load_project(str(tmp_path / "nowhere.t3d"))


def test_failed_save_leaves_the_old_file_intact(qapp, tmp_path):
    path = tmp_path / "scene.t3d"
    save_project(str(path), Document())
    before = path.read_bytes()
    broken = Document(scene=busy_scene())
    broken.scene.objects[0].transform.x = object()  # not JSON-serializable: fails part-way through writing
    with pytest.raises(TypeError):
        save_project(str(path), broken)
    assert path.read_bytes() == before
    assert os.listdir(tmp_path) == ["scene.t3d"]  # no temp file left behind


def test_saving_into_a_missing_folder_raises(qapp, tmp_path):
    with pytest.raises(OSError):
        save_project(str(tmp_path / "missing" / "scene.t3d"), Document())


# ----- Tolerance --------------------------------------------------------------------------------


def test_unknown_fields_are_ignored(qapp, tmp_path):
    scene = {
        "future_section": {"a": 1},
        "camera": {"fov": 30.0, "lens_shift": 2.0},
        "objects": [{"id": 4, "text": {"content": "X", "kerning": 3}, "sticker": True}],
    }
    path = write_json(tmp_path / "scene.t3d", project_dict(scene=scene, extra="ignored"))
    loaded = load_project(path).document.scene
    assert loaded.camera.fov == 30.0
    assert loaded.objects[0].id == 4 and loaded.objects[0].text.content == "X"


def test_missing_fields_take_defaults(qapp, tmp_path):
    scene = {"light": {"azimuth": 10.0}, "objects": [{"id": 2, "transform": {"x": 4.0}}]}
    loaded = load_project(write_json(tmp_path / "scene.t3d", project_dict(scene=scene))).document.scene
    assert loaded.light == dataclasses.replace(model.LightParams(), azimuth=10.0)
    assert loaded.objects[0].transform == TransformParams(x=4.0)
    assert loaded.objects[0].text == TextParams()
    assert loaded.ground == model.GroundParams() and loaded.shadow == model.ShadowParams()


def test_missing_scene_and_photo_give_the_default_document(qapp, tmp_path):
    path = write_json(tmp_path / "bare.t3d", {"format": "text3d-studio-project", "version": 1})
    loaded = load_project(path)
    assert loaded.document.scene == Scene()
    assert loaded.document.background is None and loaded.warnings == []


def test_empty_object_list_falls_back_to_one_default_object(qapp, tmp_path):
    path = write_json(tmp_path / "scene.t3d", project_dict(scene={"objects": []}))
    assert load_project(path).document.scene.objects == [TextObject()]


def test_duplicate_or_missing_ids_are_renumbered(qapp, tmp_path):
    objects = [{"text": {"content": "A"}}, {"text": {"content": "B"}}, {"id": 1, "text": {"content": "C"}}]
    scene = load_project(write_json(tmp_path / "scene.t3d", project_dict(scene={"objects": objects}))).document.scene
    assert [o.id for o in scene.objects] == [0, 1, 2]
    assert [o.text.content for o in scene.objects] == ["A", "B", "C"]


# ----- Photo path hardening ---------------------------------------------------------------------


@pytest.mark.parametrize("path, windows, allowed", [
    # Windows: drive-letter paths only
    ("C:\\Users\\me\\p.png", True, True),
    ("d:/photos/p.png", True, True),
    ("\\\\host\\share\\p.png", True, False),
    ("//host/share/p.png", True, False),
    ("/\\host\\share\\p.png", True, False),
    ("\\\\?\\UNC\\host\\share\\p.png", True, False),
    ("\\\\?\\C:\\photos\\p.png", True, False),
    ("\\\\.\\C:\\photos\\p.png", True, False),
    ("\\??\\C:\\photos\\p.png", True, False),
    ("\\photos\\p.png", True, False),
    ("C:p.png", True, False),
    ("photos\\p.png", True, False),
    # POSIX: absolute local paths
    ("/Users/me/p.png", False, True),
    ("/Volumes/Backup/p.png", False, True),
    ("/network-photos/p.png", False, True),
    ("//host/share/p.png", False, False),
    ("/net/host/p.png", False, False),
    ("/NET/host/p.png", False, False),
    ("/Network/Servers/host/p.png", False, False),
    ("/network/servers/p.png", False, False),
    ("\\\\host\\share\\p.png", False, False),
    ("photos/p.png", False, False),
])
def test_photo_path_allowlist(path, windows, allowed):
    assert is_allowed_photo_path(path, windows=windows) is allowed


@pytest.mark.parametrize("network", [
    "\\\\host\\share\\p.png", "//host/share/p.png", "\\\\?\\UNC\\host\\share\\p.png",
    "/net/host/p.png", "/Network/Servers/host/p.png", "C:\\photos\\p.png",
])
@pytest.mark.parametrize("field", ["relative_path", "absolute_path"])
def test_network_photo_paths_are_rejected_without_touching_them(qapp, tmp_path, monkeypatch, network, field):
    from text3d_studio import project_io

    touched = []
    real_isfile = os.path.isfile
    monkeypatch.setattr(project_io.os.path, "isfile", lambda p: touched.append(p) or real_isfile(p))
    monkeypatch.setattr(project_io, "load_image", lambda p: touched.append(p))
    path = write_json(tmp_path / "scene.t3d", project_dict(photo={field: network}))
    loaded = load_project(path)
    assert touched == []
    assert loaded.document.background is None and loaded.missing_photo == network
    assert "not a local path" in loaded.warnings[0]


def test_non_image_photo_path_is_rejected(qapp, tmp_path):
    secret = tmp_path / "notes.txt"
    secret.write_text("private")
    entry = {"relative_path": "notes.txt", "absolute_path": str(secret)}
    loaded = load_project(write_json(tmp_path / "scene.t3d", project_dict(photo=entry)))
    assert loaded.document.background is None
    assert "not a supported image type" in loaded.warnings[0]


def test_image_extension_check_ignores_case(qapp, tmp_path):
    photo_file(tmp_path, "PHOTO.PNG")
    loaded = load_project(write_json(tmp_path / "scene.t3d", project_dict(photo={"relative_path": "PHOTO.PNG"})))
    assert loaded.warnings == [] and loaded.document.background is not None


def test_rejected_relative_path_still_falls_back_to_a_safe_absolute_one(qapp, tmp_path):
    photo = photo_file(tmp_path)
    entry = {"relative_path": "//host/share/p.png", "absolute_path": photo}
    loaded = load_project(write_json(tmp_path / "scene.t3d", project_dict(photo=entry)))
    assert loaded.warnings == [] and loaded.document.background_path == photo


def test_relative_photo_outside_the_project_folder_loads(qapp, tmp_path):
    (tmp_path / "photos").mkdir()
    (tmp_path / "projects").mkdir()
    photo_file(tmp_path / "photos", "p.png")
    entry = {"relative_path": os.path.join("..", "photos", "p.png")}
    loaded = load_project(write_json(tmp_path / "projects" / "scene.t3d", project_dict(photo=entry)))
    assert loaded.warnings == []
    assert loaded.document.background_path == str(tmp_path / "photos" / "p.png")


def test_wall_and_mounts_round_trip_and_default_when_missing(qapp, tmp_path):
    path = str(tmp_path / "scene.t3d")
    save_project(path, Document(scene=busy_scene()))
    scene = load_project(path).document.scene
    assert (scene.wall.angle, scene.wall.offset) == (-35.0, 7.5)
    assert [o.transform.mount for o in scene.objects] == ["floor", "wall"]
    assert (scene.objects[1].transform.wall_x, scene.objects[1].transform.wall_y) == (-2.5, 1.25)
    older = load_project(write_json(tmp_path / "old.t3d", project_dict(scene={"objects": [{"transform": {"x": 1.0}}]})))
    assert older.document.scene.wall == model.WallParams()
    assert older.document.scene.objects[0].transform.mount == "floor"


@pytest.mark.parametrize("wall, mounts, expected", [
    ({"angle": 5.0}, ["floor"], False),  # saved before walls caught shadows: no wall text, no wall
    ({"angle": 5.0}, ["floor", "wall"], True),  # wall text means the photo showed a wall
    ({"enabled": False}, ["wall"], False),
    ({"enabled": True}, ["floor"], True),
    ({"enabled": "yes"}, ["floor"], False),  # not a bool: worked out from the mounts
    ({"enabled": 1}, ["wall"], True),
])
def test_whether_the_photo_shows_a_wall_loads_or_is_worked_out(qapp, tmp_path, wall, mounts, expected):
    objects = [{"id": i, "transform": {"mount": mount}} for i, mount in enumerate(mounts)]
    path = write_json(tmp_path / "wall.t3d", project_dict(scene={"wall": wall, "objects": objects}))
    assert load_project(path).document.scene.wall.enabled is expected


# ----- Occlusion mask ---------------------------------------------------------------------------


def soft_mask(height=30, width=50):
    mask = np.zeros((height, width), np.uint8)
    mask[5:20, 10:30] = 255
    mask[5:20, 30:35] = np.linspace(200, 20, 5).astype(np.uint8)  # a feathered edge
    return mask


@pytest.mark.parametrize("embed_photo", [False, True])
def test_mask_round_trips_pixel_exact_as_an_embedded_png(qapp, tmp_path, embed_photo):
    document = Document(background=photo_pixels(), background_path=photo_file(tmp_path))
    document.mask = soft_mask()
    path = str(tmp_path / "masked.t3d")
    save_project(path, document, embed_photo=embed_photo)
    entry = json.loads(open(path).read())["mask"]
    assert set(entry) == {"embedded_png"}
    loaded = load_project(path)
    assert loaded.warnings == []
    assert np.array_equal(loaded.document.mask, document.mask)


def test_no_mask_is_saved_as_null_and_loads_as_none(qapp, tmp_path):
    path = str(tmp_path / "plain.t3d")
    save_project(path, Document(background=photo_pixels()))
    assert json.loads(open(path).read())["mask"] is None
    assert load_project(path).document.mask is None
    older = load_project(write_json(tmp_path / "old.t3d", project_dict()))  # saved before masks existed
    assert older.document.mask is None and older.warnings == []


@pytest.mark.parametrize("entry", [{"embedded_png": "not base64!"}, {"embedded_png": "aGVsbG8="}, {"other": 1}, "png"])
def test_a_damaged_mask_is_left_out_with_a_warning(qapp, tmp_path, entry):
    loaded = load_project(write_json(tmp_path / "bad.t3d", project_dict(mask=entry, photo=embedded_photo())))
    assert loaded.document.background is not None
    assert loaded.document.mask is None
    assert any("mask" in warning and "could not be read" in warning for warning in loaded.warnings)


def embedded_photo():
    from text3d_studio.project_io import _encode_png

    return {"embedded_png": _encode_png(photo_pixels())}


def png_header_only(width, height):
    """A PNG that claims to be width x height but holds no pixel data: a crafted file's way to ask for a huge
    allocation. Only its header may be read."""
    import base64
    import struct
    import zlib

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    chunk = b"IHDR" + ihdr
    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(ihdr)) + chunk + struct.pack(">I", zlib.crc32(chunk))
    return {"embedded_png": base64.b64encode(png).decode("ascii")}


@pytest.mark.parametrize("width, height", [(100000, 100000), (2**31 - 1, 1), (50, 30)])
def test_an_oversized_or_empty_mask_is_refused_from_its_header(qapp, tmp_path, width, height):
    loaded = load_project(write_json(tmp_path / "huge.t3d", project_dict(mask=png_header_only(width, height),
                                                                          photo=embedded_photo())))
    assert loaded.document.mask is None and loaded.document.background is not None
    assert any("mask" in warning for warning in loaded.warnings)


def test_a_mask_that_does_not_fit_the_photo_is_left_out_with_a_warning(qapp, tmp_path):
    document = Document(background=photo_pixels())
    document.mask = soft_mask(height=40, width=40)
    path = str(tmp_path / "mismatch.t3d")
    save_project(path, document, embed_photo=True)
    loaded = load_project(path)
    assert loaded.document.mask is None and loaded.document.background is not None
    assert any("size" in warning for warning in loaded.warnings)


def test_a_mask_whose_photo_is_missing_or_rejected_is_left_out_with_a_warning(qapp, tmp_path):
    document = Document(background=photo_pixels(), background_path=photo_file(tmp_path))
    document.mask = soft_mask()
    path = str(tmp_path / "masked.t3d")
    save_project(path, document)
    os.remove(document.background_path)
    loaded = load_project(path)
    assert loaded.missing_photo is not None
    assert loaded.document.mask is None
    assert any("mask" in warning for warning in loaded.warnings)
    no_photo = load_project(write_json(tmp_path / "nophoto.t3d", project_dict(mask={"embedded_png": "aGVsbG8="})))
    assert no_photo.document.mask is None and any("mask" in warning for warning in no_photo.warnings)


@pytest.mark.parametrize("value, expected", [
    (1e9, 0.5), (-3.0, 0.005), (float("nan"), model.MaskParams().brush_size), ("big", model.MaskParams().brush_size),
    (True, model.MaskParams().brush_size), (0.2, 0.2),
])
def test_brush_values_are_clamped_or_defaulted_on_load(qapp, tmp_path, value, expected):
    path = tmp_path / "brush.t3d"
    path.write_text(json.dumps(project_dict(scene={"mask": {"brush_size": value, "feather": 7}})))
    brush = load_project(str(path)).document.scene.mask
    assert brush.brush_size == expected
    assert brush.feather == 1.0


SECTION_NAMES = {type(getattr(Scene(), f.name)): f.name for f in dataclasses.fields(Scene) if f.name != "objects"}


@pytest.mark.parametrize("cls", list(model.FIELD_RANGES))
@pytest.mark.parametrize("value, end", [(1e12, 1), (-1e12, 0), ("NaN", None), ("x", None), (None, None), ([1], None)])
def test_every_ranged_field_loads_clamped_or_defaulted(qapp, tmp_path, cls, value, end):
    section = SECTION_NAMES[cls]
    ranges = model.FIELD_RANGES[cls]
    text = json.dumps(project_dict(scene={section: {name: value for name in ranges}}))
    path = tmp_path / "ranged.t3d"
    path.write_text(text.replace('"NaN"', "NaN"))
    params = getattr(load_project(str(path)).document.scene, section)
    for name, bounds in ranges.items():
        expected = getattr(cls(), name) if end is None else bounds[end]
        assert getattr(params, name) == expected, name


def test_every_ranged_field_has_a_default_inside_its_range():
    for cls, ranges in model.FIELD_RANGES.items():
        for name, (low, high) in ranges.items():
            assert low <= getattr(cls(), name) <= high, (cls.__name__, name)


def test_side_material_round_trips_and_an_unknown_one_loads_as_same(qapp, tmp_path):
    scene = Scene(objects=[TextObject(id=0), TextObject(id=1)])
    scene.objects[0].material.side_kind = "stone"
    path = str(tmp_path / "sides.t3d")
    save_project(path, Document(scene=scene))
    assert [o.material.side_kind for o in load_project(path).document.scene.objects] == ["stone", "same"]
    for value in ("cheese", 7, None, ["wood"]):
        crafted = project_dict(scene={"objects": [{"material": {"kind": "metal", "side_kind": value}}]})
        loaded = load_project(write_json(tmp_path / "crafted.t3d", crafted))
        assert loaded.document.scene.objects[0].material.side_kind == "same", value


def test_outline_round_trips_and_crafted_values_are_clamped(qapp, tmp_path):
    scene = Scene()
    scene.objects[0].material = dataclasses.replace(
        scene.objects[0].material, outline_width=0.6, outline_color=(0.2, 0.9, 0.4), outline_metallic=1.0
    )
    path = str(tmp_path / "outline.t3d")
    save_project(path, Document(scene=scene))
    assert load_project(path).document.scene.objects[0].material == scene.objects[0].material
    crafted = project_dict(scene={"objects": [{"material": {
        "outline_width": 1e300, "outline_metallic": -4, "outline_color": [3.0, -1.0, 0.5]}}]})
    material = load_project(write_json(tmp_path / "crafted.t3d", crafted)).document.scene.objects[0].material
    assert (material.outline_width, material.outline_metallic, material.outline_color) == (1.0, 0.0, (1.0, 0.0, 0.5))
    crafted = project_dict(scene={"objects": [{"material": {"outline_width": "wide", "outline_color": [1, 2]}}]})
    material = load_project(write_json(tmp_path / "crafted.t3d", crafted)).document.scene.objects[0].material
    assert (material.outline_width, material.outline_color) == (0.0, model.MaterialParams().outline_color)


def test_neon_glow_round_trips_and_crafted_values_are_clamped(qapp, tmp_path):
    scene = Scene()
    scene.objects[0].material = dataclasses.replace(
        scene.objects[0].material, kind="neon", emission_color=(0.1, 1.0, 0.3), emission_strength=2.5
    )
    path = str(tmp_path / "neon.t3d")
    save_project(path, Document(scene=scene))
    assert load_project(path).document.scene.objects[0].material == scene.objects[0].material
    crafted = project_dict(scene={"objects": [{"material": {"kind": "neon", "emission_strength": 1e9,
                                                             "emission_color": [0.5, 9, -3]}}]})
    material = load_project(write_json(tmp_path / "crafted.t3d", crafted)).document.scene.objects[0].material
    assert (material.emission_strength, material.emission_color) == (5.0, (0.5, 1.0, 0.0))
