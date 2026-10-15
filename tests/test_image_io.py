import numpy as np
import pytest

from text3d_studio.image_io import load_image, save_image


def gradient_image(width=64, height=40):
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    pixels[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    pixels[..., 3] = 255
    return pixels


def test_png_roundtrip_is_lossless(qapp, tmp_path):
    pixels = gradient_image()
    path = str(tmp_path / "a.png")
    save_image(pixels, path)
    np.testing.assert_array_equal(load_image(path), pixels)


def test_jpg_roundtrip_preserves_size_and_is_close(qapp, tmp_path):
    pixels = gradient_image(width=81, height=33)  # odd sizes exercise row padding
    path = str(tmp_path / "a.jpg")
    save_image(pixels, path, quality=95)
    loaded = load_image(path)
    assert loaded.shape == pixels.shape
    assert np.abs(loaded[..., :3].astype(int) - pixels[..., :3]).mean() < 4


def test_load_rejects_non_image_file(qapp, tmp_path):
    path = tmp_path / "bad.png"
    path.write_text("not an image")
    with pytest.raises(ValueError):
        load_image(str(path))


def test_save_to_missing_directory_raises(qapp, tmp_path):
    with pytest.raises(OSError):
        save_image(gradient_image(), str(tmp_path / "missing" / "out.png"))
