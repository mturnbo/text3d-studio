"""Image loading/saving via QImage (no widget dependency), exchanged as numpy RGBA arrays."""

import numpy as np
from PyQt6.QtGui import QImage

SUPPORTED_LOAD = "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)"


def qimage_to_rgba(image: QImage) -> np.ndarray:
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    width, height = image.width(), image.height()
    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes())
    rows = np.frombuffer(ptr, dtype=np.uint8).reshape(height, image.bytesPerLine())
    return rows[:, : width * 4].reshape(height, width, 4).copy()


def rgba_to_qimage(pixels: np.ndarray) -> QImage:
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    height, width = pixels.shape[:2]
    image = QImage(pixels.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
    return image.copy()  # detach from the numpy buffer


def load_image(path: str) -> np.ndarray:
    image = QImage(path)
    if image.isNull():
        raise ValueError(f"Could not read image: {path}")
    return qimage_to_rgba(image)


def save_image(pixels: np.ndarray, path: str, quality: int = 95) -> None:
    image = rgba_to_qimage(pixels)
    if path.lower().endswith((".jpg", ".jpeg")):
        image = image.convertToFormat(QImage.Format.Format_RGB888)
    if not image.save(path, quality=quality):
        raise OSError(f"Could not write image: {path}")
