from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxMeta:
    original_shape: Tuple[int, int]
    input_shape: Tuple[int, int]
    scale: float
    pad_x: int
    pad_y: int


def letterbox(image: np.ndarray, size: int, color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, LetterboxMeta]:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HWC BGR image")
    if size <= 0:
        raise ValueError("size must be positive")

    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=image.dtype)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized

    meta = LetterboxMeta(
        original_shape=(height, width),
        input_shape=(size, size),
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return canvas, meta


def preprocess_image(image: np.ndarray, size: int) -> Tuple[np.ndarray, LetterboxMeta]:
    boxed, meta = letterbox(image, size)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)
    return tensor, meta
