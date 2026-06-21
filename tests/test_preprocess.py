import numpy as np

from src.yolo11.preprocess import letterbox, preprocess_image


def test_letterbox_returns_expected_shape():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    resized, meta = letterbox(image, 640)

    assert resized.shape == (640, 640, 3)
    assert meta.original_shape == (480, 640)
    assert meta.input_shape == (640, 640)
    assert meta.scale == 1.0
    assert meta.pad_x == 0
    assert meta.pad_y == 80


def test_preprocess_image_returns_nhwc_float32_batch():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, meta = preprocess_image(image, 640)

    assert tensor.shape == (1, 640, 640, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    assert meta.original_shape == (480, 640)


def test_preprocess_image_converts_bgr_to_rgb():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:, :] = [255, 0, 0]

    tensor, _ = preprocess_image(image, 2)

    assert tensor[0, 0, 0].tolist() == [0.0, 0.0, 1.0]
