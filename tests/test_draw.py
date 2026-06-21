import numpy as np

from src.yolo11.draw import draw_detections
from src.yolo11.postprocess import Detection


def test_draw_detections_returns_modified_copy():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [Detection(0, 0.95, (10, 10, 50, 50))]

    drawn = draw_detections(image, detections, ["person"])

    assert drawn.shape == image.shape
    assert np.array_equal(image, np.zeros((100, 100, 3), dtype=np.uint8))
    assert not np.array_equal(drawn, image)


def test_draw_detections_handles_missing_label():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [Detection(5, 0.95, (10, 10, 50, 50))]

    drawn = draw_detections(image, detections, ["person"])

    assert not np.array_equal(drawn, image)
