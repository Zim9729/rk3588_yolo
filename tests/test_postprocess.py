import numpy as np
import pytest

from src.yolo11.postprocess import Detection, nms, postprocess_outputs
from src.yolo11.preprocess import LetterboxMeta


def test_nms_keeps_highest_score_for_overlapping_boxes():
    detections = [
        Detection(0, 0.90, (10, 10, 100, 100)),
        Detection(0, 0.80, (12, 12, 102, 102)),
    ]

    kept = nms(detections, 0.45)

    assert len(kept) == 1
    assert kept[0].score == 0.90


def test_nms_is_class_aware():
    detections = [
        Detection(0, 0.90, (10, 10, 100, 100)),
        Detection(1, 0.80, (12, 12, 102, 102)),
    ]

    kept = nms(detections, 0.45)

    assert len(kept) == 2


def test_postprocess_accepts_ultralytics_shape():
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    output[0, 0:4, 0] = [320, 320, 100, 100]
    output[0, 4, 0] = 0.90
    meta = LetterboxMeta((640, 640), (640, 640), 1.0, 0, 0)

    detections = postprocess_outputs([output], meta, 0.25, 0.45)

    assert len(detections) == 1
    assert detections[0].class_id == 0
    assert detections[0].box == pytest.approx((270, 270, 370, 370))


def test_postprocess_accepts_row_major_shape():
    output = np.zeros((1, 8400, 84), dtype=np.float32)
    output[0, 0, 0:4] = [320, 320, 100, 100]
    output[0, 0, 4] = 0.90
    meta = LetterboxMeta((640, 640), (640, 640), 1.0, 0, 0)

    detections = postprocess_outputs([output], meta, 0.25, 0.45)

    assert len(detections) == 1
    assert detections[0].class_id == 0
    assert detections[0].box == pytest.approx((270, 270, 370, 370))


def test_postprocess_unletterboxes_to_original_shape():
    output = np.zeros((1, 1, 84), dtype=np.float32)
    output[0, 0, 0:4] = [320, 320, 100, 100]
    output[0, 0, 4] = 0.90
    meta = LetterboxMeta((480, 640), (640, 640), 1.0, 0, 80)

    detections = postprocess_outputs([output], meta, 0.25, 0.45)

    assert detections[0].box == pytest.approx((270, 190, 370, 290))


def test_postprocess_rescales_explicit_normalized_coordinates():
    output = np.zeros((1, 5, 6), dtype=np.float32)
    output[0, 0:4, 0] = [0.5, 0.5, 0.25, 0.25]
    output[0, 4, 0] = 0.90
    meta = LetterboxMeta((640, 640), (640, 640), 1.0, 0, 0)

    detections = postprocess_outputs([output], meta, 0.25, 0.45, "normalized")

    assert detections[0].box == pytest.approx((240, 240, 400, 400))


def test_postprocess_does_not_guess_when_pixel_coordinates_are_explicit():
    output = np.zeros((1, 5, 6), dtype=np.float32)
    output[0, 0:4, 0] = [1, 1, 1, 1]
    output[0, 4, 0] = 0.90
    meta = LetterboxMeta((640, 640), (640, 640), 1.0, 0, 0)

    detections = postprocess_outputs([output], meta, 0.25, 0.45, "pixels")

    assert detections[0].box == pytest.approx((0.5, 0.5, 1.5, 1.5))


def test_postprocess_rejects_unsupported_shape():
    output = np.zeros((1, 2, 3, 4), dtype=np.float32)
    meta = LetterboxMeta((640, 640), (640, 640), 1.0, 0, 0)

    with pytest.raises(ValueError, match="Unsupported YOLO output shape"):
        postprocess_outputs([output], meta, 0.25, 0.45)
