import pytest

from src.yolo11.rknn_infer import RknnLiteDetector


def test_rknn_lite_detector_rejects_missing_model(tmp_path):
    detector = RknnLiteDetector(tmp_path / "missing.rknn")

    with pytest.raises(FileNotFoundError, match="RKNN model not found"):
        detector.load()


def test_rknn_lite_detector_can_be_released_without_load(tmp_path):
    model = tmp_path / "model.rknn"
    model.write_bytes(b"fake")
    detector = RknnLiteDetector(model)

    detector.release()
