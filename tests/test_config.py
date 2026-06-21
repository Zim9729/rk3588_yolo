from pathlib import Path

import pytest

from src.yolo11.config import load_config, load_labels


def test_load_labels_reads_non_empty_labels(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("person\ncar\n", encoding="utf-8")

    assert load_labels(labels) == ["person", "car"]


def test_load_labels_rejects_empty_file(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No labels"):
        load_labels(labels)


def test_load_config_resolves_labels_path():
    config = load_config(Path("configs/coco.yaml"))

    assert config.img_size == 640
    assert config.labels_path.name == "coco80.txt"
    assert config.conf_threshold > 0
    assert config.nms_threshold > 0
    assert config.input_format == "nhwc"
