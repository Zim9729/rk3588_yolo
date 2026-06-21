from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import yaml


@dataclass(frozen=True)
class YoloConfig:
    img_size: int
    labels_path: Path
    conf_threshold: float
    nms_threshold: float
    input_format: str


def load_labels(path: Union[Path, str]) -> List[str]:
    labels_path = Path(path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if not labels:
        raise ValueError(f"No labels found in: {labels_path}")
    return labels


def load_config(path: Union[Path, str]) -> YoloConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = ["img_size", "labels", "conf_threshold", "nms_threshold", "input_format"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing config keys in {config_path}: {', '.join(missing)}")

    labels_path = Path(data["labels"])
    if not labels_path.is_absolute():
        labels_path = (config_path.parent / labels_path).resolve()

    img_size = int(data["img_size"])
    conf_threshold = float(data["conf_threshold"])
    nms_threshold = float(data["nms_threshold"])
    input_format = str(data["input_format"]).lower()

    if img_size <= 0:
        raise ValueError("img_size must be positive")
    if not 0 <= conf_threshold <= 1:
        raise ValueError("conf_threshold must be between 0 and 1")
    if not 0 <= nms_threshold <= 1:
        raise ValueError("nms_threshold must be between 0 and 1")
    if input_format != "nhwc":
        raise ValueError("Only nhwc input_format is supported")

    load_labels(labels_path)
    return YoloConfig(
        img_size=img_size,
        labels_path=labels_path,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        input_format=input_format,
    )
