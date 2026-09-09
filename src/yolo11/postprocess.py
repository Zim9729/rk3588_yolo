from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from src.yolo11.preprocess import LetterboxMeta


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    box: Tuple[float, float, float, float]


def xywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def nms(detections: Iterable[Detection], threshold: float) -> List[Detection]:
    remaining = sorted(detections, key=lambda det: det.score, reverse=True)
    kept = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            det for det in remaining
            if det.class_id != current.class_id or iou(det.box, current.box) <= threshold
        ]
    return kept


def postprocess_outputs(outputs: Sequence[np.ndarray], meta: LetterboxMeta, conf_threshold: float, nms_threshold: float) -> List[Detection]:
    if not outputs:
        return []

    predictions = _normalize_output(outputs[0])
    detections = []
    for row in predictions:
        if row.shape[0] < 5:
            continue
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if score < conf_threshold:
            continue
        box = _unletterbox(xywh_to_xyxy(row[0:4]), meta)
        detections.append(Detection(class_id, score, box))

    return nms(detections, nms_threshold)


def _normalize_output(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Unsupported YOLO output shape: {tuple(output.shape)}")

    if array.shape[0] >= 5 and array.shape[0] < array.shape[1]:
        array = array.T
    if array.shape[1] < 5:
        raise ValueError(f"Unsupported YOLO output shape: {tuple(output.shape)}")
    return array.astype(np.float32, copy=False)


def _unletterbox(box: Sequence[float], meta: LetterboxMeta) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = (x1 - meta.pad_x) / meta.scale
    y1 = (y1 - meta.pad_y) / meta.scale
    x2 = (x2 - meta.pad_x) / meta.scale
    y2 = (y2 - meta.pad_y) / meta.scale

    height, width = meta.original_shape
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return x1, y1, x2, y2
