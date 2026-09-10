from dataclasses import dataclass
from typing import List, Sequence, Tuple

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


def nms(detections: Sequence[Detection], threshold: float) -> List[Detection]:
    remaining = sorted(detections, key=lambda det: det.score, reverse=True)
    kept: List[Detection] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [
            det for det in remaining
            if det.class_id != current.class_id or iou(det.box, current.box) <= threshold
        ]
    return kept


def postprocess_outputs(outputs: Sequence[np.ndarray], meta: LetterboxMeta, conf_threshold: float, nms_threshold: float, coordinate_format: str = "auto") -> List[Detection]:
    if not outputs:
        return []

    predictions = _normalize_output(outputs[0])
    if predictions.shape[0] == 0:
        return []

    # vectorized: split boxes (xywh) and class scores
    boxes_xywh = predictions[:, :4].copy()
    class_scores = predictions[:, 4:]

    # Ultralytics INT8 RKNN export normalizes box coords to [0,1] to avoid
    # per-tensor quantization zeroing class scores. Detect and rescale.
    if coordinate_format not in {"auto", "pixels", "normalized"}:
        raise ValueError(f"Unsupported coordinate format: {coordinate_format}")
    if coordinate_format == "normalized" or (coordinate_format == "auto" and boxes_xywh.max() <= 2.0):
        boxes_xywh[:, [0, 2]] *= float(meta.input_shape[1])
        boxes_xywh[:, [1, 3]] *= float(meta.input_shape[0])

    # batch argmax: best class per anchor
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_ids)), class_ids]

    # confidence filter (single boolean mask, no Python loop)
    mask = scores >= conf_threshold
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    scores = scores[mask]

    # vectorized xywh -> xyxy
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

    # vectorized unletterbox
    x1 = (x1 - meta.pad_x) / meta.scale
    y1 = (y1 - meta.pad_y) / meta.scale
    x2 = (x2 - meta.pad_x) / meta.scale
    y2 = (y2 - meta.pad_y) / meta.scale

    height, width = meta.original_shape
    x1 = np.clip(x1, 0.0, float(width))
    y1 = np.clip(y1, 0.0, float(height))
    x2 = np.clip(x2, 0.0, float(width))
    y2 = np.clip(y2, 0.0, float(height))

    # build Detection list (only for survivors, typically < 100)
    detections = [
        Detection(int(class_ids[i]), float(scores[i]), (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])))
        for i in range(len(scores))
    ]

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
