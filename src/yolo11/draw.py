from typing import Sequence

import cv2
import numpy as np

from src.yolo11.postprocess import Detection


def draw_detections(image: np.ndarray, detections: Sequence[Detection], labels: Sequence[str]) -> np.ndarray:
    if image is None or image.ndim != 3:
        raise ValueError("image must be an HWC image")

    output = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.box]
        class_name = labels[detection.class_id] if 0 <= detection.class_id < len(labels) else str(detection.class_id)
        text = f"{class_name} {detection.score:.2f}"
        color = _color_for_class(detection.class_id)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_y = max(y1, text_size[1] + baseline)
        cv2.rectangle(
            output,
            (x1, text_y - text_size[1] - baseline),
            (x1 + text_size[0], text_y + baseline),
            color,
            -1,
        )
        cv2.putText(output, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _color_for_class(class_id: int):
    palette = (
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
    )
    return palette[class_id % len(palette)]
