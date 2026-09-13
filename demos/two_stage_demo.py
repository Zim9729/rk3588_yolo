import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yolo11.config import load_config, load_labels
from src.yolo11.postprocess import Detection, postprocess_outputs
from src.yolo11.preprocess import preprocess_image
from src.yolo11.rknn_infer import RknnLiteDetector

PART_NAMES = {"tb": "碳滑板", "yj": "羊角", "yw": "异物"}
# 与 JC24 toml 中 stage2_overlap_threshold 一致：yw 框落入部件区域面积占比阈值
OVERLAP_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage pantograph foreign-object detection (fp16 RKNN).")
    parser.add_argument("--image", default="3C_Test.jpg", help="Input image path.")
    parser.add_argument("--stage1-model", default="models_arm/3C_stage1_fp16.rknn")
    parser.add_argument("--stage2-model", default="models_arm/3C_stage2_fp16.rknn")
    parser.add_argument("--stage2-config", default="configs/stage2.yaml")
    parser.add_argument("--output", default="outputs/two_stage_result.jpg")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold for both stages.")
    parser.add_argument("--overlap", type=float, default=OVERLAP_THRESHOLD,
                        help="IoB threshold for attributing yw to a part.")
    return parser.parse_args()


def iob(inner_box, outer_box) -> float:
    """Intersection over the inner (foreign body) box area."""
    ix1 = max(inner_box[0], outer_box[0])
    iy1 = max(inner_box[1], outer_box[1])
    ix2 = min(inner_box[2], outer_box[2])
    iy2 = min(inner_box[3], outer_box[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, inner_box[2] - inner_box[0]) * max(0.0, inner_box[3] - inner_box[1])
    return inter / area if area > 0 else 0.0


def run_stage(detector, image, img_size, conf, nms) -> list[Detection]:
    tensor, meta = preprocess_image(image, img_size)
    outputs = detector.infer(tensor)
    return postprocess_outputs(outputs, meta, conf, nms, "pixels")


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    stage2_config = load_config(args.stage2_config)
    stage2_labels = load_labels(stage2_config.labels_path)
    conf = stage2_config.conf_threshold if args.conf is None else args.conf
    nms = stage2_config.nms_threshold
    img_size = stage2_config.img_size

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    img_h, img_w = image.shape[:2]

    results = []  # (box, text, color) in full-image coords
    canvas = image.copy()

    with RknnLiteDetector(Path(args.stage1_model)) as det1, \
         RknnLiteDetector(Path(args.stage2_model)) as det2:

        stage1_dets = run_stage(det1, image, img_size, conf, nms)
        print(f"[stage1] {len(stage1_dets)} Pantograph_Area:")
        for det in stage1_dets:
            x1, y1, x2, y2 = det.box
            print(f"  Pantograph_Area conf={det.score:.3f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
            results.append((det.box, f"Pantograph_Area {det.score:.2f}", (0, 200, 0)))

        for area_idx, area in enumerate(stage1_dets):
            ax1, ay1, ax2, ay2 = [int(round(v)) for v in area.box]
            ax1, ay1 = max(0, ax1), max(0, ay1)
            ax2, ay2 = min(img_w, ax2), min(img_h, ay2)
            crop = image[ay1:ay2, ax1:ax2]
            if crop.size == 0:
                continue

            stage2_dets = run_stage(det2, crop, img_size, conf, nms)
            # 映射回原图坐标
            dets = [
                Detection(d.class_id, d.score,
                          (d.box[0] + ax1, d.box[1] + ay1, d.box[2] + ax1, d.box[3] + ay1))
                for d in stage2_dets
            ]
            parts = [d for d in dets if stage2_labels[d.class_id] != "yw"]
            foreigns = [d for d in dets if stage2_labels[d.class_id] == "yw"]

            print(f"[stage2] area#{area_idx}: {len(parts)} parts, {len(foreigns)} yw raw")
            for det in parts:
                name = stage2_labels[det.class_id]
                x1, y1, x2, y2 = det.box
                print(f"  {name}({PART_NAMES.get(name, name)}) conf={det.score:.3f} "
                      f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                results.append((det.box, f"{name} {det.score:.2f}", (255, 128, 0)))

            for det in foreigns:
                best_part, best_iob = None, 0.0
                for part in parts:
                    overlap = iob(det.box, part.box)
                    if overlap > best_iob:
                        best_iob, best_part = overlap, part
                if best_part is None or best_iob < args.overlap:
                    continue  # 不落在任何部件上的异物不输出
                part_name = stage2_labels[best_part.class_id]
                label = f"{PART_NAMES.get(part_name, part_name)}异物"
                x1, y1, x2, y2 = det.box
                print(f"  {label} (yw->{part_name}, iob={best_iob:.2f}) conf={det.score:.3f} "
                      f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                results.append((det.box, f"{part_name}_yw {det.score:.2f}", (0, 0, 255)))

    for box, text, color in results:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, text, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write output image: {output_path}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
