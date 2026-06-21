import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yolo11.config import load_config, load_labels
from src.yolo11.draw import draw_detections
from src.yolo11.postprocess import postprocess_outputs
from src.yolo11.preprocess import preprocess_image
from src.yolo11.rknn_infer import RknnLiteDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO11 RKNN image inference on RK3588.")
    parser.add_argument("--model", default="models/yolo11n.rknn", help="Path to RKNN model.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--config", default="configs/coco.yaml", help="YAML config path.")
    parser.add_argument("--output", default="outputs/result.jpg", help="Output image path.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override.")
    parser.add_argument("--nms", type=float, default=None, help="NMS threshold override.")
    return parser.parse_args()


def run_image_demo(model_path: Path, image_path: Path, config_path: Path, output_path: Path, conf_threshold=None, nms_threshold=None) -> int:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    config = load_config(config_path)
    labels = load_labels(config.labels_path)
    conf = config.conf_threshold if conf_threshold is None else conf_threshold
    nms = config.nms_threshold if nms_threshold is None else nms_threshold

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    input_tensor, meta = preprocess_image(image, config.img_size)
    with RknnLiteDetector(model_path) as detector:
        outputs = detector.infer(input_tensor)

    detections = postprocess_outputs(outputs, meta, conf, nms)
    result = draw_detections(image, detections, labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Failed to write output image: {output_path}")

    print(f"Detections: {len(detections)}")
    print(f"Output: {output_path}")
    return len(detections)


def main() -> int:
    args = parse_args()
    run_image_demo(
        model_path=Path(args.model),
        image_path=Path(args.image),
        config_path=Path(args.config),
        output_path=Path(args.output),
        conf_threshold=args.conf,
        nms_threshold=args.nms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
