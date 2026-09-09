import argparse
import time
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
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs (not counted).")
    parser.add_argument("--runs", type=int, default=10, help="Number of timed runs.")
    return parser.parse_args()


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


def _print_stats(label: str, times: list[float]) -> None:
    avg = sum(times) / len(times)
    print(f"  {label:12s} avg={avg:7.2f} ms  min={min(times):7.2f}  max={max(times):7.2f}")


def run_image_demo(model_path: Path, image_path: Path, config_path: Path, output_path: Path, conf_threshold=None, nms_threshold=None, warmup: int = 3, runs: int = 10) -> int:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    config = load_config(config_path)
    labels = load_labels(config.labels_path)
    conf = config.conf_threshold if conf_threshold is None else conf_threshold
    nms = config.nms_threshold if nms_threshold is None else nms_threshold

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    pre_times, infer_times, post_times, draw_times = [], [], [], []
    detections = []

    with RknnLiteDetector(model_path) as detector:
        total = warmup + runs
        for i in range(total):
            (input_tensor, meta), pre_t = _timed(preprocess_image, image, config.img_size)
            outputs, infer_t = _timed(detector.infer, input_tensor)
            detections, post_t = _timed(postprocess_outputs, outputs, meta, conf, nms)
            _, draw_t = _timed(draw_detections, image, detections, labels)

            if i >= warmup:
                pre_times.append(pre_t)
                infer_times.append(infer_t)
                post_times.append(post_t)
                draw_times.append(draw_t)

    result = draw_detections(image, detections, labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError(f"Failed to write output image: {output_path}")

    total_times = [pre_times[i] + infer_times[i] + post_times[i] + draw_times[i] for i in range(runs)]

    print(f"Detections: {len(detections)}")
    print(f"Output: {output_path}")
    print(f"Timing ({runs} runs, {warmup} warmup):")
    _print_stats("preprocess", pre_times)
    _print_stats("infer", infer_times)
    _print_stats("postprocess", post_times)
    _print_stats("draw", draw_times)
    _print_stats("total", total_times)
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
        warmup=args.warmup,
        runs=args.runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
