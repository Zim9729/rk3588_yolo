import argparse
from pathlib import Path
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Convert an ONNX model to RKNN for RK3588.")
    parser.add_argument("--onnx", default="models/yolo11n.onnx", help="Input ONNX model path.")
    parser.add_argument("--output", default="models/yolo11n.rknn", help="Output RKNN model path.")
    parser.add_argument("--target", default="rk3588", help="RKNN target platform.")
    parser.add_argument("--dataset", default=None, help="Calibration dataset text file for quantization.")
    parser.add_argument("--quantized", action="store_true", help="Enable quantized RKNN build.")
    return parser.parse_args()


def _check_ret(ret: int, step: str) -> None:
    if ret != 0:
        raise RuntimeError(f"RKNN {step} failed with code: {ret}")


def convert_to_rknn(onnx_path: Path, output_path: Path, target: str, dataset: Optional[Path], quantized: bool) -> Path:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if quantized and dataset is None:
        raise ValueError("--dataset is required when --quantized is set")
    if quantized and dataset is not None and not dataset.exists():
        raise FileNotFoundError(f"Calibration dataset not found: {dataset}")

    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "rknn-toolkit2 is required for ONNX to RKNN conversion. "
            "Install the Rockchip RKNN Toolkit 2 wheel for your PC environment."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        _check_ret(rknn.config(target_platform=target), "config")
        _check_ret(rknn.load_onnx(model=str(onnx_path)), "load_onnx")
        dataset_arg = str(dataset) if dataset is not None else None
        _check_ret(rknn.build(do_quantization=quantized, dataset=dataset_arg), "build")
        _check_ret(rknn.export_rknn(str(output_path)), "export_rknn")
    finally:
        rknn.release()
    return output_path


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset) if args.dataset else None
    output = convert_to_rknn(Path(args.onnx), Path(args.output), args.target, dataset, args.quantized)
    print(f"Exported RKNN model: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
