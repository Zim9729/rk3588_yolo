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
    parser.add_argument("--quantized-dtype", default="w8a8", choices=["w8a8"], help="Quantized dtype. RK3588 only supports w8a8 (INT8). Use --auto-hybrid for mixed precision.")
    parser.add_argument("--quantized-method", default="channel", choices=["layer", "channel"], help="Quantization granularity: layer (per-layer) or channel (per-channel, higher precision).")
    parser.add_argument("--auto-hybrid", action="store_true", help="Enable auto hybrid quantization (mixed INT8+FP16). Higher accuracy but slower than pure INT8.")
    return parser.parse_args()


def _check_ret(ret: int, step: str) -> None:
    if ret != 0:
        raise RuntimeError(f"RKNN {step} failed with code: {ret}")


def _validate_conversion_paths(onnx_path: Path, output_path: Path) -> None:
    if output_path.suffix.lower() != ".rknn":
        raise ValueError(f"RKNN output path must end in .rknn: {output_path}")
    if onnx_path.resolve() == output_path.resolve():
        raise ValueError("RKNN output path must not overwrite the input ONNX model")


def convert_to_rknn(onnx_path: Path, output_path: Path, target: str, dataset: Optional[Path], quantized: bool, quantized_dtype: str = "w8a8", quantized_method: str = "channel", auto_hybrid: bool = False) -> Path:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    _validate_conversion_paths(onnx_path, output_path)
    if auto_hybrid and not quantized:
        raise ValueError("--auto-hybrid requires --quantized")
    if dataset is not None and not quantized:
        raise ValueError("--dataset requires --quantized")
    if quantized and dataset is None:
        raise ValueError("--dataset is required when --quantized is set")
    if quantized and dataset is not None and not dataset.exists():
        raise FileNotFoundError(f"Calibration dataset not found: {dataset}")

    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise RuntimeError(
            "rknn-toolkit2 is required for ONNX to RKNN conversion. "
            "Run in WSL/Ubuntu: uv sync --extra export"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        config_kwargs = {
            "target_platform": target,
            "mean_values": [[0, 0, 0]],
            "std_values": [[255, 255, 255]],
        }
        if quantized:
            config_kwargs["quantized_dtype"] = quantized_dtype
            config_kwargs["quantized_method"] = quantized_method
        _check_ret(rknn.config(**config_kwargs), "config")
        _check_ret(rknn.load_onnx(model=str(onnx_path)), "load_onnx")
        dataset_arg = str(dataset) if dataset is not None else None
        try:
            ret = rknn.build(do_quantization=quantized, dataset=dataset_arg, auto_hybrid=auto_hybrid)
        except TypeError as exc:
            if auto_hybrid:
                raise RuntimeError("--auto-hybrid requires rknn-toolkit2 2.3.2 or newer") from exc
            raise
        _check_ret(ret, "build")
        _check_ret(rknn.export_rknn(str(output_path)), "export_rknn")
    finally:
        rknn.release()
    return output_path


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset) if args.dataset else None
    output = convert_to_rknn(
        Path(args.onnx),
        Path(args.output),
        args.target,
        dataset,
        args.quantized,
        quantized_dtype=args.quantized_dtype,
        quantized_method=args.quantized_method,
        auto_hybrid=args.auto_hybrid,
    )
    print(f"Exported RKNN model: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
