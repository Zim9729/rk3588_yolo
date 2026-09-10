import argparse
import json
from pathlib import Path
from typing import Optional

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Convert an ONNX model to RKNN for RK3588.")
    parser.add_argument("--onnx", default="models/yolo11n.onnx", help="Input ONNX model path.")
    parser.add_argument("--output", default="models/yolo11n.rknn", help="Output RKNN model path.")
    parser.add_argument("--target", default="rk3588", help="RKNN target platform.")
    parser.add_argument("--dataset", default=None, help="Calibration dataset text file for quantization.")
    parser.add_argument("--quantized", action="store_true", help="Enable quantized RKNN build.")
    parser.add_argument("--quantized-dtype", default="w8a8", choices=["w8a8"], help="Quantized dtype exposed by this script (INT8 weights and activations).")
    parser.add_argument("--quantized-method", default="channel", choices=["layer", "channel"], help="Quantization granularity: layer (per-tensor weights) or channel (per-channel weights).")
    parser.add_argument("--quantized-algorithm", default="normal", choices=["normal", "mmse", "kl_divergence"], help="RKNN calibration algorithm.")
    parser.add_argument("--auto-hybrid", action="store_true", help="Enable RKNN automatic hybrid quantization for sensitive layers.")
    return parser.parse_args()


def _check_ret(ret: int, step: str) -> None:
    if ret != 0:
        raise RuntimeError(f"RKNN {step} failed with code: {ret}")


def _validate_conversion_paths(onnx_path: Path, output_path: Path) -> None:
    if output_path.suffix.lower() != ".rknn":
        raise ValueError(f"RKNN output path must end in .rknn: {output_path}")
    if onnx_path.resolve() == output_path.resolve():
        raise ValueError("RKNN output path must not overwrite the input ONNX model")


def _read_onnx_info(onnx_path: Path) -> dict:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    if len(input_shape) != 4 or input_shape[2] <= 0 or input_shape[3] <= 0:
        raise ValueError(f"RKNN conversion requires a static NCHW ONNX input, got: {input_shape}")
    return {
        "input_shape": input_shape,
        "output_coordinates": metadata.get("rknn_output_coordinates", "pixels"),
    }


def convert_to_rknn(onnx_path: Path, output_path: Path, target: str, dataset: Optional[Path], quantized: bool, quantized_dtype: str = "w8a8", quantized_method: str = "channel", quantized_algorithm: str = "normal", auto_hybrid: bool = False) -> Path:
    _validate_conversion_paths(onnx_path, output_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if auto_hybrid and not quantized:
        raise ValueError("--auto-hybrid requires --quantized")
    if dataset is not None and not quantized:
        raise ValueError("--dataset requires --quantized")
    if quantized and dataset is None:
        raise ValueError("--dataset is required when --quantized is set")
    if quantized and dataset is not None and not dataset.exists():
        raise FileNotFoundError(f"Calibration dataset not found: {dataset}")

    onnx_info = _read_onnx_info(onnx_path)
    if quantized and onnx_info["output_coordinates"] != "normalized":
        raise ValueError(
            "INT8 and hybrid RKNN conversion requires normalized YOLO output coordinates. "
            "Re-export ONNX with scripts/export_yolo11_onnx.py --normalize-coordinates."
        )

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
        precision = "hybrid" if auto_hybrid else "int8" if quantized else "float16"
        runtime_metadata = {
            "precision": precision,
            "output_coordinates": onnx_info["output_coordinates"],
            "input_shape": onnx_info["input_shape"],
        }
        config_kwargs = {
            "target_platform": target,
            "mean_values": [[0, 0, 0]],
            "std_values": [[255, 255, 255]],
            "float_dtype": "float16",
            "custom_string": json.dumps(runtime_metadata, separators=(",", ":")),
        }
        if quantized:
            config_kwargs["quantized_dtype"] = quantized_dtype
            config_kwargs["quantized_method"] = quantized_method
            config_kwargs["quantized_algorithm"] = quantized_algorithm
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
        metadata_path = output_path.with_suffix(f"{output_path.suffix}.yaml")
        metadata_path.write_text(yaml.safe_dump(runtime_metadata, sort_keys=False), encoding="utf-8")
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
        quantized_algorithm=args.quantized_algorithm,
        auto_hybrid=args.auto_hybrid,
    )
    print(f"Exported RKNN model: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
