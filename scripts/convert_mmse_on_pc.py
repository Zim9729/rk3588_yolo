#!/usr/bin/env python3
"""PC 端 mmse 量化脚本 —— 在 Windows/WSL 上运行，生成 mmse 量化的 RKNN 模型。

板端（aarch64）运行 mmse/kl_divergence 会 OOM，需在 PC 端执行。
生成的 .rknn 模型拷贝到板端即可推理。

前置条件（PC 端）：
  uv sync --extra export    # 安装 rknn-toolkit2（x86_64）

用法：
  uv run python scripts/convert_mmse_on_pc.py --onnx models/3C_stage1_int8.onnx --output models/3C_stage1_int8_mmse.rknn --dataset data/calibration/dataset.txt

生成的 models/3C_stage1_int8_mmse.rknn 拷贝到板端后用同样的评测脚本测试。
"""
import argparse
import json
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Convert ONNX to RKNN with mmse quantization (PC-side).")
    p.add_argument("--onnx", required=True, help="Input ONNX with normalized coordinates.")
    p.add_argument("--output", required=True, help="Output RKNN path.")
    p.add_argument("--target", default="rk3588")
    p.add_argument("--dataset", required=True, help="Calibration dataset.txt")
    p.add_argument("--quantized-method", default="channel", choices=["channel", "layer"])
    return p.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    output_path = Path(args.output)
    dataset_path = Path(args.dataset)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    try:
        from rknn.api import RKNN
    except ImportError:
        raise RuntimeError(
            "rknn-toolkit2 not installed. Run on PC: uv sync --extra export"
        )

    # read ONNX metadata
    import onnx
    model = onnx.load(str(onnx_path), load_external_data=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    output_coords = metadata.get("rknn_output_coordinates", "pixels")
    if output_coords != "normalized":
        raise ValueError(
            "mmse quantization requires normalized-coordinate ONNX. "
            "Re-export with --normalize-coordinates."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        runtime_metadata = {
            "precision": "int8_mmse",
            "output_coordinates": output_coords,
            "input_shape": input_shape,
        }
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=args.target,
            quantized_dtype="w8a8",
            quantized_method=args.quantized_method,
            quantized_algorithm="mmse",
            custom_string=json.dumps(runtime_metadata, separators=(",", ":")),
        )
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"load_onnx failed: {ret}")
        print("Building with mmse quantization (this may take 10-30 min on PC)...")
        ret = rknn.build(do_quantization=True, dataset=str(dataset_path))
        if ret != 0:
            raise RuntimeError(f"build failed: {ret}")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"export_rknn failed: {ret}")
        metadata_path = output_path.with_suffix(f"{output_path.suffix}.yaml")
        metadata_path.write_text(yaml.safe_dump(runtime_metadata, sort_keys=False), encoding="utf-8")
        print(f"\nExported: {output_path}")
        print(f"Metadata: {metadata_path}")
        print(f"\n拷贝到板端后运行评测:")
        print(f"  uv run python scripts/eval_precision_loss.py \\")
        print(f"    --models models/3C_stage1_fp16.rknn {output_path.name} \\")
        print(f"    --names FP16 INT8-mmse")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
