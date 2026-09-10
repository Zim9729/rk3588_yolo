import argparse
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export an Ultralytics YOLO11 model to ONNX.")
    parser.add_argument("--model", default="3C/best.pt", help="Input Ultralytics model path or model name.")
    parser.add_argument("--output", default="models/yolo11n.onnx", help="Output ONNX model path.")
    parser.add_argument("--img-size", type=int, default=640, help="Export image size.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--simplify", action="store_true", help="Simplify exported ONNX model.")
    parser.add_argument("--normalize-coordinates", action="store_true", help="Normalize output xywh coordinates for downstream RKNN INT8 or hybrid quantization.")
    return parser.parse_args()


def _validate_export_paths(model_path: str, output_path: Path) -> None:
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"ONNX output path must end in .onnx: {output_path}")
    source_path = Path(model_path)
    if source_path.exists() and source_path.resolve() == output_path.resolve():
        raise ValueError("ONNX output path must not overwrite the input model")


def normalize_output_coordinates(onnx_path: Path, img_size: int) -> None:
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(onnx_path))
    if len(model.graph.output) != 1:
        raise ValueError("Coordinate normalization requires an ONNX model with exactly one output")
    output = model.graph.output[0]
    shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
    if len(shape) != 3 or shape[1] < 5:
        raise ValueError(f"Expected YOLO output shape [batch, channels, anchors], got: {shape}")

    output_name = output.name
    pixel_name = f"{output_name}_pixel_coordinates"
    producer_found = False
    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name == output_name:
                node.output[i] = pixel_name
                producer_found = True
        for i, name in enumerate(node.input):
            if name == output_name:
                node.input[i] = pixel_name
    if not producer_found:
        raise ValueError(f"Could not find the producer of ONNX output: {output_name}")

    prefix = "rknn_coordinate_normalization"
    initializers = [
        helper.make_tensor(f"{prefix}_box_starts", TensorProto.INT64, [1], [0]),
        helper.make_tensor(f"{prefix}_box_ends", TensorProto.INT64, [1], [4]),
        helper.make_tensor(f"{prefix}_score_starts", TensorProto.INT64, [1], [4]),
        helper.make_tensor(f"{prefix}_score_ends", TensorProto.INT64, [1], [np.iinfo(np.int64).max]),
        helper.make_tensor(f"{prefix}_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor(f"{prefix}_steps", TensorProto.INT64, [1], [1]),
        helper.make_tensor(f"{prefix}_scale", TensorProto.FLOAT, [1, 4, 1], [img_size, img_size, img_size, img_size]),
    ]
    boxes = f"{prefix}_boxes"
    normalized_boxes = f"{prefix}_normalized_boxes"
    scores = f"{prefix}_scores"
    model.graph.initializer.extend(initializers)
    model.graph.node.extend(
        [
            helper.make_node("Slice", [pixel_name, f"{prefix}_box_starts", f"{prefix}_box_ends", f"{prefix}_axes", f"{prefix}_steps"], [boxes]),
            helper.make_node("Div", [boxes, f"{prefix}_scale"], [normalized_boxes]),
            helper.make_node("Slice", [pixel_name, f"{prefix}_score_starts", f"{prefix}_score_ends", f"{prefix}_axes", f"{prefix}_steps"], [scores]),
            helper.make_node("Concat", [normalized_boxes, scores], [output_name], axis=1),
        ]
    )

    metadata = {item.key: item for item in model.metadata_props}
    for key, value in {"rknn_output_coordinates": "normalized", "rknn_input_size": str(img_size)}.items():
        item = metadata.get(key) or model.metadata_props.add()
        item.key, item.value = key, value
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))


def export_onnx(model_path: str, output_path: Path, img_size: int, opset: int, simplify: bool, normalize_coordinates: bool = False) -> Path:
    _validate_export_paths(model_path, output_path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required. Run: uv sync --extra export") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    exported = Path(model.export(format="onnx", imgsz=img_size, opset=opset, simplify=simplify, nms=False))
    if exported.resolve() != output_path.resolve():
        shutil.copy2(exported, output_path)
    if normalize_coordinates:
        normalize_output_coordinates(output_path, img_size)
    return output_path


def main() -> int:
    args = parse_args()
    output = export_onnx(args.model, Path(args.output), args.img_size, args.opset, args.simplify, args.normalize_coordinates)
    print(f"Exported ONNX model: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
