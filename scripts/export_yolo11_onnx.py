import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export an Ultralytics YOLO11 model to ONNX.")
    parser.add_argument("--model", default="yolo11n.pt", help="Input Ultralytics model path or model name.")
    parser.add_argument("--output", default="models/yolo11n.onnx", help="Output ONNX model path.")
    parser.add_argument("--img-size", type=int, default=640, help="Export image size.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version.")
    parser.add_argument("--simplify", action="store_true", help="Simplify exported ONNX model.")
    return parser.parse_args()


def _validate_export_paths(model_path: str, output_path: Path) -> None:
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"ONNX output path must end in .onnx: {output_path}")
    source_path = Path(model_path)
    if source_path.exists() and source_path.resolve() == output_path.resolve():
        raise ValueError("ONNX output path must not overwrite the input model")


def export_onnx(model_path: str, output_path: Path, img_size: int, opset: int, simplify: bool) -> Path:
    _validate_export_paths(model_path, output_path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required. Run: uv sync --extra export") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    exported = Path(model.export(format="onnx", imgsz=img_size, opset=opset, simplify=simplify))
    if exported.resolve() != output_path.resolve():
        shutil.copy2(exported, output_path)
    return output_path


def main() -> int:
    args = parse_args()
    output = export_onnx(args.model, Path(args.output), args.img_size, args.opset, args.simplify)
    print(f"Exported ONNX model: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
