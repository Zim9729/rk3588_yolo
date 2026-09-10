import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("command", "error"),
    [
        (["scripts/convert_onnx_to_rknn.py", "--onnx", "models/yolo11n.onnx", "--output", "models/yolo11n.onnx"], "must end in .rknn"),
        (["scripts/convert_onnx_to_rknn.py", "--onnx", "models/yolo11n.rknn", "--output", "models/yolo11n.rknn"], "must not overwrite the input ONNX model"),
        (["scripts/convert_onnx_to_rknn.py", "--onnx", "models/yolo11n.onnx", "--auto-hybrid"], "--auto-hybrid requires --quantized"),
        (["scripts/convert_onnx_to_rknn.py", "--onnx", "models/yolo11n.onnx", "--dataset", "data/calibration/dataset.txt"], "--dataset requires --quantized"),
        (["scripts/export_yolo11_onnx.py", "--model", "yolo11n.pt", "--output", "yolo11n.pt"], "must end in .onnx"),
        (["scripts/export_yolo11_onnx.py", "--model", "models/yolo11n.onnx", "--output", "models/yolo11n.onnx"], "must not overwrite the input model"),
    ],
)
def test_conversion_scripts_reject_unsafe_argument_combinations(command, error):
    result = subprocess.run(
        [sys.executable, *command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert error in result.stderr


def test_export_script_help_exits_successfully():
    result = subprocess.run(
        [sys.executable, "scripts/export_yolo11_onnx.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--output" in result.stdout


def test_convert_script_help_exits_successfully():
    result = subprocess.run(
        [sys.executable, "scripts/convert_onnx_to_rknn.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--onnx" in result.stdout
    assert "--target" in result.stdout
    assert "--quantized-dtype" in result.stdout
    assert "--quantized-method" in result.stdout


def test_normalize_output_coordinates_marks_and_updates_onnx(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    from scripts.export_yolo11_onnx import normalize_output_coordinates

    source = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 1])
    constant = helper.make_tensor("predictions", TensorProto.FLOAT, [1, 5, 1], [320, 320, 100, 100, 0.9])
    graph = helper.make_graph([helper.make_node("Constant", [], ["output0"], value=constant)], "test", [source], [output])
    model_path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)]), model_path)

    normalize_output_coordinates(model_path, 640)

    model = onnx.load(model_path)
    metadata = {item.key: item.value for item in model.metadata_props}
    assert metadata["rknn_output_coordinates"] == "normalized"
    assert [node.op_type for node in model.graph.node[-4:]] == ["Slice", "Div", "Slice", "Concat"]
    onnx.checker.check_model(model)


def test_image_demo_help_exits_successfully():
    result = subprocess.run(
        [sys.executable, "demos/image_demo.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "--image" in result.stdout
