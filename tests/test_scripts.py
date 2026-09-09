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
