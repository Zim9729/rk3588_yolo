import subprocess
import sys


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
