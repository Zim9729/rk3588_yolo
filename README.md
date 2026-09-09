# RK3588 YOLO11 Python Demo

[English](README.md) | [简体中文](README.zh-CN.md)

This project provides a minimal deployment path for running YOLO11 on the RK3588 NPU with Python.

The flow is:

1. Export an Ultralytics YOLO11 model to ONNX on a PC.
2. Convert ONNX to RKNN for RK3588.
3. Copy the `.rknn` model to the RK3588 board.
4. Run image-file inference with `rknn-toolkit-lite2`.

## Directory Overview

```text
configs/                 Default YAML config
data/labels/             COCO labels
data/calibration/        Quantization calibration notes
models/                  Generated model artifacts
scripts/                 PC-side export and conversion scripts
src/yolo11/              Reusable preprocessing, postprocessing, drawing, and RKNN runtime code
demos/                   RK3588 board-side demo entry points
tests/                   Local tests for non-hardware code
```

## Windows PC Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment management. Install uv first, then sync the PC-side (export) dependencies:

```bash
uv sync --extra export
```

This installs ultralytics, onnx, onnxsim, pytest, and the shared base packages (numpy, opencv-python, PyYAML).

`rknn-toolkit2` (used for ONNX -> RKNN conversion) is pinned in the `export` extra but only has Linux x86_64 wheels on PyPI, so it is **skipped on Windows**. To convert on Windows, run the conversion step inside WSL/Ubuntu:

```bash
# inside WSL/Ubuntu, from the project root
uv sync --extra export   # now installs rknn-toolkit2
```

## Export YOLO11n to ONNX

If `yolo11n.pt` is not present, Ultralytics may download it from the network.

```bash
uv run python scripts/export_yolo11_onnx.py --model yolo11n.pt --output models/yolo11n.onnx --img-size 640 --opset 12 --simplify
```

For a custom model, replace `--model` with your local `.pt` path.

## Convert ONNX to RKNN

Non-quantized conversion for first validation:

```bash
uv run python scripts/convert_onnx_to_rknn.py --onnx models/yolo11n.onnx --output models/yolo11n.rknn --target rk3588
```

Quantized conversion (INT8, default):

```bash
uv run python scripts/convert_onnx_to_rknn.py --onnx models/yolo11n.onnx --output models/yolo11n.rknn --target rk3588 --dataset data/calibration/dataset.txt --quantized
```

High-accuracy quantized conversion (INT8 + auto hybrid mixed precision, per-channel):

```bash
uv run python scripts/convert_onnx_to_rknn.py --onnx models/yolo11n.onnx --output models/yolo11n.rknn --target rk3588 --dataset data/calibration/dataset.txt --quantized --quantized-method channel --auto-hybrid
```

Quantization options:

| Option | Values | Description |
|---|---|---|
| `--quantized-dtype` | `w8a8` (default) | RK3588 only supports INT8 quantization |
| `--quantized-method` | `channel` (default), `layer` | `channel`: per-channel quantization, higher precision. `layer`: per-layer, faster conversion |
| `--auto-hybrid` | flag | Enable mixed INT8+FP16 quantization. Higher accuracy but slower than pure INT8 |

Create `data/calibration/dataset.txt` with one representative image path per line. Use 200-500 images that match your deployment scenes for best quantization accuracy.

## RK3588 Board Setup

On the RK3588 board (aarch64 Linux), sync the board-side dependencies with uv:

```bash
uv sync --extra rk3588
```

This installs the shared base packages plus `rknn-toolkit-lite2` (aarch64) and `setuptools`.

If the `rknn-toolkit-lite2` wheel for your board image is not on PyPI, install the matching Rockchip wheel manually:

```bash
uv pip install rknn_toolkit_lite2-<version>-cp312-cp312-linux_aarch64.whl
```

Copy these files to the RK3588 board:

```text
models/yolo11n.rknn
configs/coco.yaml
data/labels/coco80.txt
src/
demos/
```

## Run Image Inference on RK3588

```bash
uv run python demos/image_demo.py --model models/yolo11n.rknn --image test.jpg --output outputs/result.jpg
```

Expected result:

- Detection count is printed.
- `outputs/result.jpg` is created with boxes, labels, and confidence scores.

## Custom Labels and Models

To use a custom model:

1. Export your custom `.pt` model to ONNX.
2. Convert it to RKNN.
3. Create a labels file with one class name per line.
4. Update `configs/coco.yaml` or pass another config file.

Example config:

```yaml
img_size: 640
labels: ../data/labels/my_labels.txt
conf_threshold: 0.25
nms_threshold: 0.45
input_format: nhwc
```

## Common Errors

### `ModuleNotFoundError: ultralytics`

Install PC dependencies:

```bash
uv sync --extra export
```

### `rknn-toolkit2 is required`

Install the Rockchip `rknn-toolkit2` wheel on the PC conversion environment. If Windows is unsupported for your wheel, use Ubuntu or WSL.

### `rknn-toolkit-lite2 is required`

Install the Rockchip `rknn-toolkit-lite2` wheel on the RK3588 board.

### `Unsupported YOLO output shape`

The exported ONNX output layout may differ from the expected Ultralytics YOLO11 detection layout. Check the printed tensor shape and adjust `src/yolo11/postprocess.py` if your model has a custom head.

### Low accuracy after quantization

Use more representative calibration images and verify `data/calibration/dataset.txt` paths are correct.

## Local Verification

Hardware inference is not expected to run on the Windows PC. Local checks verify syntax, config loading, preprocessing, postprocessing, and CLI help.

```bash
uv run pytest tests -v
uv run python -m compileall src demos scripts tests
uv run python scripts/export_yolo11_onnx.py --help
uv run python scripts/convert_onnx_to_rknn.py --help
uv run python demos/image_demo.py --help
```
