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
scripts/                 PC-side export, conversion, and calibration scripts
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

## Convert to RKNN

There are two paths to get an RKNN model. **Path A** is a one-step export via Ultralytics (recommended for INT8). **Path B** is a two-step export via the project's own scripts (more control, supports mixed precision).

### Path A: `yolo export format=rknn` (one step, recommended for INT8)

Ultralytics handles ONNX export + RKNN conversion in one command. For `quantize=8`, it automatically wraps the model with coordinate normalization (`_NormalizeCoords`) so that RKNN's per-tensor INT8 scale does not zero out class scores.

```bash
# FP16 (no quantization, raw pixel coordinates in output)
uv run yolo export model=3C/best.pt format=rknn quantize=16

# INT8 (with coordinate normalization, requires calibration data)
uv run yolo export model=3C/best.pt format=rknn quantize=8 data=data/calibration/calibration.yaml
```

Output goes to `<model>_rknn_model/` (e.g. `3C/best_rknn_model/best-rk3588.rknn`) with a `metadata.yaml` that records the `quantize` value. The intermediate ONNX is deleted after INT8 export.

### Path B: `scripts/convert_onnx_to_rknn.py` (two steps, supports mixed precision)

The floating-point and quantized paths intentionally use different ONNX output contracts:

```bash
# Floating-point RKNN (RKNN Toolkit builds float16; output coordinates remain pixels)
uv run python scripts/export_yolo11_onnx.py --model 3C/best.pt --output models/best-fp16.onnx --img-size 640 --opset 17 --simplify
uv run python scripts/convert_onnx_to_rknn.py --onnx models/best-fp16.onnx --output models/best-fp16.rknn --target rk3588

# Pure INT8 (normalized output coordinates; calibration required)
uv run python scripts/export_yolo11_onnx.py --model 3C/best.pt --output models/best-int8.onnx --img-size 640 --opset 17 --simplify --normalize-coordinates
uv run python scripts/convert_onnx_to_rknn.py --onnx models/best-int8.onnx --output models/best-int8.rknn --target rk3588 --dataset data/calibration/dataset.txt --quantized

# Automatic hybrid quantization (normalized output coordinates; calibration required)
uv run python scripts/export_yolo11_onnx.py --model 3C/best.pt --output models/best-hybrid.onnx --img-size 640 --opset 17 --simplify --normalize-coordinates
uv run python scripts/convert_onnx_to_rknn.py --onnx models/best-hybrid.onnx --output models/best-hybrid.rknn --target rk3588 --dataset data/calibration/dataset.txt --quantized --quantized-method channel --auto-hybrid
```

`--normalize-coordinates` reproduces the output normalization used by Ultralytics' official `format=rknn quantize=8` path: only the four `xywh` channels are divided by the input size; class scores are unchanged. The conversion script rejects INT8/hybrid conversion of an unmarked pixel-coordinate ONNX to prevent silent score collapse. It writes `<model>.rknn.yaml` with precision, input shape, and coordinate format for inference.

### Quantization modes comparison

| Mode | Path | Command flag | RKNN build | Coord normalization | Calibration needed |
|---|---|---|---|---|---|
| FP16 | A | `quantize=16` | Floating point | No | No |
| FP16 | B | no `--quantized` | `float_dtype=float16` | No | No |
| INT8 | A | `quantize=8` | W8A8 | **Yes** | Yes |
| INT8 | B | `--quantized` | W8A8 | **Yes, in ONNX** | Yes |
| Auto hybrid | B | `--quantized --auto-hybrid` | RKNN-selected INT8/float layers | **Yes, in ONNX** | Yes |

### `convert_onnx_to_rknn.py` options

| Option | Values | Description |
|---|---|---|
| `--onnx` | path | Input ONNX model (default: `models/yolo11n.onnx`) |
| `--output` | path | Output RKNN model (default: `models/yolo11n.rknn`) |
| `--target` | string | Target platform (default: `rk3588`) |
| `--dataset` | path | Calibration dataset text file (required for `--quantized`) |
| `--quantized` | flag | Enable INT8 quantization |
| `--quantized-dtype` | `w8a8` (default) | INT8 weights and activations exposed by this script |
| `--quantized-method` | `channel` (default), `layer` | Per-channel or per-tensor weight quantization |
| `--quantized-algorithm` | `normal`, `mmse`, `kl_divergence` | Calibration threshold algorithm |
| `--auto-hybrid` | flag | Let RKNN Toolkit automatically retain sensitive operations in floating point |

### Calibration dataset

Create `data/calibration/dataset.txt` with one representative image path per line. Use 200-500 images that match your deployment scenes for best quantization accuracy.

Generate it from a local image directory:

```bash
uv run python scripts/generate_calibration_dataset.py --input /path/to/images --count 300 --wsl
```

| Option | Description |
|---|---|
| `--input` | Directory containing calibration images (scans subdirectories) |
| `--output` | Output `dataset.txt` path (default: `data/calibration/dataset.txt`) |
| `--count` | Max number of images, `0` for all (default: 300) |
| `--wsl` | Convert Windows paths to WSL `/mnt/` format |
| `--shuffle` | Randomly sample instead of taking the first N |

For quick testing only, download random sample images (not representative of deployment scenes):

```bash
uv run python scripts/download_calibration_samples.py
```

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
models/yolo11n.rknn.yaml    # generated by the two-step conversion path
configs/coco.yaml
data/labels/coco80.txt
src/
demos/
```

## Run Image Inference on RK3588

```bash
uv run python demos/image_demo.py --model models/best-int8.rknn --image demos/test.jpg --output outputs/result.jpg --conf 0.25
```

The demo reads `<model>.rknn.yaml` automatically. For a model copied without metadata, pass `--coordinate-format normalized` for INT8/hybrid models produced by this two-step path, or `--coordinate-format pixels` for floating-point models. `auto` falls back to value-range detection only when metadata is unavailable.

Expected result:

- Detection count is printed.
- `outputs/result.jpg` is created with boxes, labels, and confidence scores.
- Per-stage timing (preprocess / infer / postprocess / draw) is printed with avg/min/max.

Timing options:

| Option | Default | Description |
|---|---|---|
| `--warmup` | 3 | Warmup runs (not counted) |
| `--runs` | 10 | Timed runs for statistics |

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
uv run python -m pytest tests -v
uv run python -m compileall src demos scripts tests
uv run python scripts/export_yolo11_onnx.py --help
uv run python scripts/convert_onnx_to_rknn.py --help
uv run python scripts/generate_calibration_dataset.py --help
uv run python demos/image_demo.py --help
```
