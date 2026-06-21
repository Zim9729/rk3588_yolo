# RK3588 YOLO11 Python Demo

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

Install PC-side dependencies:

```bash
python -m pip install -r requirements-export.txt
```

`rknn-toolkit2` is distributed by Rockchip as version-specific wheels. Install the wheel that matches your Python and OS if you want to convert ONNX to RKNN on Windows.

If the Windows wheel is unavailable or incompatible, run the same conversion command in Ubuntu or WSL with a compatible Rockchip `rknn-toolkit2` package.

## Export YOLO11n to ONNX

If `yolo11n.pt` is not present, Ultralytics may download it from the network.

```bash
python scripts/export_yolo11_onnx.py --model yolo11n.pt --output models/yolo11n.onnx --img-size 640 --opset 12 --simplify
```

For a custom model, replace `--model` with your local `.pt` path.

## Convert ONNX to RKNN

Non-quantized conversion for first validation:

```bash
python scripts/convert_onnx_to_rknn.py --onnx models/yolo11n.onnx --output models/yolo11n.rknn --target rk3588
```

Quantized conversion:

```bash
python scripts/convert_onnx_to_rknn.py --onnx models/yolo11n.onnx --output models/yolo11n.rknn --target rk3588 --dataset data/calibration/dataset.txt --quantized
```

Create `data/calibration/dataset.txt` with one representative image path per line. Use images that match your deployment scenes.

## RK3588 Board Setup

Install board-side Python dependencies:

```bash
python3 -m pip install -r requirements-rk3588.txt
```

Install `rknn-toolkit-lite2` from the Rockchip wheel that matches your board image, Python version, and RKNN runtime.

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
python demos/image_demo.py --model models/yolo11n.rknn --image test.jpg --output outputs/result.jpg
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
python -m pip install -r requirements-export.txt
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
python -m pytest tests -v
python -m compileall src demos scripts tests
python scripts/export_yolo11_onnx.py --help
python scripts/convert_onnx_to_rknn.py --help
python demos/image_demo.py --help
```
