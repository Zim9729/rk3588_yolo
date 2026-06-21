# RK3588 YOLO11 Python Demo Design

## Summary

Build a minimal Python deployment demo for running YOLO11 on RK3588 NPU. The project starts from an official YOLO11n model, exports ONNX on the PC, converts ONNX to RKNN for RK3588, and runs image-file inference on the RK3588 board with configurable labels.

## Goals

- Provide a fast path from YOLO11n to RKNN on a Windows PC.
- Provide a Python image inference demo for RK3588 using RKNN Toolkit Lite 2.
- Support default COCO 80-class labels and custom labels through configuration.
- Keep the first version small and easy to debug.
- Document Windows conversion limitations and Linux or WSL fallback guidance.

## Non-goals

- Do not implement C++ deployment in this version.
- Do not implement camera or video inference in this version.
- Do not implement training or dataset annotation tooling.
- Do not hide RKNN Toolkit installation requirements behind opaque scripts.

## Target Environments

### Windows PC

The PC side exports YOLO11n to ONNX and attempts ONNX-to-RKNN conversion. If the installed RKNN Toolkit 2 package does not support the local Windows environment, the project documentation will direct the user to run the same conversion script in Ubuntu or WSL.

### RK3588 Board

The board side runs Python inference with `rknn-toolkit-lite2`, `numpy`, OpenCV, and `PyYAML`. The expected artifact copied to the board is a `.rknn` model.

## Project Layout

```text
rk3588_yolo/
  README.md
  requirements-export.txt
  requirements-rk3588.txt
  configs/
    coco.yaml
  data/
    labels/
      coco80.txt
    calibration/
      README.md
  models/
    README.md
  scripts/
    export_yolo11_onnx.py
    convert_onnx_to_rknn.py
  src/
    yolo11/
      __init__.py
      preprocess.py
      postprocess.py
      draw.py
      rknn_infer.py
  demos/
    image_demo.py
```

## Components

### Export Script

`scripts/export_yolo11_onnx.py` exports an Ultralytics YOLO11 model to ONNX. By default it uses `yolo11n.pt`, image size `640`, and output path `models/yolo11n.onnx`. If `yolo11n.pt` is not present, Ultralytics may download it from the network; users can also pass a local model path. The script accepts command-line arguments for model path, output path, image size, opset, and simplification.

### RKNN Conversion Script

`scripts/convert_onnx_to_rknn.py` converts ONNX to RKNN for `rk3588`. It supports an optional calibration dataset list for quantization and a non-quantized mode for initial debugging. It reports common environment problems clearly, especially missing `rknn-toolkit2`.

### Preprocessing

`src/yolo11/preprocess.py` implements letterbox resizing, BGR-to-RGB conversion, normalization, and shape metadata needed to map detection boxes back to the original image.

### Postprocessing

`src/yolo11/postprocess.py` decodes YOLO11 detection outputs, filters by confidence, applies class-aware non-maximum suppression, and rescales boxes to the original image size. It handles common output shapes used by Ultralytics ONNX exports.

### Drawing

`src/yolo11/draw.py` draws boxes, labels, and confidence scores on an image and writes the result to disk.

### RKNN Runtime Wrapper

`src/yolo11/rknn_infer.py` owns RKNN Lite initialization, runtime setup for RK3588 NPU, inference, and cleanup. It exposes a small interface for loading a model and running one image tensor.

### Image Demo

`demos/image_demo.py` ties together configuration, image loading, preprocessing, RKNN inference, postprocessing, and result drawing. It accepts model path, image path, config path, output path, confidence threshold, and NMS threshold.

## Data Flow

1. On the PC, run the export script to produce `models/yolo11n.onnx`.
2. On the PC, run the conversion script to produce `models/yolo11n.rknn` for RK3588.
3. Copy the project, `.rknn` model, labels, and test image to the RK3588 board.
4. On the board, run `demos/image_demo.py`.
5. The demo loads the image, applies letterbox preprocessing, runs RKNN NPU inference, decodes detections, draws boxes, and writes the output image.

## Configuration

`configs/coco.yaml` contains model input size, labels path, default thresholds, and optional normalization settings. The default labels file is `data/labels/coco80.txt`. Custom models can reuse the same demo by replacing the RKNN model path and labels path.

## Error Handling

- Missing model files fail with explicit path messages.
- Missing image files fail with explicit path messages.
- Missing RKNN Toolkit packages show installation guidance.
- Unsupported output tensor shapes include the actual tensor shapes in the error message.
- Runtime initialization failure includes the RKNN Lite return code.

## Testing and Verification

The first implementation will include lightweight local checks where possible:

- Run Python syntax checks for all generated Python files.
- Validate config and labels loading without requiring RK3588 hardware.
- Keep RKNN board inference as a documented manual verification step because it depends on board hardware and NPU runtime.

## Acceptance Criteria

- A user can install PC-side export dependencies and export `yolo11n.onnx`.
- A user with RKNN Toolkit 2 installed can convert `yolo11n.onnx` to `yolo11n.rknn` for RK3588.
- A user can copy `yolo11n.rknn` to an RK3588 board and run image-file inference with Python.
- The result image contains detected boxes, labels, and confidence scores.
- COCO labels work by default, and custom labels can be configured without code changes.

## Implementation Order

1. Create dependency files, README, model notes, calibration notes, config, and COCO labels.
2. Implement preprocessing, postprocessing, drawing, and RKNN runtime wrapper.
3. Implement export and conversion scripts.
4. Implement image demo.
5. Run local syntax/config checks.
6. Document RK3588 manual verification steps.
