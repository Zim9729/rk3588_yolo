# RK3588 YOLO11 Python Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal Python project that exports YOLO11n to ONNX, converts ONNX to RKNN for RK3588, and runs image-file inference on RK3588.

**Architecture:** The project separates PC-side model preparation from RK3588 board-side runtime inference. `scripts/` contains export and conversion entry points, while `src/yolo11/` contains reusable preprocessing, postprocessing, drawing, configuration, and RKNN runtime code used by `demos/image_demo.py`.

**Tech Stack:** Python 3.8+, Ultralytics, ONNX, RKNN Toolkit 2, RKNN Toolkit Lite 2, NumPy, OpenCV, PyYAML, pytest.

---

## Spec Reference

- `docs/superpowers/specs/2026-06-21-rk3588-yolo11-python-demo-design.md`

## File Structure and Responsibilities

- `README.md`: End-to-end usage for Windows PC export/conversion and RK3588 image inference.
- `requirements-export.txt`: PC-side dependencies for YOLO11 ONNX export and tests.
- `requirements-rk3588.txt`: RK3588 board-side dependencies for Python inference.
- `configs/coco.yaml`: Default demo configuration.
- `data/labels/coco80.txt`: COCO class names.
- `data/calibration/README.md`: Calibration dataset guidance for quantized RKNN conversion.
- `models/README.md`: Expected generated model artifacts and copy instructions.
- `scripts/export_yolo11_onnx.py`: Export YOLO11 model to ONNX.
- `scripts/convert_onnx_to_rknn.py`: Convert ONNX to RKNN for RK3588.
- `src/yolo11/__init__.py`: Package marker and exported version.
- `src/yolo11/config.py`: Load and validate YAML config and labels.
- `src/yolo11/preprocess.py`: Letterbox image preprocessing and scale metadata.
- `src/yolo11/postprocess.py`: YOLO11 output normalization, confidence filtering, NMS, and box rescaling.
- `src/yolo11/draw.py`: Draw detections onto images.
- `src/yolo11/rknn_infer.py`: RKNN Lite runtime wrapper.
- `demos/image_demo.py`: Board-side image inference CLI.
- `tests/test_config.py`: Config and labels tests.
- `tests/test_preprocess.py`: Letterbox preprocessing tests.
- `tests/test_postprocess.py`: Detection postprocessing and NMS tests.

## Task 1: Create Project Metadata, Config, Labels, and Tests

**Files:**
- Create: `requirements-export.txt`
- Create: `requirements-rk3588.txt`
- Create: `configs/coco.yaml`
- Create: `data/labels/coco80.txt`
- Create: `data/calibration/README.md`
- Create: `models/README.md`
- Create: `src/yolo11/__init__.py`
- Create: `src/yolo11/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/test_config.py` with tests that verify:

```python
from pathlib import Path

from src.yolo11.config import load_config, load_labels


def test_load_labels_reads_non_empty_labels(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("person\ncar\n", encoding="utf-8")

    assert load_labels(labels) == ["person", "car"]


def test_load_config_resolves_labels_path():
    config = load_config(Path("configs/coco.yaml"))

    assert config.img_size == 640
    assert config.labels_path.name == "coco80.txt"
    assert config.conf_threshold > 0
    assert config.nms_threshold > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL because `src.yolo11.config` does not exist.

- [ ] **Step 3: Create dependency and data files**

Create PC dependencies:

```text
ultralytics>=8.3.0
onnx>=1.14.0
onnxsim>=0.4.36
opencv-python>=4.8.0
numpy>=1.23.0
PyYAML>=6.0.0
pytest>=7.0.0
```

Create RK3588 dependencies:

```text
numpy>=1.23.0
opencv-python>=4.8.0
PyYAML>=6.0.0
```

Do not list `rknn-toolkit2` or `rknn-toolkit-lite2` as pip-installable defaults unless a Rockchip wheel path is available. Document those separately in `README.md`.

Create `configs/coco.yaml`:

```yaml
img_size: 640
labels: ../data/labels/coco80.txt
conf_threshold: 0.25
nms_threshold: 0.45
input_format: nhwc
```

Create `data/labels/coco80.txt` with 80 COCO class names, one per line.

- [ ] **Step 4: Implement config loader**

Create `src/yolo11/config.py` with:

- `YoloConfig` dataclass.
- `load_labels(path: Union[Path, str]) -> List[str]`.
- `load_config(path: Union[Path, str]) -> YoloConfig`.
- Relative labels paths resolved from the config file directory.
- Clear `FileNotFoundError` and `ValueError` messages.

- [ ] **Step 5: Run config tests**

Run: `python -m pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit or checkpoint**

If Git is initialized:

```bash
git add requirements-export.txt requirements-rk3588.txt configs/coco.yaml data/labels/coco80.txt data/calibration/README.md models/README.md src/yolo11/__init__.py src/yolo11/config.py tests/test_config.py
git commit -m "feat: add yolo11 demo configuration"
```

If Git is not initialized, skip commit and continue.

## Task 2: Implement Letterbox Preprocessing

**Files:**
- Create: `src/yolo11/preprocess.py`
- Create: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing preprocessing tests**

Create tests for:

```python
import numpy as np

from src.yolo11.preprocess import letterbox, preprocess_image


def test_letterbox_returns_expected_shape():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    resized, meta = letterbox(image, 640)

    assert resized.shape == (640, 640, 3)
    assert meta.original_shape == (480, 640)
    assert meta.input_shape == (640, 640)


def test_preprocess_image_returns_nhwc_float32_batch():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor, meta = preprocess_image(image, 640)

    assert tensor.shape == (1, 640, 640, 3)
    assert tensor.dtype == np.float32
    assert meta.original_shape == (480, 640)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_preprocess.py -v`

Expected: FAIL because preprocessing functions do not exist.

- [ ] **Step 3: Implement preprocessing**

Create:

- `LetterboxMeta` dataclass with `original_shape`, `input_shape`, `scale`, `pad_x`, `pad_y`.
- `letterbox(image, size, color=(114, 114, 114))`.
- `preprocess_image(image, size)` returning NHWC float32 batch normalized to `0..1` RGB.

- [ ] **Step 4: Run preprocessing tests**

Run: `python -m pytest tests/test_preprocess.py -v`

Expected: PASS.

- [ ] **Step 5: Run config and preprocessing tests together**

Run: `python -m pytest tests/test_config.py tests/test_preprocess.py -v`

Expected: PASS.

- [ ] **Step 6: Commit or checkpoint**

If Git is initialized:

```bash
git add src/yolo11/preprocess.py tests/test_preprocess.py
git commit -m "feat: add yolo11 preprocessing"
```

## Task 3: Implement YOLO11 Postprocessing

**Files:**
- Create: `src/yolo11/postprocess.py`
- Create: `tests/test_postprocess.py`

- [ ] **Step 1: Write failing postprocessing tests**

Create tests for:

```python
import numpy as np

from src.yolo11.postprocess import Detection, nms, postprocess_outputs
from src.yolo11.preprocess import LetterboxMeta


def test_nms_keeps_highest_score_for_overlapping_boxes():
    detections = [
        Detection(0, 0.90, (10, 10, 100, 100)),
        Detection(0, 0.80, (12, 12, 102, 102)),
    ]

    kept = nms(detections, 0.45)

    assert len(kept) == 1
    assert kept[0].score == 0.90


def test_postprocess_accepts_ultralytics_shape():
    output = np.zeros((1, 84, 2), dtype=np.float32)
    output[0, 0:4, 0] = [320, 320, 100, 100]
    output[0, 4, 0] = 0.90
    meta = LetterboxMeta((640, 640), (640, 640), 1.0, 0, 0)

    detections = postprocess_outputs([output], meta, 0.25, 0.45)

    assert len(detections) == 1
    assert detections[0].class_id == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_postprocess.py -v`

Expected: FAIL because postprocessing does not exist.

- [ ] **Step 3: Implement postprocessing**

Create:

- `Detection` dataclass with `class_id`, `score`, `box`.
- Tensor normalization for output shapes `(1, 84, N)`, `(1, N, 84)`, `(84, N)`, `(N, 84)`.
- `xywh_to_xyxy` conversion.
- Box unletterboxing using `LetterboxMeta`.
- IoU and class-aware NMS.
- Clear error for unsupported shapes.

- [ ] **Step 4: Run postprocessing tests**

Run: `python -m pytest tests/test_postprocess.py -v`

Expected: PASS.

- [ ] **Step 5: Run all tests so far**

Run: `python -m pytest tests -v`

Expected: PASS.

- [ ] **Step 6: Commit or checkpoint**

If Git is initialized:

```bash
git add src/yolo11/postprocess.py tests/test_postprocess.py
git commit -m "feat: add yolo11 postprocessing"
```

## Task 4: Implement Drawing and RKNN Runtime Wrapper

**Files:**
- Create: `src/yolo11/draw.py`
- Create: `src/yolo11/rknn_infer.py`

- [ ] **Step 1: Implement drawing helper**

Create `draw_detections(image, detections, labels)` that:

- Copies the input image.
- Draws rectangle, class label, and score.
- Falls back to class ID if labels are missing.

- [ ] **Step 2: Implement RKNN runtime wrapper**

Create `RknnLiteDetector` with:

- Lazy import of `rknnlite.api.RKNNLite` so PC-side tests do not require board runtime.
- `load()` to load `.rknn` and initialize runtime.
- `infer(input_tensor)` to call `rknn.inference(inputs=[input_tensor])`.
- `release()` and context-manager support.
- Explicit errors for missing model path and RKNN Lite import failure.

- [ ] **Step 3: Run syntax check**

Run: `python -m compileall src demos scripts tests`

Expected: PASS for existing directories; if `demos` or `scripts` do not exist yet, create them before this command or defer full compileall to Task 6.

- [ ] **Step 4: Commit or checkpoint**

If Git is initialized:

```bash
git add src/yolo11/draw.py src/yolo11/rknn_infer.py
git commit -m "feat: add drawing and rknn runtime wrapper"
```

## Task 5: Implement Export and RKNN Conversion Scripts

**Files:**
- Create: `scripts/export_yolo11_onnx.py`
- Create: `scripts/convert_onnx_to_rknn.py`

- [ ] **Step 1: Implement ONNX export CLI**

`scripts/export_yolo11_onnx.py` must support:

```text
--model yolo11n.pt
--output models/yolo11n.onnx
--img-size 640
--opset 12
--simplify
```

Use `ultralytics.YOLO(model).export(format="onnx", imgsz=..., opset=..., simplify=...)` and move or copy the produced ONNX to the requested output path when needed.

- [ ] **Step 2: Implement RKNN conversion CLI**

`scripts/convert_onnx_to_rknn.py` must support:

```text
--onnx models/yolo11n.onnx
--output models/yolo11n.rknn
--target rk3588
--dataset data/calibration/dataset.txt
--quantized
```

Use `rknn.api.RKNN` with:

- `config(target_platform="rk3588")`.
- `load_onnx(model=...)`.
- `build(do_quantization=..., dataset=...)`.
- `export_rknn(...)`.
- Clear return-code checking after each RKNN call.

- [ ] **Step 3: Run CLI help checks**

Run:

```bash
python scripts/export_yolo11_onnx.py --help
python scripts/convert_onnx_to_rknn.py --help
```

Expected: Both commands print usage and exit 0 without requiring model files.

- [ ] **Step 4: Commit or checkpoint**

If Git is initialized:

```bash
git add scripts/export_yolo11_onnx.py scripts/convert_onnx_to_rknn.py
git commit -m "feat: add yolo11 export and rknn conversion scripts"
```

## Task 6: Implement Image Demo and Documentation

**Files:**
- Create: `demos/image_demo.py`
- Create: `README.md`

- [ ] **Step 1: Implement image demo CLI**

`demos/image_demo.py` must support:

```text
--model models/yolo11n.rknn
--image path/to/image.jpg
--config configs/coco.yaml
--output outputs/result.jpg
--conf 0.25
--nms 0.45
```

The flow is:

1. Load config and labels.
2. Read image with OpenCV.
3. Preprocess image.
4. Run RKNN inference.
5. Postprocess outputs.
6. Draw detections.
7. Create output directory if needed.
8. Save result image.
9. Print detection count and output path.

- [ ] **Step 2: Write README**

`README.md` must include:

- Project purpose.
- Directory overview.
- Windows PC setup.
- YOLO11n ONNX export command.
- RKNN conversion command.
- RKNN Toolkit 2 Windows caveat and Ubuntu/WSL fallback.
- RK3588 setup.
- Image inference command.
- Custom labels/model instructions.
- Calibration dataset instructions.
- Common errors and fixes.

- [ ] **Step 3: Run local verification**

Run:

```bash
python -m pytest tests -v
python -m compileall src demos scripts tests
python scripts/export_yolo11_onnx.py --help
python scripts/convert_onnx_to_rknn.py --help
python demos/image_demo.py --help
```

Expected: All local checks pass. Hardware inference is not expected to run on the Windows PC.

- [ ] **Step 4: Document board verification**

Add RK3588 manual verification command to README:

```bash
python demos/image_demo.py --model models/yolo11n.rknn --image test.jpg --output outputs/result.jpg
```

Expected on board: output image is created and detections are printed.

- [ ] **Step 5: Commit or checkpoint**

If Git is initialized:

```bash
git add demos/image_demo.py README.md
git commit -m "feat: add rk3588 image inference demo"
```

## Final Verification

Run from project root:

```bash
python -m pytest tests -v
python -m compileall src demos scripts tests
python scripts/export_yolo11_onnx.py --help
python scripts/convert_onnx_to_rknn.py --help
python demos/image_demo.py --help
```

Expected:

- All tests pass.
- All Python files compile.
- All CLI help commands exit 0.
- README documents the manual RK3588 inference verification step.

## Known Constraints

- Current workspace is not a Git repository. Commit steps require `git init` first or can be skipped.
- RKNN conversion depends on a compatible Rockchip `rknn-toolkit2` installation.
- RK3588 inference cannot be fully verified on the Windows PC without board hardware and RKNN Lite runtime.
