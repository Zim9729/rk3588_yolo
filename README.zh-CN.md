# RK3588 YOLO11 Python 示例

[English](README.md) | [简体中文](README.zh-CN.md)

本项目提供一条在 RK3588 NPU 上用 Python 运行 YOLO11 的最小部署路径。

整体流程：

1. 在 PC 上将 Ultralytics YOLO11 模型导出为 ONNX。
2. 将 ONNX 转换为 RK3588 专用的 RKNN 模型。
3. 将 `.rknn` 模型拷贝到 RK3588 板端。
4. 用 `rknn-toolkit-lite2` 运行图片推理。

## 目录结构

```text
configs/                 默认 YAML 配置
data/labels/             COCO 标签
data/calibration/        量化校准说明
models/                  生成的模型产物
scripts/                 PC 端导出与转换脚本
src/yolo11/              可复用的预处理、后处理、绘图与 RKNN 运行时代码
demos/                   RK3588 板端示例入口
tests/                   非硬件代码的本地测试
```

## Windows PC 环境准备

本项目使用 [uv](https://docs.astral.sh/uv/) 管理环境。先安装 uv，然后同步 PC 端（导出）依赖：

```bash
uv sync --extra export
```

这会安装 ultralytics、onnx、onnxsim、pytest，以及共享的基础包（numpy、opencv-python、PyYAML）。

`rknn-toolkit2`（用于 ONNX -> RKNN 转换）已固定在 `export` extra 中，但 PyPI 上只有 Linux x86_64 的 wheel，因此在 **Windows 上会被跳过**。要在 Windows 上做转换，请在 WSL/Ubuntu 中执行：

```bash
# 在 WSL/Ubuntu 内，进入项目根目录
uv sync --extra export   # 此时才会安装 rknn-toolkit2
```

## 导出 YOLO11n 为 ONNX

如果 `yolo11n.pt` 不存在，Ultralytics 会从网络下载。

```bash
uv run python scripts/export_yolo11_onnx.py --model yolo11n.pt --output models/yolo11n.onnx --img-size 640 --opset 12 --simplify
```

自定义模型请把 `--model` 换成你本地的 `.pt` 路径。

## 将 ONNX 转换为 RKNN

首次验证用非量化转换：

```bash
uv run python scripts/convert_onnx_to_rknn.py --onnx models/yolo11n.onnx --output models/yolo11n.rknn --target rk3588
```

量化转换（INT8，默认）：

```bash
uv run python scripts/convert_onnx_to_rknn.py --onnx models/yolo11n.onnx --output models/yolo11n.rknn --target rk3588 --dataset data/calibration/dataset.txt --quantized
```

高精度量化转换（INT8 + 自动混合精度，逐通道）：

```bash
uv run python scripts/convert_onnx_to_rknn.py --onnx models/yolo11n.onnx --output models/yolo11n.rknn --target rk3588 --dataset data/calibration/dataset.txt --quantized --quantized-method channel --auto-hybrid
```

量化选项：

| 选项 | 取值 | 说明 |
|---|---|---|
| `--quantized-dtype` | `w8a8`（默认） | RK3588 仅支持 INT8 量化 |
| `--quantized-method` | `channel`（默认）、`layer` | `channel`：逐通道量化，精度更高。`layer`：逐层量化，转换更快 |
| `--auto-hybrid` | 开关 | 启用 INT8 + FP16 混合量化。精度更高，但比纯 INT8 慢 |

创建 `data/calibration/dataset.txt`，每行放一张代表性图片的路径。建议用 200-500 张与部署场景匹配的图片，以获得最佳量化精度。

## RK3588 板端环境准备

在 RK3588 板（aarch64 Linux）上，用 uv 同步板端依赖：

```bash
uv sync --extra rk3588
```

这会安装共享基础包，外加 `rknn-toolkit-lite2`（aarch64）和 `setuptools`。

如果你的板子镜像对应的 `rknn-toolkit-lite2` wheel 不在 PyPI 上，请手动安装匹配的 Rockchip wheel：

```bash
uv pip install rknn_toolkit_lite2-<version>-cp312-cp312-linux_aarch64.whl
```

将以下文件拷贝到 RK3588 板端：

```text
models/yolo11n.rknn
configs/coco.yaml
data/labels/coco80.txt
src/
demos/
```

## 在 RK3588 上运行图片推理

```bash
uv run python demos/image_demo.py --model models/yolo11n.rknn --image demos/test.jpg --output outputs/result.jpg
```

预期结果：

- 打印检测到的目标数量。
- 生成 `outputs/result.jpg`，包含检测框、标签和置信度。

## 自定义标签与模型

使用自定义模型：

1. 将你的 `.pt` 模型导出为 ONNX。
2. 转换为 RKNN。
3. 创建标签文件，每行一个类别名。
4. 更新 `configs/coco.yaml`，或传入其他配置文件。

配置示例：

```yaml
img_size: 640
labels: ../data/labels/my_labels.txt
conf_threshold: 0.25
nms_threshold: 0.45
input_format: nhwc
```

## 常见错误

### `ModuleNotFoundError: ultralytics`

安装 PC 端依赖：

```bash
uv sync --extra export
```

### `rknn-toolkit2 is required`

在 PC 转换环境中安装 Rockchip `rknn-toolkit2` wheel。如果 Windows 不支持该 wheel，请在 Ubuntu 或 WSL 中执行。

### `rknn-toolkit-lite2 is required`

在 RK3588 板端安装 Rockchip `rknn-toolkit-lite2` wheel。

### `Unsupported YOLO output shape`

导出的 ONNX 输出布局可能与预期的 Ultralytics YOLO11 检测布局不一致。请查看打印的张量形状，若你的模型使用了自定义检测头，请调整 `src/yolo11/postprocess.py`。

### 量化后精度低

使用更多有代表性的校准图片，并确认 `data/calibration/dataset.txt` 中的路径正确。

## 本地验证

Windows PC 上不运行硬件推理。本地检查用于验证语法、配置加载、预处理、后处理以及 CLI 帮助。

```bash
uv run pytest tests -v
uv run python -m compileall src demos scripts tests
uv run python scripts/export_yolo11_onnx.py --help
uv run python scripts/convert_onnx_to_rknn.py --help
uv run python demos/image_demo.py --help
```
