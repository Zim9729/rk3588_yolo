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
scripts/                 PC 端导出、转换与校准脚本
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

## 转换为 RKNN

有两条路径生成 RKNN 模型。**路径 A** 通过 Ultralytics 一步导出（INT8 推荐）。**路径 B** 通过本项目脚本分两步导出（控制更灵活，支持混合精度）。

### 路径 A：`yolo export format=rknn`（一步导出，INT8 推荐）

Ultralytics 一步完成 ONNX 导出 + RKNN 转换。`quantize=8` 时会自动为模型添加坐标归一化层（`_NormalizeCoords`），避免 RKNN 的 per-tensor INT8 量化把类别分数压零。

```bash
# FP16（不量化，输出原始像素坐标）
uv run yolo export model=3C/best.pt format=rknn quantize=16

# INT8（带坐标归一化，需要校准数据）
uv run yolo export model=3C/best.pt format=rknn quantize=8 data=data/calibration/calibration.yaml
```

输出到 `<model>_rknn_model/` 目录（如 `3C/best_rknn_model/best-rk3588.rknn`），附带 `metadata.yaml` 记录 `quantize` 值。INT8 导出后会删除中间 ONNX。

### 路径 B：`scripts/convert_onnx_to_rknn.py`（两步导出，支持混合精度）

浮点和量化路径有意使用不同的 ONNX 输出约定：

```bash
# 浮点 RKNN（RKNN Toolkit 构建为 float16；输出坐标保持像素值）
uv run python scripts/export_yolo11_onnx.py --model 3C/best.pt --output models/best-fp16.onnx --img-size 640 --opset 17 --simplify
uv run python scripts/convert_onnx_to_rknn.py --onnx models/best-fp16.onnx --output models/best-fp16.rknn --target rk3588

# 纯 INT8（输出坐标归一化；需要校准）
uv run python scripts/export_yolo11_onnx.py --model 3C/best.pt --output models/best-int8.onnx --img-size 640 --opset 17 --simplify --normalize-coordinates
uv run python scripts/convert_onnx_to_rknn.py --onnx models/best-int8.onnx --output models/best-int8.rknn --target rk3588 --dataset data/calibration/dataset.txt --quantized

# 自动混合量化（输出坐标归一化；需要校准）
uv run python scripts/export_yolo11_onnx.py --model 3C/best.pt --output models/best-hybrid.onnx --img-size 640 --opset 17 --simplify --normalize-coordinates
uv run python scripts/convert_onnx_to_rknn.py --onnx models/best-hybrid.onnx --output models/best-hybrid.rknn --target rk3588 --dataset data/calibration/dataset.txt --quantized --quantized-method channel --auto-hybrid
```

`--normalize-coordinates` 复现 Ultralytics 官方 `format=rknn quantize=8` 路径的输出归一化：仅将四个 `xywh` 通道除以输入尺寸，类别分数保持不变。转换脚本会拒绝使用未标记的像素坐标 ONNX 进行 INT8/混合量化，防止类别分数被静默压缩。转换完成后会生成 `<model>.rknn.yaml`，记录精度、输入形状和坐标格式，供推理端读取。

### 量化模式对比

| 模式 | 路径 | 命令参数 | RKNN 构建 | 坐标归一化 | 需要校准 |
|---|---|---|---|---|---|
| FP16 | A | `quantize=16` | 浮点 | 否 | 否 |
| FP16 | B | 不加 `--quantized` | `float_dtype=float16` | 否 | 否 |
| INT8 | A | `quantize=8` | W8A8 | **是** | 是 |
| INT8 | B | `--quantized` | W8A8 | **是，在 ONNX 中** | 是 |
| 自动混合 | B | `--quantized --auto-hybrid` | RKNN 自动选择 INT8/浮点层 | **是，在 ONNX 中** | 是 |

### `convert_onnx_to_rknn.py` 参数

| 选项 | 取值 | 说明 |
|---|---|---|
| `--onnx` | 路径 | 输入 ONNX 模型（默认 `models/yolo11n.onnx`） |
| `--output` | 路径 | 输出 RKNN 模型（默认 `models/yolo11n.rknn`） |
| `--target` | 字符串 | 目标平台（默认 `rk3588`） |
| `--dataset` | 路径 | 校准数据集文本文件（`--quantized` 时必填） |
| `--quantized` | 开关 | 启用 INT8 量化 |
| `--quantized-dtype` | `w8a8`（默认） | 本脚本开放的 INT8 权重和激活量化 |
| `--quantized-method` | `channel`（默认）、`layer` | 权重逐通道或逐张量量化 |
| `--quantized-algorithm` | `normal`、`mmse`、`kl_divergence` | 校准阈值算法 |
| `--auto-hybrid` | 开关 | 让 RKNN Toolkit 自动将敏感算子保留为浮点 |

### 校准数据集

创建 `data/calibration/dataset.txt`，每行放一张代表性图片的路径。建议用 200-500 张与部署场景匹配的图片，以获得最佳量化精度。

从本地图片目录生成校准集：

```bash
uv run python scripts/generate_calibration_dataset.py --input /path/to/images --count 300 --wsl
```

| 选项 | 说明 |
|---|---|
| `--input` | 包含校准图片的目录（递归扫描子目录） |
| `--output` | 输出 `dataset.txt` 路径（默认 `data/calibration/dataset.txt`） |
| `--count` | 最多取多少张，`0` 表示全部（默认 300） |
| `--wsl` | Windows 路径转 WSL `/mnt/` 格式 |
| `--shuffle` | 随机采样而非取前 N 张 |

仅用于快速测试时，可下载随机图片（不代表实际部署场景）：

```bash
uv run python scripts/download_calibration_samples.py
```

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
models/yolo11n.rknn.yaml    # 两步转换路径生成
configs/coco.yaml
data/labels/coco80.txt
src/
demos/
```

## 在 RK3588 板端同时做导出与推理

如果想在板子上直接完成 ONNX 导出 + RKNN 转换 + 推理（无需 PC），用 `all` extra 一次装齐导出与推理依赖：

```bash
uv sync --extra all
```

这会安装 `export`（ultralytics、onnx、onnxsim）与 `rk3588`（rknn-toolkit-lite2）的全部依赖。

**注意：** `rknn-toolkit2`（ONNX→RKNN 转换）的 aarch64 wheel **不在 PyPI**，`uv sync --extra all` 不会自动安装它。需用本仓库提供的脚本从 Rockchip GitHub 下载并安装：

```bash
bash scripts/install_rknn_toolkit2_aarch64.sh
# 或指定版本
RKNN_TOOLKIT2_VERSION=2.3.2 bash scripts/install_rknn_toolkit2_aarch64.sh
```

脚本会自动检测 Python 版本（cp310/cp311/cp312）和架构（aarch64），下载匹配的 wheel 并用 `uv pip install` 安装。安装完成后即可在板端执行完整流程：

```bash
# 1. 导出 ONNX
uv run python scripts/export_yolo11_onnx.py --model 3C/best.pt --output models/best-int8.onnx --img-size 640 --opset 17 --simplify --normalize-coordinates
# 2. 转换 RKNN（需先准备 data/calibration/dataset.txt）
uv run python scripts/convert_onnx_to_rknn.py --onnx models/best-int8.onnx --output models/best-int8.rknn --target rk3588 --dataset data/calibration/dataset.txt --quantized
# 3. 推理
uv run python demos/image_demo.py --model models/best-int8.rknn --image demos/test.jpg --output outputs/result.jpg --conf 0.25
```

## 在 RK3588 上运行图片推理

```bash
uv run python demos/image_demo.py --model models/best-int8.rknn --image demos/test.jpg --output outputs/result.jpg --conf 0.25
```

示例会自动读取 `<model>.rknn.yaml`。如果模型拷贝时没有携带元数据，两步路径生成的 INT8/混合模型请传 `--coordinate-format normalized`，浮点模型请传 `--coordinate-format pixels`。只有元数据不可用时，`auto` 才退回到数值范围判断。

预期结果：

- 打印检测到的目标数量。
- 生成 `outputs/result.jpg`，包含检测框、标签和置信度。
- 打印分阶段耗时（预处理 / 推理 / 后处理 / 绘图），含 avg/min/max。

计时选项：

| 选项 | 默认 | 说明 |
|---|---|---|
| `--warmup` | 3 | 预热次数（不计时） |
| `--runs` | 10 | 计时次数 |

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
uv run python -m pytest tests -v
uv run python -m compileall src demos scripts tests
uv run python scripts/export_yolo11_onnx.py --help
uv run python scripts/convert_onnx_to_rknn.py --help
uv run python scripts/generate_calibration_dataset.py --help
uv run python demos/image_demo.py --help
```
