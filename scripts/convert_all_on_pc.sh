#!/usr/bin/env bash
# ============================================================
# Windows PC (WSL/Ubuntu) 一键转换脚本
# 生成路径 A 和路径 B 的所有精度 RKNN 模型
#
# 前置条件：
#   1. Windows 已安装 WSL2 + Ubuntu 22.04/24.04
#   2. WSL 内已安装 uv:  curl -LsSf https://astral.sh/uv/install.sh | sh
#   3. 项目目录在 WSL 中可访问（如 /mnt/d/... 或 WSL 原生路径）
#
# 用法（在 WSL 终端中，项目根目录下）：
#   bash scripts/convert_all_on_pc.sh
#
# 可选环境变量：
#   MODEL=models/3C/3C_SDG_stage1_1_best_v2.pt   # 源模型路径
#   IMG_SIZE=640                                  # 输入尺寸
#   CALIB_DIR=data/calibration                    # 校准图片目录
#   OUTPUT_DIR=models                             # 输出目录
# ============================================================
set -euo pipefail

# ---- 参数 ----
MODEL="${MODEL:-models/3C/3C_SDG_stage1_1_best_v2.pt}"
IMG_SIZE="${IMG_SIZE:-640}"
CALIB_DIR="${CALIB_DIR:-data/calibration}"
XML_DIR="${XML_DIR:-data/xml}"
OUTPUT_DIR="${OUTPUT_DIR:-models}"
TARGET="${TARGET:-rk3588}"

# 派生路径
MODEL_STEM=$(basename "$MODEL" .pt)
ONNX_FP16="$OUTPUT_DIR/${MODEL_STEM}_pathB_fp16.onnx"
ONNX_INT8="$OUTPUT_DIR/${MODEL_STEM}_pathB_int8.onnx"
RKNN_DIR_A="$OUTPUT_DIR/${MODEL_STEM}_rknn_model"

echo "============================================================"
echo "  RKNN 模型一键转换（路径 A + 路径 B，所有精度）"
echo "============================================================"
echo "  源模型:     $MODEL"
echo "  输入尺寸:   $IMG_SIZE"
echo "  校准目录:   $CALIB_DIR"
echo "  输出目录:   $OUTPUT_DIR"
echo "  目标平台:   $TARGET"
echo "============================================================"
echo ""

# ---- Step 0: 环境检查 ----
echo "[Step 0] 环境检查..."
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv 未安装。请运行: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
  echo "WARNING: 当前架构 $ARCH，rknn-toolkit2 仅支持 x86_64 Linux"
  echo "         请在 WSL/Ubuntu 中运行此脚本"
  exit 1
fi
echo "  架构: $ARCH ✓"
echo ""

# ---- Step 1: 安装依赖 ----
echo "[Step 1] 安装 PC 端依赖 (uv sync --extra export)..."
uv sync --extra export
echo "  依赖安装完成 ✓"
echo ""

# ---- Step 2: 生成校准数据集 ----
echo "[Step 2] 生成校准数据集 dataset.txt..."
DATASET_TXT="$CALIB_DIR/dataset.txt"
if [ -f "$DATASET_TXT" ]; then
  echo "  dataset.txt 已存在，跳过"
else
  uv run python scripts/generate_calibration_dataset.py \
    --input "$CALIB_DIR" --output "$DATASET_TXT" --count 0
fi
# 转为绝对路径（RKNN 需要）
uv run python -c "
from pathlib import Path
p = Path('$DATASET_TXT')
lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
abs_lines = [str(Path(l).resolve()) for l in lines]
p.write_text('\n'.join(abs_lines) + '\n', encoding='utf-8')
print(f'  {len(abs_lines)} 张校准图片（绝对路径）')
"
echo ""

# ---- Step 3: 创建校准 yaml（路径 A 需要）----
echo "[Step 3] 创建校准 yaml..."
CALIB_YAML="configs/calibration.yaml"
PROJECT_ROOT=$(pwd)
cat > "$CALIB_YAML" << EOF
path: $PROJECT_ROOT/$CALIB_DIR
train: .
val: .
names:
  0: Pantograph_Area
EOF
echo "  $CALIB_YAML ✓"
echo ""

# ---- Step 4: 路径 A — FP16 ----
echo "[Step 4] 路径 A: FP16 (yolo export quantize=16)..."
uv run yolo export model="$MODEL" format=rknn quantize=16 name="$TARGET"
cp "$RKNN_DIR_A/"*-rk3588.rknn "$OUTPUT_DIR/${MODEL_STEM}_pathA_fp16.rknn"
cp "$RKNN_DIR_A/metadata.yaml" "$OUTPUT_DIR/${MODEL_STEM}_pathA_fp16.rknn.yaml"
echo "  路径 A FP16 完成 ✓"
echo ""

# ---- Step 5: 路径 A — INT8 ----
echo "[Step 5] 路径 A: INT8 (yolo export quantize=8)..."
uv run yolo export model="$MODEL" format=rknn quantize=8 name="$TARGET" data="$CALIB_YAML"
cp "$RKNN_DIR_A/"*-rk3588.rknn "$OUTPUT_DIR/${MODEL_STEM}_pathA_int8.rknn"
cp "$RKNN_DIR_A/metadata.yaml" "$OUTPUT_DIR/${MODEL_STEM}_pathA_int8.rknn.yaml"
echo "  路径 A INT8 完成 ✓"
echo ""

# ---- Step 6: 路径 B — 导出 ONNX ----
echo "[Step 6] 路径 B: 导出 ONNX..."
# FP16 用：不归一化坐标
uv run python scripts/export_yolo11_onnx.py \
  --model "$MODEL" --output "$ONNX_FP16" \
  --img-size "$IMG_SIZE" --opset 17 --simplify
echo "  FP16 ONNX ✓: $ONNX_FP16"
# INT8/hybrid/mmse 用：归一化坐标
uv run python scripts/export_yolo11_onnx.py \
  --model "$MODEL" --output "$ONNX_INT8" \
  --img-size "$IMG_SIZE" --opset 17 --simplify --normalize-coordinates
echo "  INT8 ONNX ✓: $ONNX_INT8"
echo ""

# ---- Step 7: 路径 B — FP16 RKNN ----
echo "[Step 7] 路径 B: FP16 RKNN..."
uv run python scripts/convert_onnx_to_rknn.py \
  --onnx "$ONNX_FP16" \
  --output "$OUTPUT_DIR/${MODEL_STEM}_pathB_fp16.rknn" \
  --target "$TARGET"
echo "  路径 B FP16 完成 ✓"
echo ""

# ---- Step 8: 路径 B — INT8 (normal, per-channel) ----
echo "[Step 8] 路径 B: INT8 normal per-channel..."
uv run python scripts/convert_onnx_to_rknn.py \
  --onnx "$ONNX_INT8" \
  --output "$OUTPUT_DIR/${MODEL_STEM}_pathB_int8.rknn" \
  --target "$TARGET" --dataset "$DATASET_TXT" --quantized
echo "  路径 B INT8 (channel) 完成 ✓"
echo ""

# ---- Step 9: 路径 B — INT8 (normal, per-tensor/layer) ----
echo "[Step 9] 路径 B: INT8 normal per-tensor (layer)..."
uv run python scripts/convert_onnx_to_rknn.py \
  --onnx "$ONNX_INT8" \
  --output "$OUTPUT_DIR/${MODEL_STEM}_pathB_int8_layer.rknn" \
  --target "$TARGET" --dataset "$DATASET_TXT" --quantized --quantized-method layer
echo "  路径 B INT8 (layer) 完成 ✓"
echo ""

# ---- Step 10: 路径 B — Hybrid (auto-hybrid) ----
echo "[Step 10] 路径 B: Hybrid (auto-hybrid)..."
uv run python scripts/convert_onnx_to_rknn.py \
  --onnx "$ONNX_INT8" \
  --output "$OUTPUT_DIR/${MODEL_STEM}_pathB_hybrid.rknn" \
  --target "$TARGET" --dataset "$DATASET_TXT" --quantized --auto-hybrid
echo "  路径 B Hybrid 完成 ✓"
echo ""

# ---- Step 11: 路径 B — INT8 (mmse) — PC 端可跑，板端会 OOM ----
echo "[Step 11] 路径 B: INT8 mmse (PC 端专用，约 10-30 分钟)..."
uv run python scripts/convert_onnx_to_rknn.py \
  --onnx "$ONNX_INT8" \
  --output "$OUTPUT_DIR/${MODEL_STEM}_pathB_int8_mmse.rknn" \
  --target "$TARGET" --dataset "$DATASET_TXT" --quantized --quantized-algorithm mmse
echo "  路径 B INT8 (mmse) 完成 ✓"
echo ""

# ---- Step 12: 路径 B — INT8 (kl_divergence) — PC 端可跑 ----
echo "[Step 12] 路径 B: INT8 kl_divergence (PC 端专用)..."
uv run python scripts/convert_onnx_to_rknn.py \
  --onnx "$ONNX_INT8" \
  --output "$OUTPUT_DIR/${MODEL_STEM}_pathB_int8_kl.rknn" \
  --target "$TARGET" --dataset "$DATASET_TXT" --quantized --quantized-algorithm kl_divergence
echo "  路径 B INT8 (kl) 完成 ✓"
echo ""

# ---- 汇总 ----
echo "============================================================"
echo "  全部转换完成！生成的 RKNN 模型："
echo "============================================================"
echo ""
echo "路径 A（Ultralytics 一步导出）："
ls -lh "$OUTPUT_DIR/${MODEL_STEM}_pathA_"*.rknn 2>/dev/null | awk '{printf "  %s  %s\n", $5, $9}'
echo ""
echo "路径 B（两步导出，脚本控制）："
ls -lh "$OUTPUT_DIR/${MODEL_STEM}_pathB_"*.rknn 2>/dev/null | awk '{printf "  %s  %s\n", $5, $9}'
echo ""
echo "============================================================"
echo "  拷贝到板端后运行评测："
echo "============================================================"
echo ""
echo "  # 把模型和评测脚本拷到板端"
echo "  scp $OUTPUT_DIR/${MODEL_STEM}_path*.rknn hzx@<板子IP>:$OUTPUT_DIR/"
echo "  scp $OUTPUT_DIR/${MODEL_STEM}_path*.rknn.yaml hzx@<板子IP>:$OUTPUT_DIR/"
echo ""
echo "  # 板端评测（需先 uv sync --extra all）"
echo "  uv run python scripts/eval_precision_loss.py \\"
echo "    --models \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathA_fp16.rknn \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathA_int8.rknn \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathB_fp16.rknn \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathB_int8.rknn \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathB_int8_layer.rknn \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathB_hybrid.rknn \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathB_int8_mmse.rknn \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathB_int8_kl.rknn \\"
echo "    --names PathA-FP16 PathA-INT8 PathB-FP16 PathB-INT8 PathB-INT8-layer PathB-Hybrid PathB-INT8-mmse PathB-INT8-kl \\"
echo "    --output-dir outputs/eval_all"
echo ""
echo "  # 单张测试图对比"
echo "  uv run python scripts/benchmark_precisions.py \\"
echo "    --models \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathA_fp16.rknn \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathA_int8.rknn \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathB_fp16.rknn \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathB_int8.rknn \\"
echo "      $OUTPUT_DIR/${MODEL_STEM}_pathB_int8_mmse.rknn \\"
echo "    --names PathA-FP16 PathA-INT8 PathB-FP16 PathB-INT8 PathB-INT8-mmse \\"
echo "    --image 3C_Test.jpg \\"
echo "    --output-dir outputs/benchmark_all"
echo ""
