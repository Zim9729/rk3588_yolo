# RK3588 YOLO11 RKNN 全量评测总结

> 模型：`3C_SDG_stage1_1_best_v2.pt`  
> 类别：`Pantograph_Area`（1 类）  
> 评测对象：板端转换与 WSL2 转换的 Path A / Path B 全部有效 RKNN 模型  
> 报告生成时间：2026-09-12 00:45:39

## 1. 最终结论

### 最佳综合模型

**推荐：`WSL2-B-INT8`**  
模型文件：`models_wsl2/3C_SDG_stage1_1_best_v2_pathB_int8.rknn`

选择理由：

- 300 张 XML 真值集：`TP=300`、`FP=0`、`FN=0`，Precision/Recall/F1 均为 `1.0000`。
- 平均 IoU 为 `0.8623`；相比 FP16 的 `0.9276`，绝对下降 `0.0653`，损失主要来自框回归，而不是分类或召回。
- `3C_Test.jpg` 只输出 **1 个框**，符合已确认的正确行为；板端转换的 normal INT8 输出 2 个重叠框。
- 预热 5 次、计时 30 次后，纯 NPU 推理平均 `42.40 ms`，相对 WSL2 Path B FP16 的 `83.83 ms` 加速约 `1.98x`。
- 模型大小 `4.58 MiB`，相对 FP16 的 `7.19 MiB` 减少约 `36.4%`。
- Path B 保留 ONNX 中间产物，并允许选择量化算法和粒度，复现和后续调优能力优于 Path A。

### 精度优先模型

若定位精度比速度更重要，选择 **`WSL2-B-FP16`** 或任一 FP16：平均 IoU `0.9276`，300 张图 0 误检、0 漏检，单图结果正确；代价是推理约为 INT8 的 `1.98x`，模型也更大。

### 不推荐模型

- `WSL2-B-INT8-mmse`：平均 IoU `0.8112`，出现 `1` 个 FP；本数据集上明显劣于 normal。
- `WSL2-B-INT8-kl`：平均 IoU `0.8406`，且 `3C_Test.jpg` 输出 2 个框。
- `WSL2-B-INT8-layer`：平均 IoU `0.8648` 为 INT8 中最高，但 300 图出现 `1` 个 FP，综合稳定性不如 normal per-channel。
- 板端量化的 INT8/Hybrid：全量集指标可用，但在目标单图上均产生重复框。

## 2. 统一测试条件

| 项目 | 配置 |
|---|---|
| RK3588 Runtime | `librknnrt 1.6.0`，driver `0.9.8` |
| RKNN Toolkit / 模型版本 | `2.3.2` |
| 输入尺寸 | `640 × 640` |
| 类别 | `Pantograph_Area` |
| 置信度阈值 | `0.25` |
| NMS IoU 阈值 | `0.45` |
| 全量评测 | `data/calibration` 中 300 张图，真值来自 `data/xml` |
| TP 匹配阈值 | IoU ≥ `0.5` |
| 单图基准 | `3C_Test.jpg`，warmup=5，runs=30 |
| 正确单图行为 | 1 个检测框（由人工确认） |

说明：单图表中的延迟是预热后的**纯 NPU 推理时间**。300 图评测脚本记录的 `推理avg` 实际覆盖预处理、NPU 推理和后处理，因此不作为主要速度排名依据。测试期间存在 `RKNN model 2.3.2` 与板端 runtime `1.6.0` 的版本不匹配警告，生产部署应优先升级 Runtime/驱动后复测。

## 3. 全部模型综合结果

| 模型 | 转换环境 | 路径 | 量化方式 | 大小 MiB | TP/FP/FN | P | R | F1 | 平均IoU | 平均score | 300图端到端avg ms | 300图p50 ms | 单图检测数 | 单图NPU avg ms | p50 ms |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ARM-A-FP16 | 板端转换 | Path A | 不量化 | 7.19 | 300 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | 0.9276 | 0.8560 | 79.64 | 76.07 | 1（正确） | 87.70 | 84.32 |
| ARM-A-INT8 | 板端转换 | Path A | normal / per-channel | 4.58 | 300 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | 0.8628 | 0.8469 | 58.15 | 58.94 | 2（重复框） | 41.89 | 41.78 |
| ARM-B-FP16 | 板端转换 | Path B | 不量化 | 7.19 | 300 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | 0.9276 | 0.8560 | 106.75 | 109.59 | 1（正确） | 84.36 | 75.75 |
| ARM-B-INT8 | 板端转换 | Path B | normal / per-channel | 4.58 | 300 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | 0.8633 | 0.8469 | 58.60 | 59.54 | 2（重复框） | 45.20 | 45.97 |
| ARM-B-INT8-layer | 板端转换 | Path B | normal / per-tensor | 4.55 | 300 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | 0.8646 | 0.8466 | 58.23 | 59.51 | 2（重复框） | 44.87 | 46.95 |
| ARM-B-Hybrid | 板端转换 | Path B | normal / auto-hybrid | 4.64 | 300 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | 0.8633 | 0.8469 | 60.70 | 62.10 | 2（重复框） | 44.39 | 43.95 |
| WSL2-A-FP16 | WSL2 转换 | Path A | 不量化 | 7.19 | 300 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | 0.9276 | 0.8560 | 104.41 | 108.91 | 1（正确） | 84.83 | 75.71 |
| WSL2-A-INT8 | WSL2 转换 | Path A | normal / per-channel | 4.58 | 300 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | 0.8624 | 0.8471 | 58.65 | 59.38 | 1（正确） | 41.23 | 42.05 |
| WSL2-B-FP16 | WSL2 转换 | Path B | 不量化 | 7.19 | 300 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | 0.9276 | 0.8560 | 106.18 | 109.05 | 1（正确） | 83.83 | 75.62 |
| WSL2-B-INT8 | WSL2 转换 | Path B | normal / per-channel | 4.58 | 300 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | 0.8623 | 0.8471 | 58.44 | 59.91 | 1（正确） | 42.40 | 43.23 |
| WSL2-B-INT8-layer | WSL2 转换 | Path B | normal / per-tensor | 4.55 | 300 / 1 / 0 | 0.9967 | 1.0000 | 0.9983 | 0.8648 | 0.8472 | 58.23 | 58.73 | 1（正确） | 43.37 | 43.57 |
| WSL2-B-Hybrid | WSL2 转换 | Path B | normal / auto-hybrid | 4.66 | 300 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | 0.8619 | 0.8467 | 60.13 | 62.06 | 1（正确） | 44.15 | 44.58 |
| WSL2-B-INT8-mmse | WSL2 转换 | Path B | mmse / per-channel | 4.58 | 300 / 1 / 0 | 0.9967 | 1.0000 | 0.9983 | 0.8112 | 0.8222 | 58.64 | 59.26 | 1（正确） | 41.88 | 43.28 |
| WSL2-B-INT8-kl | WSL2 转换 | Path B | kl_divergence / per-channel | 4.58 | 300 / 0 / 0 | 1.0000 | 1.0000 | 1.0000 | 0.8406 | 0.8260 | 58.15 | 59.51 | 2（重复框） | 43.87 | 43.96 |

## 4. 300 张 XML 真值集分析

### 4.1 FP16

所有 FP16 模型的结果一致：`TP=300`、`FP=0`、`FN=0`、平均 IoU `0.9276`、平均 score `0.8560`。这说明 Path A 和 Path B 在浮点模型上没有可观测精度差异。

### 4.2 normal INT8

- 板端 Path A：IoU `0.8628`；板端 Path B：IoU `0.8633`。
- WSL2 Path A：IoU `0.8624`；WSL2 Path B：IoU `0.8623`。
- 四个 normal INT8 在 300 图上均为 0 FP、0 FN，IoU 差异最大仅约 `0.0010`，整体精度近似等价。
- 量化损失集中在定位：平均 IoU 相比 FP16 下降约 `0.0643–0.0653`，平均 score 只下降约 `0.009`。

### 4.3 layer、Hybrid、MMSE 与 KL

- 板端 `INT8-layer`：IoU `0.8646`，300 图无 FP，但目标单图有重复框。
- WSL2 `INT8-layer`：IoU `0.8648`，略高于 normal，但出现 1 个 FP。
- WSL2 Hybrid：IoU `0.8619`，速度也没有优于 normal，自动混合精度没有带来收益。
- MMSE：IoU `0.8112`，较 normal 再下降 `0.0511`，平均 score 也降至 `0.8222`。MMSE 并不保证检测模型更准，本模型检测头存在权重离群值，且其最小均方误差目标不等价于最大化最终框 IoU。
- KL：IoU `0.8406`，低于 normal，目标单图也产生重复框。

## 5. `3C_Test.jpg` 单图分析

### 5.1 检测数

- 板端转换：FP16 为 1 个框；Path A INT8、Path B INT8、INT8-layer、Hybrid 均为 **2 个框**。
- WSL2 转换：FP16、Path A INT8、Path B normal INT8、INT8-layer、Hybrid、MMSE 均为 **1 个框**；只有 KL 为 **2 个框**。
- 板端 normal INT8 的两个框 IoU 约 `0.4386`，略低于 NMS 阈值 `0.45`，因此没有互相抑制。

### 5.2 主要框坐标

| 模型组 | score | box `(x1, y1, x2, y2)` | 与 FP16 框 IoU |
|---|---:|---|---:|
| FP16 | 0.8110 | `(654.0, 1582.5, 1886.0, 1833.5)` | 1.0000 |
| normal INT8 主框 | 0.8184 | `(637.0, 1612.2, 1911.0, 1881.2)` | 0.7189 |
| 板端 normal INT8 重复框 | 0.7409 | `(679.5, 1513.1, 1868.5, 1782.1)` | 约 0.6058 |
| WSL2 MMSE 主框 | 0.8153 | `(715.4, 1622.7, 1883.7, 1885.2)` | 0.6661 |
| WSL2 KL 最佳匹配框 | 0.8007 | `(678.5, 1553.0, 1950.8, 1821.6)` | 0.7972 |

KL 的最佳匹配框与 FP16 更接近，但它仍输出第二个高分框，因此不能只看单个最佳框 IoU。MMSE 虽然只输出一个框，但定位误差更大，这与其 300 图平均 IoU 最低相一致。

### 5.3 单图输出图位置

- `ARM-A-FP16`：`../outputs/final_summary/single_image/ARM-A-FP16_result.jpg`
- `ARM-A-INT8`：`../outputs/final_summary/single_image/ARM-A-INT8_result.jpg`
- `ARM-B-FP16`：`../outputs/final_summary/single_image/ARM-B-FP16_result.jpg`
- `ARM-B-INT8`：`../outputs/final_summary/single_image/ARM-B-INT8_result.jpg`
- `ARM-B-INT8-layer`：`../outputs/final_summary/single_image/ARM-B-INT8-layer_result.jpg`
- `ARM-B-Hybrid`：`../outputs/final_summary/single_image/ARM-B-Hybrid_result.jpg`
- `WSL2-A-FP16`：`../outputs/final_summary/single_image/WSL2-A-FP16_result.jpg`
- `WSL2-A-INT8`：`../outputs/final_summary/single_image/WSL2-A-INT8_result.jpg`
- `WSL2-B-FP16`：`../outputs/final_summary/single_image/WSL2-B-FP16_result.jpg`
- `WSL2-B-INT8`：`../outputs/final_summary/single_image/WSL2-B-INT8_result.jpg`
- `WSL2-B-INT8-layer`：`../outputs/final_summary/single_image/WSL2-B-INT8-layer_result.jpg`
- `WSL2-B-Hybrid`：`../outputs/final_summary/single_image/WSL2-B-Hybrid_result.jpg`
- `WSL2-B-INT8-mmse`：`../outputs/final_summary/single_image/WSL2-B-INT8-mmse_result.jpg`
- `WSL2-B-INT8-kl`：`../outputs/final_summary/single_image/WSL2-B-INT8-kl_result.jpg`

HTML 版报告已内嵌全部输出图，可直接离线展示。

## 6. Path A 与 Path B 对比

| 维度 | Path A | Path B |
|---|---|---|
| 工作流 | Ultralytics `yolo export format=rknn` 一步完成 | `.pt → ONNX → RKNN` 分步完成 |
| FP16 精度 | 与 Path B 相同 | 与 Path A 相同 |
| normal INT8 精度 | 与 Path B 基本相同 | 与 Path A 基本相同 |
| 中间 ONNX | 一般不作为稳定产物保留 | 明确保留，便于复现和诊断 |
| 算法/粒度控制 | 较少 | normal/mmse/KL、channel/layer、Hybrid |
| 推荐用途 | 快速转换 | 正式部署、实验对比和后续优化 |

## 7. 板端转换与 WSL2 转换对比

- 300 图整体指标几乎相同：板端 Path B normal INT8 IoU `0.8633`，WSL2 为 `0.8623`。
- 目标单图行为不同：板端 normal INT8 为 2 框，WSL2 normal INT8 为 1 框。
- 观测到模型编译器构建不同：板端模型为 `2.3.2 (@2025-04-03...)`，WSL2 模型为 `2.3.2 (e045de294f@2025-04-07...)`。但不能仅据此断定差异完全由编译器造成；校准图片遍历顺序、ONNX 产物、Toolkit wheel 构建和转换环境也可能影响量化参数。
- 基于本次实际结果，**WSL2 生成的 normal per-channel INT8 更适合部署**。

## 8. 最终选择矩阵

| 目标 | 推荐模型 | 理由 |
|---|---|---|
| 综合部署（首选） | `WSL2-B-INT8` | 300 图 0 FP/0 FN；单图 1 框；速度快；体积小；流程可复现 |
| 定位精度最高 | `WSL2-B-FP16` | IoU 0.9276，0 FP/0 FN，单图正确 |
| 只追求单图最低延迟 | 不建议仅按最低数字选型 | 板端 INT8 虽快，但目标单图重复框；应先满足正确性 |
| 快速简单转换 | `WSL2-A-INT8` | 与 Path B normal 基本等价，且单图正确 |
| 继续量化研究 | 以 `WSL2-B-INT8` 为基线 | 当前 MMSE/KL/Hybrid 均未改善综合指标 |

## 9. 原始产物

- 统一 300 图 Excel：`../outputs/final_summary/dataset_eval/precision_loss_eval.xlsx`
- 统一 300 图原始 HTML：`../outputs/final_summary/dataset_eval/precision_loss_eval.html`
- 统一单图 Excel：`../outputs/final_summary/single_image/precision_benchmark.xlsx`
- 统一单图原始 HTML：`../outputs/final_summary/single_image/precision_benchmark.html`
- 本综合报告 HTML：`RKNN_COMPREHENSIVE_EVALUATION.html`
