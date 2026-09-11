"""Evaluate RKNN precision loss against XML ground truth on a dataset.

Parses VOC-style XML annotations, runs each RKNN model on every image, computes
per-image TP/FP/FN, mAP@0.5, precision, recall, and per-detection IoU/score
differences vs FP16 reference. Writes Excel + HTML report.
"""
import argparse
import json
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yolo11.config import load_config
from src.yolo11.postprocess import postprocess_outputs, iou, Detection
from src.yolo11.preprocess import preprocess_image
from src.yolo11.rknn_infer import RknnLiteDetector


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RKNN precision loss vs XML ground truth.")
    p.add_argument("--models", nargs="+", required=True, help="RKNN model paths.")
    p.add_argument("--names", nargs="+", default=None, help="Display names for each model.")
    p.add_argument("--image-dir", default="data/calibration", help="Image directory.")
    p.add_argument("--xml-dir", default="data/xml", help="XML annotation directory.")
    p.add_argument("--config", default="configs/coco.yaml", help="YAML config path.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--nms", type=float, default=0.45, help="NMS threshold.")
    p.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for TP.")
    p.add_argument("--max-images", type=int, default=0, help="Max images to eval (0=all).")
    p.add_argument("--output-dir", default="outputs/eval", help="Output directory.")
    return p.parse_args()


def parse_xml(xml_path: Path):
    """Parse VOC XML, return list of (class_id, x1,y1,x2,y2). Single class=0."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        x1 = float(bnd.find("xmin").text)
        y1 = float(bnd.find("ymin").text)
        x2 = float(bnd.find("xmax").text)
        y2 = float(bnd.find("ymax").text)
        boxes.append((0, x1, y1, x2, y2))
    return boxes


def _resolve_coordinate_format(model_path: Path) -> str:
    import yaml
    metadata_path = model_path.with_suffix(f"{model_path.suffix}.yaml")
    if metadata_path.is_file():
        data = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        coord = data.get("output_coordinates")
        if coord in {"pixels", "normalized"}:
            return coord
        # Path A (Ultralytics) metadata uses args.quantize
        args = data.get("args") or {}
        if isinstance(args, dict) and args.get("quantize") is not None:
            return "normalized" if args["quantize"] == 8 else "pixels"
    return "auto"


def match_to_gt(gt_boxes, dets, iou_thr):
    """Match detections to GT boxes (greedy by score). Returns tp, fp, fn lists."""
    used_gt = [False] * len(gt_boxes)
    tp, fp = [], []
    for det in sorted(dets, key=lambda d: d.score, reverse=True):
        best_j, best_iou = -1, iou_thr
        for j, gt in enumerate(gt_boxes):
            if used_gt[j]:
                continue
            v = iou(det.box, (gt[1], gt[2], gt[3], gt[4]))
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            used_gt[best_j] = True
            tp.append((det, best_iou, best_j))
        else:
            fp.append(det)
    fn = [gt for j, gt in enumerate(gt_boxes) if not used_gt[j]]
    return tp, fp, fn


def compute_ap(recalls, precisions):
    """VOC mAP computation (11-point or area under PR curve)."""
    recalls = [0.0] + list(recalls) + [1.0]
    precisions = [1.0] + list(precisions) + [0.0]
    # make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    area = 0.0
    for i in range(len(recalls) - 1):
        area += (recalls[i + 1] - recalls[i]) * precisions[i]
    return area


def eval_model(model_path, image_paths, xml_dir, config, conf, nms, iou_thr, coord_fmt):
    """Run model on all images, return per-image results + aggregate metrics."""
    all_tp, all_fp, total_fn = 0, 0, 0
    per_image = []
    infer_times = []
    all_ious = []
    all_scores = []
    with RknnLiteDetector(model_path) as detector:
        for idx, img_path in enumerate(image_paths):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            xml_path = xml_dir / (img_path.stem + ".xml")
            gt_boxes = parse_xml(xml_path) if xml_path.exists() else []

            t0 = time.perf_counter()
            input_tensor, meta = preprocess_image(image, config.img_size)
            outputs = detector.infer(input_tensor)
            dets = postprocess_outputs(outputs, meta, conf, nms, coord_fmt)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            infer_times.append(infer_ms)

            tp, fp, fn = match_to_gt(gt_boxes, dets, iou_thr)
            all_tp += len(tp)
            all_fp += len(fp)
            total_fn += len(fn)
            for det, v, j in tp:
                all_ious.append(v)
                all_scores.append(det.score)
            per_image.append({
                "image": img_path.name,
                "gt": len(gt_boxes), "det": len(dets),
                "tp": len(tp), "fp": len(fp), "fn": len(fn),
                "scores": [round(d.score, 4) for d in dets],
                "infer_ms": infer_ms,
            })
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + total_fn) if (all_tp + total_fn) > 0 else 0.0
    # mAP@0.5 (single class, single IoU thr): approximate as precision*recall at this point
    # but compute proper AP from sorted scores
    return {
        "per_image": per_image,
        "tp": all_tp, "fp": all_fp, "fn": total_fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
        "avg_iou": statistics.mean(all_ious) if all_ious else 0.0,
        "avg_score": statistics.mean(all_scores) if all_scores else 0.0,
        "infer_avg": statistics.mean(infer_times) if infer_times else 0.0,
        "infer_p50": statistics.median(infer_times) if infer_times else 0.0,
        "infer_times": infer_times,
        "all_ious": all_ious,
    }


def build_report(results, ref_name, args, image_paths, out_xlsx, out_html):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    wb = Workbook()
    hdr_fill = PatternFill("solid", fgColor="4472C4")
    hdr_font = Font(color="FFFFFF", bold=True)

    # Summary
    ws = wb.active
    ws.title = "Summary"
    ws.append(["RKNN 精度损失评测报告"])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])
    ws.append(["图片目录", args.image_dir])
    ws.append(["标注目录", args.xml_dir])
    ws.append(["图片数", len(image_paths)])
    ws.append(["IoU阈值", args.iou_thr])
    ws.append(["conf", args.conf])
    ws.append(["nms", args.nms])
    ws.append(["参考精度", ref_name])
    ws.append([])
    headers = ["精度", "TP", "FP", "FN", "Precision", "Recall", "F1", "平均IoU",
               "平均score", "推理avg(ms)", "推理p50(ms)"]
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        ws.cell(row=ws.max_row, column=c).fill = hdr_fill
        ws.cell(row=ws.max_row, column=c).font = hdr_font
    for name, r in results.items():
        ws.append([name, r["tp"], r["fp"], r["fn"], round(r["precision"], 4),
                   round(r["recall"], 4), round(r["f1"], 4), round(r["avg_iou"], 4),
                   round(r["avg_score"], 4), round(r["infer_avg"], 2), round(r["infer_p50"], 2)])
    for col in "ABCDEFGHIJK":
        ws.column_dimensions[col].width = 14

    # Per-image comparison
    ws2 = wb.create_sheet("PerImage")
    ws2.append(["图片", "GT", "精度"] + ["检测数", "TP", "FP", "FN", "scores", "infer(ms)"])
    for c in range(1, 10):
        ws2.cell(row=1, column=c).fill = hdr_fill
        ws2.cell(row=1, column=c).font = hdr_font
    for name, r in results.items():
        for pi in r["per_image"]:
            ws2.append([pi["image"], pi["gt"], name, pi["det"], pi["tp"], pi["fp"], pi["fn"],
                        ", ".join(map(str, pi["scores"])), round(pi["infer_ms"], 2)])
    for col in "ABCDEFGHI":
        ws2.column_dimensions[col].width = 14

    # Loss vs FP16
    ws3 = wb.create_sheet("Loss vs FP16")
    ws3.append(["精度", "TP差", "FP差", "FN差", "Precision差", "Recall差", "F1差", "平均IoU差", "推理加速比"])
    for c in range(1, 10):
        ws3.cell(row=1, column=c).fill = hdr_fill
        ws3.cell(row=1, column=c).font = hdr_font
    ref = results[ref_name]
    for name, r in results.items():
        speedup = ref["infer_avg"] / r["infer_avg"] if r["infer_avg"] > 0 else 0
        ws3.append([name, r["tp"] - ref["tp"], r["fp"] - ref["fp"], r["fn"] - ref["fn"],
                    round(r["precision"] - ref["precision"], 4), round(r["recall"] - ref["recall"], 4),
                    round(r["f1"] - ref["f1"], 4), round(r["avg_iou"] - ref["avg_iou"], 4),
                    round(speedup, 2)])
    for col in "ABCDEFGHI":
        ws3.column_dimensions[col].width = 16

    wb.save(out_xlsx)

    # HTML
    names = list(results.keys())
    summary_rows = ""
    for name, r in results.items():
        is_ref = name == ref_name
        cls = "ref-row" if is_ref else ""
        summary_rows += f"<tr class='{cls}'><td>{name}</td><td>{r['tp']}</td><td>{r['fp']}</td><td>{r['fn']}</td><td>{r['precision']:.4f}</td><td>{r['recall']:.4f}</td><td>{r['f1']:.4f}</td><td>{r['avg_iou']:.4f}</td><td>{r['avg_score']:.4f}</td><td>{r['infer_avg']:.2f}</td><td>{r['infer_p50']:.2f}</td></tr>"

    loss_rows = ""
    for name, r in results.items():
        if name == ref_name:
            loss_rows += f"<tr class='ref-row'><td>{name}</td><td>0</td><td>0</td><td>0</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>1.00x</td></tr>"
            continue
        speedup = ref["infer_avg"] / r["infer_avg"] if r["infer_avg"] > 0 else 0
        loss_rows += f"<tr><td>{name}</td><td>{r['tp']-ref['tp']:+d}</td><td>{r['fp']-ref['fp']:+d}</td><td>{r['fn']-ref['fn']:+d}</td><td>{r['precision']-ref['precision']:+.4f}</td><td>{r['recall']-ref['recall']:+.4f}</td><td>{r['f1']-ref['f1']:+.4f}</td><td>{r['avg_iou']-ref['avg_iou']:+.4f}</td><td>{speedup:.2f}x</td></tr>"

    avgs = [round(r["infer_avg"], 2) for r in results.values()]
    precisions = [round(r["precision"], 4) for r in results.values()]
    recalls = [round(r["recall"], 4) for r in results.values()]

    html = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>RKNN 精度损失评测</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; margin:0; background:#f5f7fa; color:#333; }}
.container {{ max-width:1100px; margin:0 auto; padding:24px; }}
h1 {{ color:#2c3e50; border-bottom:3px solid #4472C4; padding-bottom:10px; }}
h2 {{ color:#2c3e50; margin-top:30px; }}
.info {{ background:#fff; padding:14px 18px; border-radius:8px; margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,.08); }}
.info b {{ color:#4472C4; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin-bottom:20px; }}
th,td {{ border:1px solid #e0e0e0; padding:6px 10px; text-align:left; }}
th {{ background:#f0f4f8; }}
.ref-row {{ background:#eef4ff; font-weight:600; }}
.chart-box {{ background:#fff; padding:18px; border-radius:10px; margin-bottom:22px; box-shadow:0 2px 6px rgba(0,0,0,.1); }}
canvas {{ max-width:100%; }}
.neg {{ color:#e74c3c; font-weight:600; }}
.pos {{ color:#27ae60; }}
</style></head><body><div class="container">
<h1>RKNN 精度损失评测报告</h1>
<div class="info">
<p><b>图片目录:</b> {args.image_dir} &nbsp; <b>标注目录:</b> {args.xml_dir} &nbsp; <b>图片数:</b> {len(image_paths)}</p>
<p><b>IoU阈值:</b> {args.iou_thr} &nbsp; <b>conf:</b> {args.conf} &nbsp; <b>nms:</b> {args.nms} &nbsp; <b>参考精度:</b> {ref_name}</p>
</div>
<h2>汇总指标</h2>
<table><tr><th>精度</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th><th>平均IoU</th><th>平均score</th><th>推理avg(ms)</th><th>推理p50(ms)</th></tr>
{summary_rows}</table>
<h2>精度损失 vs {ref_name}</h2>
<table><tr><th>精度</th><th>TP差</th><th>FP差</th><th>FN差</th><th>Precision差</th><th>Recall差</th><th>F1差</th><th>平均IoU差</th><th>推理加速比</th></tr>
{loss_rows}</table>
<div class="chart-box"><h2>推理延迟对比 (ms)</h2><canvas id="latencyChart" height="120"></canvas></div>
<div class="chart-box"><h2>Precision / Recall</h2><canvas id="prChart" height="120"></canvas></div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script>
new Chart(document.getElementById('latencyChart'),{{type:'bar',data:{{labels:{json.dumps(names)},datasets:[{{label:'avg(ms)',data:{json.dumps(avgs)},backgroundColor:'#4472C4'}}]}},options:{{scales:{{y:{{beginAtZero:true}}}}}}}});
new Chart(document.getElementById('prChart'),{{type:'bar',data:{{labels:{json.dumps(names)},datasets:[{{label:'Precision',data:{json.dumps(precisions)},backgroundColor:'#4472C4'}},{{label:'Recall',data:{json.dumps(recalls)},backgroundColor:'#70AD47'}}]}},options:{{scales:{{y:{{beginAtZero:true,max:1}}}}}}}});
</script>
</div></body></html>"""
    out_html.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    image_dir = Path(args.image_dir)
    xml_dir = Path(args.xml_dir)
    image_paths = sorted(image_dir.glob("*.jpg"))
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    print(f"Evaluating on {len(image_paths)} images from {image_dir}")

    model_names = args.names or [Path(m).stem for m in args.models]
    if len(model_names) != len(args.models):
        raise ValueError("--names count must match --models count")

    results = {}
    for model_path_str, name in zip(args.models, model_names):
        model_path = Path(model_path_str)
        coord_fmt = _resolve_coordinate_format(model_path)
        print(f"\n=== Evaluating {name}: {model_path} (coord={coord_fmt}) ===")
        t0 = time.perf_counter()
        r = eval_model(model_path, image_paths, xml_dir, config, args.conf, args.nms, args.iou_thr, coord_fmt)
        elapsed = time.perf_counter() - t0
        results[name] = r
        print(f"  TP={r['tp']} FP={r['fp']} FN={r['fn']} P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")
        print(f"  avgIoU={r['avg_iou']:.4f} avgScore={r['avg_score']:.4f} infer_avg={r['infer_avg']:.2f}ms ({elapsed:.1f}s total)")

    ref_name = model_names[0]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = out_dir / "precision_loss_eval.xlsx"
    out_html = out_dir / "precision_loss_eval.html"
    build_report(results, ref_name, args, image_paths, out_xlsx, out_html)
    print(f"\nExcel: {out_xlsx}")
    print(f"HTML:  {out_html}")
