"""Benchmark FP16 / INT8 / hybrid RKNN models on a test image.

Runs each model, records per-stage latency and detections, compares INT8/hybrid
against FP16 as reference, and writes an Excel + HTML report.
"""
import argparse
import base64
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yolo11.config import load_config, load_labels
from src.yolo11.draw import draw_detections
from src.yolo11.postprocess import postprocess_outputs, iou
from src.yolo11.preprocess import preprocess_image
from src.yolo11.rknn_infer import RknnLiteDetector


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark RKNN precisions on one image.")
    p.add_argument("--models", nargs="+", required=True, help="RKNN model paths.")
    p.add_argument("--names", nargs="+", default=None, help="Display names for each model.")
    p.add_argument("--image", default="3C_Test.jpg", help="Test image path.")
    p.add_argument("--config", default="configs/coco.yaml", help="YAML config path.")
    p.add_argument("--warmup", type=int, default=5, help="Warmup runs (not counted).")
    p.add_argument("--runs", type=int, default=30, help="Timed runs.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--nms", type=float, default=0.45, help="NMS threshold.")
    p.add_argument("--output-dir", default="outputs/benchmark", help="Output directory.")
    return p.parse_args()


def _resolve_coordinate_format(model_path: Path) -> str:
    metadata_path = model_path.with_suffix(f"{model_path.suffix}.yaml")
    if metadata_path.is_file():
        data = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        coord = data.get("output_coordinates")
        if coord in {"pixels", "normalized"}:
            return coord
    return "auto"


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000.0


def run_model(model_path: Path, image: np.ndarray, config, conf, nms, warmup, runs):
    coord_fmt = _resolve_coordinate_format(model_path)
    pre_times, infer_times, post_times = [], [], []
    detections = []
    with RknnLiteDetector(model_path) as detector:
        total = warmup + runs
        for i in range(total):
            (input_tensor, meta), pre_t = _timed(preprocess_image, image, config.img_size)
            outputs, infer_t = _timed(detector.infer, input_tensor)
            dets, post_t = _timed(postprocess_outputs, outputs, meta, conf, nms, coord_fmt)
            if i >= warmup:
                pre_times.append(pre_t)
                infer_times.append(infer_t)
                post_times.append(post_t)
                detections = dets
    total_times = [pre_times[i] + infer_times[i] + post_times[i] for i in range(runs)]
    return {
        "detections": detections,
        "coord_fmt": coord_fmt,
        "pre_ms": pre_times,
        "infer_ms": infer_times,
        "post_ms": post_times,
        "total_ms": total_times,
    }


def _stats(times):
    return {
        "avg": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "p50": statistics.median(times),
        "std": statistics.pstdev(times),
    }


def match_detections(ref_dets, other_dets, iou_thr=0.5):
    """Match other_dets to ref_dets by greedy IoU. Returns list of match dicts."""
    matches = []
    used = [False] * len(other_dets)
    for rd in ref_dets:
        best_j, best_iou = -1, iou_thr
        for j, od in enumerate(other_dets):
            if used[j]:
                continue
            v = iou(rd.box, od.box)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
            od = other_dets[best_j]
            box_diff = [abs(a - b) for a, b in zip(rd.box, od.box)]
            matches.append({
                "ref_score": rd.score, "other_score": od.score,
                "score_diff": abs(rd.score - od.score),
                "iou": best_iou,
                "box_diff_x1": box_diff[0], "box_diff_y1": box_diff[1],
                "box_diff_x2": box_diff[2], "box_diff_y2": box_diff[3],
            })
    unmatched_ref = sum(1 for i, rd in enumerate(ref_dets) if not any(used[j] for j in range(len(other_dets)) if iou(rd.box, other_dets[j].box) > iou_thr))
    unmatched_other = sum(1 for j in range(len(other_dets)) if not used[j])
    return matches, len(ref_dets) - len(matches), unmatched_other


def img_to_base64(img_path: Path, max_w=480):
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_excel(results, ref_name, image_path, out_xlsx):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    hdr_fill = PatternFill("solid", fgColor="4472C4")
    hdr_font = Font(color="FFFFFF", bold=True)

    # Summary sheet
    ws.append(["RKNN 精度对比报告"])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([])
    ws.append(["测试图片", str(image_path)])
    ws.append(["参考精度", ref_name])
    ws.append([])
    headers = ["精度", "检测数", "推理 avg(ms)", "推理 min(ms)", "推理 max(ms)",
               "推理 p50(ms)", "推理 std(ms)", "总耗时 avg(ms)", "坐标格式"]
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        ws.cell(row=ws.max_row, column=c).fill = hdr_fill
        ws.cell(row=ws.max_row, column=c).font = hdr_font
    for name, r in results.items():
        ist = _stats(r["infer_ms"])
        tst = _stats(r["total_ms"])
        ws.append([name, len(r["detections"]), round(ist["avg"], 3), round(ist["min"], 3),
                   round(ist["max"], 3), round(ist["p50"], 3), round(ist["std"], 3),
                   round(tst["avg"], 3), r["coord_fmt"]])
    for col in "ABCDEFGHI":
        ws.column_dimensions[col].width = 16

    # Detections sheet
    ws2 = wb.create_sheet("Detections")
    ws2.append(["精度", "序号", "class_id", "class_name", "score", "x1", "y1", "x2", "y2"])
    for c in range(1, 10):
        ws2.cell(row=1, column=c).fill = hdr_fill
        ws2.cell(row=1, column=c).font = hdr_font
    labels = load_config("configs/coco.yaml").labels_path
    label_names = load_labels(labels)
    for name, r in results.items():
        for i, d in enumerate(r["detections"]):
            ws2.append([name, i, d.class_id, label_names[d.class_id] if d.class_id < len(label_names) else "?",
                        round(d.score, 4), round(d.box[0], 1), round(d.box[1], 1),
                        round(d.box[2], 1), round(d.box[3], 1)])
    for col in "ABCDEFGHI":
        ws2.column_dimensions[col].width = 14

    # Accuracy comparison sheet
    ws3 = wb.create_sheet("Accuracy vs FP16")
    ws3.append(["精度", "匹配检测数", "未匹配(参考)", "未匹配(其他)", "平均IoU", "平均分数差", "最大分数差", "平均框坐标差(px)"])
    for c in range(1, 9):
        ws3.cell(row=1, column=c).fill = hdr_fill
        ws3.cell(row=1, column=c).font = hdr_font
    ref_dets = results[ref_name]["detections"]
    for name, r in results.items():
        if name == ref_name:
            ws3.append([name, len(ref_dets), 0, 0, 1.0, 0.0, 0.0, 0.0])
            continue
        matches, un_ref, un_other = match_detections(ref_dets, r["detections"])
        if matches:
            avg_iou = statistics.mean(m["iou"] for m in matches)
            avg_sd = statistics.mean(m["score_diff"] for m in matches)
            max_sd = max(m["score_diff"] for m in matches)
            avg_box = statistics.mean([m["box_diff_x1"] + m["box_diff_y1"] + m["box_diff_x2"] + m["box_diff_y2"] for m in matches]) / 4
        else:
            avg_iou = avg_sd = max_sd = avg_box = 0.0
        ws3.append([name, len(matches), un_ref, un_other, round(avg_iou, 4), round(avg_sd, 4), round(max_sd, 4), round(avg_box, 2)])
    for col in "ABCDEFGH":
        ws3.column_dimensions[col].width = 18

    # Per-match detail sheet
    ws4 = wb.create_sheet("Match Details")
    ws4.append(["精度", "参考score", "其他score", "分数差", "IoU", "x1差", "y1差", "x2差", "y2差"])
    for c in range(1, 10):
        ws4.cell(row=1, column=c).fill = hdr_fill
        ws4.cell(row=1, column=c).font = hdr_font
    for name, r in results.items():
        if name == ref_name:
            continue
        matches, _, _ = match_detections(ref_dets, r["detections"])
        for m in matches:
            ws4.append([name, round(m["ref_score"], 4), round(m["other_score"], 4), round(m["score_diff"], 4),
                        round(m["iou"], 4), round(m["box_diff_x1"], 1), round(m["box_diff_y1"], 1),
                        round(m["box_diff_x2"], 1), round(m["box_diff_y2"], 1)])
    for col in "ABCDEFGHI":
        ws4.column_dimensions[col].width = 13

    wb.save(out_xlsx)


def build_html(results, ref_name, image_path, out_html, annotated_paths, bench_args):
    labels = load_labels(load_config("configs/coco.yaml").labels_path)
    img_b64 = img_to_base64(image_path)

    rows_html = ""
    for name, r in results.items():
        ist = _stats(r["infer_ms"])
        tst = _stats(r["total_ms"])
        dets_html = ""
        for d in r["detections"]:
            lbl = labels[d.class_id] if d.class_id < len(labels) else "?"
            dets_html += f"<tr><td>{d.class_id}</td><td>{lbl}</td><td>{d.score:.4f}</td><td>{d.box[0]:.1f}</td><td>{d.box[1]:.1f}</td><td>{d.box[2]:.1f}</td><td>{d.box[3]:.1f}</td></tr>"
        ann_b64 = img_to_base64(annotated_paths[name]) if annotated_paths.get(name) else ""
        is_ref = (name == ref_name)
        badge = '<span class="badge ref">参考</span>' if is_ref else ""
        rows_html += f"""
        <div class="card {'ref-card' if is_ref else ''}">
          <h2>{name} {badge}</h2>
          <div class="card-body">
            <div class="metrics">
              <div class="metric"><span class="label">检测数</span><span class="value">{len(r['detections'])}</span></div>
              <div class="metric"><span class="label">推理 avg</span><span class="value">{ist['avg']:.2f} ms</span></div>
              <div class="metric"><span class="label">推理 min</span><span class="value">{ist['min']:.2f} ms</span></div>
              <div class="metric"><span class="label">推理 max</span><span class="value">{ist['max']:.2f} ms</span></div>
              <div class="metric"><span class="label">推理 p50</span><span class="value">{ist['p50']:.2f} ms</span></div>
              <div class="metric"><span class="label">总耗时 avg</span><span class="value">{tst['avg']:.2f} ms</span></div>
              <div class="metric"><span class="label">坐标格式</span><span class="value">{r['coord_fmt']}</span></div>
            </div>
            <div class="img-row">
              <div class="img-box"><h4>检测结果</h4>{f'<img src="data:image/jpeg;base64,{ann_b64}"/>' if ann_b64 else '<p>无</p>'}</div>
            </div>
            <table class="det-table"><tr><th>class</th><th>名称</th><th>score</th><th>x1</th><th>y1</th><th>x2</th><th>y2</th></tr>{dets_html}</table>
          </div>
        </div>"""

    # accuracy comparison table
    ref_dets = results[ref_name]["detections"]
    acc_rows = ""
    for name, r in results.items():
        if name == ref_name:
            acc_rows += f"<tr class='ref-row'><td>{name}</td><td>{len(ref_dets)}</td><td>0</td><td>0</td><td>1.0000</td><td>0.0000</td><td>0.0000</td><td>0.00</td></tr>"
            continue
        matches, un_ref, un_other = match_detections(ref_dets, r["detections"])
        if matches:
            avg_iou = statistics.mean(m["iou"] for m in matches)
            avg_sd = statistics.mean(m["score_diff"] for m in matches)
            max_sd = max(m["score_diff"] for m in matches)
            avg_box = statistics.mean([m["box_diff_x1"] + m["box_diff_y1"] + m["box_diff_x2"] + m["box_diff_y2"] for m in matches]) / 4
        else:
            avg_iou = avg_sd = max_sd = avg_box = 0.0
        acc_rows += f"<tr><td>{name}</td><td>{len(matches)}</td><td>{un_ref}</td><td>{un_other}</td><td>{avg_iou:.4f}</td><td>{avg_sd:.4f}</td><td>{max_sd:.4f}</td><td>{avg_box:.2f}</td></tr>"

    # latency bar chart data
    names = list(results.keys())
    avgs = [round(_stats(r["infer_ms"])["avg"], 2) for r in results.values()]
    p50s = [round(_stats(r["infer_ms"])["p50"], 2) for r in results.values()]

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>RKNN 精度对比报告</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 0; background: #f5f7fa; color: #333; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #4472C4; padding-bottom: 10px; }}
.info {{ background: #fff; padding: 14px 18px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.info b {{ color: #4472C4; }}
.card {{ background: #fff; border-radius: 10px; margin-bottom: 22px; box-shadow: 0 2px 6px rgba(0,0,0,.1); overflow: hidden; }}
.card h2 {{ margin: 0; padding: 14px 20px; background: #f0f4f8; color: #2c3e50; }}
.ref-card {{ border: 2px solid #4472C4; }}
.ref-card h2 {{ background: #4472C4; color: #fff; }}
.badge {{ font-size: 12px; padding: 2px 8px; border-radius: 10px; margin-left: 8px; }}
.badge.ref {{ background: #fff; color: #4472C4; }}
.card-body {{ padding: 18px 20px; }}
.metrics {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }}
.metric {{ background: #f8f9fa; border-radius: 6px; padding: 10px 14px; min-width: 120px; }}
.metric .label {{ display: block; font-size: 12px; color: #888; }}
.metric .value {{ display: block; font-size: 18px; font-weight: 600; color: #2c3e50; }}
.img-row {{ display: flex; gap: 16px; margin-bottom: 14px; flex-wrap: wrap; }}
.img-box {{ flex: 1; min-width: 280px; }}
.img-box h4 {{ margin: 0 0 8px; color: #555; }}
.img-box img {{ max-width: 100%; border-radius: 6px; border: 1px solid #eee; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #e0e0e0; padding: 6px 10px; text-align: left; }}
th {{ background: #f0f4f8; }}
.det-table {{ margin-top: 8px; }}
.ref-row {{ background: #eef4ff; font-weight: 600; }}
.chart-box {{ background: #fff; padding: 18px; border-radius: 10px; margin-bottom: 22px; box-shadow: 0 2px 6px rgba(0,0,0,.1); }}
canvas {{ max-width: 100%; }}
</style></head><body><div class="container">
<h1>RKNN 精度对比报告</h1>
<div class="info">
  <p><b>测试模型:</b> 3C_SDG_stage1_1_best_v2.pt &nbsp; <b>测试图片:</b> {image_path} &nbsp; <b>参考精度:</b> {ref_name}</p>
  <p><b>warmup:</b> {bench_args.warmup} &nbsp; <b>runs:</b> {bench_args.runs} &nbsp; <b>conf:</b> {bench_args.conf} &nbsp; <b>nms:</b> {bench_args.nms}</p>
  <div class="img-row" style="margin-top:10px;"><div class="img-box"><h4>原图</h4>{f'<img src="data:image/jpeg;base64,{img_b64}"/>' if img_b64 else '<p>无</p>'}</div></div>
</div>
<div class="chart-box"><h2>推理延迟对比 (ms)</h2><canvas id="latencyChart" height="120"></canvas></div>
<h2 style="color:#2c3e50;">精度差异对比 (以 {ref_name} 为参考)</h2>
<table style="margin-bottom:22px;">
<tr><th>精度</th><th>匹配检测数</th><th>未匹配(参考)</th><th>未匹配(其他)</th><th>平均IoU</th><th>平均分数差</th><th>最大分数差</th><th>平均框坐标差(px)</th></tr>
{acc_rows}
</table>
{rows_html}
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<script>
const ctx = document.getElementById('latencyChart');
new Chart(ctx, {{
  type: 'bar',
  data: {{
    labels: {json.dumps(names)},
    datasets: [
      {{ label: 'avg (ms)', data: {json.dumps(avgs)}, backgroundColor: '#4472C4' }},
      {{ label: 'p50 (ms)', data: {json.dumps(p50s)}, backgroundColor: '#70AD47' }},
    ]
  }},
  options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true }} }} }}
}});
</script>
</div></body></html>"""
    out_html.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    config = load_config(args.config)
    labels = load_labels(config.labels_path)

    model_names = args.names or [Path(m).stem for m in args.models]
    if len(model_names) != len(args.models):
        raise ValueError("--names count must match --models count")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    annotated_paths = {}
    for model_path_str, name in zip(args.models, model_names):
        model_path = Path(model_path_str)
        print(f"\n=== Benchmarking {name}: {model_path} ===")
        r = run_model(model_path, image, config, args.conf, args.nms, args.warmup, args.runs)
        results[name] = r
        ist = _stats(r["infer_ms"])
        print(f"  detections: {len(r['detections'])}, infer avg: {ist['avg']:.2f} ms, p50: {ist['p50']:.2f} ms")
        # save annotated image
        ann = draw_detections(image, r["detections"], labels)
        ann_path = out_dir / f"{name}_result.jpg"
        cv2.imwrite(str(ann_path), ann)
        annotated_paths[name] = ann_path

    ref_name = model_names[0]  # FP16 as reference (first in list)
    out_xlsx = out_dir / "precision_benchmark.xlsx"
    out_html = out_dir / "precision_benchmark.html"
    build_excel(results, ref_name, image_path, out_xlsx)
    build_html(results, ref_name, image_path, out_html, annotated_paths, args)
    print(f"\nExcel: {out_xlsx}")
    print(f"HTML:  {out_html}")
