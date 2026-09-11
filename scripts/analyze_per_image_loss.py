"""Per-image precision loss analysis: find images with largest IoU/score
differences between FP16 reference and INT8, visualize top-N worst cases."""
import argparse
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.yolo11.config import load_config, load_labels
from src.yolo11.draw import draw_detections
from src.yolo11.postprocess import postprocess_outputs, iou
from src.yolo11.preprocess import preprocess_image
from src.yolo11.rknn_infer import RknnLiteDetector


def parse_args():
    p = argparse.ArgumentParser(description="Per-image precision loss analysis.")
    p.add_argument("--ref-model", required=True, help="Reference RKNN model (FP16).")
    p.add_argument("--cmp-model", required=True, help="Compared RKNN model (INT8).")
    p.add_argument("--ref-name", default="FP16")
    p.add_argument("--cmp-name", default="INT8")
    p.add_argument("--image-dir", default="data/calibration")
    p.add_argument("--xml-dir", default="data/xml")
    p.add_argument("--config", default="configs/coco.yaml")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms", type=float, default=0.45)
    p.add_argument("--top-n", type=int, default=10, help="Visualize top-N worst images.")
    p.add_argument("--output-dir", default="outputs/per_image_analysis")
    return p.parse_args()


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        boxes.append((float(bnd.find("xmin").text), float(bnd.find("ymin").text),
                      float(bnd.find("xmax").text), float(bnd.find("ymax").text)))
    return boxes


def _coord_fmt(model_path):
    import yaml
    mp = model_path.with_suffix(f"{model_path.suffix}.yaml")
    if mp.is_file():
        data = yaml.safe_load(mp.read_text(encoding="utf-8")) or {}
        c = data.get("output_coordinates")
        if c in {"pixels", "normalized"}:
            return c
    return "auto"


def run_inference(detector, image, img_size, conf, nms, coord_fmt):
    input_tensor, meta = preprocess_image(image, img_size)
    outputs = detector.infer(input_tensor)
    dets = postprocess_outputs(outputs, meta, conf, nms, coord_fmt)
    return dets, meta


def match_pairs(ref_dets, cmp_dets, iou_thr=0.3):
    """Match detections between ref and cmp. Returns list of (rd, cd, iou_val)."""
    pairs = []
    used = [False] * len(cmp_dets)
    for rd in sorted(ref_dets, key=lambda d: d.score, reverse=True):
        best_j, best_v = -1, iou_thr
        for j, cd in enumerate(cmp_dets):
            if used[j]:
                continue
            v = iou(rd.box, cd.box)
            if v > best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
            pairs.append((rd, cmp_dets[best_j], best_v))
    unmatched_ref = [rd for i, rd in enumerate(ref_dets) if not any(p[0] is rd for p in pairs)]
    unmatched_cmp = [cd for j, cd in enumerate(cmp_dets) if not used[j]]
    return pairs, unmatched_ref, unmatched_cmp


def draw_comparison(image, gt_boxes, ref_dets, cmp_dets, labels, ref_name, cmp_name):
    """Draw GT (green), ref (blue), cmp (red) on the same image."""
    vis = image.copy()
    # GT boxes - green dashed
    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 4)
        cv2.putText(vis, "GT", (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
    # ref detections - blue
    for d in ref_dets:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 100, 0), 3)
        lbl = f"{ref_name} {d.score:.3f}"
        cv2.putText(vis, lbl, (int(x1), int(y1) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
    # cmp detections - red
    for d in cmp_dets:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        lbl = f"{cmp_name} {d.score:.3f}"
        cv2.putText(vis, lbl, (int(x1), int(y2) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis


def main():
    args = parse_args()
    config = load_config(args.config)
    labels = load_labels(config.labels_path)
    image_dir = Path(args.image_dir)
    xml_dir = Path(args.xml_dir)
    image_paths = sorted(image_dir.glob("*.jpg"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = Path(args.ref_model)
    cmp_path = Path(args.cmp_model)
    ref_fmt = _coord_fmt(ref_path)
    cmp_fmt = _coord_fmt(cmp_path)

    per_image_stats = []

    print(f"Running {args.ref_name} on {len(image_paths)} images...")
    ref_all = {}
    with RknnLiteDetector(ref_path) as det:
        for i, img_path in enumerate(image_paths):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            dets, meta = run_inference(det, image, config.img_size, args.conf, args.nms, ref_fmt)
            xml_path = xml_dir / (img_path.stem + ".xml")
            gt = parse_xml(xml_path) if xml_path.exists() else []
            ref_all[img_path.stem] = {"dets": dets, "gt": gt}

    print(f"Running {args.cmp_name} on {len(image_paths)} images...")
    cmp_all = {}
    with RknnLiteDetector(cmp_path) as det:
        for i, img_path in enumerate(image_paths):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            dets, meta = run_inference(det, image, config.img_size, args.conf, args.nms, cmp_fmt)
            cmp_all[img_path.stem] = {"dets": dets}

    print("Comparing per-image...")
    for stem in ref_all:
        rd = ref_all[stem]["dets"]
        cd = cmp_all.get(stem, {}).get("dets", [])
        gt = ref_all[stem]["gt"]
        pairs, un_ref, un_cmp = match_pairs(rd, cd)

        # IoU of ref vs GT
        ref_gt_ious = []
        for g in gt:
            best = max([iou(d.box, g) for d in rd], default=0.0)
            ref_gt_ious.append(best)
        cmp_gt_ious = []
        for g in gt:
            best = max([iou(d.box, g) for d in cd], default=0.0)
            cmp_gt_ious.append(best)

        # box coordinate differences between ref and cmp
        box_diffs = []
        score_diffs = []
        for r, c, v in pairs:
            bd = [abs(a - b) for a, b in zip(r.box, c.box)]
            box_diffs.append(sum(bd) / 4)
            score_diffs.append(abs(r.score - c.score))

        avg_ref_gt_iou = statistics.mean(ref_gt_ious) if ref_gt_ious else 0
        avg_cmp_gt_iou = statistics.mean(cmp_gt_ious) if cmp_gt_ious else 0
        avg_box_diff = statistics.mean(box_diffs) if box_diffs else 0
        avg_score_diff = statistics.mean(score_diffs) if score_diffs else 0
        iou_loss = avg_ref_gt_iou - avg_cmp_gt_iou

        per_image_stats.append({
            "stem": stem,
            "gt_count": len(gt),
            "ref_count": len(rd),
            "cmp_count": len(cd),
            "matched": len(pairs),
            "unmatched_ref": len(un_ref),
            "unmatched_cmp": len(un_cmp),
            "ref_gt_iou": avg_ref_gt_iou,
            "cmp_gt_iou": avg_cmp_gt_iou,
            "iou_loss": iou_loss,
            "box_diff_px": avg_box_diff,
            "score_diff": avg_score_diff,
            "ref_scores": [d.score for d in rd],
            "cmp_scores": [d.score for d in cd],
        })

    # sort by IoU loss descending
    per_image_stats.sort(key=lambda x: x["iou_loss"], reverse=True)

    # print top-10
    print(f"\n{'='*120}")
    print(f"Top-{args.top_n} images with largest IoU loss ({args.cmp_name} vs {args.ref_name}):")
    print(f"{'Image':<55} {'GT':>3} {'RefIoU':>7} {'CmpIoU':>7} {'IoULoss':>8} {'BoxDiff':>8} {'ScoreDiff':>9}")
    print(f"{'-'*120}")
    for s in per_image_stats[:args.top_n]:
        print(f"{s['stem']:<55} {s['gt_count']:>3} {s['ref_gt_iou']:>7.4f} {s['cmp_gt_iou']:>7.4f} {s['iou_loss']:>8.4f} {s['box_diff_px']:>8.2f} {s['score_diff']:>9.4f}")

    # visualize top-N
    print(f"\nVisualizing top-{args.top_n} worst images...")
    for idx, s in enumerate(per_image_stats[:args.top_n]):
        img_path = image_dir / (s["stem"] + ".jpg")
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        vis = draw_comparison(image, ref_all[s["stem"]]["gt"],
                              ref_all[s["stem"]]["dets"], cmp_all[s["stem"]]["dets"],
                              labels, args.ref_name, args.cmp_name)
        # add title
        title = f"#{idx+1} {s['stem']} | GT IoU: ref={s['ref_gt_iou']:.4f} cmp={s['cmp_gt_iou']:.4f} loss={s['iou_loss']:.4f} | box_diff={s['box_diff_px']:.1f}px"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(vis, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out_path = out_dir / f"worst_{idx+1:02d}_{s['stem'][:40]}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  saved: {out_path}")

    # also visualize best (smallest loss) for comparison
    print(f"\nVisualizing top-{min(5, args.top_n)} best images (smallest loss)...")
    best_stats = sorted(per_image_stats, key=lambda x: x["iou_loss"])[:5]
    for idx, s in enumerate(best_stats):
        img_path = image_dir / (s["stem"] + ".jpg")
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        vis = draw_comparison(image, ref_all[s["stem"]]["gt"],
                              ref_all[s["stem"]]["dets"], cmp_all[s["stem"]]["dets"],
                              labels, args.ref_name, args.cmp_name)
        title = f"BEST #{idx+1} {s['stem']} | IoU loss={s['iou_loss']:.4f} | box_diff={s['box_diff_px']:.1f}px"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(vis, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out_path = out_dir / f"best_{idx+1:02d}_{s['stem'][:40]}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  saved: {out_path}")

    # distribution stats
    all_losses = [s["iou_loss"] for s in per_image_stats]
    all_box_diffs = [s["box_diff_px"] for s in per_image_stats if s["box_diff_px"] > 0]
    all_score_diffs = [s["score_diff"] for s in per_image_stats if s["score_diff"] > 0]
    print(f"\n{'='*60}")
    print(f"IoU loss distribution ({len(all_losses)} images):")
    print(f"  mean={statistics.mean(all_losses):.4f}  median={statistics.median(all_losses):.4f}")
    print(f"  min={min(all_losses):.4f}  max={max(all_losses):.4f}")
    print(f"  stdev={statistics.pstdev(all_losses):.4f}")
    print(f"Box coordinate diff (px) distribution ({len(all_box_diffs)} matched):")
    print(f"  mean={statistics.mean(all_box_diffs):.2f}  median={statistics.median(all_box_diffs):.2f}")
    print(f"  min={min(all_box_diffs):.2f}  max={max(all_box_diffs):.2f}")
    print(f"Score diff distribution ({len(all_score_diffs)} matched):")
    print(f"  mean={statistics.mean(all_score_diffs):.4f}  median={statistics.median(all_score_diffs):.4f}")
    print(f"  min={min(all_score_diffs):.4f}  max={max(all_score_diffs):.4f}")

    # save CSV
    import csv
    csv_path = out_dir / "per_image_iou_loss.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "gt", "ref_count", "cmp_count", "matched", "unmatched_ref", "unmatched_cmp",
                    "ref_gt_iou", "cmp_gt_iou", "iou_loss", "box_diff_px", "score_diff",
                    "ref_scores", "cmp_scores"])
        for s in per_image_stats:
            w.writerow([s["stem"], s["gt_count"], s["ref_count"], s["cmp_count"],
                        s["matched"], s["unmatched_ref"], s["unmatched_cmp"],
                        f"{s['ref_gt_iou']:.4f}", f"{s['cmp_gt_iou']:.4f}", f"{s['iou_loss']:.4f}",
                        f"{s['box_diff_px']:.2f}", f"{s['score_diff']:.4f}",
                        ";".join(f"{x:.4f}" for x in s["ref_scores"]),
                        ";".join(f"{x:.4f}" for x in s["cmp_scores"])])
    print(f"\nCSV: {csv_path}")
    print(f"Visualizations: {out_dir}/")


if __name__ == "__main__":
    main()
