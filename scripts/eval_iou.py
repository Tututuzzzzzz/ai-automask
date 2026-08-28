"""Quantitative evaluation against ground truth.

Reports the metrics the brief is graded on, plus the one metric that decides
whether the *system* (not just the model) works:

  IoU              region overlap at alpha > 0.5
  Boundary F1      edge agreement within a tolerance - the number that actually
                   tracks "does the cut-out look clean", which IoU hides on
                   large objects (a 3 px halo costs almost no IoU)
  MAE / SAD        error on the *soft* alpha, so semi-transparency counts
  Trimap MAE       MAE restricted to a band around the true boundary - the
                   hardest pixels, where hand-cutting takes designers the longest
  Confidence-IoU   Spearman correlation between our self-assessed confidence and
    correlation    the true IoU. If this is not strongly positive, the
                   READY/REVIEW/FAILED routing is decoration and a human still
                   has to check everything.

Also prints the decision matrix: how good the masks we auto-published actually
were (READY-mean-IoU), versus the ones we routed to a human.

    python scripts/eval_iou.py                       # data/samples + its ground_truth/
    python scripts/eval_iou.py --folder data/samples --out data/eval
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.imaging import SUPPORTED_EXT  # noqa: E402
from app.pipeline import ProcessOptions, process_image  # noqa: E402
from app.postprocess.quality import boundary_f1, iou  # noqa: E402
from app.segmentation import registry  # noqa: E402
from app.storage import JobStore  # noqa: E402


def trimap_band(gt: np.ndarray, width_px: int) -> np.ndarray:
    """Unknown band around the true boundary, the standard matting eval region."""
    k = int(max(3, width_px) | 1)
    kernel = np.ones((k, k), np.uint8)
    hard = (gt > 0.5).astype(np.uint8)
    return (cv2.dilate(hard, kernel) - cv2.erode(hard, kernel)) > 0


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without pulling in scipy.stats."""
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(np.asarray(a, dtype=np.float64)))
    rb = np.argsort(np.argsort(np.asarray(b, dtype=np.float64)))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float(ra.dot(rb) / denom) if denom > 0 else float("nan")


def evaluate(folder: Path, out_dir: Path, limit: int = 0, workers: int | None = None) -> dict:
    gt_dir = folder / "ground_truth"
    if not gt_dir.is_dir():
        raise SystemExit(f"no ground_truth/ folder inside {folder}; run scripts/make_dataset.py first")

    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
    if limit:
        files = files[:limit]

    hints: dict[str, str] = {}
    manifest = folder / "manifest.csv"
    if manifest.exists():
        with manifest.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                if row.get("file"):
                    hints[row["file"]] = (row.get("category") or "auto").strip()

    print(f"Warming models on {settings.resolved_device()} ...")
    warm = registry.warmup()
    print(f"  primary={warm.get('primary')} cross_check={warm.get('cross_check')}")

    store = JobStore.create("eval_")
    rows: list[dict] = []
    t0 = time.perf_counter()

    for i, path in enumerate(files, start=1):
        gt_path = gt_dir / f"{path.stem}_gt.png"
        if not gt_path.exists():
            print(f"  [skip] no ground truth for {path.name}")
            continue
        result = process_image(
            store, source_name=path.name, path=str(path),
            options=ProcessOptions(category=hints.get(path.name, "auto"),
                                   emit_shadow_maps=False, emit_displacement=False,
                                   emit_trimap=False, emit_cutout=False),
            slug=f"{i:03d}_{path.stem[:44]}",
        )
        if result.status == "error":
            print(f"  [err ] {path.name}: {result.error}")
            continue

        mask_path = store.root / f"{i:03d}_{path.stem[:44]}_mask.png"
        pred = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            print(f"  [err ] could not read mask/gt for {path.name}")
            continue
        if pred.shape != gt.shape:
            # This must never happen - it is the resolution guarantee in the brief.
            print(f"  [FAIL] resolution mismatch {pred.shape} vs {gt.shape} for {path.name}")
            continue

        p = pred.astype(np.float32) / 255.0
        g = gt.astype(np.float32) / 255.0
        pb, gb = p > 0.5, g > 0.5

        band_px = max(6, int(0.006 * max(g.shape)))
        band = trimap_band(g, band_px)
        tol = max(2, int(0.002 * max(g.shape)))

        row = {
            "file": path.name,
            "category_true": hints.get(path.name, ""),
            "category_pred": result.category or "",
            "verdict": result.verdict or "",
            "confidence": round(result.confidence or 0.0, 4),
            "iou": round(iou(pb, gb), 4),
            "boundary_f1": round(boundary_f1(pb.astype(np.uint8), gb.astype(np.uint8), tol=tol), 4),
            "mae": round(float(np.abs(p - g).mean()), 5),
            "sad_k": round(float(np.abs(p - g).sum() / 1000.0), 2),
            "trimap_mae": round(float(np.abs(p - g)[band].mean()) if band.any() else 0.0, 5),
            "dice": round(float(2 * (pb & gb).sum() / max(1, pb.sum() + gb.sum())), 4),
            "latency_ms": round(result.timings_ms.get("total", 0.0), 1),
            "segment_ms": round(result.timings_ms.get("segment", 0.0), 1),
            "width": result.width, "height": result.height,
            "print_area": result.print_area.kind if result.print_area else "",
            "print_area_conf": round(result.print_area.confidence, 3) if result.print_area else "",
        }
        rows.append(row)
        print(f"  {row['verdict']:6s} conf {row['confidence']:.3f}  IoU {row['iou']:.4f}  "
              f"bF1 {row['boundary_f1']:.4f}  triMAE {row['trimap_mae']:.4f}  "
              f"{row['latency_ms']:6.0f}ms  {path.name}")

    wall = time.perf_counter() - t0
    if not rows:
        raise SystemExit("nothing evaluated")

    def col(name: str) -> np.ndarray:
        return np.asarray([r[name] for r in rows], dtype=np.float64)

    summary = {
        "images": len(rows),
        "mean_iou": round(float(col("iou").mean()), 4),
        "median_iou": round(float(np.median(col("iou"))), 4),
        "min_iou": round(float(col("iou").min()), 4),
        "mean_boundary_f1": round(float(col("boundary_f1").mean()), 4),
        "mean_dice": round(float(col("dice").mean()), 4),
        "mean_mae": round(float(col("mae").mean()), 5),
        "mean_trimap_mae": round(float(col("trimap_mae").mean()), 5),
        "mean_latency_ms": round(float(col("latency_ms").mean()), 1),
        "p95_latency_ms": round(float(np.percentile(col("latency_ms"), 95)), 1),
        "mean_segment_ms": round(float(col("segment_ms").mean()), 1),
        "wall_s": round(wall, 2),
        "throughput_img_per_min": round(len(rows) / (wall / 60.0), 2),
        "iou_at_50": round(float((col("iou") >= 0.50).mean()), 4),
        "iou_at_75": round(float((col("iou") >= 0.75).mean()), 4),
        "iou_at_90": round(float((col("iou") >= 0.90).mean()), 4),
        "iou_at_95": round(float((col("iou") >= 0.95).mean()), 4),
        "confidence_iou_spearman": round(spearman([r["confidence"] for r in rows],
                                                  [r["iou"] for r in rows]), 4),
    }

    # The routing question: were the auto-published masks actually good?
    by_verdict: dict[str, dict] = {}
    for verdict in ("READY", "REVIEW", "FAILED"):
        sel = [r for r in rows if r["verdict"] == verdict]
        if not sel:
            continue
        ious = np.asarray([r["iou"] for r in sel])
        by_verdict[verdict] = {
            "count": len(sel),
            "share": round(len(sel) / len(rows), 4),
            "mean_iou": round(float(ious.mean()), 4),
            "min_iou": round(float(ious.min()), 4),
            "mean_boundary_f1": round(float(np.mean([r["boundary_f1"] for r in sel])), 4),
        }
    summary["by_verdict"] = by_verdict

    cat_rows = [r for r in rows if r["category_true"]]
    summary["category_accuracy"] = (
        round(sum(1 for r in cat_rows if r["category_pred"] == r["category_true"]) / len(cat_rows), 4)
        if cat_rows else None
    )
    by_category: dict[str, dict] = {}
    for cat in sorted({r["category_true"] for r in cat_rows}):
        sel = [r for r in cat_rows if r["category_true"] == cat]
        by_category[cat] = {
            "count": len(sel),
            "mean_iou": round(float(np.mean([r["iou"] for r in sel])), 4),
            "mean_boundary_f1": round(float(np.mean([r["boundary_f1"] for r in sel])), 4),
            "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in sel])), 1),
        }
    summary["by_category"] = by_category

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "eval_per_image.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "summary": summary,
        "environment": {
            "device": settings.resolved_device(),
            "primary_model": warm.get("primary"),
            "cross_check": warm.get("cross_check"),
            "infer_size": settings.infer_size,
            "fp16": settings.use_fp16,
            "ensemble": settings.ensemble,
        },
        "per_image": rows,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== summary =====================================================")
    for k in ("images", "mean_iou", "median_iou", "min_iou", "mean_boundary_f1", "mean_dice",
              "mean_mae", "mean_trimap_mae", "iou_at_75", "iou_at_90", "iou_at_95",
              "confidence_iou_spearman", "category_accuracy",
              "mean_segment_ms", "mean_latency_ms", "p95_latency_ms", "throughput_img_per_min"):
        print(f"  {k:26s} {summary.get(k)}")
    print("\n  routing quality (did the verdict match reality?)")
    for verdict, stats in by_verdict.items():
        print(f"    {verdict:6s} n={stats['count']:2d} ({stats['share']*100:4.1f}%)  "
              f"mean IoU {stats['mean_iou']:.4f}  worst {stats['min_iou']:.4f}")
    print("\n  per category")
    for cat, stats in by_category.items():
        print(f"    {cat:11s} n={stats['count']:2d}  IoU {stats['mean_iou']:.4f}  "
              f"bF1 {stats['mean_boundary_f1']:.4f}  {stats['mean_latency_ms']:.0f}ms")
    print(f"\n  masks    -> {store.root}")
    print(f"  csv/json -> {out_dir}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, default=ROOT / "data" / "samples")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "eval")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    evaluate(args.folder, args.out, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
