"""Run the pipeline over a local folder of product bases.

The offline counterpart to the API: same pipeline object, same artifacts, same
report. Useful for (a) the demo, (b) processing a drop-folder on a build agent,
and (c) the judges' hidden test - point it at the folder and every mask lands in
one job directory with an HTML report.

    python scripts/run_folder.py data/samples --report
    python scripts/run_folder.py /path/to/hidden_test --category auto --workers 2
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.batch import BatchJob, WorkItem, run_batch  # noqa: E402
from app.config import settings  # noqa: E402
from app.imaging import SUPPORTED_EXT  # noqa: E402
from app.pipeline import ProcessOptions  # noqa: E402
from app.report import render_report  # noqa: E402
from app.segmentation import registry  # noqa: E402
from app.storage import JobStore  # noqa: E402

VERDICT_ICON = {"READY": "[READY ]", "REVIEW": "[REVIEW]", "FAILED": "[FAILED]"}


def category_hints(folder: Path) -> dict[str, str]:
    """Read per-file categories from a manifest.csv if the folder ships one."""
    manifest = folder / "manifest.csv"
    if not manifest.exists():
        return {}
    hints: dict[str, str] = {}
    with manifest.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("file") or "").strip()
            cat = (row.get("category") or "").strip()
            if name and cat:
                hints[name] = cat
    return hints


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--category", default=None,
                    help="force a category for every image (default: manifest hint, else auto)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--report", action="store_true", help="write report.html into the job folder")
    ap.add_argument("--no-maps", action="store_true", help="skip shadow/highlight/displacement")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    folder: Path = args.folder
    if not folder.is_dir():
        print(f"not a directory: {folder}")
        return 1

    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no images in {folder} (looked for {sorted(SUPPORTED_EXT)})")
        return 1

    hints = category_hints(folder)
    items = [
        WorkItem(source=p.name, path=str(p),
                 category=args.category or hints.get(p.name, "auto"))
        for p in files
    ]

    print(f"Warming models on {settings.resolved_device()} ...")
    t0 = time.perf_counter()
    warm = registry.warmup()
    print(f"  primary={warm.get('primary')} cross_check={warm.get('cross_check')} "
          f"({time.perf_counter() - t0:.1f}s)")
    if warm.get("errors"):
        for e in warm["errors"]:
            print(f"  [warn] {e}")

    store = JobStore.create("folder_")
    opts = ProcessOptions(
        emit_shadow_maps=False if args.no_maps else None,
        emit_displacement=False if args.no_maps else None,
    )
    job = BatchJob(job_id=store.job_id, total=len(items))

    print(f"\nProcessing {len(items)} images -> {store.root}")
    results, summary = run_batch(store, items, opts, job=job, workers=args.workers)

    for r in sorted(results, key=lambda x: x.id):
        icon = VERDICT_ICON.get(r.verdict or "FAILED", "[  ?   ]")
        note = (r.reasons[0][:74] if r.reasons else "")
        print(f"  {icon} {r.confidence if r.confidence is not None else 0:.3f} "
              f"{r.id[:34]:34s} {r.category or '-':10s} "
              f"{r.timings_ms.get('total', 0):6.0f}ms  {note}")

    print(f"\n  total            {summary.total}")
    print(f"  READY            {summary.ready}")
    print(f"  REVIEW           {summary.review}")
    print(f"  FAILED           {summary.failed}")
    print(f"  automation rate  {summary.automation_rate * 100:.1f}%")
    print(f"  mean confidence  {summary.mean_confidence:.3f}")
    print(f"  mean latency     {summary.mean_latency_ms:.0f} ms")
    print(f"  throughput       {summary.throughput_img_per_min:.1f} img/min "
          f"(wall {summary.total_wall_ms / 1000:.1f}s, {args.workers or settings.batch_workers} workers)")

    if args.report:
        html = render_report(store.job_id, results, summary, meta={
            "model": warm.get("primary"), "cross_check": warm.get("cross_check") or "off",
            "device": settings.resolved_device(), "infer_size": settings.infer_size,
            "source": str(folder),
        })
        path = store.path_for("report.html")
        store.save_text("report.html", html, mime="text/html")
        print(f"\n  report -> {path}")
    print(f"  artifacts -> {store.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
