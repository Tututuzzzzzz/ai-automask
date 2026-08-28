"""Latency / throughput benchmark.

Answers the questions the performance criterion asks: how long does one image
take, where does the time go, how does that scale with resolution and with the
knobs a deployer can actually turn (model size, inference resolution, fp16,
cross-check, worker count).

    python scripts/bench.py                       # full sweep on data/samples
    python scripts/bench.py --folder data/real --repeat 3
    python scripts/bench.py --quick               # one configuration only

Output is a table plus data/eval/bench.json for the slides.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.batch import WorkItem, run_batch  # noqa: E402
from app.config import settings  # noqa: E402
from app.imaging import SUPPORTED_EXT, load_path  # noqa: E402
from app.pipeline import ProcessOptions, process_image  # noqa: E402
from app.segmentation import registry  # noqa: E402
from app.storage import JobStore  # noqa: E402

STAGES = ["load", "segment", "cross_check", "classify", "refine", "qc", "print_area", "artifacts"]


def device_info() -> dict:
    info = {"device": settings.resolved_device(), "fp16": settings.use_fp16,
            "infer_size": settings.infer_size}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    import os

    info["cpu_count"] = os.cpu_count()
    return info


def bench_single(files: list[Path], repeat: int, opts: ProcessOptions) -> dict:
    """Sequential, one image at a time - the p50/p95 a synchronous caller sees."""
    store = JobStore.create("bench_")
    totals: list[float] = []
    stage_ms: dict[str, list[float]] = {s: [] for s in STAGES}
    pixels: list[int] = []

    for r in range(repeat):
        for i, path in enumerate(files):
            res = process_image(store, path.name, path=str(path), options=opts,
                                slug=f"b{r}_{i:03d}")
            if res.status != "ok":
                continue
            totals.append(res.timings_ms.get("total", 0.0))
            pixels.append((res.width or 0) * (res.height or 0))
            for s in STAGES:
                if s in res.timings_ms:
                    stage_ms[s].append(res.timings_ms[s])

    if not totals:
        return {}
    arr = np.asarray(totals)
    out = {
        "n": len(totals),
        "mean_ms": round(float(arr.mean()), 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "min_ms": round(float(arr.min()), 1),
        "max_ms": round(float(arr.max()), 1),
        "mean_megapixels": round(float(np.mean(pixels)) / 1e6, 2),
        "ms_per_megapixel": round(float(arr.mean() / max(1e-6, np.mean(pixels) / 1e6)), 1),
        "stages_mean_ms": {s: round(statistics.fmean(v), 1) for s, v in stage_ms.items() if v},
    }
    # Share of the budget per stage: shows immediately whether the GPU or the
    # CPU post-processing is the bottleneck for a given configuration.
    total_stage = sum(out["stages_mean_ms"].values()) or 1.0
    out["stages_pct"] = {s: round(100 * v / total_stage, 1) for s, v in out["stages_mean_ms"].items()}
    return out


def bench_batch(files: list[Path], workers: int, opts: ProcessOptions) -> dict:
    store = JobStore.create("benchb_")
    items = [WorkItem(source=p.name, path=str(p)) for p in files]
    t0 = time.perf_counter()
    _results, summary = run_batch(store, items, opts, workers=workers)
    wall = time.perf_counter() - t0
    return {
        "workers": workers,
        "images": summary.total,
        "wall_s": round(wall, 2),
        "img_per_min": round(summary.total / (wall / 60.0), 2),
        "mean_latency_ms": summary.mean_latency_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, default=ROOT / "data" / "samples")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="current settings only, no sweep")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "eval" / "bench.json")
    args = ap.parse_args()

    files = sorted(p for p in args.folder.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no images in {args.folder}")
        return 1

    # Fail fast on an unreadable folder before spending minutes on the sweep.
    load_path(files[0])

    info = device_info()
    print("environment")
    for k, v in info.items():
        print(f"  {k:16s} {v}")

    print("\nwarming up...")
    registry.warmup()

    light = ProcessOptions(emit_shadow_maps=False, emit_displacement=False,
                           emit_overlay=False, emit_trimap=False, emit_cutout=False)
    full = ProcessOptions(emit_shadow_maps=True, emit_displacement=True)

    results: dict = {"environment": info, "images": len(files), "repeat": args.repeat,
                     "configurations": {}}

    def record(label: str, payload: dict) -> None:
        results["configurations"][label] = payload
        if not payload:
            print(f"  {label:34s} (no data)")
            return
        print(f"  {label:34s} mean {payload['mean_ms']:7.0f} ms   "
              f"p95 {payload['p95_ms']:7.0f} ms   {payload['ms_per_megapixel']:6.0f} ms/MP")

    print("\nsingle-image latency (sequential)")
    record("mask only", bench_single(files, args.repeat, light))
    record("mask + all extra layers", bench_single(files, args.repeat, full))

    if not args.quick:
        # Each of these is a knob a deployer can turn; the numbers are the
        # trade-off table that belongs in a capacity plan.
        original = (settings.ensemble, settings.infer_size, settings.use_fp16,
                    settings.refine_edges)
        try:
            settings.ensemble = False
            record("no cross-check model", bench_single(files, args.repeat, light))
            settings.ensemble = True

            settings.refine_edges = False
            record("no guided-filter refinement", bench_single(files, args.repeat, light))
            settings.refine_edges = True

            for size in (768, 1280):
                settings.infer_size = size
                registry.get("birefnet").unload()
                registry.get("birefnet").infer_size = size
                registry._primary = None
                registry.warmup()
                record(f"infer_size={size}", bench_single(files, args.repeat, light))
        finally:
            (settings.ensemble, settings.infer_size, settings.use_fp16,
             settings.refine_edges) = original
            bire = registry.get("birefnet")
            bire.unload()
            bire.infer_size = settings.infer_size
            registry._primary = None
            registry.warmup()

    print("\nbatch throughput")
    results["batch"] = []
    for workers in ([2] if args.quick else [1, 2, 4]):
        payload = bench_batch(files, workers, light)
        results["batch"].append(payload)
        print(f"  workers={payload['workers']}  {payload['img_per_min']:6.1f} img/min   "
              f"wall {payload['wall_s']:6.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    best = results["configurations"].get("mask only") or {}
    if best.get("stages_pct"):
        print("\nwhere the time goes (mask only)")
        for stage, pct in sorted(best["stages_pct"].items(), key=lambda kv: -kv[1]):
            print(f"  {stage:14s} {pct:5.1f}%  ({best['stages_mean_ms'][stage]:.0f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
