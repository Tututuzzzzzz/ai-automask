"""Batch processing: CSV/JSON ingest, concurrent execution, automation report.

The operational question this module answers is not "can the AI cut out a mug",
it is **"of the 1,200 bases we dropped in last night, how many can go live
without a designer touching them?"** That number - the automation rate - is the
whole business case, so it is a first-class output here, not a footnote.

Concurrency model: a thread pool of ``AUTOMASK_BATCH_WORKERS``. Segmentation is
serialised by a per-backend lock (see ``SegmentationBackend.predict``), so the
pool's real job is overlapping the CPU-bound half of the pipeline - HTTP fetch,
JPEG decode, guided-filter refinement, PNG encode - with the GPU-bound half.
On the reference laptop GPU that lifts throughput by roughly 1.6x over
sequential without ever putting two CUDA graphs in flight at once.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .config import CATEGORIES, settings
from .pipeline import ProcessOptions, process_image
from .schemas import BatchSummary, MaskResult
from .storage import JobStore

log = logging.getLogger("automask.batch")

URL_KEYS = ("image_url", "url", "image", "src", "link", "base_url", "image_link")
CATEGORY_KEYS = ("category", "product_type", "type", "product_category")
SKU_KEYS = ("sku", "id", "code", "product_id", "base_id")


@dataclass
class WorkItem:
    source: str                      # URL or filename, used for display
    data: bytes | None = None        # in-memory upload
    url: str | None = None
    path: str | None = None
    category: str = "auto"
    sku: str | None = None

    def slug(self, index: int) -> str:
        import re

        stem = (self.sku or self.source).rsplit("/", 1)[-1]
        stem = re.sub(r"\.[a-zA-Z0-9]{2,5}$", "", stem)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "image"
        return f"{index:03d}_{stem[:48]}"


# ------------------------------------------------------------------- ingestion
def _norm_key(key: str | None) -> str:
    """Normalise a column name so real-world exports match our aliases.

    Sheets and PIM exports arrive as "Image URL", "image-url", "Product Type" -
    all of which mean the same thing. Matching on the raw string means a batch of
    500 rows silently yields zero usable items, which is exactly the failure a
    tolerant ingest layer exists to prevent.
    """
    return (key or "").strip().lower().replace(" ", "_").replace("-", "_").lstrip("\ufeff")


def _pick(row: dict, keys: Iterable[str]) -> str | None:
    normalised = {_norm_key(k): v for k, v in row.items()}
    for key in keys:
        val = normalised.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return None


def _normalise_category(raw: str | None) -> str:
    if not raw:
        return "auto"
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "tshirt": "apparel", "t_shirt": "apparel", "shirt": "apparel", "hoodie": "apparel",
        "clothing": "apparel", "tote": "apparel", "garment": "apparel",
        "mug": "drinkware", "cup": "drinkware", "tumbler": "drinkware", "bottle": "drinkware",
        "canvas": "wall_art", "poster": "wall_art", "print": "wall_art", "wallart": "wall_art",
        "frame": "wall_art", "framed_print": "wall_art",
        "phone_case": "accessory", "case": "accessory", "cap": "accessory",
        "hat": "accessory", "mousepad": "accessory",
    }
    key = aliases.get(key, key)
    return key if key in CATEGORIES else "auto"


def parse_csv(data: bytes) -> tuple[list[WorkItem], list[str]]:
    """Parse a CSV of image URLs. Tolerant about column naming and delimiters."""
    warnings: list[str] = []
    text = data.decode("utf-8-sig", errors="replace")
    if not text.strip():
        return [], ["CSV is empty"]
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)

    items: list[WorkItem] = []
    if reader.fieldnames and not any(_norm_key(f) in URL_KEYS for f in reader.fieldnames):
        # Headerless file, or a header we do not recognise: treat column 0 as the URL.
        warnings.append(
            f"No recognised URL column in {reader.fieldnames!r}; falling back to the first column."
        )
        for line in csv.reader(io.StringIO(text), dialect=dialect):
            if not line or not line[0].strip():
                continue
            first = line[0].strip()
            if not first.lower().startswith(("http://", "https://")):
                continue
            cat = _normalise_category(line[1] if len(line) > 1 else None)
            items.append(WorkItem(source=first, url=first, category=cat))
        return items, warnings

    for i, row in enumerate(reader, start=2):
        url = _pick(row, URL_KEYS)
        if not url:
            warnings.append(f"row {i}: no image URL, skipped")
            continue
        items.append(WorkItem(
            source=url,
            url=url,
            category=_normalise_category(_pick(row, CATEGORY_KEYS)),
            sku=_pick(row, SKU_KEYS),
        ))
    return items, warnings


def parse_json(data: bytes) -> tuple[list[WorkItem], list[str]]:
    """Accepts a bare list of URLs, a list of objects, or {"items": [...]}."""
    warnings: list[str] = []
    try:
        payload = json.loads(data.decode("utf-8-sig", errors="replace"))
    except json.JSONDecodeError as exc:
        return [], [f"invalid JSON: {exc}"]

    if isinstance(payload, dict):
        for key in ("items", "images", "data", "rows", "bases"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        return [], ["JSON must be a list, or an object containing an 'items' list"]

    items: list[WorkItem] = []
    for i, entry in enumerate(payload):
        if isinstance(entry, str):
            if entry.strip().lower().startswith(("http://", "https://")):
                items.append(WorkItem(source=entry.strip(), url=entry.strip()))
            else:
                warnings.append(f"item {i}: not an http(s) URL, skipped")
            continue
        if not isinstance(entry, dict):
            warnings.append(f"item {i}: unsupported entry type {type(entry).__name__}")
            continue
        url = _pick(entry, URL_KEYS)
        if not url:
            warnings.append(f"item {i}: no image URL, skipped")
            continue
        items.append(WorkItem(
            source=url, url=url,
            category=_normalise_category(_pick(entry, CATEGORY_KEYS)),
            sku=_pick(entry, SKU_KEYS),
        ))
    return items, warnings


def parse_manifest(filename: str, data: bytes) -> tuple[list[WorkItem], list[str]]:
    name = (filename or "").lower()
    if name.endswith(".json"):
        return parse_json(data)
    if name.endswith((".csv", ".tsv", ".txt")):
        return parse_csv(data)
    # Sniff: JSON manifests start with a bracket or brace.
    head = data.lstrip()[:1]
    return (parse_json(data) if head in (b"[", b"{") else parse_csv(data))


# -------------------------------------------------------------------- summary
def summarise(results: list[MaskResult], wall_ms: float) -> BatchSummary:
    total = len(results)
    ready = sum(1 for r in results if r.verdict == "READY")
    review = sum(1 for r in results if r.verdict == "REVIEW")
    failed = sum(1 for r in results if r.verdict == "FAILED")
    errors = sum(1 for r in results if r.status == "error")
    confs = [r.confidence for r in results if r.confidence is not None]
    lats = [r.timings_ms.get("total", 0.0) for r in results if r.timings_ms]
    return BatchSummary(
        total=total,
        ready=ready,
        review=review,
        failed=failed,
        errors=errors,
        automation_rate=round(ready / total, 4) if total else 0.0,
        touchless_rate=round((ready + review) / total, 4) if total else 0.0,
        mean_confidence=round(sum(confs) / len(confs), 4) if confs else 0.0,
        mean_latency_ms=round(sum(lats) / len(lats), 2) if lats else 0.0,
        total_wall_ms=round(wall_ms, 2),
        throughput_img_per_min=round(total / (wall_ms / 60000.0), 2) if wall_ms > 0 else 0.0,
    )


# ------------------------------------------------------------------- execution
@dataclass
class BatchJob:
    """In-memory job record. The API polls this for progress."""

    job_id: str
    total: int
    state: str = "queued"
    processed: int = 0
    results: list[MaskResult] = field(default_factory=list)
    summary: BatchSummary | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    report_url: str | None = None
    started_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, result: MaskResult) -> None:
        with self._lock:
            self.results.append(result)
            self.processed += 1


class JobRegistry:
    """Bounded in-process job table. Swap for Redis when you run more than one
    replica - the API surface stays identical."""

    def __init__(self, capacity: int = 200) -> None:
        self._jobs: dict[str, BatchJob] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self.capacity = capacity

    def put(self, job: BatchJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            while len(self._order) > self.capacity:
                self._jobs.pop(self._order.pop(0), None)

    def get(self, job_id: str) -> BatchJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[BatchJob]:
        with self._lock:
            return [self._jobs[i] for i in reversed(self._order[-limit:]) if i in self._jobs]


jobs = JobRegistry()


def run_batch(
    store: JobStore,
    items: list[WorkItem],
    options: ProcessOptions | None = None,
    job: BatchJob | None = None,
    workers: int | None = None,
    on_result: Callable[[MaskResult], None] | None = None,
) -> tuple[list[MaskResult], BatchSummary]:
    """Process every item, preserving input order in the returned list."""
    opts = options or ProcessOptions()
    workers = max(1, workers or settings.batch_workers)
    t0 = time.perf_counter()
    slots: list[MaskResult | None] = [None] * len(items)

    def work(index: int, item: WorkItem) -> tuple[int, MaskResult]:
        item_opts = ProcessOptions(**{**opts.__dict__, "category": item.category or opts.category,
                                      "sku": item.sku})
        res = process_image(
            store,
            source_name=item.source,
            data=item.data,
            url=item.url,
            path=item.path,
            options=item_opts,
            slug=item.slug(index + 1),
        )
        return index, res

    if job:
        job.state = "running"
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="automask") as pool:
            futures = [pool.submit(work, i, it) for i, it in enumerate(items)]
            for fut in as_completed(futures):
                index, res = fut.result()
                slots[index] = res
                if job:
                    job.add(res)
                if on_result:
                    on_result(res)
    except Exception as exc:  # noqa: BLE001
        log.exception("batch aborted")
        if job:
            job.state = "error"
            job.error = f"{type(exc).__name__}: {exc}"
        raise

    results = [r for r in slots if r is not None]
    wall = (time.perf_counter() - t0) * 1000.0
    summary = summarise(results, wall)

    store.save_json("batch_results", {
        "job_id": store.job_id,
        "summary": summary.model_dump(),
        "results": [r.model_dump() for r in results],
    })
    store.save_text("batch_results.csv", results_to_csv(results), mime="text/csv")

    if job:
        job.summary = summary
        job.state = "done"
    return results, summary


def results_to_csv(results: list[MaskResult]) -> str:
    """Flat CSV for the ops team - one row per base, importable into a sheet."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([
        "id", "source", "status", "verdict", "confidence", "category", "category_source",
        "width", "height", "model", "total_ms", "coverage", "edge_sharpness", "ensemble_iou",
        "holes", "solidity", "border_contact", "print_area_kind", "print_area_confidence",
        "mask_file", "reasons", "error",
    ])
    for r in results:
        m = r.metrics
        pa = r.print_area
        writer.writerow([
            r.id, r.source, r.status, r.verdict or "", r.confidence if r.confidence is not None else "",
            r.category or "", r.category_source or "", r.width or "", r.height or "",
            r.model_used or "", r.timings_ms.get("total", ""),
            m.coverage if m else "", m.edge_sharpness if m else "",
            m.ensemble_iou if (m and m.ensemble_iou is not None) else "",
            m.holes if m else "", m.solidity if m else "", m.border_contact if m else "",
            pa.kind if pa else "", pa.confidence if pa else "",
            r.artifacts.alpha_mask or "", " | ".join(r.reasons), r.error or "",
        ])
    return buf.getvalue()
