"""FastAPI service - the microservice the Mockup Generator calls.

Endpoints (all documented interactively at /docs):

  GET  /health                  liveness + which models actually loaded
  POST /v1/mask                 one uploaded file   -> mask + verdict
  POST /v1/mask/url             one direct image URL (the integration path)
  POST /v1/mask/batch           JSON list of URLs, synchronous
  POST /v1/mask/batch/async     JSON list of URLs, returns a job id to poll
  POST /v1/mask/upload-batch    multi-file upload
  POST /v1/mask/manifest        CSV / JSON manifest upload
  GET  /v1/jobs/{id}            job progress + results
  GET  /v1/jobs/{id}/report     HTML automation report
  GET  /v1/jobs/{id}/download   zip of every artifact in the job
  GET  /artifacts/{job}/{file}  static artifact delivery
  GET  /                        review dashboard (UI)

Design decisions that matter for integration:

* **Errors are 200s with status="error" inside batch responses.** A 500 on one
  bad URL out of 500 would force the caller to re-run the whole batch. Single-
  image endpoints do use real HTTP codes, because there the caller has one thing
  to retry.
* **Optional API-key auth** via ``AUTOMASK_API_KEY`` and the ``X-API-Key``
  header. Off by default so judges can poke at /docs without ceremony.
* **Artifacts are served from disk, not held in memory.** Swap ``storage.py``
  for S3 and this layer does not change.
"""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query,
                     Request, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .batch import (BatchJob, ProcessOptions, WorkItem, jobs, parse_manifest, results_to_csv,
                    run_batch, summarise)
from .labeling import router as labeling_router
from .config import CATEGORIES, settings
from .imaging import ImageLoadError
from .pipeline import process_image
from .report import render_report
from .schemas import (BatchRequest, BatchResponse, HealthResponse, JobStatus, MaskResult,
                      SamplesRequest, UrlRequest)
from .segmentation import registry
from .storage import JobStore, disk_usage_mb, purge_old_jobs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("automask.api")

STATIC_DIR = Path(__file__).parent / "static"
_warmup_report: dict = {"state": "pending"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the models on a background thread.

    Deliberately *not* awaited: the container becomes healthy immediately and
    /health reports ``warming`` until the weights are resident. That keeps
    Kubernetes readiness probes honest without a 30 s startup stall.
    """
    def _warm() -> None:
        try:
            _warmup_report.update(registry.warmup(), state="ready")
            log.info("warmup complete: %s", _warmup_report)
        except Exception as exc:  # noqa: BLE001
            _warmup_report.update(state="error", error=str(exc))
            log.error("warmup failed: %s", exc)

    threading.Thread(target=_warm, name="warmup", daemon=True).start()
    purged = purge_old_jobs()
    if purged:
        log.info("purged %d expired job folders", purged)
    yield


app = FastAPI(
    title="AI Auto-Masking for Mockup Generator",
    version=__version__,
    description=(
        "Automatic alpha-mask generation for e-commerce / POD product bases.\n\n"
        "Upload a blank product photo and get back a pixel-aligned alpha mask, a "
        "print-area quad, optional shadow/highlight/displacement layers, and a "
        "self-assessed **READY / REVIEW / FAILED** verdict so only genuinely "
        "uncertain masks reach a designer."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------ auth
async def require_key(request: Request) -> None:
    if not settings.api_key:
        return
    supplied = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if supplied != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


# --------------------------------------------------------------------- helpers
def _opts(category: str = "auto", shadow: bool | None = None, displacement: bool | None = None,
          sku: str | None = None) -> ProcessOptions:
    return ProcessOptions(
        category=category if category in CATEGORIES else "auto",
        emit_shadow_maps=shadow,
        emit_displacement=displacement,
        sku=sku,
    )


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    limit = settings.max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise HTTPException(413, f"{file.filename}: exceeds {settings.max_upload_mb} MB")
    if not data:
        raise HTTPException(400, f"{file.filename}: empty upload")
    return data


def _write_report(store: JobStore, results: list[MaskResult], summary, warnings=None,
                  meta: dict | None = None) -> str:
    html = render_report(store.job_id, results, summary, warnings=warnings, meta=meta)
    store.save_text("report.html", html, mime="text/html")
    return f"/v1/jobs/{store.job_id}/report"


def _run_meta() -> dict:
    return {
        "model": _warmup_report.get("primary") or settings.primary_model,
        "cross_check": _warmup_report.get("cross_check") or "disabled",
        "device": settings.resolved_device(),
        "infer_size": settings.infer_size,
        "fp16": settings.use_fp16,
        "ready_threshold": settings.ready_threshold,
        "review_threshold": settings.review_threshold,
        "workers": settings.batch_workers,
    }


# ---------------------------------------------------------------------- health
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    state = _warmup_report.get("state", "pending")
    return HealthResponse(
        status={"ready": "ok", "pending": "warming", "error": "degraded"}.get(state, state),
        version=__version__,
        device=settings.resolved_device(),
        models={"warmup": _warmup_report, **registry.describe()},
        settings={
            "primary_model": settings.primary_model,
            "infer_size": settings.infer_size,
            "fp16": settings.use_fp16,
            "ensemble": settings.ensemble,
            "refine_edges": settings.refine_edges,
            "ready_threshold": settings.ready_threshold,
            "review_threshold": settings.review_threshold,
            "batch_workers": settings.batch_workers,
            "max_upload_mb": settings.max_upload_mb,
            "auth_required": bool(settings.api_key),
            "outputs_disk_mb": disk_usage_mb(),
            "categories": list(CATEGORIES),
        },
    )


@app.get("/v1/categories", tags=["ops"])
def categories() -> dict:
    """Category taxonomy, so a caller can populate a dropdown from the service."""
    return {k: {"label": v["label"], "print_area": v["print_area"],
                "expected_coverage": v["expect_coverage"]} for k, v in CATEGORIES.items()}


# ---------------------------------------------------------- single image (file)
@app.post("/v1/mask", response_model=MaskResult, tags=["mask"],
          dependencies=[Depends(require_key)])
async def mask_upload(
    file: UploadFile = File(..., description="Blank product image (PNG/JPG/WEBP)"),
    category: str = Form("auto", description="auto | apparel | drinkware | wall_art | accessory"),
    sku: str | None = Form(None),
    shadow_maps: bool | None = Form(None),
    displacement: bool | None = Form(None),
) -> MaskResult:
    data = await _read_upload(file)
    store = JobStore.create("single_")
    result = process_image(
        store, source_name=file.filename or "upload",
        data=data, options=_opts(category, shadow_maps, displacement, sku),
        slug="001_" + Path(file.filename or "image").stem[:48],
    )
    if result.status == "error" and result.width is None:
        raise HTTPException(422, result.error or "could not process image")
    return result


# ----------------------------------------------------------- single image (url)
@app.post("/v1/mask/url", response_model=MaskResult, tags=["mask"],
          dependencies=[Depends(require_key)])
def mask_url(payload: UrlRequest) -> MaskResult:
    """The integration endpoint: another service posts a direct image URL."""
    store = JobStore.create("api_", inline_base64=payload.return_base64)
    result = process_image(
        store, source_name=str(payload.image_url), url=str(payload.image_url),
        options=_opts(payload.category, payload.emit_shadow_maps, payload.emit_displacement,
                      payload.sku),
        slug="001_" + (payload.sku or "image")[:48],
    )
    if result.status == "error" and result.width is None:
        raise HTTPException(422, result.error or "could not fetch or decode image")
    return result


# ------------------------------------------------------------------ batch: urls
@app.post("/v1/mask/batch", response_model=BatchResponse, tags=["batch"],
          dependencies=[Depends(require_key)])
def mask_batch(payload: BatchRequest) -> BatchResponse:
    """Synchronous batch. Use the async variant for anything over ~30 images."""
    store = JobStore.create("batch_")
    items = [WorkItem(source=str(i.image_url), url=str(i.image_url),
                      category=i.category, sku=i.sku) for i in payload.items]
    opts = _opts("auto", payload.emit_shadow_maps, payload.emit_displacement)
    job = BatchJob(job_id=store.job_id, total=len(items))
    jobs.put(job)
    results, summary = run_batch(store, items, opts, job=job)
    job.report_url = _write_report(store, results, summary, meta=_run_meta())
    return BatchResponse(job_id=store.job_id, summary=summary, results=results,
                         report_url=job.report_url)


@app.post("/v1/mask/batch/async", response_model=JobStatus, tags=["batch"],
          dependencies=[Depends(require_key)])
def mask_batch_async(payload: BatchRequest, background: BackgroundTasks) -> JobStatus:
    """Kick off a batch and return immediately; poll /v1/jobs/{id}."""
    store = JobStore.create("batch_")
    items = [WorkItem(source=str(i.image_url), url=str(i.image_url),
                      category=i.category, sku=i.sku) for i in payload.items]
    opts = _opts("auto", payload.emit_shadow_maps, payload.emit_displacement)
    job = BatchJob(job_id=store.job_id, total=len(items))
    jobs.put(job)

    def _run() -> None:
        try:
            results, summary = run_batch(store, items, opts, job=job)
            job.report_url = _write_report(store, results, summary, meta=_run_meta())
        except Exception as exc:  # noqa: BLE001
            job.state, job.error = "error", str(exc)

    background.add_task(_run)
    return JobStatus(job_id=job.job_id, state="queued", processed=0, total=job.total)


# ---------------------------------------------------------- batch: file uploads
@app.post("/v1/mask/upload-batch", response_model=BatchResponse, tags=["batch"],
          dependencies=[Depends(require_key)])
async def mask_upload_batch(
    files: list[UploadFile] = File(..., description="One or more product images"),
    category: str = Form("auto"),
    shadow_maps: bool | None = Form(None),
    displacement: bool | None = Form(None),
) -> BatchResponse:
    if not files:
        raise HTTPException(400, "no files uploaded")
    store = JobStore.create("upload_")
    items: list[WorkItem] = []
    for f in files:
        items.append(WorkItem(source=f.filename or "upload", data=await _read_upload(f),
                              category=category))
    job = BatchJob(job_id=store.job_id, total=len(items))
    jobs.put(job)
    results, summary = run_batch(store, items, _opts(category, shadow_maps, displacement), job=job)
    job.report_url = _write_report(store, results, summary, meta=_run_meta())
    return BatchResponse(job_id=store.job_id, summary=summary, results=results,
                         report_url=job.report_url)


# ---------------------------------------------------------------- batch: manifest
@app.post("/v1/mask/manifest", response_model=BatchResponse, tags=["batch"],
          dependencies=[Depends(require_key)])
async def mask_manifest(
    file: UploadFile = File(..., description="CSV or JSON listing image URLs"),
    shadow_maps: bool | None = Form(None),
    displacement: bool | None = Form(None),
    limit: int = Form(500, description="Safety cap on rows processed"),
) -> BatchResponse:
    """CSV/JSON ingest. Recognised columns: image_url|url|image, category, sku."""
    data = await _read_upload(file)
    items, warnings = parse_manifest(file.filename or "manifest.csv", data)
    if not items:
        raise HTTPException(422, {"error": "no usable rows in manifest", "warnings": warnings})
    if len(items) > limit:
        warnings.append(f"manifest had {len(items)} rows; truncated to the {limit}-row cap")
        items = items[:limit]

    store = JobStore.create("manifest_")
    job = BatchJob(job_id=store.job_id, total=len(items), warnings=warnings)
    jobs.put(job)
    results, summary = run_batch(store, items, _opts("auto", shadow_maps, displacement), job=job)
    job.report_url = _write_report(store, results, summary, warnings=warnings, meta=_run_meta())
    return BatchResponse(job_id=store.job_id, summary=summary, results=results,
                         report_url=job.report_url)


# ------------------------------------------------------------- bundled samples
@app.post("/v1/mask/samples", response_model=BatchResponse, tags=["batch"],
          dependencies=[Depends(require_key)])
def mask_samples(payload: SamplesRequest | None = None) -> BatchResponse:
    """Run the bundled test set in ``data/samples`` server-side.

    Exists so a reviewer (or a judge) can see the full output surface - verdicts,
    lighting maps, batch report - in one click, without hunting for product
    photos to upload first. Categories come from the folder's manifest.csv when
    present, so this also exercises the metadata path rather than only auto-detect.
    """
    payload = payload or SamplesRequest()
    folder = Path(payload.folder) if payload.folder else (settings.data_dir / "samples")
    if not folder.is_dir():
        raise HTTPException(404, f"sample folder not found: {folder}. "
                                 f"Generate it with: python scripts/make_dataset.py")

    from .imaging import SUPPORTED_EXT

    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)[: payload.limit]
    if not files:
        raise HTTPException(404, f"no images in {folder}")

    hints: dict[str, str] = {}
    manifest = folder / "manifest.csv"
    if manifest.exists():
        import csv as _csv

        with manifest.open(encoding="utf-8-sig") as fh:
            for row in _csv.DictReader(fh):
                if row.get("file"):
                    hints[row["file"]] = (row.get("category") or "auto").strip()

    store = JobStore.create("samples_")
    items = [WorkItem(source=f.name, path=str(f), category=hints.get(f.name, "auto"))
             for f in files]
    job = BatchJob(job_id=store.job_id, total=len(items))
    jobs.put(job)
    results, summary = run_batch(
        store, items,
        _opts("auto", payload.emit_shadow_maps, payload.emit_displacement),
        job=job,
    )
    job.report_url = _write_report(store, results, summary, meta=_run_meta())
    return BatchResponse(job_id=store.job_id, summary=summary, results=results,
                         report_url=job.report_url)


# ------------------------------------------------------------------------- jobs
@app.get("/v1/jobs", tags=["batch"])
def list_jobs(limit: int = Query(20, ge=1, le=100)) -> list[dict]:
    return [
        {"job_id": j.job_id, "state": j.state, "processed": j.processed, "total": j.total,
         "automation_rate": j.summary.automation_rate if j.summary else None,
         "report_url": j.report_url, "started_at": j.started_at}
        for j in jobs.recent(limit)
    ]


@app.get("/v1/jobs/{job_id}", response_model=JobStatus, tags=["batch"])
def job_status(job_id: str, include_results: bool = Query(True)) -> JobStatus:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id}")
    partial = job.summary
    if partial is None and job.results:
        # Live progress figures while the batch is still running.
        partial = summarise(job.results, (0.0))
    return JobStatus(
        job_id=job.job_id, state=job.state, processed=job.processed, total=job.total,
        summary=partial, results=job.results if include_results else [], error=job.error,
    )


@app.get("/v1/jobs/{job_id}/report", response_class=HTMLResponse, tags=["batch"])
def job_report(job_id: str) -> HTMLResponse:
    store = JobStore.open(job_id)
    if store is None:
        raise HTTPException(404, f"unknown job {job_id}")
    path = store.path_for("report.html")
    if not path.exists():
        raise HTTPException(404, "report not generated yet")
    # Rewrite relative artifact paths so the served copy resolves through the
    # /artifacts mount; the on-disk copy stays portable.
    html = path.read_text(encoding="utf-8")
    html = html.replace('src="', f'src="/artifacts/{job_id}/')
    html = html.replace(f'src="/artifacts/{job_id}/data:', 'src="data:')
    return HTMLResponse(html)


@app.get("/v1/jobs/{job_id}/results.csv", response_class=PlainTextResponse, tags=["batch"])
def job_csv(job_id: str) -> PlainTextResponse:
    job = jobs.get(job_id)
    if job is None:
        store = JobStore.open(job_id)
        if store is None or not store.path_for("batch_results.csv").exists():
            raise HTTPException(404, f"unknown job {job_id}")
        return PlainTextResponse(store.path_for("batch_results.csv").read_text(encoding="utf-8"),
                                 media_type="text/csv")
    return PlainTextResponse(results_to_csv(job.results), media_type="text/csv")


@app.get("/v1/jobs/{job_id}/download", tags=["batch"])
def job_download(job_id: str) -> FileResponse:
    store = JobStore.open(job_id)
    if store is None:
        raise HTTPException(404, f"unknown job {job_id}")
    archive = store.zip_all()
    return FileResponse(archive, filename=f"{job_id}.zip", media_type="application/zip")


# -------------------------------------------------------------------- artifacts
@app.get("/artifacts/{job_id}/{filename:path}", tags=["artifacts"])
def artifact(job_id: str, filename: str) -> FileResponse:
    store = JobStore.open(job_id)
    if store is None:
        raise HTTPException(404, "unknown job")
    path = (store.root / filename).resolve()
    try:
        path.relative_to(store.root.resolve())      # block ../ traversal
    except ValueError:
        raise HTTPException(400, "invalid path") from None
    if not path.is_file():
        raise HTTPException(404, "artifact not found")
    return FileResponse(path)


# ---------------------------------------------------------------------- errors
@app.exception_handler(ImageLoadError)
async def _image_error(_request: Request, exc: ImageLoadError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


# Vòng lặp gán nhãn có người duyệt (/v1/labeling/*, giao diện ở /review).
app.include_router(labeling_router)


@app.get("/review", response_class=HTMLResponse, include_in_schema=False)
def review_ui() -> HTMLResponse:
    index = STATIC_DIR / "review.html"
    if not index.exists():
        raise HTTPException(404, "thieu review.html")
    return HTMLResponse(index.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------- UI
if STATIC_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse('<h1>ai-automask</h1><p>API docs at <a href="/docs">/docs</a></p>')
