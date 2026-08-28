"""The auto-masking pipeline - one function, one image, one auditable result.

    load -> classify -> segment -> cross-check -> refine -> QC -> print area
         -> lighting maps -> artifacts -> MaskResult

Stage notes worth knowing before reading the code:

* **Cross-check is not an accuracy trick, it is the QC input.** We run a second,
  architecturally unrelated model (U2-Net, 320 px, ~0.3 s CPU) purely so the
  quality stage can ask "does anyone else agree with this boundary?". It is the
  difference between a confidence number and a guess. It can be turned off
  (``AUTOMASK_ENSEMBLE=0``) when throughput matters more than autonomy.

* **Category drives geometry, not segmentation.** The matte is class-agnostic;
  the category only selects the print-area solver, the plausible coverage band
  and the edge-hardness curve. So a wrong category degrades the print box, never
  the mask.

* **Every artifact is written at the source resolution.** The only resize in the
  whole path is inside the model wrappers, and ``SegmentationBackend.predict``
  restores the original geometry before anything else sees the array.

* **Failures are results, not exceptions.** A broken URL or an undecodable file
  comes back as a ``MaskResult`` with ``status="error"``, so one bad row never
  takes down a 500-image batch.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CATEGORIES, VERDICT_FAILED, settings
from .imaging import ImageLoadError, decode_image, fetch_url, load_path, thumbnail
from .postprocess import classify, maps, printarea, quality, refine
from .schemas import Artifacts, MaskResult
from .segmentation import registry
from .storage import JobStore

log = logging.getLogger("automask.pipeline")


@dataclass
class ProcessOptions:
    category: str = "auto"
    emit_shadow_maps: bool | None = None
    emit_displacement: bool | None = None
    emit_print_area: bool | None = None
    emit_trimap: bool = True
    emit_cutout: bool = True
    emit_overlay: bool = True
    sku: str | None = None
    ensemble: bool | None = None
    extras: dict = field(default_factory=dict)

    def flag(self, name: str) -> bool:
        override = getattr(self, name, None)
        if override is not None:
            return bool(override)
        return bool(getattr(settings, name, False))


class _Timer:
    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        self._t0 = time.perf_counter()
        self._last = self._t0

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.marks[name] = round((now - self._last) * 1000.0, 2)
        self._last = now

    def total(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000.0, 2)


# --------------------------------------------------------------------- loading
def load_source(
    *,
    data: bytes | None = None,
    url: str | None = None,
    path: str | Path | None = None,
) -> np.ndarray:
    if data is not None:
        return decode_image(data)
    if url is not None:
        return fetch_url(url)
    if path is not None:
        return load_path(path)
    raise ImageLoadError("no image source provided")


# ------------------------------------------------------------------- main call
def process_image(
    store: JobStore,
    source_name: str,
    *,
    data: bytes | None = None,
    url: str | None = None,
    path: str | Path | None = None,
    options: ProcessOptions | None = None,
    slug: str | None = None,
) -> MaskResult:
    """Run the full pipeline for one image and persist its artifacts."""
    opts = options or ProcessOptions()
    timer = _Timer()
    result = MaskResult(id=slug or store.job_id, source=source_name)

    # ---- 1. load ----------------------------------------------------------
    try:
        image = load_source(data=data, url=url, path=path)
    except ImageLoadError as exc:
        result.status = "error"
        result.verdict = VERDICT_FAILED
        result.error = str(exc)
        result.confidence = 0.0
        result.reasons = [str(exc)]
        result.timings_ms = {"total": timer.total()}
        return result
    timer.mark("load")

    h, w = image.shape[:2]
    result.width, result.height = w, h
    base = slug or "image"

    try:
        # ---- 2. segment (primary) -----------------------------------------
        primary = registry.primary()
        seg = primary.predict(image)
        result.model_used = seg.model
        result.models_tried = [seg.model]
        timer.mark("segment")

        # ---- 3. cross-check for QC ---------------------------------------
        cross_alpha = None
        use_ensemble = settings.ensemble if opts.ensemble is None else opts.ensemble
        if use_ensemble:
            cross = registry.cross_check(exclude=seg.model)
            if cross is not None:
                try:
                    cross_out = cross.predict(image)
                    cross_alpha = cross_out.alpha
                    result.models_tried.append(cross_out.model)
                except Exception as exc:  # noqa: BLE001 - QC input is optional
                    log.warning("cross-check failed: %s", exc)
        timer.mark("cross_check")

        # ---- 4. category --------------------------------------------------
        category = opts.category if opts.category in CATEGORIES else "auto"
        if category == "auto":
            # Detected on the raw matte: the refiner's hole handling depends on
            # the category, so classification has to come first.
            raw_binary = (seg.alpha > 0.5).astype(np.uint8)
            _cleaned, holes = refine.fill_small_holes(refine.keep_significant_components(raw_binary))
            detected, cat_conf, cat_detail = classify.detect_category(image, seg.alpha, holes=holes)
            category = detected
            result.category_source = "auto-detected"
            result.category = category
            cat_note = f"Category auto-detected as {category} (confidence {cat_conf:.2f})."
        else:
            result.category_source = "metadata"
            result.category = category
            cat_conf, cat_detail = 1.0, {"note": "supplied by caller"}
            cat_note = None
        timer.mark("classify")

        # ---- 5. refine ----------------------------------------------------
        alpha, refine_info = refine.refine_alpha(image, seg.alpha, category=category)
        timer.mark("refine")

        # Optional SAM boundary second opinion (off by default, see sam_refiner).
        sam_alpha = registry.sam.refine(image, alpha)
        if sam_alpha is not None:
            sam_iou = quality.iou(alpha > 0.5, sam_alpha > 0.5)
            refine_info["sam_iou"] = round(sam_iou, 4)
            result.models_tried.append("mobile_sam")
            if cross_alpha is None:
                cross_alpha = sam_alpha
            timer.mark("sam")

        # ---- 6. quality check --------------------------------------------
        verdict, confidence, metrics, reasons, qc_detail = quality.assess(
            image, alpha, category=category, refine_info=refine_info, cross_alpha=cross_alpha
        )
        if cat_note:
            reasons.append(cat_note)
        result.verdict = verdict
        result.confidence = confidence
        result.metrics = metrics
        result.reasons = reasons
        timer.mark("qc")

        # ---- 7. print area ------------------------------------------------
        pa = None
        if opts.flag("emit_print_area"):
            pa = printarea.detect_print_area(image, alpha, category=category)
            result.print_area = pa
        timer.mark("print_area")

        # ---- 8. artifacts -------------------------------------------------
        artifacts = Artifacts()
        alpha_u8 = refine.alpha_to_uint8(alpha)
        artifacts.alpha_mask = store.save_png(f"{base}_mask", alpha_u8)

        if opts.emit_cutout:
            artifacts.cutout_rgba = store.save_png(f"{base}_cutout", maps.build_cutout(image, alpha))
        if opts.emit_overlay:
            quad = pa.quad if (pa and pa.quad) else None
            overlay = maps.build_overlay(image, alpha, verdict=verdict, print_area=quad)
            artifacts.overlay = store.save_jpeg(f"{base}_overlay", thumbnail(overlay, 1024), quality=86)
        if opts.emit_trimap:
            artifacts.trimap = store.save_png(f"{base}_trimap", refine.make_trimap(alpha))

        if opts.flag("emit_shadow_maps"):
            shadow, highlight, sh_info = maps.shadow_highlight_maps(image, alpha)
            artifacts.shadow_map = store.save_png(f"{base}_shadow", shadow)
            artifacts.highlight_map = store.save_png(f"{base}_highlight", highlight)
            qc_detail["lighting"] = sh_info
        if opts.flag("emit_displacement") and CATEGORIES[category].get("displacement", True):
            disp, disp_info = maps.displacement_map(image, alpha)
            artifacts.displacement_map = store.save_png(f"{base}_displacement", disp)
            qc_detail["displacement"] = disp_info

        result.artifacts = artifacts
        timer.mark("artifacts")

        # ---- 9. sidecar metadata (what the mockup engine actually reads) ---
        sidecar = {
            "id": result.id,
            "sku": opts.sku,
            "source": source_name,
            "resolution": {"width": w, "height": h},
            "verdict": verdict,
            "confidence": confidence,
            "category": {"value": category, "source": result.category_source,
                         "confidence": cat_conf, "detail": cat_detail},
            "metrics": metrics.model_dump() if metrics else None,
            "qc_detail": qc_detail,
            "refine": refine_info,
            "print_area": pa.model_dump() if pa else None,
            "model": {"primary": seg.model, "tried": result.models_tried},
            "artifacts": artifacts.model_dump(),
        }
        store.save_json(f"{base}_meta", sidecar)

    except Exception as exc:  # noqa: BLE001 - one bad image must not kill a batch
        log.exception("pipeline failure on %s", source_name)
        result.status = "error"
        result.verdict = VERDICT_FAILED
        result.confidence = 0.0
        result.error = f"{type(exc).__name__}: {exc}"
        result.reasons = [f"Processing failed: {result.error}"]

    timings = dict(timer.marks)
    timings["total"] = timer.total()
    result.timings_ms = timings
    return result
