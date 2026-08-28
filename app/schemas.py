"""Pydantic contracts for the public API (also drives the Swagger docs)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl

Category = Literal["auto", "apparel", "drinkware", "wall_art", "accessory"]
Verdict = Literal["READY", "REVIEW", "FAILED"]


class MaskMetrics(BaseModel):
    """Explainable sub-scores behind the confidence value."""

    coverage: float = Field(..., description="Fraction of the frame occupied by the product (0-1)")
    edge_sharpness: float = Field(..., description="1.0 = crisp binary boundary, 0.0 = mush")
    uncertain_ratio: float = Field(..., description="Fraction of pixels with alpha in (0.05, 0.95)")
    boundary_complexity: float = Field(..., description="Perimeter / sqrt(area) normalised; hair & fringe raise it")
    component_penalty: float = Field(..., description="Penalty for fragmented masks (0 = single clean blob)")
    border_contact: float = Field(..., description="Fraction of the image border touched by the mask")
    ensemble_iou: float | None = Field(None, description="IoU against the cross-check model; None if disabled")
    holes: int = Field(..., description="Number of interior holes retained")
    solidity: float = Field(..., description="area / convex-hull area")


class PrintArea(BaseModel):
    kind: str = Field(..., description="torso | cylinder | quad | none")
    quad: list[list[float]] = Field(default_factory=list, description="4 corner points [x, y] in source pixels, TL-TR-BR-BL")
    bbox: list[float] = Field(default_factory=list, description="[x, y, w, h] axis-aligned bounds")
    area_px: int = 0
    coverage_of_product: float = 0.0
    perspective_matrix: list[list[float]] | None = Field(
        None, description="3x3 homography mapping a unit design canvas onto the quad"
    )
    confidence: float = 0.0


class Artifacts(BaseModel):
    alpha_mask: str | None = Field(None, description="8-bit greyscale PNG - the deliverable mask")
    cutout_rgba: str | None = Field(None, description="Source image with the alpha channel applied")
    overlay: str | None = Field(None, description="Original vs mask overlay for human review")
    shadow_map: str | None = None
    highlight_map: str | None = None
    displacement_map: str | None = None
    print_area_preview: str | None = None
    trimap: str | None = None


class MaskResult(BaseModel):
    id: str
    source: str = Field(..., description="Filename or URL the image came from")
    status: Literal["ok", "error"] = "ok"
    verdict: Verdict | None = None
    confidence: float | None = Field(None, description="0-1 self-assessed mask reliability")
    reasons: list[str] = Field(default_factory=list, description="Human-readable QC explanations")
    category: str | None = None
    category_source: str | None = Field(None, description="metadata | auto-detected")
    width: int | None = None
    height: int | None = None
    model_used: str | None = None
    models_tried: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    metrics: MaskMetrics | None = None
    print_area: PrintArea | None = None
    artifacts: Artifacts = Field(default_factory=Artifacts)
    error: str | None = None


class UrlRequest(BaseModel):
    """Single-image API call used by upstream services."""

    image_url: HttpUrl = Field(..., description="Direct URL of the blank product image")
    category: Category = "auto"
    sku: str | None = Field(None, description="Optional caller-side identifier echoed back")
    emit_shadow_maps: bool | None = None
    emit_displacement: bool | None = None
    return_base64: bool = Field(False, description="Inline the PNG artifacts as base64 instead of URLs")


class BatchItem(BaseModel):
    image_url: HttpUrl
    category: Category = "auto"
    sku: str | None = None


class BatchRequest(BaseModel):
    items: list[BatchItem] = Field(..., min_length=1, max_length=500)
    emit_shadow_maps: bool | None = None
    emit_displacement: bool | None = None


class SamplesRequest(BaseModel):
    """Run the bundled (or any server-local) folder of bases."""

    folder: str | None = Field(None, description="Server-side folder; defaults to data/samples")
    limit: int = Field(50, ge=1, le=500)
    emit_shadow_maps: bool | None = None
    emit_displacement: bool | None = None


class BatchSummary(BaseModel):
    total: int
    ready: int
    review: int
    failed: int
    errors: int
    automation_rate: float = Field(..., description="READY / total - the KPI the design team cares about")
    touchless_rate: float = Field(..., description="(READY + REVIEW) / total - anything not rejected outright")
    mean_confidence: float
    mean_latency_ms: float
    total_wall_ms: float
    throughput_img_per_min: float


class BatchResponse(BaseModel):
    job_id: str
    summary: BatchSummary
    results: list[MaskResult]
    report_url: str | None = None


class JobStatus(BaseModel):
    job_id: str
    state: Literal["queued", "running", "done", "error"]
    processed: int
    total: int
    summary: BatchSummary | None = None
    results: list[MaskResult] = Field(default_factory=list)
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    models: dict[str, Any]
    settings: dict[str, Any]
