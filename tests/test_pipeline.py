"""Tests for the invariants that would silently break a production integration.

These are deliberately about *contracts*, not about model accuracy (that lives in
scripts/eval_iou.py, which needs ground truth). What is asserted here:

  * the mask always matches the source resolution, exactly - the one hard
    requirement in the brief that a downstream engine cannot work around
  * EXIF-rotated input does not produce a mask rotated against the image
  * a bad file / dead URL becomes a result row, never an exception, so one row
    cannot kill a 500-image batch
  * CSV and JSON manifests parse with the column-name variants real ops teams use
  * the QC scorer refuses to hand out READY for empty and full-frame masks
  * the print-area quad lies inside the product and its homography is invertible

Run with:  pytest -q
Most tests use the always-available GrabCut backend so the suite does not need
600 MB of weights or a GPU in CI.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import batch as batch_mod  # noqa: E402
from app.config import CATEGORIES, settings  # noqa: E402
from app.imaging import ImageLoadError, decode_image, encode_png  # noqa: E402
from app.pipeline import ProcessOptions, process_image  # noqa: E402
from app.postprocess import printarea, quality, refine  # noqa: E402
from app.segmentation.grabcut import GrabCutBackend  # noqa: E402
from app.storage import JobStore  # noqa: E402


# --------------------------------------------------------------------- helpers
def synthetic_base(w: int = 420, h: int = 520, colour=(60, 90, 200)) -> np.ndarray:
    """A dark rounded product on a light sweep - enough for GrabCut to find."""
    img = np.full((h, w, 3), 236, np.uint8)
    img[:, :, 2] = 240
    cv2.rectangle(img, (int(w * 0.22), int(h * 0.18)), (int(w * 0.78), int(h * 0.82)),
                  colour, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def png_bytes(img: np.ndarray) -> bytes:
    return encode_png(img)


@pytest.fixture(scope="module")
def store() -> JobStore:
    return JobStore.create("test_")


@pytest.fixture(scope="module")
def fast_opts() -> ProcessOptions:
    # Skip the optional layers: they are exercised in their own tests and make
    # every other test ~3x slower for no extra coverage.
    return ProcessOptions(emit_shadow_maps=False, emit_displacement=False, emit_overlay=False)


# ------------------------------------------------------------------ resolution
@pytest.mark.parametrize("size", [(240, 320), (640, 480), (1001, 733)])
def test_mask_matches_source_resolution_exactly(store, fast_opts, size):
    h, w = size
    img = synthetic_base(w, h)
    result = process_image(store, "res.png", data=png_bytes(img), options=fast_opts,
                           slug=f"res_{w}x{h}")
    assert result.status == "ok", result.error
    assert (result.width, result.height) == (w, h)

    mask_path = store.path_for(f"res_{w}x{h}_mask.png")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    assert mask is not None, "mask PNG was not written"
    assert mask.shape[:2] == (h, w), f"mask {mask.shape[:2]} != source {(h, w)}"
    assert mask.dtype == np.uint8
    assert mask.ndim == 2, "the alpha mask must be single-channel 8-bit greyscale"


def test_odd_resolution_survives_the_backend_resize(fast_opts, store):
    """Odd dimensions are where a naive resize-and-restore silently drifts."""
    img = synthetic_base(413, 617)
    result = process_image(store, "odd.png", data=png_bytes(img), options=fast_opts, slug="odd")
    mask = cv2.imread(str(store.path_for("odd_mask.png")), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (617, 413) == (result.height, result.width)


def test_backend_contract_restores_geometry():
    """Any backend, however it resizes internally, must return the input shape."""
    backend = GrabCutBackend()
    img = synthetic_base(333, 211)
    out = backend.predict(img)
    assert out.alpha.shape == img.shape[:2]
    assert out.alpha.dtype == np.float32
    assert 0.0 <= float(out.alpha.min()) and float(out.alpha.max()) <= 1.0


# ------------------------------------------------------------------------ EXIF
def test_exif_orientation_applied_once():
    """An Orientation=6 JPEG must decode upright.

    If the tag were ignored, or applied twice, the mask would be 90 degrees out
    against the image the mockup engine loads - which is the classic way this
    kind of service ships a broken base.
    """
    from PIL import Image

    portrait = synthetic_base(200, 400)               # taller than wide
    landscape = np.ascontiguousarray(np.rot90(portrait, k=-1))   # 400x200 stored
    pil = Image.fromarray(landscape)
    buf = io.BytesIO()
    # Orientation 6 = "rotate 90 CW to display", i.e. back to portrait.
    exif = pil.getexif()
    exif[274] = 6
    pil.save(buf, format="JPEG", exif=exif.tobytes(), quality=95)

    decoded = decode_image(buf.getvalue())
    assert decoded.shape[0] > decoded.shape[1], "EXIF rotation was not applied on load"
    assert decoded.shape[:2] == portrait.shape[:2]


def test_rgba_input_flattens_without_black_halo():
    rgba = np.dstack([synthetic_base(120, 120), np.full((120, 120), 255, np.uint8)])
    rgba[:10, :10, 3] = 0
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    out = decode_image(buf.getvalue())
    assert out.shape == (120, 120, 3)
    # Transparent corner must flatten to neutral grey, not to black.
    assert out[:10, :10].mean() > 100


# ------------------------------------------------------------------ robustness
def test_corrupt_upload_is_a_result_not_an_exception(store, fast_opts):
    result = process_image(store, "broken.png", data=b"not an image at all",
                           options=fast_opts, slug="broken")
    assert result.status == "error"
    assert result.verdict == "FAILED"
    assert result.confidence == 0.0
    assert result.error and "decode" in result.error.lower()


def test_unreachable_url_is_a_result_not_an_exception(store, fast_opts):
    result = process_image(store, "gone", url="http://127.0.0.1:1/nope.png",
                           options=fast_opts, slug="gone")
    assert result.status == "error"
    assert result.verdict == "FAILED"
    assert "cannot fetch" in (result.error or "")


def test_oversized_image_is_rejected_with_a_clear_message():
    original = settings.max_side
    settings.max_side = 64
    try:
        with pytest.raises(ImageLoadError) as exc:
            decode_image(png_bytes(synthetic_base(200, 200)))
        assert "long edge" in str(exc.value)
    finally:
        settings.max_side = original


def test_tiny_image_is_rejected():
    with pytest.raises(ImageLoadError):
        decode_image(png_bytes(np.full((16, 16, 3), 200, np.uint8)))


# -------------------------------------------------------------------- manifest
def test_csv_manifest_accepts_real_world_column_names():
    csv = ("sku,Image URL,product_type\n"
           "TEE-001,https://cdn.example.com/a.jpg,T-Shirt\n"
           "MUG-002,https://cdn.example.com/b.png,mug\n"
           "BAD-003,,canvas\n")
    items, warnings = batch_mod.parse_csv(csv.encode())
    assert [i.sku for i in items] == ["TEE-001", "MUG-002"]
    assert [i.category for i in items] == ["apparel", "drinkware"]
    assert any("row 4" in w for w in warnings)


def test_csv_manifest_without_a_recognised_header_falls_back_to_column_one():
    csv = "https://cdn.example.com/a.jpg\nhttps://cdn.example.com/b.jpg\n"
    items, warnings = batch_mod.parse_csv(csv.encode())
    assert len(items) == 2
    assert warnings, "the fallback should be reported to the caller"


def test_semicolon_delimited_csv_is_sniffed():
    csv = "url;category\nhttps://cdn.example.com/a.jpg;wall_art\n"
    items, _ = batch_mod.parse_csv(csv.encode())
    assert len(items) == 1 and items[0].category == "wall_art"


@pytest.mark.parametrize("payload", [
    '["https://cdn.example.com/a.jpg"]',
    '{"items": [{"image_url": "https://cdn.example.com/a.jpg", "category": "hoodie"}]}',
    '{"images": [{"src": "https://cdn.example.com/a.jpg"}]}',
])
def test_json_manifest_shapes(payload):
    items, _ = batch_mod.parse_json(payload.encode())
    assert len(items) == 1
    assert items[0].url.endswith("a.jpg")


def test_json_manifest_maps_aliases_to_the_taxonomy():
    items, _ = batch_mod.parse_json(
        json.dumps([{"url": "https://x/y.jpg", "category": "Phone Case"}]).encode())
    assert items[0].category == "accessory"


def test_unknown_category_degrades_to_auto():
    items, _ = batch_mod.parse_json(
        json.dumps([{"url": "https://x/y.jpg", "category": "spaceship"}]).encode())
    assert items[0].category == "auto"


def test_work_item_slug_is_filesystem_safe():
    item = batch_mod.WorkItem(source="https://cdn.example.com/a b/../weird name!.JPG")
    slug = item.slug(3)
    assert slug.startswith("003_")
    assert all(c.isalnum() or c in "._-" for c in slug)


# -------------------------------------------------------------------------- QC
def test_empty_mask_fails_and_never_reads_ready():
    img = synthetic_base(200, 200)
    verdict, conf, _m, reasons, _d = quality.assess(img, np.zeros((200, 200), np.float32))
    assert verdict == "FAILED"
    assert conf == 0.0
    assert "No product detected" in reasons[0]


def test_full_frame_mask_fails():
    img = synthetic_base(200, 200)
    verdict, _c, _m, reasons, _d = quality.assess(img, np.ones((200, 200), np.float32))
    assert verdict == "FAILED"
    assert "whole frame" in reasons[0]


def test_fragmented_mask_never_reads_ready():
    """A main blob plus a scatter of ragged shards is a broken mask.

    Note the shards are deliberately unequal and irregular. Piece *count* alone
    is not the defect - see the multi-instance test below - so the fixture has
    to exhibit real fragmentation for this assertion to mean anything.
    """
    # Pieces must be large enough to count as significant (>= 5% of the main
    # blob) but clearly unequal, which is what separates a torn mask from a
    # photo of several products.
    img = np.full((520, 520, 3), 235, np.uint8)
    alpha = np.zeros((520, 520), np.float32)
    for (x, y, w, h) in ((40, 40, 260, 260), (330, 90, 170, 100), (150, 400, 90, 60)):
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
        alpha[y:y + h, x:x + w] = 1.0
    verdict, _c, _m, reasons, detail = quality.assess(img, alpha, category="auto")
    assert verdict != "READY"
    assert any("separate pieces" in r for r in reasons)
    assert "split into" in detail.get("ready_veto", [])


def test_multi_instance_photo_is_not_treated_as_fragmented():
    """Two whole products of comparable size is a composition, not a defect.

    The real base library photographs the back of every mug base as *two* mugs
    side by side. Treating piece count as fragmentation sent every one of those
    to manual review, which is what this test exists to prevent regressing.
    """
    img = np.full((400, 700, 3), 240, np.uint8)
    alpha = np.zeros((400, 700), np.float32)
    for x0 in (60, 380):
        cv2.rectangle(img, (x0, 90), (x0 + 260, 320), (55, 55, 60), -1)
        alpha[90:320, x0:x0 + 260] = 1.0
    _v, _c, _m, reasons, detail = quality.assess(img, alpha, category="drinkware")
    assert "split into" not in detail.get("ready_veto", [])
    assert detail["topology"]["multi_instance"] is True
    assert detail["topology"]["component_penalty"] == 0.0
    assert any("multi-item composition" in r for r in reasons)


def test_solidity_is_measured_correctly_for_a_perfect_rectangle():
    """Regression: a 4-point contour must not score 0 solidity."""
    alpha = np.zeros((300, 300), np.float32)
    alpha[50:250, 60:240] = 1.0
    _score, detail = quality.topology_signals(alpha, 0, CATEGORIES["wall_art"])
    assert detail["solidity"] > 0.95, detail


def test_ensemble_iou_is_reported_when_a_cross_check_is_supplied():
    img = synthetic_base(300, 300)
    alpha = np.zeros((300, 300), np.float32)
    alpha[60:240, 60:240] = 1.0
    shifted = np.zeros_like(alpha)
    shifted[64:244, 64:244] = 1.0
    _v, _c, metrics, _r, detail = quality.assess(img, alpha, cross_alpha=shifted)
    assert metrics.ensemble_iou is not None
    assert 0.8 < metrics.ensemble_iou < 1.0
    assert "cross_check" in detail


def test_confidence_is_bounded_and_metrics_serialise():
    img = synthetic_base(300, 300)
    alpha = np.zeros((300, 300), np.float32)
    alpha[60:240, 60:240] = 1.0
    _v, conf, metrics, _r, _d = quality.assess(img, alpha)
    assert 0.0 <= conf <= 1.0
    assert set(metrics.model_dump()) >= {"coverage", "edge_sharpness", "solidity", "holes"}


def test_iou_helper_handles_the_empty_case():
    z = np.zeros((10, 10), bool)
    assert quality.iou(z, z) == 1.0
    o = np.ones((10, 10), bool)
    assert quality.iou(z, o) == 0.0


# --------------------------------------------------------------------- refiner
def test_refiner_keeps_a_structural_hole_and_drops_specks():
    """A mug handle's aperture must survive; a 3 px speck must not."""
    img = np.full((400, 400, 3), 240, np.uint8)
    cv2.rectangle(img, (100, 80), (300, 340), (50, 50, 50), -1)
    cv2.circle(img, (200, 200), 45, (240, 240, 240), -1)
    alpha = np.zeros((400, 400), np.float32)
    alpha[80:340, 100:300] = 1.0
    cv2.circle(alpha, (200, 200), 45, 0.0, -1)          # structural hole
    alpha[10:13, 10:13] = 1.0                            # speck
    out, info = refine.refine_alpha(img, alpha, category="drinkware")
    assert info["holes_kept"] == 1
    assert out[11, 11] == 0.0, "isolated speck should have been removed"
    assert out[200, 200] < 0.5, "the structural hole should still be open"


def test_refiner_never_changes_geometry():
    img = synthetic_base(257, 389)
    alpha = np.zeros((389, 257), np.float32)
    alpha[80:300, 60:200] = 1.0
    out, _info = refine.refine_alpha(img, alpha, category="apparel")
    assert out.shape == alpha.shape
    assert out.dtype == np.float32


def test_strand_suppression_ignores_a_clean_garment():
    """No thin dark structures -> the mask must come back untouched."""
    img = np.full((600, 600, 3), 238, np.uint8)
    cv2.rectangle(img, (150, 120), (450, 500), (246, 246, 248), -1)
    alpha = np.zeros((600, 600), np.float32)
    alpha[120:500, 150:450] = 1.0
    out, info = refine.suppress_thin_occluders(img, alpha)
    assert info["applied"] is False
    assert np.array_equal(out, alpha)


def test_alpha_to_uint8_round_trips_the_extremes():
    a = np.array([[0.0, 0.5, 1.0]], np.float32)
    assert refine.alpha_to_uint8(a).tolist() == [[0, 128, 255]]


# ------------------------------------------------------------------ print area
def test_print_area_stays_inside_the_product():
    alpha = np.zeros((600, 500), np.float32)
    alpha[100:500, 80:420] = 1.0
    img = np.full((600, 500, 3), 235, np.uint8)
    img[100:500, 80:420] = (60, 60, 60)
    pa = printarea.detect_print_area(img, alpha, category="wall_art")
    assert pa.kind == "quad"
    assert len(pa.quad) == 4
    poly = np.zeros((600, 500), np.uint8)
    cv2.fillConvexPoly(poly, np.array(pa.quad, np.int32), 1)
    outside = int(((poly > 0) & (alpha < 0.5)).sum())
    assert outside == 0, f"{outside} print-area pixels fall outside the product"


def test_print_area_homography_maps_the_unit_square_onto_the_quad():
    alpha = np.zeros((400, 400), np.float32)
    alpha[60:340, 60:340] = 1.0
    img = np.full((400, 400, 3), 240, np.uint8)
    img[60:340, 60:340] = (40, 40, 40)
    pa = printarea.detect_print_area(img, alpha, category="wall_art")
    m = np.array(pa.perspective_matrix, np.float64)
    assert m.shape == (3, 3)
    assert abs(np.linalg.det(m)) > 1e-9, "homography must be invertible"
    for (u, v), expected in zip([(0, 0), (1, 0), (1, 1), (0, 1)], pa.quad):
        p = m @ np.array([u, v, 1.0])
        got = (p[0] / p[2], p[1] / p[2])
        assert np.allclose(got, expected, atol=1e-3)


def test_largest_inscribed_rect_is_exact():
    grid = np.zeros((10, 10), np.uint8)
    grid[2:8, 3:6] = 1                    # 6 rows x 3 cols = 18
    rect = printarea.largest_inscribed_rect(grid)
    assert rect == (3, 2, 3, 6)


def test_print_area_none_for_an_empty_mask():
    pa = printarea.detect_print_area(np.full((100, 100, 3), 200, np.uint8),
                                     np.zeros((100, 100), np.float32))
    assert pa.kind == "none"


# ---------------------------------------------------------------------- config
def test_every_category_declares_the_keys_the_pipeline_reads():
    required = {"label", "print_area", "expect_coverage", "expect_holes", "soft_edges",
                "displacement", "shape"}
    for name, cfg in CATEGORIES.items():
        assert required <= set(cfg), f"{name} is missing {required - set(cfg)}"
        lo, hi = cfg["expect_coverage"]
        assert 0.0 < lo < hi <= 1.0
        assert set(cfg["shape"]) == {"bbox_fill", "aspect", "quad_fit_min"}


def test_batch_summary_reports_the_automation_rate():
    from app.schemas import MaskResult

    results = [
        MaskResult(id="a", source="a", verdict="READY", confidence=0.9, timings_ms={"total": 100}),
        MaskResult(id="b", source="b", verdict="REVIEW", confidence=0.6, timings_ms={"total": 200}),
        MaskResult(id="c", source="c", verdict="FAILED", confidence=0.1, timings_ms={"total": 300}),
        MaskResult(id="d", source="d", verdict="READY", confidence=0.95, timings_ms={"total": 100}),
    ]
    summary = batch_mod.summarise(results, wall_ms=1000.0)
    assert summary.total == 4
    assert summary.automation_rate == 0.5
    assert summary.touchless_rate == 0.75
    assert summary.mean_latency_ms == 175.0
    assert summary.throughput_img_per_min == 240.0


def test_results_csv_has_a_row_per_image_and_a_stable_header():
    from app.schemas import MaskResult

    rows = batch_mod.results_to_csv([
        MaskResult(id="a", source="a.png", verdict="READY", confidence=0.9),
        MaskResult(id="b", source="b.png", status="error", verdict="FAILED", error="boom"),
    ]).strip().splitlines()
    assert len(rows) == 3
    assert rows[0].startswith("id,source,status,verdict,confidence")
    assert "boom" in rows[2]
