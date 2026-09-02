"""Mask Quality Check - the self-assessment that produces READY / REVIEW / FAILED.

This is the part that makes the system deployable rather than merely clever. A
segmentation model always returns *something*; without an honest confidence
signal a designer still has to eyeball all 1000 masks, and the bottleneck the
brief describes never actually moves.

Design constraint: we have **no ground truth at inference time**. So every
signal here is *no-reference* - computed from the image and the mask alone:

  1. edge_alignment  - do the mask's boundary pixels sit on real image edges?
                       (Canny on the source, distance transform, sample the
                       boundary.) This is the strongest single-model signal:
                       a lazy or bloated mask cuts through flat pixels.
  2. ensemble_iou    - does an architecturally independent model (U2-Net) agree?
                       Two models agreeing on a boundary is the closest thing to
                       ground truth we can get for free.
  3. edge_sharpness  - is the alpha transition a tight 1-3 px ramp, or 30 px of
                       mush that will composite as a grey halo?
  4. topology        - component count, hole count vs. category expectation,
                       solidity. Catches shredded and inverted masks.
  5. coverage        - plausible product-to-frame ratio for the category.
                       Catches "matted the whole backdrop" and "found nothing".
  6. border_contact  - a product running off the frame cannot be a clean base.

Weights were tuned so that the failure modes a designer actually complains
about (halo, clipped sleeve, shadow eaten as product) land in REVIEW rather
than sneaking through as READY. Everything is reported back to the caller, so
the verdict is auditable instead of a black-box number.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import CATEGORIES, VERDICT_FAILED, VERDICT_READY, VERDICT_REVIEW, settings
from ..schemas import MaskMetrics

# Relative weights of the quality terms (sum normalised at use).
WEIGHTS = {
    "edge_alignment": 0.20,
    "ensemble": 0.20,
    "boundary_contrast": 0.14,
    "sharpness": 0.12,
    "bg_consistency": 0.10,
    "topology": 0.10,
    "coverage": 0.07,
    "border": 0.05,
    "shape_prior": 0.02,
}


def _binary(alpha: np.ndarray, t: float = 0.5) -> np.ndarray:
    return (alpha > t).astype(np.uint8)


def largest_component(binary: np.ndarray) -> np.ndarray:
    """The biggest connected piece of a mask.

    Silhouette statistics - bounding-box fill, aspect, quad fit, outline
    complexity - describe *a product*. Measured across a multi-item photo they
    describe the photo's layout instead: two mugs side by side leave a diagonal
    gap that drags bbox fill from 0.75 down to 0.43, and every mug in the real
    base library then looks "atypical for drinkware". Measuring one instance
    keeps the prior meaning what it was calibrated to mean.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return binary
    idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == idx).astype(np.uint8)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a_b, b_b = a.astype(bool), b.astype(bool)
    union = np.logical_or(a_b, b_b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a_b, b_b).sum() / union)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tol: int = 3) -> float:
    """F1 of boundary pixels within *tol* px - the metric that reflects
    'does the edge look right', which plain IoU hides on large objects."""
    pe = cv2.Canny(pred.astype(np.uint8) * 255, 50, 150) > 0
    ge = cv2.Canny(gt.astype(np.uint8) * 255, 50, 150) > 0
    if pe.sum() == 0 or ge.sum() == 0:
        return 0.0 if (pe.sum() != ge.sum()) else 1.0
    dt_g = cv2.distanceTransform((~ge).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((~pe).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((dt_g[pe] <= tol).mean())
    recall = float((dt_p[ge] <= tol).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------- signals
def edge_alignment(image: np.ndarray, alpha: np.ndarray) -> tuple[float, dict]:
    """Fraction of mask-boundary pixels that sit on a real photometric edge.

    A correct cut-out traces a discontinuity in the image. A sloppy one - bleeding
    into the sweep, or clipping into flat fabric - traces nothing.

    Measured on the *gradient field* rather than on Canny output. Canny needs two
    thresholds, and no fixed pair works across both a white-on-white studio sweep
    and a high-contrast lifestyle shot: tuned for one, it reports "this image has
    no edges" on the other and the signal silently degrades to a constant. Sobel
    magnitude compared against the image's own median gradient is scale-free and
    needs no tuning: a genuine product boundary is many times stronger than the
    median gradient of a photo, which is dominated by flat regions.

    A 1-pixel dilation of the gradient field before sampling absorbs the
    half-pixel disagreement between where the matte puts the edge and where the
    intensity step actually is.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    contour = cv2.morphologyEx(_binary(alpha), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if contour.sum() < 32:
        return 0.0, {"note": "mask has no boundary"}

    # Tolerance in pixels, scaled with resolution: a 4000 px photo's boundary may
    # legitimately be 3-4 px from the gradient peak after upsampling the matte.
    tol = max(1, int(round(0.0015 * max(image.shape[:2]))))
    k = 2 * tol + 1
    local_max = cv2.dilate(grad, np.ones((k, k), np.float32))

    median_grad = float(np.median(grad))
    # Guard the degenerate case of a synthetic flat image where the median is 0.
    ref = max(median_grad * 2.5, float(np.percentile(grad, 99.0)) * 0.06, 1.5)
    aligned = float((local_max[contour] >= ref).mean())
    return aligned, {
        "aligned_ratio": round(aligned, 4),
        "tolerance_px": tol,
        "median_gradient": round(median_grad, 3),
        "threshold": round(ref, 3),
    }


def edge_sharpness(alpha: np.ndarray) -> tuple[float, float]:
    """Return (score, uncertain_ratio).

    Measures the *width* of the alpha ramp: uncertain pixels divided by boundary
    length gives an average ramp thickness in pixels. 1-3 px is a crisp,
    correctly anti-aliased matte; >12 px composites as a visible halo.
    """
    uncertain = ((alpha > 0.05) & (alpha < 0.95))
    n_unc = int(uncertain.sum())
    total = alpha.size
    uncertain_ratio = n_unc / float(total)

    contour = cv2.morphologyEx(_binary(alpha), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    perim = int(contour.sum())
    if perim < 32:
        return 0.0, uncertain_ratio
    ramp_px = n_unc / float(perim)
    # 2 px -> 1.0, 14 px -> 0.0
    score = float(np.clip(1.0 - (ramp_px - 2.0) / 12.0, 0.0, 1.0))
    return score, uncertain_ratio


def background_consistency(image: np.ndarray, alpha: np.ndarray) -> tuple[float, dict]:
    """Is the ring just *outside* the mask actually background?

    This catches the failure no other signal here can see: a mask that is a
    clean, sharp, well-aligned cut of the *wrong thing* - the white print inside
    a dark picture frame, a label peeled off a bottle. Every edge-based signal is
    happy with those, because the boundary really is a strong edge. But the
    pixels immediately outside a *correct* product mask must look like the
    backdrop, and for a sub-region of a larger object they do not.

    Background is modelled from the frame border (robust median + MAD in Lab),
    which is where studio sweeps and walls live. A busy background widens the MAD
    and softens the signal automatically, so it degrades gracefully instead of
    firing constantly on lifestyle shots.
    """
    h, w = alpha.shape[:2]
    binary = _binary(alpha)
    if binary.sum() < 64 or binary.mean() > 0.96:
        return 0.5, {"note": "mask too small or too large to test"}

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    edge = max(4, int(0.02 * max(h, w)))
    border = np.concatenate([
        lab[:edge].reshape(-1, 3), lab[-edge:].reshape(-1, 3),
        lab[:, :edge].reshape(-1, 3), lab[:, -edge:].reshape(-1, 3),
    ])
    # Exclude border pixels the mask claims are product (a product running off
    # the frame), so the object cannot poison its own background model.
    border_mask = np.concatenate([
        binary[:edge].reshape(-1), binary[-edge:].reshape(-1),
        binary[:, :edge].reshape(-1), binary[:, -edge:].reshape(-1),
    ])
    border = border[border_mask == 0]
    if len(border) < 256:
        return 0.5, {"note": "no clean border to model the background from"}

    med = np.median(border, axis=0)
    mad = np.median(np.abs(border - med), axis=0) * 1.4826 + 2.0

    ring_r = max(3, int(0.008 * max(h, w)))
    ring = (cv2.dilate(binary, np.ones((ring_r * 2 + 1,) * 2, np.uint8)) - binary) > 0
    if ring.sum() < 128:
        return 0.5, {"note": "no outside ring available"}

    vals = lab[ring]
    dist = np.abs(vals - med) / mad
    # Chroma has to match; luminance is allowed to be *darker*, because the
    # product's own contact shadow falls on the backdrop right where we sample.
    # Without this, every correctly-masked product with a drop shadow looked
    # like it was sitting on a different material.
    chroma_ok = dist[:, 1] <= 3.0
    chroma_ok &= dist[:, 2] <= 3.0
    lum_ok = (dist[:, 0] <= 3.0) | (vals[:, 0] < med[0])
    matches = float((chroma_ok & lum_ok).mean())
    score = float(np.clip((matches - 0.35) / 0.5, 0.0, 1.0))
    return score, {"outside_is_background": round(matches, 4),
                   "ring_px": int(ring.sum()), "ring_radius_px": ring_r}


def boundary_contrast(image: np.ndarray, alpha: np.ndarray) -> tuple[float, dict]:
    """Is there a real photometric step *across* the mask boundary?

    Complementary to ``edge_alignment``: that one asks "is the outline sitting
    on a Canny edge", this one asks "are the colours on the two sides of the
    outline actually different, relative to the local noise floor". It is the
    signal that catches a mask which has swallowed a neighbouring object - at the
    point where the mask crosses the middle of that object, inside and outside
    look identical, so the boundary there is arbitrary.

    Implementation: two masked box-filters give, for every boundary pixel, the
    mean Lab colour of the mask side and of the background side of its local
    neighbourhood. Their distance is compared against the local Lab standard
    deviation, so a white mug on a white sweep - where the true step is small but
    the noise floor is smaller still - is not punished for low absolute contrast.
    """
    h, w = alpha.shape[:2]
    binary = _binary(alpha)
    if binary.sum() < 64 or binary.mean() > 0.98:
        return 0.5, {"note": "no usable boundary"}

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    r = max(2, int(0.004 * max(h, w)))
    k = (r * 2 + 1, r * 2 + 1)

    inside = binary.astype(np.float32)
    outside = 1.0 - inside

    def masked_mean(weight: np.ndarray) -> np.ndarray:
        den = cv2.boxFilter(weight, cv2.CV_32F, k, normalize=True)
        num = np.dstack([cv2.boxFilter(lab[..., c] * weight, cv2.CV_32F, k, normalize=True)
                         for c in range(3)])
        return num / np.maximum(den[..., None], 1e-3)

    mean_in = masked_mean(inside)
    mean_out = masked_mean(outside)

    # Local noise floor from the variance of L over the same window.
    mean_l = cv2.boxFilter(lab[..., 0], cv2.CV_32F, k)
    mean_l2 = cv2.boxFilter(lab[..., 0] * lab[..., 0], cv2.CV_32F, k)
    noise = np.sqrt(np.maximum(mean_l2 - mean_l * mean_l, 0.0)) + 1.5

    contour = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    # Ignore the frame edge: there is no "outside" there to compare against.
    contour[:r + 1, :] = contour[-r - 1:, :] = False
    contour[:, :r + 1] = contour[:, -r - 1:] = False
    if contour.sum() < 64:
        return 0.5, {"note": "boundary lies on the image frame"}

    step = np.linalg.norm(mean_in - mean_out, axis=2)
    ratio = step[contour] / noise[contour]
    # A ratio below 1 means the two sides are indistinguishable given local noise.
    flat = float((ratio < 1.0).mean())
    score = float(np.clip(1.0 - (flat - 0.06) / 0.34, 0.0, 1.0))
    return score, {
        "flat_boundary_fraction": round(flat, 4),
        "median_step_over_noise": round(float(np.median(ratio)), 3),
        "sample_px": int(contour.sum()),
    }


def shape_prior(alpha: np.ndarray, cfg: dict) -> tuple[float, dict]:
    """Does the silhouette look like the product class it claims to be?

    Cheap, explainable, class-conditional sanity: a mug is a boxy upright body,
    a canvas is a quad that fills its bounding box, a garment fills 35-90% of
    its box. When a mask swallows an adjacent object the bounding box balloons
    and the fill ratio collapses - which is exactly what this measures.

    Deliberately soft (4% of the confidence weight): a real product library
    always contains something unusual, and this should nudge such an image
    toward REVIEW, never reject it on its own.
    """
    spec = cfg.get("shape") or {}
    if not spec:
        return 1.0, {}
    full = _binary(alpha)
    binary = largest_component(full)
    multi = int(binary.sum()) < int(full.sum())
    ys, xs = np.nonzero(binary)
    if len(ys) < 64:
        return 0.0, {"note": "empty mask"}
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bh, bw = (y1 - y0 + 1), (x1 - x0 + 1)
    fill = float(binary[y0:y1 + 1, x0:x1 + 1].mean())
    aspect = bw / float(bh)

    # Solidity of the same instance, used to decide whether the bbox-fill test
    # applies at all.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(c)) if len(c) >= 3 else 0.0
        if hull_area > 0:
            solidity = float(cv2.contourArea(c) / hull_area)

    detail = {"bbox_fill": round(fill, 4), "aspect": round(aspect, 3),
              "solidity": round(solidity, 4),
              "measured_on": "largest_instance" if multi else "whole_mask"}

    def band_score(value: float, bounds, softness: float) -> float:
        lo, hi = bounds
        if lo <= value <= hi:
            return 1.0
        gap = (lo - value) if value < lo else (value - hi)
        return float(np.clip(1.0 - gap / softness, 0.0, 1.0))

    # Bounding-box fill only means something for a *single* compact product.
    # Two mugs photographed side by side and overlapping fuse into one connected
    # piece whose bbox fill collapses to ~0.50 - numerically identical to the
    # failure case where a mask swallowed adjacent desk clutter (0.496). The two
    # are separated by solidity, not by fill: the fused mug pair measures 0.98,
    # the mug-plus-clutter blob 0.81. So a highly solid silhouette is exempted
    # from the fill test and judged on aspect and quad fit alone.
    if solidity >= 0.93:
        score = 1.0
        detail["fill_test"] = "skipped (solid silhouette)"
    else:
        score = band_score(fill, spec["bbox_fill"], 0.30)
    score = min(score, band_score(aspect, spec["aspect"], 1.20))

    quad_min = spec.get("quad_fit_min")
    if quad_min:
        # Reuse the print-area quad fitter: its IoU-against-silhouette figure is
        # precisely "how quad-shaped is this mask".
        from .printarea import _downscale, _quad_area

        small, _scale = _downscale(binary)
        _quad, fit = _quad_area(small, inset=0.0)
        detail["quad_fit"] = round(fit, 4)
        score = min(score, float(np.clip(1.0 - (quad_min - fit) / 0.25, 0.0, 1.0)))
    return float(np.clip(score, 0.0, 1.0)), detail


def topology_signals(alpha: np.ndarray, holes: int, cfg: dict) -> tuple[float, dict]:
    binary = _binary(alpha)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True) if n > 1 else []
    if not areas:
        return 0.0, {"components": 0, "solidity": 0.0, "component_penalty": 1.0}

    biggest = areas[0]
    significant = [a for a in areas if a >= biggest * 0.05]

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    def _solidity(contour) -> float:
        # >= 3 points, not >= 5. A perfect axis-aligned rectangle traces to
        # exactly 4 points under CHAIN_APPROX_SIMPLE, and the old guard scored
        # every such shape 0.0 - i.e. the flattest, most solid product possible
        # was reported as maximally ragged.
        if contour is None or len(contour) < 3:
            return 0.0
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        return float(cv2.contourArea(contour) / hull_area) if hull_area > 0 else 0.0

    main = contours[0] if contours else None
    solidity = _solidity(main)

    # A mask with several pieces is not automatically broken. Product catalogues
    # are full of legitimate multi-item compositions - the "back" shot of a mug
    # base photographs two mugs side by side, a set is photographed as a set. So
    # separate the two cases instead of punishing piece count alone:
    #
    #   fragmented     -> many pieces, or pieces of wildly different size, or
    #                     pieces that are individually ragged
    #   multi-instance -> a few pieces, comparable in size, each individually a
    #                     solid compact shape
    #
    # Only the first is a defect. Getting this wrong sent every two-mug photo in
    # the real base library to manual review for no reason.
    size_ratio = float(min(significant) / max(significant)) if len(significant) > 1 else 1.0
    piece_solidity = [_solidity(c) for c in contours[: len(significant)]]
    worst_piece_solidity = float(min(piece_solidity)) if piece_solidity else 0.0
    multi_instance = (
        1 < len(significant) <= 4
        and size_ratio >= 0.25
        and worst_piece_solidity >= 0.50
    )
    if multi_instance:
        component_penalty = 0.0
    else:
        extra = max(0, len(significant) - 1)
        component_penalty = float(np.clip(extra / 4.0, 0.0, 1.0))

    hole_penalty = 0.0
    if holes > 12:                       # shredded matte
        hole_penalty = float(np.clip((holes - 12) / 30.0, 0.0, 1.0))
    elif holes > 0 and not cfg["expect_holes"]:
        hole_penalty = 0.35              # a canvas should be a solid quad

    # Very low solidity means a spidery or fragmented shape; apparel with spread
    # arms legitimately sits around 0.7-0.85, so only punish below that.
    solidity_penalty = float(np.clip((0.62 - solidity) / 0.4, 0.0, 1.0))

    score = float(np.clip(1.0 - 0.5 * component_penalty - 0.3 * hole_penalty - 0.4 * solidity_penalty, 0.0, 1.0))
    return score, {
        "components": len(significant),
        "solidity": round(solidity, 4),
        "component_penalty": round(component_penalty, 4),
        "hole_penalty": round(hole_penalty, 4),
        "multi_instance": multi_instance,
        "size_ratio": round(size_ratio, 4),
        "worst_piece_solidity": round(worst_piece_solidity, 4),
    }


def coverage_signal(alpha: np.ndarray, cfg: dict) -> tuple[float, float]:
    coverage = float(_binary(alpha).mean())
    lo, hi = cfg["expect_coverage"]
    if lo <= coverage <= hi:
        score = 1.0
    elif coverage < lo:
        score = float(np.clip(coverage / max(lo, 1e-6), 0.0, 1.0))
    else:
        score = float(np.clip((1.0 - coverage) / max(1.0 - hi, 1e-6), 0.0, 1.0))
    return score, coverage


def border_signal(alpha: np.ndarray) -> tuple[float, float]:
    """Products that run off the frame make bad mockup bases: the mockup engine
    has nothing to composite against at the crop line."""
    b = _binary(alpha)
    edge_px = np.concatenate([b[0], b[-1], b[:, 0], b[:, -1]])
    contact = float(edge_px.mean())
    # Up to ~2% border contact is a harmless nick; 30%+ means a hard crop.
    score = float(np.clip(1.0 - (contact - 0.02) / 0.28, 0.0, 1.0))
    return score, contact


def boundary_complexity(alpha: np.ndarray) -> float:
    """Normalised perimeter. Hair and knit fringe raise it; it is not a defect
    by itself, but it tells the operator why confidence is lower."""
    # Per instance, for the same reason as the shape prior: two mugs have twice
    # the perimeter of one, which is a fact about the photo, not about the cut.
    b = largest_component(_binary(alpha))
    area = float(b.sum())
    if area < 32:
        return 0.0
    # arcLength on the traced contours, not the pixel count of a morphological
    # gradient: the gradient ring is ~2 px thick and would double the perimeter,
    # making every clean rectangle look like it had a fringe.
    contours, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perim = float(sum(cv2.arcLength(c, True) for c in contours))
    if perim <= 0:
        return 0.0
    # A circle sits at perimeter/sqrt(area) = 3.54, a square at 4.0, a t-shirt
    # with sleeves around 5-7. Hair and knit fringe push well past 10.
    return float(np.clip((perim / np.sqrt(area) - 4.2) / 8.0, 0.0, 1.0))


# ------------------------------------------------------------------- aggregate
def assess(
    image: np.ndarray,
    alpha: np.ndarray,
    category: str = "auto",
    refine_info: dict | None = None,
    cross_alpha: np.ndarray | None = None,
) -> tuple[str, float, MaskMetrics, list[str], dict]:
    """Score a mask and return (verdict, confidence, metrics, reasons, detail)."""
    cfg = CATEGORIES.get(category, CATEGORIES["auto"])
    refine_info = refine_info or {}
    reasons: list[str] = []
    detail: dict = {}

    coverage = float(_binary(alpha).mean())

    # ---- hard failures: no point scoring these -----------------------------
    if coverage < 0.004:
        m = MaskMetrics(coverage=coverage, edge_sharpness=0.0, uncertain_ratio=0.0, boundary_complexity=0.0,
                        component_penalty=1.0, border_contact=0.0, ensemble_iou=None, holes=0, solidity=0.0)
        return VERDICT_FAILED, 0.0, m, ["No product detected - mask is essentially empty."], detail
    if coverage > 0.985:
        m = MaskMetrics(coverage=coverage, edge_sharpness=0.0, uncertain_ratio=0.0, boundary_complexity=0.0,
                        component_penalty=1.0, border_contact=1.0, ensemble_iou=None, holes=0, solidity=1.0)
        return (VERDICT_FAILED, 0.0, m,
                ["Mask covers the whole frame - the model failed to separate product from background."], detail)

    # ---- signals -----------------------------------------------------------
    s_align, align_detail = edge_alignment(image, alpha)
    s_sharp, uncertain_ratio = edge_sharpness(alpha)
    s_topo, topo_detail = topology_signals(alpha, int(refine_info.get("holes_kept", 0)), cfg)
    s_cov, coverage = coverage_signal(alpha, cfg)
    s_border, border_contact = border_signal(alpha)
    s_bg, bg_detail = background_consistency(image, alpha)
    s_shape, shape_detail = shape_prior(alpha, cfg)
    s_contrast, contrast_detail = boundary_contrast(image, alpha)
    complexity = boundary_complexity(alpha)

    ens_iou: float | None = None
    s_ens = None
    if cross_alpha is not None:
        ens_iou = iou(_binary(alpha), _binary(cross_alpha))
        b_f1 = boundary_f1(_binary(alpha), _binary(cross_alpha), tol=max(2, int(0.002 * max(image.shape[:2]))))
        detail["cross_check"] = {"iou": round(ens_iou, 4), "boundary_f1": round(b_f1, 4)}
        # Agreement is informative in both dimensions: same blob (IoU) and same
        # edge placement (boundary F1). Weight the region term higher because a
        # 320 px cross-check model legitimately has a coarser boundary.
        s_ens = float(np.clip(0.72 * ens_iou + 0.28 * b_f1, 0.0, 1.0))

    detail["edge_alignment"] = align_detail
    detail["topology"] = topo_detail
    detail["background"] = bg_detail
    detail["shape"] = shape_detail
    detail["boundary_contrast"] = contrast_detail

    # ---- weighted aggregate ------------------------------------------------
    terms = {
        "edge_alignment": s_align,
        "sharpness": s_sharp,
        "topology": s_topo,
        "coverage": s_cov,
        "border": s_border,
        "bg_consistency": s_bg,
        "shape_prior": s_shape,
        "boundary_contrast": s_contrast,
    }
    if s_ens is not None:
        terms["ensemble"] = s_ens
    weight_sum = sum(WEIGHTS[k] for k in terms)
    confidence = sum(WEIGHTS[k] * v for k, v in terms.items()) / weight_sum

    # Complex boundaries (hair, fringe) are genuinely harder; shade confidence a
    # little so those masks route to a human rather than shipping blind.
    confidence *= (1.0 - 0.10 * complexity)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    detail["terms"] = {k: round(v, 4) for k, v in terms.items()}
    detail["weights"] = {k: WEIGHTS[k] for k in terms}

    # ---- human-readable reasons -------------------------------------------
    if s_align < 0.45:
        reasons.append(f"Only {s_align*100:.0f}% of the mask outline sits on a real image edge - "
                       f"the boundary may be bleeding into the background or clipping the product.")
    if s_sharp < 0.55:
        reasons.append("Alpha transition is wide (soft/hazy edge) - risk of a grey halo when composited.")
    if s_ens is not None and s_ens < 0.78:
        reasons.append(
            f"Cross-check model disagrees (IoU {ens_iou:.2f}, agreement score {s_ens:.2f}) - "
            f"ambiguous product boundary."
        )
    if topo_detail.get("component_penalty", 0) > 0.2:
        reasons.append(f"Mask is split into {topo_detail.get('components')} separate pieces.")
    elif topo_detail.get("multi_instance"):
        reasons.append(
            f"Photo contains {topo_detail.get('components')} products of comparable size - "
            f"read as a multi-item composition, not a fragmented mask."
        )
    if topo_detail.get("hole_penalty", 0) > 0.2:
        reasons.append(f"{refine_info.get('holes_kept', 0)} interior holes - unexpected for this product type.")
    if s_contrast < 0.5:
        reasons.append(
            f"{contrast_detail.get('flat_boundary_fraction', 0)*100:.0f}% of the outline has no "
            f"colour step across it - the mask probably cuts through an object instead of "
            f"following the product's edge."
        )
    if s_bg < 0.55:
        reasons.append(
            f"Only {bg_detail.get('outside_is_background', 0)*100:.0f}% of the pixels just outside "
            f"the mask look like background - the mask may be a sub-region of a larger object "
            f"(e.g. the print inside a frame) rather than the whole product."
        )
    if s_shape < 0.6:
        reasons.append(
            f"Silhouette is atypical for {category} (bbox fill {shape_detail.get('bbox_fill')}, "
            f"aspect {shape_detail.get('aspect')}) - the mask may include a neighbouring object."
        )
    if s_cov < 0.7:
        lo, hi = cfg["expect_coverage"]
        reasons.append(f"Product occupies {coverage*100:.1f}% of the frame, outside the expected "
                       f"{lo*100:.0f}-{hi*100:.0f}% for {category}.")
    if border_contact > 0.06:
        reasons.append(f"Product is cropped by the frame on {border_contact*100:.0f}% of the border.")
    if complexity > 0.4:
        reasons.append("Highly complex outline (hair / fringe / fine detail) - human spot-check recommended.")
    if refine_info.get("shadow_suppressed"):
        reasons.append("A cast shadow was detected and removed from the mask.")

    # ---- verdict -----------------------------------------------------------
    # Hard gates. These specific defects make a mask unusable as a mockup base
    # however good the aggregate score looks, so they veto READY outright and
    # send the image to a human instead.
    veto = [k for k in ("split into", "cropped by the frame", "sub-region of a larger object",
                        "atypical for", "cuts through an object", "Cross-check model disagrees")
            if any(k in r for r in reasons)]
    if veto:
        detail["ready_veto"] = veto
    if confidence >= settings.ready_threshold and not veto:
        verdict = VERDICT_READY
        if not reasons:
            reasons.append("Clean single-piece mask, edges align with the source image, models agree.")
    elif confidence >= settings.review_threshold:
        verdict = VERDICT_REVIEW
    else:
        verdict = VERDICT_FAILED
        reasons.insert(0, "Confidence below the acceptance floor - refusing to auto-publish this mask.")

    metrics = MaskMetrics(
        coverage=round(coverage, 5),
        edge_sharpness=round(s_sharp, 4),
        uncertain_ratio=round(uncertain_ratio, 5),
        boundary_complexity=round(complexity, 4),
        component_penalty=round(topo_detail.get("component_penalty", 0.0), 4),
        border_contact=round(border_contact, 4),
        ensemble_iou=round(ens_iou, 4) if ens_iou is not None else None,
        holes=int(refine_info.get("holes_kept", 0)),
        solidity=round(topo_detail.get("solidity", 0.0), 4),
    )
    return verdict, round(confidence, 4), metrics, reasons, detail
