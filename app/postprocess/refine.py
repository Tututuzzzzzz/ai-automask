"""Alpha-matte refinement.

The neural matte is good but is produced at a fixed working resolution, so its
boundary is inherently soft/aliased once upscaled to a 4000 px product photo.
This module recovers that lost boundary detail using the *source image* as a
guide, then cleans up topology (specks, spurious holes, cast shadows).

Pipeline order matters and is deliberate:

    raw alpha
      -> topology cleanup      (drop specks, keep legitimate holes)
      -> cast-shadow suppression
      -> trimap band extraction
      -> guided-filter matting inside the band only   <- edge detail comes from here
      -> category-aware contrast curve                (crisp for canvas, soft for hair)
      -> final clamp

Everything runs at full source resolution: the exported mask is never resized.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import CATEGORIES, settings


# ------------------------------------------------------------------ guided filter
def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He et al. guided filter, single-channel guide, box-filter implementation.

    Implemented locally rather than calling cv2.ximgproc.guidedFilter so the
    service only needs plain opencv-python (no contrib build) - one less
    deployment surprise inside a customer's Docker image.
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    ksize = (radius * 2 + 1, radius * 2 + 1)

    mean_i = cv2.boxFilter(guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize)
    corr_i = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize)
    corr_ip = cv2.boxFilter(guide * src, cv2.CV_32F, ksize)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
    return mean_a * guide + mean_b


# --------------------------------------------------------------- topology cleanup
def keep_significant_components(binary: np.ndarray, min_ratio: float = 0.04) -> np.ndarray:
    """Drop blobs far smaller than the main product.

    Expressed as a ratio of the largest component rather than an absolute pixel
    count, so it behaves identically on an 800 px thumbnail and a 5000 px hero
    shot. Multi-blob products are real (a pair of shoes, a mug shot beside its
    lid), so anything >= 4% of the main blob survives.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(areas.max())
    keep = [i + 1 for i, a in enumerate(areas) if a >= max(24, biggest * min_ratio)]
    out = np.zeros_like(binary)
    for lbl in keep:
        out[labels == lbl] = 1
    return out


def fill_small_holes(binary: np.ndarray, max_hole_ratio: float = 0.004) -> tuple[np.ndarray, int]:
    """Fill pinhole artefacts, keep structural holes (mug handle, arm gaps).

    Returns the cleaned mask plus the number of holes deliberately retained.
    That count feeds the QC score: "0 holes on a mug" is suspicious, "40 holes
    on a flat canvas" means the matte is shredded.
    """
    inv = (1 - binary).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    total = binary.size
    out = binary.copy()
    kept = 0
    border = set(labels[0].tolist()) | set(labels[-1].tolist())
    border |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    for i in range(1, n):
        if i in border:            # touches the frame -> genuine background
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area / total < max_hole_ratio:
            out[labels == i] = 1
        else:
            kept += 1
    return out, kept


# ------------------------------------------------------------- shadow suppression
def suppress_cast_shadow(image: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, bool]:
    """Remove the drop shadow that studio shots bake onto the sweep.

    Symptom in the wild: the matte extends a soft grey wedge below a mug because
    the model read the contact shadow as part of the object. That wedge is
    (a) low alpha, (b) desaturated, (c) darker than the backdrop but nowhere
    near as dark as a real product edge. We detect that signature specifically.
    """
    soft = ((alpha > 0.08) & (alpha < 0.75))
    if soft.sum() < 200:
        return alpha, False

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0

    if (alpha > 0.9).sum() < 200:
        return alpha, False
    bg = alpha < 0.02
    if bg.sum() < 200:
        return alpha, False
    bg_val = float(np.median(val[bg]))

    shadowish = soft & (sat < 0.18) & (val < bg_val - 0.02) & (val > bg_val - 0.42)
    if shadowish.sum() < 300:
        return alpha, False

    # Only strip shadow outside the dilated product body, otherwise genuine soft
    # edges (fabric fuzz, hair) get eaten.
    body = cv2.dilate((alpha > 0.6).astype(np.uint8), np.ones((9, 9), np.uint8), iterations=2).astype(bool)
    target = shadowish & ~body
    if target.sum() < 300:
        return alpha, False

    out = alpha.copy()
    out[target] = 0.0
    return out, True


# --------------------------------------------------------- thin-occluder removal
def suppress_thin_occluders(image: np.ndarray, alpha: np.ndarray,
                            strength: float = 1.0) -> tuple[np.ndarray, dict]:
    """Cut hair strands (and stray threads) out of a garment mask.

    The brief names this failure explicitly: "toc model vat ngang" - a model's
    hair falling across the shoulder. A saliency model returns the smooth
    garment silhouette *including* the pixels the hair covers, so the mockup
    composites the customer's artwork straight over the hair. Visually it is one
    of the most obvious giveaways of an automated base.

    Detection uses a morphological black-hat, which is the textbook operator for
    "thin dark structure on a lighter surface": it responds to features narrower
    than the structuring element and ignores broad shading, so fabric folds and
    seams - which are wide and low-contrast - do not trigger it.

    Guards, in order of how much trouble each one saves. A false positive here
    punches holes straight through a good mask, so every one of these was added
    after watching the detector misfire on a real photo:
      1. only inside the eroded mask body, never near the real boundary;
      2. the structure must be clearly darker than the *local* garment tone;
      3. it must be thin - max inscribed radius below a few pixels, so a dark
         panel or a wide seam survives;
      4. **per-component elongation and length.** This is the important one.
         Printed lettering, a logo, a buckle and a rivet are all thin and dark
         too - the first version of this function happily deleted the word
         "AIRTOUCH" off a tote bag. Hair is distinguished by being *long and
         stringy*: a strand runs a good fraction of the product's height at a
         couple of pixels wide. Letters are short. So each connected component
         is measured and only long, high-aspect ones are removed;
      5. a total budget - if the detection wants more than 8% of the mask, it has
         found a pattern, not hair, and the whole operation is abandoned;
      6. the product must be locally *thick* where the structure lies. A dark
         line running down the middle of a tote strap is a crease in the strap;
         removing it slits the strap in half. Hair, by contrast, lies across a
         wide panel.
    """
    info: dict = {"applied": False}
    h, w = alpha.shape[:2]
    body = (alpha > 0.6).astype(np.uint8)
    if body.sum() < 2000:
        return alpha, info

    # Work at a bounded resolution: strand width scales with the photo, and the
    # kernel sizes below assume roughly 1200 px on the long edge.
    scale = min(1.0, 1400.0 / max(h, w))
    if scale < 1.0:
        small_img = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_body = cv2.resize(body, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    else:
        small_img, small_body = image, body

    sh, sw = small_body.shape[:2]
    lab_l = cv2.cvtColor(small_img, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) / 255.0

    strand_px = max(3, int(0.010 * max(sh, sw)))          # widest strand we accept
    k = int(strand_px | 1)
    blackhat = cv2.morphologyEx(lab_l, cv2.MORPH_BLACKHAT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    interior = cv2.erode(small_body, np.ones((k * 3 | 1,) * 2, np.uint8))
    if interior.sum() < 500:
        return alpha, info

    # Local garment tone, estimated only from mask pixels.
    weight = small_body.astype(np.float32)
    sigma = max(6.0, 0.05 * max(sh, sw))
    num = cv2.GaussianBlur(lab_l * weight, (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    local = num / np.maximum(den, 1e-4)
    contrast = local - lab_l                              # positive where darker than tone

    inside = interior > 0
    resp = blackhat[inside]
    if resp.size < 200:
        return alpha, info
    # Robust threshold: strands are the extreme tail of the black-hat response.
    thr = float(np.percentile(resp, 99.0))
    if thr < 0.045:                                       # nothing thin and dark here
        info["blackhat_p99"] = round(thr, 4)
        return alpha, info

    detect = (blackhat > max(0.045, thr * 0.55)) & inside & (contrast > 0.12)
    detect = cv2.morphologyEx(detect.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if detect.sum() < 0.0004 * sh * sw:                   # too little to be hair
        info["blackhat_p99"] = round(thr, 4)
        info["detected_px"] = int(detect.sum())
        return alpha, info

    # Thinness gate: a genuine strand field has a small maximum inscribed radius.
    dt = cv2.distanceTransform(detect, cv2.DIST_L2, 3)
    max_radius = float(dt.max())
    if max_radius > strand_px * 1.6:
        info["rejected"] = f"structure too thick ({max_radius:.1f}px > {strand_px * 1.6:.1f}px)"
        return alpha, info

    # Per-component shape gate - see guards 4 and 6 in the docstring.
    min_length = 0.12 * max(sh, sw)
    # How "thick" the product is at each pixel. Used to refuse to carve a strand
    # out of a narrow feature: a 2 px crease running down the middle of a 15 px
    # tote strap is a fold in the strap, not hair lying across it.
    body_thickness = cv2.distanceTransform(small_body, cv2.DIST_L2, 3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(detect, connectivity=8)
    keep = np.zeros_like(detect)
    kept, rejected = 0, 0
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 12:
            rejected += 1
            continue
        # Half the contour perimeter approximates the length of a thin ribbon;
        # dividing by its half-width gives elongation.
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            rejected += 1
            continue
        length = 0.5 * float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
        radius = float(cv2.distanceTransform(comp, cv2.DIST_L2, 3).max())
        elongation = length / max(1.0, 2.0 * radius)
        # Diagonal extent, so a component that merely wanders in a tight cluster
        # (a letter, a logo) cannot pass on perimeter alone.
        extent = float(np.hypot(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]))
        local_thickness = float(np.median(body_thickness[comp > 0])) if comp.any() else 0.0
        # A strand must be thin *relative to the feature it lies on*: measured on
        # real photos, a crease inside a tote strap gives a ratio near 4, while
        # hair across a chest panel gives 40+.
        wide_enough = local_thickness >= max(12.0, 8.0 * max(1.0, radius))
        if (elongation >= 6.0 and length >= min_length and extent >= min_length * 0.8
                and wide_enough):
            keep |= comp
            kept += 1
        else:
            rejected += 1
    info["components_kept"] = kept
    info["components_rejected"] = rejected
    if kept == 0:
        info["rejected"] = ("no component is long and stringy enough to be hair "
                            f"({rejected} rejected as print/hardware/seams)")
        return alpha, info
    detect = keep

    # Total budget: hair crossing a garment is a few percent of the mask. More
    # than that and we are looking at a printed pattern or a shadow field.
    share = float(detect.sum()) / max(1, int(small_body.sum()))
    if share > 0.08:
        info["rejected"] = f"detection covers {share*100:.1f}% of the mask - too much to be hair"
        return alpha, info
    info["detected_share_of_mask"] = round(share, 4)

    # Grow slightly to catch the anti-aliased halo around each strand.
    detect = cv2.dilate(detect, np.ones((3, 3), np.uint8))
    soft = cv2.GaussianBlur(detect.astype(np.float32), (0, 0), 1.0)
    soft = np.clip(soft * strength, 0.0, 1.0)
    if scale < 1.0:
        soft = cv2.resize(soft, (w, h), interpolation=cv2.INTER_LINEAR)

    out = alpha * (1.0 - soft)
    info.update({
        "applied": True,
        "blackhat_p99": round(thr, 4),
        "strand_kernel_px": k,
        "max_radius_px": round(max_radius, 2),
        "removed_px": int((soft > 0.5).sum()),
        "removed_ratio_of_mask": round(float((soft > 0.5).sum()) / max(1, int(body.sum())), 5),
    })
    return out.astype(np.float32), info


# -------------------------------------------------------------------- main entry
def refine_alpha(image: np.ndarray, alpha: np.ndarray, category: str = "auto") -> tuple[np.ndarray, dict]:
    """Refine a raw matte into a production-grade alpha channel.

    Returns (alpha_float32, info); *info* carries diagnostics for the QC stage
    and the API response.
    """
    cfg = CATEGORIES.get(category, CATEGORIES["auto"])
    info: dict = {"category": category, "shadow_suppressed": False, "holes_kept": 0, "guided": False}

    h, w = image.shape[:2]
    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    # ---- 1. topology, decided on a binary view ----------------------------
    binary = (alpha > 0.5).astype(np.uint8)
    if int(binary.sum()) == 0:
        info["empty"] = True
        return alpha, info

    binary = keep_significant_components(binary)
    binary, holes = fill_small_holes(binary, max_hole_ratio=0.004 if cfg["expect_holes"] else 0.05)
    info["holes_kept"] = holes

    # Re-impose those decisions on the soft matte without flattening its
    # gradient: only touch pixels the cleanup actually changed.
    removed = (alpha > 0.5) & (binary == 0)
    if removed.any():
        killed = cv2.dilate(removed.astype(np.uint8), np.ones((3, 3), np.uint8))
        alpha = np.where(killed > 0, 0.0, alpha)
    added = (alpha <= 0.5) & (binary == 1)
    if added.any():
        alpha = np.where(added, 1.0, alpha)

    # ---- 2. cast shadow ---------------------------------------------------
    alpha, did = suppress_cast_shadow(image, alpha)
    info["shadow_suppressed"] = did

    # ---- 2b. thin occluders (hair over a garment) -------------------------
    if cfg.get("strand_occluders") and settings.suppress_strands:
        alpha, strand_info = suppress_thin_occluders(image, alpha)
        info["strands"] = strand_info

    # ---- 3. guided-filter matting, restricted to the trimap band ---------
    if settings.refine_edges:
        # Band width scales with resolution: 9 px on a 1200 px image becomes
        # 30 px on a 4000 px one, which is exactly where the matte is soft.
        scale = max(1.0, max(h, w) / 1200.0)
        k = int(max(3, round(settings.trimap_band * scale)) | 1)
        kernel = np.ones((k, k), np.uint8)
        fg = cv2.erode((alpha > 0.92).astype(np.uint8), kernel)
        bg = cv2.erode((alpha < 0.08).astype(np.uint8), kernel)
        unknown = (fg == 0) & (bg == 0)
        info["unknown_band_px"] = int(unknown.sum())

        if unknown.sum() > 32:
            guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            radius = int(max(2, round(settings.guided_radius * scale)))
            filtered = np.clip(guided_filter(guide, alpha, radius=radius, eps=settings.guided_eps), 0.0, 1.0)
            alpha = np.where(unknown, filtered, alpha)
            info["guided"] = True
            info["guided_radius"] = radius

    # ---- 4. category-aware contrast curve --------------------------------
    # Hard-edged goods (canvas, phone case) must not ship a 20 px gradient;
    # soft goods (hair over a shoulder, knit fringe) must keep theirs.
    gamma = 1.35 if cfg["soft_edges"] else 2.2
    alpha = np.clip((alpha - 0.5) * gamma + 0.5, 0.0, 1.0)

    # Kill residual haze that would read as a grey halo in the composite.
    alpha[alpha < 0.02] = 0.0
    alpha[alpha > 0.985] = 1.0

    return alpha.astype(np.float32), info


def alpha_to_uint8(alpha: np.ndarray) -> np.ndarray:
    return np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)


def make_trimap(alpha: np.ndarray, band: int = 12) -> np.ndarray:
    """Export the trimap so designers see exactly where the AI was unsure."""
    k = int(max(3, band) | 1)
    kernel = np.ones((k, k), np.uint8)
    fg = cv2.erode((alpha > 0.92).astype(np.uint8), kernel)
    bg = cv2.erode((alpha < 0.08).astype(np.uint8), kernel)
    tri = np.full(alpha.shape, 128, dtype=np.uint8)
    tri[fg > 0] = 255
    tri[bg > 0] = 0
    return tri
