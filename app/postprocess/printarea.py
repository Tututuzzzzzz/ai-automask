"""Print-area detection - where the customer's design is allowed to land.

The brief asks for two things from the vision stage: the alpha mask *and* the
Print Area. They are different problems. The mask is "which pixels are the
product"; the print area is "which pixels are the flat, printable, unobstructed
part of the product" - the chest panel of a tee, the front wall of a mug, the
face of a canvas.

Approach per category, all derived from the mask + image, no manual coordinates:

* ``quad``     (canvas, poster, phone case, mousepad)
  The product *is* the printable surface. Fit a 4-corner polygon to the contour
  so perspective is preserved, then inset for bleed.

* ``torso``    (t-shirt, hoodie, tote)
  Restrict to a torso prior (central columns, upper-middle rows), subtract skin
  and hair pixels so a model's arm or hair crossing the chest is excluded, then
  solve for the largest axis-aligned rectangle that fits entirely inside what
  is left. "Largest inscribed rectangle" is exact here (classic maximal-
  rectangle DP), which is why the box never overhangs a sleeve.

* ``cylinder`` (mug, tumbler, bottle)
  Detect and remove the handle side using the interior hole, keep the straight
  body, then inscribe a rectangle and pull the vertical edges in to respect the
  wrap-around falloff of a cylindrical surface.

Output is a quad (TL, TR, BR, BL) plus the 3x3 homography that maps a unit
design canvas onto it - which is precisely what libvips / ImageMagick /
node-canvas need to composite the artwork. No mockup-engine changes required.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import CATEGORIES
from ..schemas import PrintArea

_WORK = 480  # print-area geometry is solved on a downscaled grid, then scaled up


# ------------------------------------------------------------------- utilities
def _downscale(mask: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = mask.shape[:2]
    scale = min(1.0, _WORK / float(max(h, w)))
    if scale >= 1.0:
        return mask.copy(), 1.0
    small = cv2.resize(mask, (max(8, int(w * scale)), max(8, int(h * scale))), interpolation=cv2.INTER_NEAREST)
    return small, scale


def largest_inscribed_rect(binary: np.ndarray) -> tuple[int, int, int, int] | None:
    """Maximal all-ones axis-aligned rectangle (x, y, w, h).

    Classic histogram / largest-rectangle-in-histogram DP, O(H*W). Exact, which
    matters: an approximate answer here means a print area that overhangs the
    garment and a mockup with artwork floating in mid-air.
    """
    if binary.size == 0 or binary.max() == 0:
        return None
    h, w = binary.shape
    heights = np.zeros(w, dtype=np.int32)
    best = (0, 0, 0, 0)   # area, x, y, w, h packed below
    best_area = 0
    for y in range(h):
        row = binary[y]
        heights = np.where(row > 0, heights + 1, 0)
        # largest rectangle in histogram, stack based
        stack: list[int] = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            while stack and heights[stack[-1]] >= cur:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                height = int(heights[top])
                width = x - left
                area = height * width
                if area > best_area:
                    best_area = area
                    best = (left, y - height + 1, width, height)
            stack.append(x)
    if best_area <= 0:
        return None
    x, y, rw, rh = best
    return int(x), int(y), int(rw), int(rh)


def _skin_hair_mask(image: np.ndarray) -> np.ndarray:
    """Rough skin + dark-hair detector.

    Only used to *exclude* regions from the print area, never from the alpha
    mask, so a false positive costs a slightly smaller print box - never a
    broken cut-out. Skin in YCrCb is a famously tight, illumination-robust
    cluster; hair is caught as very-low-value, low-saturation pixels.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    skin = ((cr > 133) & (cr < 180) & (cb > 77) & (cb < 130)).astype(np.uint8)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hair = ((hsv[..., 2] < 62) & (hsv[..., 1] < 110)).astype(np.uint8)

    out = cv2.morphologyEx(np.maximum(skin, hair), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    pts = pts[order]
    # angle sort starts somewhere on the circle; rotate so TL (smallest x+y) leads
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def _rect_to_quad(x: float, y: float, w: float, h: float) -> np.ndarray:
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def _homography(quad: np.ndarray) -> list[list[float]] | None:
    """3x3 matrix mapping the unit design canvas [0,1]^2 onto *quad*."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    try:
        m = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
        return [[float(v) for v in row] for row in m]
    except cv2.error:
        return None


# ---------------------------------------------------------------- per-category
def _quad_area(binary: np.ndarray, inset: float = 0.035) -> tuple[np.ndarray | None, float]:
    """Fit a 4-corner polygon to the product silhouette (canvas / poster / case)."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    main = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(main)
    if area < 32:
        return None, 0.0

    quad = None
    peri = cv2.arcLength(main, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(main, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    fit = 1.0
    if quad is None:                      # rounded corners (phone case) -> min-area rect
        rect = cv2.minAreaRect(main)
        quad = cv2.boxPoints(rect).astype(np.float32)
        fit = 0.8
    quad = _order_quad(quad)

    # How much of the silhouette does the quad actually explain? Low values mean
    # the product is not quad-shaped and the caller should not trust this box.
    hull = np.zeros_like(binary)
    cv2.fillConvexPoly(hull, quad.astype(np.int32), 1)
    inter = float(np.logical_and(hull > 0, binary > 0).sum())
    union = float(np.logical_or(hull > 0, binary > 0).sum()) or 1.0
    fit *= inter / union

    center = quad.mean(axis=0)
    quad = center + (quad - center) * (1.0 - inset)
    return quad, float(np.clip(fit, 0.0, 1.0))


def _torso_area(binary: np.ndarray, image_small: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Largest safe rectangle on the chest panel of a garment."""
    h, w = binary.shape
    ys, xs = np.nonzero(binary)
    if len(ys) < 64:
        return None, 0.0
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    gh, gw = y1 - y0 + 1, x1 - x0 + 1

    # Torso prior: skip the collar band and the hem, stay off the sleeve columns.
    prior = np.zeros_like(binary)
    py0 = int(y0 + 0.16 * gh)
    py1 = int(y0 + 0.78 * gh)
    # Column centre estimated from the widest run per row in the mid-body rows,
    # which is robust to one arm being raised.
    mid = binary[int(y0 + 0.35 * gh):int(y0 + 0.65 * gh)]
    centres = []
    widths = []
    for row in mid:
        idx = np.nonzero(row)[0]
        if len(idx) < 4:
            continue
        centres.append(0.5 * (idx.min() + idx.max()))
        widths.append(idx.max() - idx.min())
    cx = float(np.median(centres)) if centres else 0.5 * (x0 + x1)
    body_w = float(np.median(widths)) if widths else gw
    half = 0.34 * body_w
    prior[py0:py1, int(max(0, cx - half)):int(min(w, cx + half))] = 1

    candidate = ((binary > 0) & (prior > 0)).astype(np.uint8)
    skin = _skin_hair_mask(image_small)
    # Dilate the exclusion so the print box keeps clear of an arm, not just
    # touches it.
    skin = cv2.dilate(skin, np.ones((9, 9), np.uint8))

    # Sanity check before trusting the exclusion. The "hair" half of the detector
    # keys on dark, desaturated pixels - which is a perfect description of a
    # BLACK T-SHIRT. Applied naively it deletes the entire chest panel of every
    # dark garment. If the exclusion claims most of the torso, the garment itself
    # is dark and the detector carries no information here, so drop it.
    inside = candidate > 0
    excluded_share = float((skin[inside] > 0).mean()) if inside.any() else 0.0
    if excluded_share <= 0.35:
        candidate = (candidate & (skin == 0)).astype(np.uint8)
    # Erode once so the rectangle never sits exactly on the alpha boundary.
    candidate = cv2.erode(candidate, np.ones((5, 5), np.uint8))

    rect = largest_inscribed_rect(candidate)
    if rect is None:
        return None, 0.0
    x, y, rw, rh = rect
    if rw < 8 or rh < 8:
        return None, 0.0

    # Real POD print areas are portrait-ish (e.g. 12x16 in). If the inscribed
    # box is very wide, trim it to a plausible print ratio around its centre.
    max_ratio = 1.05
    if rw > rh * max_ratio:
        new_w = rh * max_ratio
        x += (rw - new_w) / 2.0
        rw = new_w

    quad = _rect_to_quad(x, y, rw, rh)
    # Confidence: how much of the torso prior we managed to fill.
    prior_area = float((prior > 0).sum()) or 1.0
    conf = float(np.clip((rw * rh) / prior_area, 0.0, 1.0)) * 0.6 + 0.4
    return quad, conf


def _cylinder_area(binary: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Front wall of a mug / tumbler, handle excluded."""
    h, w = binary.shape
    ys, xs = np.nonzero(binary)
    if len(ys) < 64:
        return None, 0.0
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()

    # The handle shows up as an interior hole (its aperture) or as a bump on one
    # side. Column occupancy over the vertical middle isolates the solid body:
    band = binary[int(y0 + 0.25 * (y1 - y0)):int(y0 + 0.75 * (y1 - y0)) + 1]
    if band.size == 0:
        return None, 0.0
    occ = band.mean(axis=0)
    body_cols = np.nonzero(occ > 0.92)[0]     # columns filled top-to-bottom = body
    if len(body_cols) < 8:
        body_cols = np.nonzero(occ > 0.7)[0]
    if len(body_cols) < 8:
        bx0, bx1 = x0, x1
    else:
        # Take the widest contiguous run: the handle is separated from the body
        # by the low-occupancy gap of its aperture.
        splits = np.split(body_cols, np.nonzero(np.diff(body_cols) > 2)[0] + 1)
        run = max(splits, key=len)
        bx0, bx1 = int(run[0]), int(run[-1])

    body = np.zeros_like(binary)
    body[y0:y1 + 1, bx0:bx1 + 1] = binary[y0:y1 + 1, bx0:bx1 + 1]
    body = cv2.erode(body, np.ones((5, 5), np.uint8))
    rect = largest_inscribed_rect(body)
    if rect is None:
        return None, 0.0
    x, y, rw, rh = rect

    # A cylinder's printable wrap is narrower than its silhouette: the extreme
    # left/right of the silhouette curves away from the camera, so pull in ~14%
    # per side, and keep clear of the rim and base.
    x += 0.14 * rw
    rw *= 0.72
    y += 0.14 * rh
    rh *= 0.70
    if rw < 8 or rh < 8:
        return None, 0.0
    quad = _rect_to_quad(x, y, rw, rh)
    body_area = float((body > 0).sum()) or 1.0
    conf = float(np.clip((rw * rh) / body_area, 0.0, 1.0)) * 0.5 + 0.45
    return quad, conf


# -------------------------------------------------------------------- entry pt
def detect_print_area(image: np.ndarray, alpha: np.ndarray, category: str = "auto") -> PrintArea:
    """Compute the print area for a refined alpha matte."""
    cfg = CATEGORIES.get(category, CATEGORIES["auto"])
    kind = cfg["print_area"]
    if kind == "auto":
        kind = "quad"

    binary_full = (alpha > 0.5).astype(np.uint8)
    if binary_full.sum() < 64:
        return PrintArea(kind="none")

    small, scale = _downscale(binary_full)
    inv = 1.0 / scale if scale else 1.0
    image_small = image
    if scale < 1.0:
        image_small = cv2.resize(image, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)

    if kind == "torso":
        quad, conf = _torso_area(small, image_small)
        if quad is None:                        # e.g. a flat-lay tee prior missed
            quad, conf = _quad_area(small, inset=0.22)
            kind = "quad"
    elif kind == "cylinder":
        quad, conf = _cylinder_area(small)
        if quad is None:
            quad, conf = _quad_area(small, inset=0.2)
            kind = "quad"
    else:
        quad, conf = _quad_area(small)

    if quad is None:
        return PrintArea(kind="none")

    quad = quad.astype(np.float32) * inv
    h, w = alpha.shape[:2]
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    quad = _order_quad(quad)

    xs, ys = quad[:, 0], quad[:, 1]
    bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min()), float(ys.max() - ys.min())]
    poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(poly, quad.astype(np.int32), 1)
    area_px = int(poly.sum())
    product_px = int(binary_full.sum()) or 1

    # Sanity: the print area must actually be inside the product.
    inside = float(np.logical_and(poly > 0, binary_full > 0).sum()) / max(area_px, 1)
    conf *= inside

    return PrintArea(
        kind=kind,
        quad=[[float(p[0]), float(p[1])] for p in quad],
        bbox=bbox,
        area_px=area_px,
        coverage_of_product=round(area_px / product_px, 4),
        perspective_matrix=_homography(quad),
        confidence=round(float(np.clip(conf, 0.0, 1.0)), 4),
    )
