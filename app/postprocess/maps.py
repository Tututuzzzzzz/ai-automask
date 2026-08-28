"""Lighting decomposition: shadow, highlight and displacement maps (bonus track).

A mask alone gives you a *flat* mockup: the design sits on the garment like a
sticker. What sells a POD preview is that the artwork picks up the same folds
and lighting as the blank. That needs three extra layers, and all three can be
derived from the same photo we already segmented - no extra capture, no manual
Photoshop passes.

Model used (intrinsic-image style, deliberately simple and stable):

    L(x)  =  albedo(x)  *  shading(x)

We estimate ``albedo`` as the *large-scale* luminance of the product - an
alpha-weighted low-pass, so the backdrop never bleeds into the estimate - and
call whatever is left ``shading``:

    shading = L / albedo
    shadow    = clamp(1 - shading)      -> multiply layer
    highlight = clamp(shading - 1)      -> screen / linear-dodge layer

The displacement map is the *mid-frequency* residual (bilateral-filtered so
sensor noise does not become fake fabric texture), remapped around mid-grey.
That is exactly the convention a displacement/mesh-warp node expects: 128 = no
shift, darker = push in, brighter = pull out.

Compositing recipe for the mockup engine (libvips / ImageMagick / canvas):

    1. warp the design onto the print-area quad  (homography from printarea.py)
    2. displace it with displacement_map         (fold geometry)
    3. multiply by shadow_map                    (creases go dark)
    4. screen with highlight_map                 (sheen comes back)
    5. mask with alpha_mask                      (nothing leaks off the product)
"""
from __future__ import annotations

import cv2
import numpy as np


def _alpha_weighted_lowpass(value: np.ndarray, alpha: np.ndarray, sigma: float) -> np.ndarray:
    """Low-pass ``value`` using only pixels the mask says belong to the product.

    Plain Gaussian blur would drag the white studio sweep into the albedo
    estimate along the product edge, producing a bright halo in every derived
    map. Normalising by the blurred alpha removes that entirely.
    """
    w = alpha.astype(np.float32)
    num = cv2.GaussianBlur(value * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)


def _luminance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    return lab[..., 0].astype(np.float32) / 255.0


def shadow_highlight_maps(
    image: np.ndarray,
    alpha: np.ndarray,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (shadow_u8, highlight_u8, info). White = strong effect in both."""
    h, w = image.shape[:2]
    lum = _luminance(image)
    a = np.clip(alpha, 0.0, 1.0)

    if a.sum() < 64:
        zero = np.zeros((h, w), dtype=np.uint8)
        return zero, zero, {"note": "empty mask"}

    # Albedo scale: ~8% of the long edge captures garment-level shading while
    # leaving folds and creases in the residual.
    sigma = max(6.0, 0.08 * max(h, w))
    albedo = _alpha_weighted_lowpass(lum, a, sigma)
    albedo = np.maximum(albedo, 0.02)

    shading = lum / albedo
    shading = cv2.bilateralFilter(shading.astype(np.float32), 0, 0.08, max(3.0, 0.004 * max(h, w)))

    shadow = np.clip((1.0 - shading) * strength, 0.0, 1.0)
    highlight = np.clip((shading - 1.0) * strength, 0.0, 1.0)

    # Normalise against the actual dynamic range of this photo, so a soft-lit
    # mug and a hard-lit hoodie both produce usable layers instead of one being
    # nearly black. Robust percentiles keep specular pinpoints from flattening
    # everything else.
    body = a > 0.5
    for arr in (shadow, highlight):
        vals = arr[body]
        if vals.size:
            p = float(np.percentile(vals, 99.0))
            if p > 1e-3:
                np.divide(arr, p, out=arr)
    np.clip(shadow, 0.0, 1.0, out=shadow)
    np.clip(highlight, 0.0, 1.0, out=highlight)

    shadow *= a
    highlight *= a

    info = {
        "albedo_sigma_px": round(sigma, 1),
        "shadow_mean": round(float(shadow[body].mean()) if body.any() else 0.0, 4),
        "highlight_mean": round(float(highlight[body].mean()) if body.any() else 0.0, 4),
    }
    to_u8 = lambda x: np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)  # noqa: E731
    return to_u8(shadow), to_u8(highlight), info


def displacement_map(image: np.ndarray, alpha: np.ndarray, gain: float = 1.0) -> tuple[np.ndarray, dict]:
    """Mid-frequency fold geometry, encoded around mid-grey (128 = no shift)."""
    h, w = image.shape[:2]
    lum = _luminance(image)
    a = np.clip(alpha, 0.0, 1.0)
    if a.sum() < 64:
        return np.full((h, w), 128, dtype=np.uint8), {"note": "empty mask"}

    # Denoise first: JPEG blocking and sensor noise would otherwise be baked in
    # as fabric texture that does not exist.
    d = max(3, int(0.0025 * max(h, w)) | 1)
    smooth = cv2.bilateralFilter(lum, 0, 0.06, d)

    sigma = max(4.0, 0.03 * max(h, w))
    base = _alpha_weighted_lowpass(smooth, a, sigma)
    residual = smooth - base

    body = a > 0.5
    vals = residual[body]
    if vals.size:
        scale = float(np.percentile(np.abs(vals), 98.0))
    else:
        scale = 0.05
    scale = max(scale, 1e-3)
    disp = np.clip(residual / scale * gain, -1.0, 1.0) * a

    out = np.clip(disp * 127.0 + 128.0 + 0.5, 0, 255).astype(np.uint8)
    out[~body] = 128            # neutral outside the product
    info = {
        "residual_p98": round(scale, 5),
        "detail_sigma_px": round(sigma, 1),
        "amplitude": round(float(np.abs(disp[body]).mean()) if body.any() else 0.0, 4),
    }
    return out, info


def build_cutout(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """RGBA cut-out with premultiplication-safe edge colour.

    Straight (non-premultiplied) RGBA is what PNG stores, but the RGB under a
    low-alpha edge pixel is still backdrop-coloured, which shows up as a white
    fringe when a compositor filters the image. Bleeding the product colour
    outward before writing kills that fringe.
    """
    a = np.clip(alpha, 0.0, 1.0)
    rgb = image.astype(np.float32)
    core = (a > 0.85).astype(np.uint8)
    if core.sum() > 64:
        # Nearest-neighbour colour bleed via inpainting on the thin edge band.
        band = cv2.dilate(core, np.ones((5, 5), np.uint8), iterations=3) - core
        need = ((a > 0.01) & (a < 0.85) & (band > 0)).astype(np.uint8)
        if need.sum() > 16:
            filled = cv2.inpaint(image, need, 3, cv2.INPAINT_TELEA)
            rgb = np.where(need[..., None] > 0, filled.astype(np.float32), rgb)
    out = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)])
    return out


def build_overlay(
    image: np.ndarray,
    alpha: np.ndarray,
    verdict: str = "READY",
    print_area: list[list[float]] | None = None,
) -> np.ndarray:
    """Review overlay: checkerboard background, tinted product, traced outline.

    Designed for the *reviewer*, not for the report aesthetic: the checkerboard
    makes leftover semi-transparent haze obvious, the outline makes a 2 px
    clipping error visible at a glance, and the tint colour encodes the verdict.
    """
    h, w = image.shape[:2]
    a = np.clip(alpha, 0.0, 1.0)[..., None]

    tile = max(8, int(0.012 * max(h, w)))
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((yy // tile) + (xx // tile)) % 2).astype(np.float32)
    board = (200.0 + 40.0 * checker)[..., None].repeat(3, axis=2)

    tint = {
        "READY": np.array([46, 204, 113], dtype=np.float32),
        "REVIEW": np.array([243, 156, 18], dtype=np.float32),
        "FAILED": np.array([231, 76, 60], dtype=np.float32),
    }.get(verdict, np.array([52, 152, 219], dtype=np.float32))

    product = image.astype(np.float32) * 0.82 + tint * 0.18
    comp = product * a + board * (1.0 - a)
    comp = np.clip(comp, 0, 255).astype(np.uint8)

    contour = cv2.morphologyEx((alpha > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT,
                               np.ones((3, 3), np.uint8))
    thickness = max(1, int(0.0018 * max(h, w)))
    if thickness > 1:
        contour = cv2.dilate(contour, np.ones((thickness, thickness), np.uint8))
    comp[contour > 0] = tint.astype(np.uint8)

    if print_area:
        pts = np.array(print_area, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(comp, [pts], True, (30, 90, 220), max(1, thickness), cv2.LINE_AA)
    return comp
