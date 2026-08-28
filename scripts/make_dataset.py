"""Procedural test-set generator - 18 diverse product bases *with ground truth*.

Why synthesise a set when real photos exist? Because the brief is scored on mask
accuracy, and you cannot measure accuracy without ground truth. Hand-labelling
even 18 real photos at pixel precision is hours of work and still subjective at
the fabric fringe. Rendering them means the alpha channel is *known exactly*, so
``scripts/eval_iou.py`` can report real IoU / boundary-F1 / MAE numbers instead
of vibes. The real-photo set (``scripts/fetch_dataset.py``) is kept alongside it
for qualitative checks.

The scenes are chosen to reproduce, deliberately, every failure mode named in
the brief and a few more that bite in production:

  * white-on-white (mug on a white sweep)      -> low-contrast boundary
  * a model's hair crossing the shoulder       -> occluder that must be cut out
  * fabric folds and creases                   -> soft shading, no real edge
  * mug handle                                 -> a legitimate interior hole
  * tote handles                               -> two interior holes
  * canvas in perspective                      -> quad print area, not a rect
  * bottle with a translucent shoulder         -> genuine partial alpha
  * cluttered wooden desk                      -> busy background
  * hard flash shadow                          -> shadow that is not product
  * heavy JPEG + sensor noise                  -> compression artefacts
  * product clipped by the frame               -> should be flagged, not shipped

Everything is seeded, so the set is byte-identical on every machine.

    python scripts/make_dataset.py                  # -> data/samples/
    python scripts/make_dataset.py --out /tmp/x     # elsewhere
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- helpers
def smooth_noise(h: int, w: int, rng: np.random.Generator, scale: float = 0.08,
                 octaves: int = 3) -> np.ndarray:
    """Fractal value noise in [0, 1] - stands in for fabric weave and wall texture."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        s = max(2, int(min(h, w) * scale / (2 ** o)))
        small = rng.random((max(2, h // s), max(2, w // s))).astype(np.float32)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * up
        total += amp
        amp *= 0.55
    out /= total
    return np.clip(out, 0.0, 1.0)


def sweep_background(h: int, w: int, rng: np.random.Generator, base: int = 242,
                     vignette: float = 0.10) -> np.ndarray:
    """Studio sweep: soft vertical gradient plus a gentle vignette."""
    grad = np.linspace(base, base - 26, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    grad *= (1.0 - vignette * np.clip(r, 0, 1.6) ** 2)
    grad += smooth_noise(h, w, rng, scale=0.5, octaves=2) * 4.0 - 2.0
    return np.clip(grad, 0, 255)[..., None].repeat(3, axis=2)


def wall_background(h: int, w: int, rng: np.random.Generator, tint=(228, 222, 212)) -> np.ndarray:
    """Painted / plastered wall: fine texture, a broad light falloff, subtle warmth."""
    tex = smooth_noise(h, w, rng, scale=0.03, octaves=4)
    lum = 0.86 + 0.14 * smooth_noise(h, w, rng, scale=0.9, octaves=2)
    img = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        img[..., c] = tint[c] * lum * (0.94 + 0.12 * tex)
    return np.clip(img, 0, 255)


def desk_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Wooden desk with clutter: the 'complex background' hidden-test case."""
    grain = smooth_noise(h, w, rng, scale=0.006, octaves=3)
    stripes = 0.5 + 0.5 * np.sin(np.linspace(0, 38, w, dtype=np.float32))[None, :].repeat(h, axis=0)
    wood = np.zeros((h, w, 3), np.float32)
    for c, base in enumerate((162, 118, 78)):
        wood[..., c] = base * (0.82 + 0.22 * grain) * (0.94 + 0.09 * stripes)

    # Clutter: notebooks, a coaster, a pen. Enough structure to give a naive
    # background-subtraction method a very bad day.
    for _ in range(4):
        x, y = int(rng.integers(0, max(1, w - 60))), int(rng.integers(0, max(1, h - 60)))
        bw, bh = int(rng.integers(240, 620)), int(rng.integers(200, 480))
        col = tuple(float(c) for c in rng.integers(40, 210, size=3))
        ang = float(rng.uniform(-25, 25))
        box = cv2.boxPoints(((x + bw / 2, y + bh / 2), (bw, bh), ang)).astype(np.int32)
        cv2.fillConvexPoly(wood, box, col, cv2.LINE_AA)
    wood = cv2.GaussianBlur(wood, (0, 0), 1.1)
    return np.clip(wood, 0, 255)


def mask_from_poly(h: int, w: int, pts: np.ndarray, ss: int = 4) -> np.ndarray:
    """Anti-aliased polygon alpha via supersampling - gives a believable 1-2 px ramp."""
    big = np.zeros((h * ss, w * ss), np.uint8)
    cv2.fillPoly(big, [np.round(pts * ss).astype(np.int32)], 255, cv2.LINE_AA)
    return (cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)


def mask_from_draw(h: int, w: int, draw, ss: int = 4) -> np.ndarray:
    """As above but for arbitrary cv2 drawing (ellipses, thick strokes)."""
    big = np.zeros((h * ss, w * ss), np.uint8)
    draw(big, ss)
    return (cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0)


def shade(alpha: np.ndarray, colour, rng: np.random.Generator, folds: float = 0.0,
          light=(-0.35, -0.6), ambient: float = 0.80, gloss: float = 0.0) -> np.ndarray:
    """Give a flat silhouette believable volume: lambert-ish falloff + fold noise."""
    h, w = alpha.shape
    base = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        base[..., c] = colour[c]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w / 2) / (w / 2)
    ny = (yy - h / 2) / (h / 2)
    lam = ambient + (1.0 - ambient) * np.clip(1.0 - (nx * light[0] + ny * light[1]), 0, 2) / 2.0

    if folds > 0:
        # Real fabric folds are *directional*: long vertical drapes with sharp
        # cross-sections. Isotropic noise looks like camouflage instead, so the
        # noise is stretched along one axis before it modulates the lighting.
        f = smooth_noise(h, w, rng, scale=0.10, octaves=3)
        f = cv2.GaussianBlur(f, (0, 0), sigmaX=max(1.0, w * 0.004), sigmaY=max(6.0, h * 0.05))
        f = (f - f.min()) / max(1e-6, f.max() - f.min())
        lam *= (1.0 - folds) + folds * 2.0 * f

    out = base * lam[..., None]
    if gloss > 0:
        # A specular streak: what makes ceramic and steel read as hard surfaces.
        streak = np.exp(-((nx - 0.35) ** 2) / 0.012) * np.exp(-(ny ** 2) / 1.4)
        out += 255.0 * gloss * streak[..., None]
    return np.clip(out, 0, 255)


def composite(canvas: np.ndarray, layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return canvas * (1.0 - a) + layer * a


def drop_shadow(canvas: np.ndarray, alpha: np.ndarray, offset=(18, 26), blur: float = 22.0,
                strength: float = 0.34) -> np.ndarray:
    """Contact shadow on the backdrop. The pipeline must NOT call this product."""
    h, w = alpha.shape
    m = np.zeros_like(alpha)
    dx, dy = offset
    ys, ye = max(0, dy), min(h, h + dy)
    xs, xe = max(0, dx), min(w, w + dx)
    m[ys:ye, xs:xe] = alpha[ys - dy:ye - dy, xs - dx:xe - dx]
    m = cv2.GaussianBlur(m, (0, 0), blur)
    return canvas * (1.0 - strength * m[..., None])


def hair_strands(h: int, w: int, rng: np.random.Generator, n: int, x0: int, y0: int,
                 span: int, drop: int) -> np.ndarray:
    """Wispy strands crossing the garment - the classic auto-masking killer."""
    def draw(big, ss):
        for _ in range(n):
            sx = x0 + int(rng.integers(-span // 2, span // 2))
            sy = y0 + int(rng.integers(-20, 20))
            ex = sx + int(rng.integers(-span // 3, span // 3))
            ey = sy + drop + int(rng.integers(-drop // 3, drop // 3))
            cx = (sx + ex) // 2 + int(rng.integers(-70, 70))
            cy = (sy + ey) // 2
            t = np.linspace(0, 1, 60)[:, None]
            p0 = np.array([[sx, sy]], np.float32)
            p1 = np.array([[cx, cy]], np.float32)
            p2 = np.array([[ex, ey]], np.float32)
            curve = ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2
            cv2.polylines(big, [np.round(curve * ss).astype(np.int32)], False, 255,
                          int(max(1, rng.integers(2, 5)) * ss), cv2.LINE_AA)
    return mask_from_draw(h, w, draw)


def degrade(img: np.ndarray, rng: np.random.Generator, noise: float = 2.0,
            jpeg: int | None = None, blur: float = 0.0) -> np.ndarray:
    out = img.astype(np.float32)
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if noise > 0:
        out += rng.normal(0, noise, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if jpeg:
        ok, enc = cv2.imencode(".jpg", out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
    return out


# ----------------------------------------------------------------- silhouettes
def tee_polygon(h: int, w: int, cx: float, top: float, width: float, length: float,
                sleeve: float = 0.42, rng=None) -> np.ndarray:
    """A t-shirt outline: shoulders, sleeves, tapered body, curved hem."""
    hw = width / 2
    sl = width * sleeve
    pts = [
        (cx - hw * 0.42, top),                        # left collar
        (cx - hw, top + length * 0.06),               # left shoulder
        (cx - hw - sl, top + length * 0.30),          # left sleeve out
        (cx - hw - sl * 0.86, top + length * 0.40),
        (cx - hw * 0.90, top + length * 0.34),        # underarm
        (cx - hw * 0.86, top + length),               # left hem
        (cx + hw * 0.86, top + length),               # right hem
        (cx + hw * 0.90, top + length * 0.34),
        (cx + hw + sl * 0.86, top + length * 0.40),
        (cx + hw + sl, top + length * 0.30),
        (cx + hw, top + length * 0.06),
        (cx + hw * 0.42, top),
        (cx, top + length * 0.075),                   # collar dip
    ]
    return np.array(pts, np.float32)


def mug_alpha(h: int, w: int, cx: float, cy: float, bw: float, bh: float,
              handle: bool = True, flip: bool = False) -> np.ndarray:
    """Mug body (slightly tapered) plus an open handle ring - one real hole."""
    def draw(big, ss):
        s = ss
        top_hw, bot_hw = bw / 2, bw / 2 * 0.90
        body = np.array([
            (cx - top_hw, cy - bh / 2), (cx + top_hw, cy - bh / 2),
            (cx + bot_hw, cy + bh / 2), (cx - bot_hw, cy + bh / 2),
        ], np.float32)
        cv2.fillPoly(big, [np.round(body * s).astype(np.int32)], 255, cv2.LINE_AA)
        cv2.ellipse(big, (int(cx * s), int((cy - bh / 2) * s)),
                    (int(top_hw * s), int(bh * 0.075 * s)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.ellipse(big, (int(cx * s), int((cy + bh / 2) * s)),
                    (int(bot_hw * s), int(bh * 0.06 * s)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        if handle:
            side = -1 if flip else 1
            hx = cx + side * (top_hw * 0.94)
            cv2.ellipse(big, (int(hx * s), int(cy * s)),
                        (int(bw * 0.34 * s), int(bh * 0.30 * s)),
                        0, -88 if side > 0 else 92, 88 if side > 0 else 268,
                        255, int(bw * 0.10 * s), cv2.LINE_AA)
    return mask_from_draw(h, w, draw)


def rounded_quad(h: int, w: int, quad: np.ndarray, radius: float) -> np.ndarray:
    """Quad with rounded corners - phone cases, mousepads, cushions."""
    def draw(big, ss):
        pts = np.round(quad * ss).astype(np.int32)
        cv2.fillConvexPoly(big, pts, 255, cv2.LINE_AA)
        cv2.polylines(big, [pts], True, 255, int(radius * ss), cv2.LINE_AA)
    m = mask_from_draw(h, w, draw)
    k = max(3, int(radius) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return np.clip(m, 0, 1)


# --------------------------------------------------------------------- scenes
def scene_tee_flatlay(rng, dark=False):
    h, w = 1500, 1200
    bg = sweep_background(h, w, rng, base=240 if not dark else 232)
    alpha = mask_from_poly(h, w, tee_polygon(h, w, cx=600, top=250, width=620, length=980))
    colour = (48, 50, 55) if dark else (247, 247, 249)
    layer = shade(alpha, colour, rng, folds=0.22 if not dark else 0.28, ambient=0.78)
    img = drop_shadow(bg, alpha, offset=(14, 22), blur=24, strength=0.28)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=2.0, jpeg=92), alpha, "apparel"


def scene_tee_on_model(rng):
    """Worn tee: skin, hair strands over the shoulder, arms outside the mask."""
    h, w = 1600, 1200
    bg = wall_background(h, w, rng, tint=(214, 210, 205))
    # torso / neck / arms in skin tone, drawn first so the shirt sits on top
    skin = mask_from_draw(h, w, lambda big, s: (
        cv2.ellipse(big, (int(600 * s), int(250 * s)), (int(104 * s), int(132 * s)), 0, 0, 360, 255, -1, cv2.LINE_AA),
        cv2.rectangle(big, (int(548 * s), int(340 * s)), (int(652 * s), int(540 * s)), 255, -1),
        cv2.ellipse(big, (int(330 * s), int(900 * s)), (int(52 * s), int(330 * s)), 12, 0, 360, 255, -1, cv2.LINE_AA),
        cv2.ellipse(big, (int(870 * s), int(900 * s)), (int(52 * s), int(330 * s)), -12, 0, 360, 255, -1, cv2.LINE_AA),
    ))
    skin_layer = shade(skin, (214, 172, 140), rng, folds=0.05, ambient=0.84)
    img = composite(bg, skin_layer, skin)

    shirt = mask_from_poly(h, w, tee_polygon(h, w, cx=600, top=470, width=560, length=880, sleeve=0.30))
    shirt_layer = shade(shirt, (243, 244, 247), rng, folds=0.30, ambient=0.74)
    img = composite(img, shirt_layer, shirt)
    gt = shirt.copy()

    # Strands start at the hairline and fall *past* the collar onto the chest -
    # the specific case where a naive matte leaves black fringe on the garment.
    hair = hair_strands(h, w, rng, n=30, x0=600, y0=200, span=200, drop=620)
    hair_layer = shade(hair, (58, 42, 34), rng, ambient=0.9)
    img = composite(img, hair_layer, hair)
    gt = gt * (1.0 - hair)                      # hair occludes the garment
    return degrade(img, rng, noise=2.4, jpeg=90), gt, "apparel"


def scene_hoodie(rng):
    h, w = 1500, 1300
    bg = sweep_background(h, w, rng, base=236)
    pts = tee_polygon(h, w, cx=650, top=300, width=700, length=940, sleeve=0.50)
    alpha = mask_from_poly(h, w, pts)
    hood = mask_from_draw(h, w, lambda big, s: cv2.ellipse(
        big, (int(650 * s), int(320 * s)), (int(210 * s), int(120 * s)), 0, 180, 360, 255, -1, cv2.LINE_AA))
    alpha = np.clip(alpha + hood, 0, 1)
    layer = shade(alpha, (96, 112, 138), rng, folds=0.34, ambient=0.72)
    img = drop_shadow(bg, alpha, offset=(16, 24), blur=26, strength=0.30)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=2.2, jpeg=91), alpha, "apparel"


def scene_tote(rng):
    """Tote bag: two handles -> two legitimate interior holes."""
    h, w = 1400, 1200
    bg = sweep_background(h, w, rng, base=244)
    body = mask_from_poly(h, w, np.array([
        (330, 520), (870, 520), (890, 1120), (310, 1120)], np.float32))
    handles = mask_from_draw(h, w, lambda big, s: (
        cv2.ellipse(big, (int(470 * s), int(520 * s)), (int(90 * s), int(180 * s)), 0, 180, 360,
                    255, int(26 * s), cv2.LINE_AA),
        cv2.ellipse(big, (int(730 * s), int(520 * s)), (int(90 * s), int(180 * s)), 0, 180, 360,
                    255, int(26 * s), cv2.LINE_AA),
    ))
    alpha = np.clip(body + handles, 0, 1)
    layer = shade(alpha, (232, 224, 202), rng, folds=0.18, ambient=0.80)
    img = drop_shadow(bg, alpha, offset=(12, 20), blur=20, strength=0.26)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=2.0, jpeg=92), alpha, "apparel"


def scene_tee_hanger(rng):
    """Tee on a hanger against a busy wall - thin metal hook, textured backdrop."""
    h, w = 1500, 1100
    bg = wall_background(h, w, rng, tint=(196, 190, 178))
    bg = composite(bg, np.full_like(bg, 120.0),
                   mask_from_draw(h, w, lambda big, s: cv2.line(
                       big, (0, int(180 * s)), (int(w * s), int(140 * s)), 255, int(9 * s), cv2.LINE_AA)))
    hook = mask_from_draw(h, w, lambda big, s: (
        cv2.ellipse(big, (int(550 * s), int(250 * s)), (int(40 * s), int(46 * s)), 0, 150, 400,
                    255, int(8 * s), cv2.LINE_AA),
        cv2.line(big, (int(400 * s), int(360 * s)), (int(700 * s), int(360 * s)), 255, int(10 * s), cv2.LINE_AA),
    ))
    shirt = mask_from_poly(h, w, tee_polygon(h, w, cx=550, top=360, width=520, length=820, sleeve=0.34))
    alpha = np.clip(shirt + hook, 0, 1)
    layer = shade(alpha, (206, 74, 68), rng, folds=0.26, ambient=0.76)
    img = composite(bg, layer, alpha)
    return degrade(img, rng, noise=2.6, jpeg=88), alpha, "apparel"


def scene_mug_white_on_white(rng):
    """Hardest drinkware case: white ceramic on a white sweep."""
    h, w = 1300, 1100
    bg = sweep_background(h, w, rng, base=250, vignette=0.05)
    alpha = mug_alpha(h, w, cx=520, cy=680, bw=430, bh=470)
    layer = shade(alpha, (250, 250, 252), rng, ambient=0.86, gloss=0.16)
    img = drop_shadow(bg, alpha, offset=(10, 16), blur=16, strength=0.20)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=1.6, jpeg=94), alpha, "drinkware"


def scene_mug_on_desk(rng):
    """Dark mug on a cluttered wooden desk - the 'complex background' case."""
    h, w = 1200, 1500
    bg = desk_background(h, w, rng)
    alpha = mug_alpha(h, w, cx=700, cy=620, bw=400, bh=440, flip=True)
    layer = shade(alpha, (36, 38, 44), rng, ambient=0.70, gloss=0.22)
    img = drop_shadow(bg, alpha, offset=(24, 18), blur=18, strength=0.42)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=3.4, jpeg=84), alpha, "drinkware"


def scene_tumbler(rng):
    h, w = 1500, 1000
    bg = sweep_background(h, w, rng, base=238)
    alpha = mug_alpha(h, w, cx=500, cy=760, bw=330, bh=820, handle=False)
    layer = shade(alpha, (176, 180, 190), rng, ambient=0.66, gloss=0.30)
    img = drop_shadow(bg, alpha, offset=(12, 14), blur=18, strength=0.26)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=2.0, jpeg=92), alpha, "drinkware"


def scene_bottle_translucent(rng):
    """Frosted bottle: the neck is genuinely semi-transparent, so the ground
    truth alpha is fractional - a real matting problem, not just segmentation."""
    h, w = 1600, 1000
    bg = wall_background(h, w, rng, tint=(206, 214, 220))
    body = mask_from_draw(h, w, lambda big, s: (
        cv2.rectangle(big, (int(360 * s), int(560 * s)), (int(640 * s), int(1320 * s)), 255, -1),
        cv2.rectangle(big, (int(440 * s), int(330 * s)), (int(560 * s), int(600 * s)), 255, -1),
        cv2.ellipse(big, (int(500 * s), int(1320 * s)), (int(140 * s), int(40 * s)), 0, 0, 360, 255, -1, cv2.LINE_AA),
    ))
    glass = shade(body, (216, 226, 230), rng, ambient=0.74, gloss=0.26)
    # Translucent upper third: alpha ramps from 0.55 to 1.0 down the bottle.
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    trans = np.clip(0.52 + 0.55 * (yy - 0.34) / 0.35, 0.52, 1.0)
    alpha = body * trans
    img = drop_shadow(bg, body, offset=(14, 12), blur=20, strength=0.22)
    img = composite(img, glass, alpha)
    return degrade(img, rng, noise=2.2, jpeg=91), alpha, "drinkware"


def scene_canvas_straight(rng):
    h, w = 1200, 1500
    bg = wall_background(h, w, rng, tint=(222, 218, 210))
    quad = np.array([(330, 240), (1180, 250), (1180, 960), (330, 950)], np.float32)
    alpha = mask_from_poly(h, w, quad)
    layer = shade(alpha, (250, 249, 246), rng, ambient=0.90)
    img = drop_shadow(bg, alpha, offset=(16, 18), blur=18, strength=0.26)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=1.8, jpeg=93), alpha, "wall_art"


def scene_canvas_perspective(rng):
    """Canvas seen at an angle: the print area is a quad, not a rectangle."""
    h, w = 1200, 1600
    bg = wall_background(h, w, rng, tint=(206, 204, 200))
    quad = np.array([(420, 300), (1300, 190), (1310, 900), (430, 1010)], np.float32)
    alpha = mask_from_poly(h, w, quad)
    layer = shade(alpha, (246, 244, 240), rng, ambient=0.80, light=(-0.8, -0.2))
    img = drop_shadow(bg, alpha, offset=(-18, 20), blur=22, strength=0.30)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=2.0, jpeg=90), alpha, "wall_art"


def scene_framed_print(rng):
    h, w = 1400, 1100
    bg = wall_background(h, w, rng, tint=(196, 196, 198))
    outer = np.array([(240, 230), (860, 230), (860, 1120), (240, 1120)], np.float32)
    alpha = mask_from_poly(h, w, outer)
    layer = shade(alpha, (58, 46, 40), rng, ambient=0.84)
    inner = mask_from_poly(h, w, np.array([(300, 290), (800, 290), (800, 1060), (300, 1060)], np.float32))
    layer = composite(layer, shade(inner, (250, 250, 248), rng, ambient=0.94), inner)
    img = drop_shadow(bg, alpha, offset=(14, 16), blur=16, strength=0.28)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=1.9, jpeg=92), alpha, "wall_art"


def scene_poster_dark(rng):
    """Low-light, high-noise, product clipped by the frame - should be flagged."""
    h, w = 1200, 1000
    bg = wall_background(h, w, rng, tint=(74, 76, 84))
    quad = np.array([(180, -60), (900, -40), (910, 880), (170, 900)], np.float32)
    alpha = mask_from_poly(h, w, quad)
    layer = shade(alpha, (238, 236, 230), rng, ambient=0.62)
    img = composite(bg, layer, alpha)
    return degrade(img, rng, noise=6.0, jpeg=78, blur=0.6), alpha, "wall_art"


def scene_phone_case(rng):
    h, w = 1400, 1000
    bg = sweep_background(h, w, rng, base=246)
    quad = np.array([(360, 300), (700, 300), (700, 1120), (360, 1120)], np.float32)
    alpha = rounded_quad(h, w, quad, radius=54)
    layer = shade(alpha, (72, 118, 196), rng, ambient=0.78, gloss=0.14)
    img = drop_shadow(bg, alpha, offset=(10, 16), blur=16, strength=0.24)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=1.8, jpeg=93), alpha, "accessory"


def scene_cap(rng):
    h, w = 1100, 1400
    bg = sweep_background(h, w, rng, base=240)
    alpha = mask_from_draw(h, w, lambda big, s: (
        cv2.ellipse(big, (int(700 * s), int(620 * s)), (int(300 * s), int(280 * s)), 0, 180, 360,
                    255, -1, cv2.LINE_AA),
        cv2.ellipse(big, (int(880 * s), int(630 * s)), (int(250 * s), int(70 * s)), 8, 0, 180,
                    255, -1, cv2.LINE_AA),
    ))
    layer = shade(alpha, (54, 96, 76), rng, folds=0.12, ambient=0.74)
    img = drop_shadow(bg, alpha, offset=(12, 14), blur=18, strength=0.26)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=2.1, jpeg=91), alpha, "accessory"


def scene_mousepad(rng):
    h, w = 900, 1600
    bg = desk_background(h, w, rng)
    quad = np.array([(200, 260), (1420, 250), (1430, 690), (190, 700)], np.float32)
    alpha = rounded_quad(h, w, quad, radius=30)
    layer = shade(alpha, (38, 40, 46), rng, ambient=0.82)
    img = drop_shadow(bg, alpha, offset=(8, 10), blur=12, strength=0.30)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=3.0, jpeg=86), alpha, "accessory"


def scene_cushion(rng):
    h, w = 1200, 1300
    bg = sweep_background(h, w, rng, base=238)
    quad = np.array([(300, 300), (1000, 300), (1000, 980), (300, 980)], np.float32)
    alpha = rounded_quad(h, w, quad, radius=110)
    layer = shade(alpha, (226, 214, 198), rng, folds=0.24, ambient=0.76)
    img = drop_shadow(bg, alpha, offset=(14, 20), blur=22, strength=0.28)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=2.3, jpeg=90), alpha, "accessory"


def scene_tee_hard_flash(rng):
    """Hard flash: a sharp, dark cast shadow that a naive matte swallows."""
    h, w = 1400, 1200
    bg = sweep_background(h, w, rng, base=248, vignette=0.04)
    alpha = mask_from_poly(h, w, tee_polygon(h, w, cx=600, top=260, width=600, length=900))
    layer = shade(alpha, (250, 250, 250), rng, folds=0.16, ambient=0.90)
    img = drop_shadow(bg, alpha, offset=(46, 40), blur=9, strength=0.50)
    img = composite(img, layer, alpha)
    return degrade(img, rng, noise=1.7, jpeg=94), alpha, "apparel"


SCENES = [
    ("01_tee_flatlay_white", scene_tee_flatlay),
    ("02_tee_flatlay_black", lambda rng: scene_tee_flatlay(rng, dark=True)),
    ("03_tee_on_model_hair", scene_tee_on_model),
    ("04_hoodie_folds", scene_hoodie),
    ("05_tote_two_handles", scene_tote),
    ("06_tee_hanger_busy_wall", scene_tee_hanger),
    ("07_tee_hard_flash_shadow", scene_tee_hard_flash),
    ("08_mug_white_on_white", scene_mug_white_on_white),
    ("09_mug_dark_cluttered_desk", scene_mug_on_desk),
    ("10_tumbler_no_handle", scene_tumbler),
    ("11_bottle_translucent", scene_bottle_translucent),
    ("12_canvas_straight", scene_canvas_straight),
    ("13_canvas_perspective", scene_canvas_perspective),
    ("14_framed_print", scene_framed_print),
    ("15_poster_dark_clipped", scene_poster_dark),
    ("16_phone_case", scene_phone_case),
    ("17_cap", scene_cap),
    ("18_mousepad_on_desk", scene_mousepad),
    ("19_cushion", scene_cushion),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "samples")
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    out = args.out
    gt_dir = out / "ground_truth"
    out.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (name, fn) in enumerate(SCENES):
        rng = np.random.default_rng(args.seed + i * 7919)
        img, alpha, category = fn(rng)
        img_path = out / f"{name}.png"
        gt_path = gt_dir / f"{name}_gt.png"
        cv2.imwrite(str(img_path), img[:, :, ::-1])
        cv2.imwrite(str(gt_path), np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8))
        h, w = alpha.shape
        rows.append({
            "file": img_path.name, "ground_truth": f"ground_truth/{gt_path.name}",
            "category": category, "width": w, "height": h,
            "coverage": round(float((alpha > 0.5).mean()), 4),
        })
        print(f"  {name:32s} {w}x{h}  {category:10s} coverage {rows[-1]['coverage']:.3f}")

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} bases + ground truth -> {out}")
    print("Evaluate with:  python scripts/eval_iou.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
