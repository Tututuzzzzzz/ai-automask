"""Product-category detection.

Category is not cosmetic metadata: it selects the print-area solver, the plausible
coverage range used by QC, and whether soft edges are preserved or crushed. When
the caller supplies it (the normal case for an integrated pipeline - the SKU
already knows it is a mug) we trust it. When they do not, we infer it.

Two tiers:

* **Geometric / photometric heuristics** (always on, ~3 ms). Deterministic, no
  weights, no network. Reads shape and light: mug handles punch exactly one
  interior hole and produce hard specular streaks on a low-saturation body;
  canvases are near-perfect quads with 0.99 solidity; worn apparel has a
  shoulder-width bulge and skin-toned pixels adjacent to the mask.

* **CLIP zero-shot** (opt-in via ``AUTOMASK_CLIP=1``). More robust on odd
  angles and flat-lays. Off by default so a cold container does not pay a
  600 MB download and the p99 latency stays predictable.

The two are fused: CLIP wins when it is confident, geometry breaks ties, and the
final label always carries its confidence into the API response so downstream
systems can decide whether to trust it.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .printarea import _quad_area, _skin_hair_mask

_CLIP_PROMPTS = {
    "apparel": [
        "a plain blank t-shirt product photo",
        "a person wearing a blank t-shirt",
        "a blank hoodie or sweatshirt",
        "a blank cotton tote bag",
    ],
    "drinkware": [
        "a plain white ceramic coffee mug",
        "a blank stainless steel tumbler",
        "a blank water bottle product photo",
    ],
    "wall_art": [
        "a blank white canvas on a wall",
        "an empty picture frame or poster mockup",
        "a blank rectangular art print",
    ],
    "accessory": [
        "a blank phone case product photo",
        "a blank baseball cap",
        "a blank rectangular mousepad",
    ],
}


def clip_enabled() -> bool:
    return os.getenv("AUTOMASK_CLIP", "0").strip().lower() in {"1", "true", "yes", "on"}


# ------------------------------------------------------------------- heuristics
def geometric_scores(image: np.ndarray, alpha: np.ndarray, holes: int = 0) -> tuple[dict, dict]:
    """Return (scores per category, diagnostic detail)."""
    h, w = alpha.shape[:2]
    binary = (alpha > 0.5).astype(np.uint8)
    detail: dict = {}
    scores = {"apparel": 0.0, "drinkware": 0.0, "wall_art": 0.0, "accessory": 0.0}
    if binary.sum() < 64:
        return scores, {"note": "empty mask"}

    # Work small: every signal here is shape-scale, none needs full resolution.
    scale = min(1.0, 420.0 / max(h, w))
    small = cv2.resize(binary, (max(8, int(w * scale)), max(8, int(h * scale))),
                       interpolation=cv2.INTER_NEAREST) if scale < 1.0 else binary
    img_s = cv2.resize(image, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else image

    ys, xs = np.nonzero(small)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bh, bw = (y1 - y0 + 1), (x1 - x0 + 1)
    aspect = bw / float(bh)
    fill = float(small[y0:y1 + 1, x0:x1 + 1].mean())      # how box-like is it
    detail["aspect"] = round(aspect, 3)
    detail["bbox_fill"] = round(fill, 3)

    quad, quad_fit = _quad_area(small, inset=0.0)
    detail["quad_fit"] = round(quad_fit, 3)

    # ---- wall art / flat rectangular goods --------------------------------
    # A canvas fills its bounding quad almost completely and has no holes.
    scores["wall_art"] += 1.6 * max(0.0, (quad_fit - 0.86) / 0.14)
    scores["wall_art"] += 1.0 * max(0.0, (fill - 0.88) / 0.12)
    if holes > 0:
        scores["wall_art"] -= 0.8

    # ---- drinkware --------------------------------------------------------
    # Signature: exactly one sizeable interior hole (the handle aperture), an
    # upright-ish aspect, a desaturated body, and hard specular streaks.
    hsv = cv2.cvtColor(img_s, cv2.COLOR_RGB2HSV)
    body = small > 0
    sat = float(hsv[..., 1][body].mean()) / 255.0
    val = hsv[..., 2][body].astype(np.float32) / 255.0
    specular = float((val > 0.94).mean())
    detail["body_saturation"] = round(sat, 3)
    detail["specular_ratio"] = round(specular, 4)

    if holes == 1:
        scores["drinkware"] += 1.3
    elif holes == 2:
        scores["drinkware"] += 0.4
    if 0.6 <= aspect <= 1.6:
        scores["drinkware"] += 0.5
    scores["drinkware"] += 0.9 * max(0.0, (0.25 - sat) / 0.25)
    scores["drinkware"] += 0.8 * min(1.0, specular / 0.06)
    # Mug silhouettes are boxy but not perfect quads - the handle breaks the fit.
    if 0.55 < quad_fit < 0.9:
        scores["drinkware"] += 0.4

    # ---- apparel ----------------------------------------------------------
    # Shoulder bulge: the row-width profile peaks in the upper third (sleeves
    # or shoulders) and narrows below it.
    widths = []
    for row in small[y0:y1 + 1]:
        idx = np.nonzero(row)[0]
        widths.append((idx.max() - idx.min() + 1) if len(idx) else 0)
    widths = np.asarray(widths, dtype=np.float32)
    if len(widths) > 10:
        upper = float(widths[: max(1, len(widths) // 3)].max())
        lower = float(np.median(widths[len(widths) // 2:]) or 1.0)
        shoulder_ratio = upper / max(lower, 1.0)
    else:
        shoulder_ratio = 1.0
    detail["shoulder_ratio"] = round(shoulder_ratio, 3)
    scores["apparel"] += 1.2 * min(1.0, max(0.0, (shoulder_ratio - 1.12) / 0.5))

    # Skin adjacent to (or inside) the mask means a human model is wearing it.
    skin = _skin_hair_mask(img_s)
    near = cv2.dilate(small, np.ones((15, 15), np.uint8)).astype(bool)
    skin_ratio = float((skin[near] > 0).mean()) if near.any() else 0.0
    detail["skin_ratio"] = round(skin_ratio, 4)
    scores["apparel"] += 1.4 * min(1.0, skin_ratio / 0.10)

    # Fabric silhouettes are irregular: low quad fit, moderate bbox fill.
    scores["apparel"] += 0.9 * max(0.0, (0.80 - quad_fit) / 0.5)
    if holes >= 1:
        scores["apparel"] += 0.25          # arm gaps / handle gaps on a tote

    # ---- accessory (catch-all for small hard goods) -----------------------
    scores["accessory"] += 0.55            # weak prior so it wins only by default
    if 0.35 <= aspect <= 0.75 and quad_fit > 0.8:
        scores["accessory"] += 0.7         # phone case: tall rounded rectangle
    if aspect > 1.7 and quad_fit > 0.9:
        scores["accessory"] += 0.5         # mousepad: wide rectangle

    return {k: round(float(max(0.0, v)), 4) for k, v in scores.items()}, detail


# ------------------------------------------------------------------------ CLIP
class ClipClassifier:
    """Optional zero-shot classifier. Loaded lazily, failures are non-fatal."""

    repo = "openai/clip-vit-base-patch32"      # MIT licence

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self._error: str | None = None
        self._labels: list[str] = []
        self._texts: list[str] = []
        for cat, prompts in _CLIP_PROMPTS.items():
            for p in prompts:
                self._labels.append(cat)
                self._texts.append(p)

    def available(self) -> bool:
        if not clip_enabled() or self._error:
            return False
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        from ..config import settings

        self.processor = CLIPProcessor.from_pretrained(self.repo)
        model = CLIPModel.from_pretrained(self.repo)
        model.eval()
        device = settings.resolved_device()
        self.model = model.to(device)
        self.device = device
        self._torch = torch

    def scores(self, image: np.ndarray) -> dict | None:
        if not self.available():
            return None
        try:
            self.load()
            torch = self._torch
            from PIL import Image

            pil = Image.fromarray(image)
            inputs = self.processor(text=self._texts, images=pil, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs).logits_per_image[0]
                probs = logits.softmax(dim=-1).cpu().numpy()
            agg: dict[str, float] = {}
            for label, p in zip(self._labels, probs):
                agg[label] = agg.get(label, 0.0) + float(p)
            return {k: round(v, 4) for k, v in agg.items()}
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
            return None

    def describe(self) -> dict:
        return {"name": "clip-vit-base-patch32", "enabled": clip_enabled(),
                "available": self.available(), "license": "MIT", "error": self._error}


clip_classifier = ClipClassifier()


# ----------------------------------------------------------------------- fuse
def detect_category(image: np.ndarray, alpha: np.ndarray, holes: int = 0) -> tuple[str, float, dict]:
    """Infer the product category. Returns (category, confidence, detail)."""
    geo, detail = geometric_scores(image, alpha, holes=holes)
    total = sum(geo.values()) or 1.0
    fused = {k: v / total for k, v in geo.items()}
    detail["geometric"] = geo

    clip = clip_classifier.scores(image)
    if clip:
        detail["clip"] = clip
        # CLIP is the stronger signal on appearance; geometry is the stronger
        # signal on shape. 60/40 with the geometric prior as the tie-breaker.
        fused = {k: 0.6 * clip.get(k, 0.0) + 0.4 * fused.get(k, 0.0) for k in fused}

    best = max(fused, key=fused.get)
    ordered = sorted(fused.values(), reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    # Confidence blends the winning share with how clearly it beat the runner-up.
    confidence = float(np.clip(0.5 * fused[best] + 0.5 * min(1.0, margin / 0.25), 0.0, 1.0))
    detail["fused"] = {k: round(v, 4) for k, v in fused.items()}
    return best, round(confidence, 4), detail
