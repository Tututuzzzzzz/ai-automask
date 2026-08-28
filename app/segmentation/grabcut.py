"""GrabCut backend - the last line of defence.

Pure OpenCV, zero weights, always available. It is not competitive on accuracy,
but it guarantees the microservice degrades instead of dying: if every neural
backend fails to load (no weights on a fresh clone, no network in a locked-down
CI runner), the pipeline still returns a mask - and the QC stage will honestly
label it REVIEW or FAILED rather than pretending it is production-ready.
"""
from __future__ import annotations

import numpy as np

from .base import SegmentationBackend


class GrabCutBackend(SegmentationBackend):
    name = "grabcut"
    priority = 1

    def available(self) -> bool:
        try:
            import cv2  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self) -> None:
        self._loaded = True

    def _predict(self, image: np.ndarray) -> np.ndarray:
        import cv2

        h, w = image.shape[:2]
        scale = 1.0
        work = image
        if max(h, w) > 1200:  # GrabCut is O(pixels) and slow; run it small then upscale
            scale = 1200.0 / max(h, w)
            work = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        wh, ww = work.shape[:2]

        # Seed from a border-colour flood: product photos are usually shot on a
        # near-uniform sweep, so "pixels similar to the frame border" is background.
        border = np.concatenate([
            work[:6].reshape(-1, 3), work[-6:].reshape(-1, 3),
            work[:, :6].reshape(-1, 3), work[:, -6:].reshape(-1, 3),
        ]).astype(np.float32)
        bg_mean = border.mean(axis=0)
        bg_std = border.std(axis=0) + 6.0
        dist = np.abs(work.astype(np.float32) - bg_mean) / bg_std
        fgness = (dist.max(axis=2) > 3.0).astype(np.uint8)
        fgness = cv2.morphologyEx(fgness, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fgness = cv2.morphologyEx(fgness, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        gc = np.full((wh, ww), cv2.GC_PR_BGD, dtype=np.uint8)
        gc[fgness > 0] = cv2.GC_PR_FGD
        sure_fg = cv2.erode(fgness, np.ones((25, 25), np.uint8))
        gc[sure_fg > 0] = cv2.GC_FGD
        gc[:3] = gc[-3:] = cv2.GC_BGD
        gc[:, :3] = gc[:, -3:] = cv2.GC_BGD

        if (gc == cv2.GC_FGD).sum() < 50:  # nothing stood out - fall back to a centre rect
            gc[:] = cv2.GC_PR_BGD
            gc[int(0.15 * wh):int(0.85 * wh), int(0.15 * ww):int(0.85 * ww)] = cv2.GC_PR_FGD
            gc[int(0.40 * wh):int(0.60 * wh), int(0.40 * ww):int(0.60 * ww)] = cv2.GC_FGD

        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(work, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            pass
        alpha = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
        if scale != 1.0:
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return alpha

    def describe(self) -> dict:
        d = super().describe()
        d["license"] = "Apache-2.0 (OpenCV)"
        return d
