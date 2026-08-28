"""Optional SAM (MobileSAM) boundary refiner.

Purpose: promptable segmentation is very good at *snapping to the true object
boundary* when the background is busy (the hidden-test "mug on a cluttered
desk" case). We do not use it as the primary segmenter - SAM has no notion of
"which object is the product" - instead we feed it the box + point prompts
derived from the primary matte and use its crisp boundary as a second opinion.

LICENSE NOTE (matters for the commercial-use rule in the brief):
  * MobileSAM weights themselves: Apache-2.0 (ChaoningZhang/MobileSAM).
  * The loader used here (`ultralytics`) is **AGPL-3.0**.
Because AGPL is awkward for a closed-source POD backend, this refiner is
**disabled by default** (`AUTOMASK_SAM_REFINE=0`). Enable it only if you either
(a) comply with AGPL, (b) hold an Ultralytics commercial licence, or (c) swap in
the upstream MobileSAM repo, which is Apache-2.0. The pipeline is fully
functional without it - BiRefNet (MIT) + U2-Net (Apache-2.0) carry the workload.
"""
from __future__ import annotations

import os

import numpy as np

from ..config import settings


def sam_enabled() -> bool:
    return os.getenv("AUTOMASK_SAM_REFINE", "0").strip().lower() in {"1", "true", "yes", "on"}


class SamRefiner:
    name = "mobile_sam"
    license = "Apache-2.0 weights / AGPL-3.0 loader"

    def __init__(self, weights: str = "mobile_sam.pt") -> None:
        self.weights_path = settings.models_dir / weights
        self.model = None
        self._error: str | None = None

    def available(self) -> bool:
        if not sam_enabled() or self._error:
            return False
        if not self.weights_path.exists():
            return False
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return False
        return True

    def load(self) -> None:
        if self.model is not None:
            return
        from ultralytics import SAM

        self.model = SAM(str(self.weights_path))

    def refine(self, image: np.ndarray, alpha: np.ndarray) -> np.ndarray | None:
        """Return a binary float32 mask from SAM, or None if unusable."""
        if not self.available():
            return None
        try:
            self.load()
            binary = (alpha > 0.5).astype(np.uint8)
            if binary.sum() < 64:
                return None
            ys, xs = np.nonzero(binary)
            box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

            # Interior points from the eroded mask give SAM an unambiguous "this
            # is the object" hint, which suppresses its habit of returning a part.
            import cv2

            core = cv2.erode(binary, np.ones((15, 15), np.uint8))
            pys, pxs = np.nonzero(core if core.sum() > 20 else binary)
            idx = np.linspace(0, len(pys) - 1, num=min(5, len(pys))).astype(int)
            points = [[int(pxs[i]), int(pys[i])] for i in idx]

            res = self.model.predict(
                image[:, :, ::-1],                    # ultralytics expects BGR ndarray
                bboxes=[box],
                points=points,
                labels=[1] * len(points),
                verbose=False,
            )
            if not res or res[0].masks is None:
                return None
            data = res[0].masks.data.cpu().numpy()
            merged = np.clip(data.sum(axis=0), 0.0, 1.0).astype(np.float32)
            h, w = image.shape[:2]
            if merged.shape[:2] != (h, w):
                merged = cv2.resize(merged, (w, h), interpolation=cv2.INTER_NEAREST)
            return merged
        except Exception as exc:  # noqa: BLE001 - a refiner must never break the pipeline
            self._error = f"{type(exc).__name__}: {exc}"
            return None

    def describe(self) -> dict:
        return {
            "name": self.name,
            "enabled": sam_enabled(),
            "available": self.available(),
            "weights": str(self.weights_path),
            "license": self.license,
            "error": self._error,
        }
