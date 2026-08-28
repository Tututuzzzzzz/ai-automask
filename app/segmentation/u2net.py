"""U2-Net (ONNX Runtime) backend.

Two jobs:
  * **Fallback** when BiRefNet weights or CUDA are unavailable - the service must
    never hard-fail a batch just because the big model did not load.
  * **Cross-check** for the QC stage. Two architecturally independent models
    agreeing on a boundary is the single strongest confidence signal we have; it
    is what lets us call READY without a human ever looking at the mask.

Weights: xuebinqin/U-2-Net (Apache-2.0), re-hosted by rembg (MIT).
Runs on CPU in ~0.4-1.2 s at 320x320, so using it as a second opinion costs
almost nothing next to the primary model.
"""
from __future__ import annotations

import numpy as np

from ..config import settings
from .base import SegmentationBackend

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class U2NetBackend(SegmentationBackend):
    name = "u2net"
    priority = 50
    input_size = 320

    def __init__(self, weights: str | None = None) -> None:
        self.weights_path = settings.models_dir / (weights or "u2net.onnx")
        self.session = None
        self._input_name: str | None = None
        self._load_error: str | None = None

    def available(self) -> bool:
        if self._load_error:
            return False
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return self.weights_path.exists() and self.weights_path.stat().st_size > 1_000_000

    def load(self) -> None:
        if self._loaded:
            return
        import onnxruntime as ort

        if not self.available():
            self._load_error = f"weights missing at {self.weights_path}"
            raise RuntimeError(self._load_error)

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(self.weights_path), sess_options=opts, providers=providers)
        self._input_name = self.session.get_inputs()[0].name
        self._loaded = True

    def unload(self) -> None:
        self.session = None
        self._loaded = False

    def _predict(self, image: np.ndarray) -> np.ndarray:
        import cv2

        h, w = image.shape[:2]
        size = self.input_size
        inp = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        arr = inp.astype(np.float32) / 255.0
        # U2-Net's reference preprocessing scales by max, then imagenet-normalises.
        mx = arr.max() or 1.0
        arr = arr / mx
        arr = (arr - _MEAN) / _STD
        tensor = arr.transpose(2, 0, 1)[None].astype(np.float32)

        outputs = self.session.run(None, {self._input_name: tensor})
        pred = np.asarray(outputs[0])[0, 0]
        lo, hi = float(pred.min()), float(pred.max())
        pred = (pred - lo) / (hi - lo + 1e-8)
        return cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    def describe(self) -> dict:
        d = super().describe()
        d.update({"weights": str(self.weights_path), "license": "Apache-2.0", "error": self._load_error})
        return d
