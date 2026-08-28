"""Common interface every segmentation backend implements.

A backend takes an RGB uint8 image and returns a float32 alpha matte in [0, 1]
at *exactly* the input resolution. Keeping that contract in one place is what
guarantees requirement 6: "mask must match the source resolution exactly".
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegOutput:
    """Result of one backend run."""

    alpha: np.ndarray                     # float32 HxW in [0, 1]
    model: str
    latency_ms: float
    meta: dict = field(default_factory=dict)


class SegmentationBackend(ABC):
    """Lazily-loaded segmentation model."""

    name: str = "abstract"
    #: Higher wins when the pipeline picks a primary model.
    priority: int = 0
    #: Set to False by subclasses when weights / deps are missing.
    _loaded: bool = False
    #: Serialises inference. Batch workers overlap CPU-side work (decode, refine,
    #: PNG encode) with GPU work, but two threads must never enter the same CUDA
    #: graph at once on a 4 GB card.
    _infer_lock: threading.Lock | None = None

    def __init_subclass__(cls, **kwargs) -> None:
        # One lock per backend class, created at import time (single-threaded),
        # so predict() never has to race on lazily creating it.
        super().__init_subclass__(**kwargs)
        cls._infer_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load weights. Must be idempotent and raise on unrecoverable failure."""

    def unload(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def available(self) -> bool:
        """Cheap check (files on disk, importable deps) without loading weights."""

    # -- inference -----------------------------------------------------------
    @abstractmethod
    def _predict(self, image: np.ndarray) -> np.ndarray:
        """Return float32 alpha in [0, 1], same HxW as *image*."""

    def predict(self, image: np.ndarray) -> SegOutput:
        with self._infer_lock:
            if not self._loaded:
                self.load()
            t0 = time.perf_counter()
            alpha = self._predict(image)
            dt = (time.perf_counter() - t0) * 1000.0

        alpha = np.asarray(alpha, dtype=np.float32)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        h, w = image.shape[:2]
        if alpha.shape[:2] != (h, w):  # defensive: never let a backend change geometry
            import cv2

            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        np.clip(alpha, 0.0, 1.0, out=alpha)
        return SegOutput(alpha=alpha, model=self.name, latency_ms=dt)

    # -- helpers -------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "name": self.name,
            "available": self.available(),
            "loaded": self._loaded,
            "priority": self.priority,
        }
