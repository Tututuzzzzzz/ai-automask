"""Model registry: pick the best available backend, keep it warm, never hard-fail."""
from __future__ import annotations

import logging
import threading

from ..config import settings
from .base import SegmentationBackend
from .birefnet import BiRefNetBackend
from .grabcut import GrabCutBackend
from .sam_refiner import SamRefiner
from .u2net import U2NetBackend

log = logging.getLogger("automask.registry")


class ModelRegistry:
    """Owns backend instances. Thread-safe because FastAPI serves concurrently."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._backends: dict[str, SegmentationBackend] = {
            "birefnet": BiRefNetBackend(),
            "u2net": U2NetBackend(),
            "grabcut": GrabCutBackend(),
        }
        self.sam = SamRefiner()
        self._primary: SegmentationBackend | None = None

    # ------------------------------------------------------------------ access
    def get(self, name: str) -> SegmentationBackend | None:
        return self._backends.get(name)

    def names(self) -> list[str]:
        return list(self._backends)

    def primary(self) -> SegmentationBackend:
        """The highest-priority backend that actually loads.

        Tried in priority order; a load failure demotes the backend permanently
        for this process so we do not pay the failure cost on every request.
        """
        with self._lock:
            if self._primary is not None and self._primary.loaded:
                return self._primary

            preferred = settings.primary_model
            order = sorted(self._backends.values(), key=lambda b: -b.priority)
            if preferred in self._backends:
                order.sort(key=lambda b: (b.name != preferred, -b.priority))

            errors: list[str] = []
            for backend in order:
                if not backend.available():
                    errors.append(f"{backend.name}: unavailable")
                    continue
                try:
                    backend.load()
                    self._primary = backend
                    log.info("primary segmentation backend = %s", backend.name)
                    return backend
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{backend.name}: {type(exc).__name__}: {exc}")
                    log.warning("backend %s failed to load: %s", backend.name, exc)
            raise RuntimeError("No segmentation backend could be loaded -> " + " | ".join(errors))

    def cross_check(self, exclude: str) -> SegmentationBackend | None:
        """An independent second model for the QC ensemble, or None."""
        if not settings.ensemble:
            return None
        with self._lock:
            for name in ("u2net", "grabcut"):
                if name == exclude:
                    continue
                backend = self._backends[name]
                if not backend.available():
                    continue
                try:
                    backend.load()
                    return backend
                except Exception as exc:  # noqa: BLE001
                    log.warning("cross-check backend %s unavailable: %s", name, exc)
            return None

    # ------------------------------------------------------------------ warmup
    def warmup(self) -> dict:
        """Load models at boot so the first API call is not 20 s slower than the rest."""
        import numpy as np

        report: dict = {"primary": None, "cross_check": None, "errors": []}
        dummy = np.full((256, 256, 3), 200, dtype=np.uint8)
        dummy[64:192, 64:192] = 40
        try:
            primary = self.primary()
            primary.predict(dummy)
            report["primary"] = primary.name
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(str(exc))
        try:
            cc = self.cross_check(exclude=report["primary"] or "")
            if cc is not None:
                cc.predict(dummy)
                report["cross_check"] = cc.name
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(str(exc))
        return report

    def describe(self) -> dict:
        return {
            "backends": [b.describe() for b in sorted(self._backends.values(), key=lambda b: -b.priority)],
            "sam_refiner": self.sam.describe(),
            "primary": self._primary.name if self._primary else None,
        }


registry = ModelRegistry()
