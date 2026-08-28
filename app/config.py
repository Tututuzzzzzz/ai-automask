"""Central configuration, environment-driven so the service is 12-factor friendly."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# HuggingFace transformers probes for TensorFlow/Flax at import time. On mixed
# environments that probe can explode on an unrelated numpy ABI mismatch, so we
# pin the framework to PyTorch before anything imports transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "").strip() or default)
    except ValueError:
        return default


@dataclass
class Settings:
    # ---- paths -------------------------------------------------------------
    root: Path = ROOT
    models_dir: Path = field(default_factory=lambda: ROOT / "models")
    outputs_dir: Path = field(default_factory=lambda: Path(os.getenv("AUTOMASK_OUTPUT_DIR", ROOT / "outputs")))
    data_dir: Path = field(default_factory=lambda: ROOT / "data")

    # ---- inference ---------------------------------------------------------
    device: str = os.getenv("AUTOMASK_DEVICE", "auto")            # auto | cuda | cpu
    primary_model: str = os.getenv("AUTOMASK_PRIMARY_MODEL", "birefnet")
    birefnet_repo: str = os.getenv("AUTOMASK_BIREFNET_REPO", "ZhengPeng7/BiRefNet")
    birefnet_fallback_repo: str = "ZhengPeng7/BiRefNet_lite"
    infer_size: int = _env_int("AUTOMASK_INFER_SIZE", 1024)
    use_fp16: bool = _env_bool("AUTOMASK_FP16", True)
    # Tile long-edge threshold: images larger than this get a hi-res second pass
    hires_threshold: int = _env_int("AUTOMASK_HIRES_THRESHOLD", 2000)
    max_side: int = _env_int("AUTOMASK_MAX_SIDE", 6000)            # guard against absurd uploads

    # ---- ensemble / QC -----------------------------------------------------
    ensemble: bool = _env_bool("AUTOMASK_ENSEMBLE", True)          # cross-check with U2-Net
    ready_threshold: float = _env_float("AUTOMASK_READY_THRESHOLD", 0.82)
    review_threshold: float = _env_float("AUTOMASK_REVIEW_THRESHOLD", 0.55)

    # ---- edge refinement ---------------------------------------------------
    refine_edges: bool = _env_bool("AUTOMASK_REFINE_EDGES", True)
    suppress_strands: bool = _env_bool("AUTOMASK_SUPPRESS_STRANDS", True)
    trimap_band: int = _env_int("AUTOMASK_TRIMAP_BAND", 9)
    guided_radius: int = _env_int("AUTOMASK_GUIDED_RADIUS", 8)
    guided_eps: float = _env_float("AUTOMASK_GUIDED_EPS", 1e-4)

    # ---- extras ------------------------------------------------------------
    emit_shadow_maps: bool = _env_bool("AUTOMASK_SHADOW_MAPS", True)
    emit_displacement: bool = _env_bool("AUTOMASK_DISPLACEMENT", True)
    emit_print_area: bool = _env_bool("AUTOMASK_PRINT_AREA", True)

    # ---- service -----------------------------------------------------------
    batch_workers: int = _env_int("AUTOMASK_BATCH_WORKERS", 2)
    fetch_timeout: int = _env_int("AUTOMASK_FETCH_TIMEOUT", 30)
    max_upload_mb: int = _env_int("AUTOMASK_MAX_UPLOAD_MB", 40)
    api_key: str | None = os.getenv("AUTOMASK_API_KEY") or None
    retention_hours: int = _env_int("AUTOMASK_RETENTION_HOURS", 72)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def ensure_dirs(self) -> None:
        for p in (self.models_dir, self.outputs_dir, self.data_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

# Product taxonomy. Each category tunes print-area geometry + QC expectations.
CATEGORIES: dict[str, dict] = {
    "apparel": {
        "label": "Apparel (t-shirt, hoodie, tote)",
        "print_area": "torso",
        "expect_coverage": (0.10, 0.92),
        "expect_holes": True,          # arm gaps / between straps are legitimate holes
        "soft_edges": True,            # hair, fabric fuzz -> keep partial alpha
        "displacement": True,
        "strand_occluders": True,      # hair crossing the garment must be cut out
        # Silhouette plausibility, used by the QC shape prior. A garment fills a
        # good chunk of its bounding box but never all of it (sleeves, hem curve).
        "shape": {"bbox_fill": (0.34, 0.94), "aspect": (0.45, 2.10), "quad_fit_min": None},
    },
    "drinkware": {
        "label": "Drinkware (mug, tumbler, bottle)",
        "print_area": "cylinder",
        "expect_coverage": (0.05, 0.75),
        "expect_holes": True,          # mug handle creates a hole
        "soft_edges": False,
        "displacement": False,
        "strand_occluders": False,
        # A mug/tumbler is a near-rectangular body: high bbox fill, upright-ish.
        "shape": {"bbox_fill": (0.62, 1.00), "aspect": (0.28, 1.90), "quad_fit_min": None},
    },
    "wall_art": {
        "label": "Wall Art (canvas, poster, framed print)",
        "print_area": "quad",
        "expect_coverage": (0.08, 0.95),
        "expect_holes": False,
        "soft_edges": False,
        "displacement": False,
        "strand_occluders": False,
        "shape": {"bbox_fill": (0.78, 1.00), "aspect": (0.25, 4.00), "quad_fit_min": 0.95},
    },
    "accessory": {
        "label": "Accessory (phone case, cap, mousepad)",
        "print_area": "quad",
        "expect_coverage": (0.04, 0.90),
        "expect_holes": True,
        "soft_edges": False,
        # Caps and cushions are fabric and do crease, so they get a displacement
        # map. Rigid goods (drinkware, framed art) do not - a fold map on a
        # ceramic mug is noise the mockup engine would have to ignore.
        "displacement": True,
        "strand_occluders": False,
        "shape": {"bbox_fill": (0.52, 1.00), "aspect": (0.20, 4.50), "quad_fit_min": None},
    },
    "auto": {
        "label": "Auto-detect",
        "print_area": "auto",
        "expect_coverage": (0.03, 0.96),
        "expect_holes": True,
        "soft_edges": True,
        "displacement": True,
        "strand_occluders": False,
        "shape": {"bbox_fill": (0.25, 1.00), "aspect": (0.15, 5.00), "quad_fit_min": None},
    },
}

VERDICT_READY = "READY"
VERDICT_REVIEW = "REVIEW"
VERDICT_FAILED = "FAILED"
