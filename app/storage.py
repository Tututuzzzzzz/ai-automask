"""Artifact storage.

A local, content-addressed folder per job. Deliberately thin: the only thing the
rest of the codebase knows is ``save_png() -> public URL``, so swapping this for
S3 / GCS in production is a one-file change (return a presigned URL instead of a
static path). Also handles retention so a long-running demo box does not fill up.
"""
from __future__ import annotations

import base64
import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import settings
from .imaging import encode_jpeg, encode_png


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


@dataclass
class JobStore:
    """One output folder, one job (single image or batch)."""

    job_id: str
    root: Path
    inline_base64: bool = False

    @classmethod
    def create(cls, prefix: str = "job_", inline_base64: bool = False) -> JobStore:
        job_id = new_id(prefix)
        root = settings.outputs_dir / job_id
        root.mkdir(parents=True, exist_ok=True)
        return cls(job_id=job_id, root=root, inline_base64=inline_base64)

    @classmethod
    def open(cls, job_id: str) -> JobStore | None:
        root = settings.outputs_dir / job_id
        if not root.is_dir():
            return None
        return cls(job_id=job_id, root=root)

    # ------------------------------------------------------------------ writes
    def save_png(self, name: str, array: np.ndarray) -> str:
        data = encode_png(array)
        return self._write(name if name.endswith(".png") else f"{name}.png", data, "image/png")

    def save_jpeg(self, name: str, array: np.ndarray, quality: int = 88) -> str:
        data = encode_jpeg(array, quality=quality)
        return self._write(name if name.endswith(".jpg") else f"{name}.jpg", data, "image/jpeg")

    def save_json(self, name: str, payload: dict | list) -> str:
        data = json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
        return self._write(name if name.endswith(".json") else f"{name}.json", data, "application/json")

    def save_text(self, name: str, text: str, mime: str = "text/plain") -> str:
        return self._write(name, text.encode("utf-8"), mime)

    def save_bytes(self, name: str, data: bytes, mime: str = "application/octet-stream") -> str:
        return self._write(name, data, mime)

    def _write(self, name: str, data: bytes, mime: str) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if self.inline_base64 and mime.startswith("image/"):
            return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        return f"/artifacts/{self.job_id}/{name}"

    # ------------------------------------------------------------------- reads
    def path_for(self, name: str) -> Path:
        return self.root / name

    def zip_all(self) -> Path:
        """Bundle the job for hand-off to a designer or to the mockup engine."""
        archive = settings.outputs_dir / f"{self.job_id}.zip"
        if archive.exists():
            archive.unlink()
        shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=self.root)
        return archive


def purge_old_jobs(max_age_hours: int | None = None) -> int:
    """Delete job folders older than the retention window. Returns count removed."""
    max_age = (max_age_hours if max_age_hours is not None else settings.retention_hours) * 3600
    if max_age <= 0:
        return 0
    cutoff = time.time() - max_age
    removed = 0
    for child in settings.outputs_dir.iterdir():
        try:
            if child.stat().st_mtime >= cutoff:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            removed += 1
        except OSError:
            continue
    return removed


def disk_usage_mb() -> float:
    total = 0
    for p in settings.outputs_dir.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return round(total / 1e6, 2)
