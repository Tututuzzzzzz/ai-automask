"""API cho công cụ duyệt nhãn phân vùng.

Nhãn do pipeline AI sinh ra chỉ là *đề xuất*. Để bộ dữ liệu đủ tư cách gọi là
"đã gán nhãn", mỗi ảnh phải qua tay người: nhận, sửa, hoặc loại. Module này là
phần server của vòng lặp đó.

Ba quyết định, và ý nghĩa của từng cái với bộ dữ liệu cuối:

    accept  nhãn máy đúng, chép nguyên sang labels/
    fix     người sửa lại bằng cọ, bản đã sửa ghi vào labels/
    reject  ảnh không dùng được (không phải sản phẩm, mask hỏng nặng) - loại
            khỏi bộ dữ liệu, nhưng vẫn ghi lại lý do để thống kê trung thực

Mọi quyết định ghi thẳng vào ``manifest.csv`` ngay khi bấm, nên đóng trình duyệt
giữa chừng không mất việc. Nhãn đã duyệt luôn được ghi ở đúng độ phân giải ảnh
gốc - kiểm tra lại ở phía server chứ không tin vào phía trình duyệt.
"""
from __future__ import annotations

import base64
import csv
import io
import re
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .config import settings

router = APIRouter(prefix="/v1/labeling", tags=["labeling"])

# Ghi manifest là read-modify-write; hai lần bấm liên tiếp có thể chồng nhau.
_lock = threading.Lock()

DATASET_DIR = settings.data_dir / "mugs"
VALID_STATUS = {"accept", "fix", "reject"}
_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _dataset_root() -> Path:
    if not (DATASET_DIR / "manifest.csv").exists():
        raise HTTPException(
            404,
            f"Chưa có bộ dữ liệu tại {DATASET_DIR}. "
            f"Chạy: python scripts/build_mug_dataset.py",
        )
    return DATASET_DIR


def _read_manifest() -> tuple[list[dict], list[str]]:
    path = _dataset_root() / "manifest.csv"
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return rows, fields


def _write_manifest(rows: list[dict], fields: list[str]) -> None:
    path = _dataset_root() / "manifest.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)          # đổi tên nguyên tử: không bao giờ để lại file dở


def _safe_id(image_id: str) -> str:
    """Chặn path traversal - id đi thẳng vào tên file."""
    if not _ID_RE.match(image_id or ""):
        raise HTTPException(400, "id không hợp lệ")
    return image_id


# ------------------------------------------------------------------ endpoints
@router.get("/dataset")
def get_dataset() -> dict:
    """Toàn bộ manifest cho giao diện duyệt."""
    rows, _ = _read_manifest()
    return {"dataset": DATASET_DIR.name, "root": str(DATASET_DIR), "items": rows}


@router.get("/stats")
def get_stats() -> dict:
    """Tiến độ duyệt - dùng cho báo cáo và để biết khi nào chia tập được."""
    rows, _ = _read_manifest()
    total = len(rows)
    done = [r for r in rows if (r.get("review_status") or "").strip()]
    counts = {s: sum(1 for r in rows if (r.get("review_status") or "") == s)
              for s in sorted(VALID_STATUS)}
    counts["todo"] = total - len(done)
    machine_ready = sum(1 for r in rows if r.get("verdict") == "READY")
    # Tỉ lệ máy đúng: trong số ảnh đã duyệt, bao nhiêu được nhận nguyên xi.
    agreement = (counts["accept"] / len(done)) if done else 0.0
    return {
        "total": total,
        "reviewed": len(done),
        "counts": counts,
        "machine_verdict_ready": machine_ready,
        "human_accept_rate": round(agreement, 4),
        "ready_for_split": counts["todo"] == 0,
    }


@router.get("/file/{image_id}/{kind}")
def get_file(image_id: str, kind: str, auto: int = 0) -> FileResponse:
    """Phục vụ ảnh gốc / nhãn / overlay cho canvas."""
    image_id = _safe_id(image_id)
    root = _dataset_root()
    if kind == "image":
        path = root / "images" / f"{image_id}.jpg"
    elif kind == "overlay":
        path = root / "overlays" / f"{image_id}.jpg"
        if not path.exists():                       # overlay là tuỳ chọn
            path = root / "images" / f"{image_id}.jpg"
    elif kind == "label":
        # Ưu tiên nhãn đã duyệt; ?auto=1 để ép lấy lại bản máy sinh.
        reviewed = root / "labels" / f"{image_id}.png"
        path = (root / "labels_auto" / f"{image_id}.png") if auto or not reviewed.exists() \
            else reviewed
    else:
        raise HTTPException(400, "kind phải là image | label | overlay")
    if not path.is_file():
        raise HTTPException(404, f"không thấy {path.name}")
    # Nhãn đang sửa không được cache, nếu không trình duyệt trả bản cũ.
    return FileResponse(path, headers={"Cache-Control": "no-store"} if kind == "label" else None)


class Decision(BaseModel):
    id: str
    status: str = Field(..., description="accept | fix | reject")
    mask_png: str | None = Field(None, description="data:image/png;base64,... khi status=fix")
    note: str = ""


@router.post("/decide")
def decide(payload: Decision) -> dict:
    image_id = _safe_id(payload.id)
    if payload.status not in VALID_STATUS:
        raise HTTPException(400, f"status phải thuộc {sorted(VALID_STATUS)}")

    root = _dataset_root()
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    auto_path = root / "labels_auto" / f"{image_id}.png"
    out_path = labels_dir / f"{image_id}.png"
    img_path = root / "images" / f"{image_id}.jpg"
    if not img_path.exists():
        raise HTTPException(404, f"không thấy ảnh {image_id}")

    src = cv2.imread(str(img_path))
    if src is None:
        raise HTTPException(500, f"không đọc được ảnh {image_id}")
    h, w = src.shape[:2]

    if payload.status == "accept":
        if not auto_path.exists():
            raise HTTPException(404, "không thấy nhãn máy để nhận")
        mask = cv2.imread(str(auto_path), cv2.IMREAD_GRAYSCALE)

    elif payload.status == "fix":
        if not payload.mask_png:
            raise HTTPException(400, "status=fix cần mask_png")
        raw = payload.mask_png.split(",", 1)[-1]
        try:
            buf = np.frombuffer(base64.b64decode(raw), np.uint8)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"mask_png không giải mã được: {exc}") from exc
        mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise HTTPException(400, "mask_png không phải ảnh hợp lệ")

    else:  # reject - không sinh nhãn
        mask = None

    if mask is not None:
        # Ràng buộc xuyên suốt dự án: nhãn khớp đúng độ phân giải ảnh gốc.
        # Trình duyệt vẽ ở kích thước gốc rồi, nhưng vẫn kiểm ở server.
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # Nhãn huấn luyện là nhị phân: 0 hoặc 255, không có mức xám lơ lửng.
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        cv2.imwrite(str(out_path), mask)
    elif out_path.exists():
        out_path.unlink()          # đổi ý sang reject thì bỏ nhãn cũ đi

    with _lock:
        rows, fields = _read_manifest()
        for extra in ("review_status", "reviewer_note", "label_final"):
            if extra not in fields:
                fields.append(extra)
        hit = None
        for row in rows:
            if row.get("id") == image_id:
                row["review_status"] = payload.status
                row["reviewer_note"] = payload.note
                row["label_final"] = "" if mask is None else f"labels/{image_id}.png"
                hit = row
                break
        if hit is None:
            raise HTTPException(404, f"{image_id} không có trong manifest")
        _write_manifest(rows, fields)

    return {"ok": True, "id": image_id, "status": payload.status,
            "label": hit.get("label_final") or None,
            "coverage": round(float((mask > 127).mean()), 5) if mask is not None else None}


@router.post("/accept-all-ready")
def accept_all_ready() -> dict:
    """Nhận hàng loạt mọi ảnh máy chấm READY và người chưa động tới.

    Có chủ đích: trên ảnh nền trắng studio, verdict READY gần như luôn đúng, và
    bắt người bấm 100 lần liên tiếp chỉ làm họ bấm cho xong. Cách dùng đúng là
    duyệt mắt qua lưới overlay trước, sau đó mới bấm nút này, rồi tập trung công
    sức vào nhóm REVIEW. Thao tác này ghi lại được và đảo ngược được (đổi
    review_status về rỗng).
    """
    root = _dataset_root()
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with _lock:
        rows, fields = _read_manifest()
        for extra in ("review_status", "reviewer_note", "label_final"):
            if extra not in fields:
                fields.append(extra)
        for row in rows:
            if row.get("verdict") != "READY" or (row.get("review_status") or "").strip():
                continue
            image_id = row["id"]
            auto_path = root / "labels_auto" / f"{image_id}.png"
            if not auto_path.exists():
                continue
            mask = cv2.imread(str(auto_path), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(labels_dir / f"{image_id}.png"),
                        np.where(mask >= 128, 255, 0).astype(np.uint8))
            row["review_status"] = "accept"
            row["reviewer_note"] = "bulk-accept: verdict READY"
            row["label_final"] = f"labels/{image_id}.png"
            n += 1
        _write_manifest(rows, fields)
    return {"accepted": n}


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui() -> HTMLResponse:
    path = Path(__file__).parent / "static" / "review.html"
    if not path.exists():
        raise HTTPException(404, "thiếu review.html")
    return HTMLResponse(path.read_text(encoding="utf-8"))
