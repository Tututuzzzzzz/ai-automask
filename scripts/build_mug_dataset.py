"""Xây dựng bộ dữ liệu cốc sứ có nhãn phân vùng, từ tài nguyên sẵn có của
Mockup Generator.

Bối cảnh: thư mục ``src/main/resources/mask/*/`` của Mockup Generator đã chứa
hàng trăm ảnh chụp phôi cốc sứ thật (1200x1200, nền trắng studio, đủ màu, nhiều
góc). Đây chính là nguồn dữ liệu đúng trọng tâm cho bài toán phân vùng - không
cần đi thu thập lại.

Script làm 4 việc:

1. **Gom và khử trùng lặp.** Nhiều base dùng chung y hệt một file ảnh; nếu không
   khử, cùng một tấm ảnh sẽ rơi vào cả tập train lẫn test và mọi chỉ số đánh giá
   đều vô nghĩa. Khử theo MD5 nội dung file, giữ bản đầu tiên và ghi lại toàn bộ
   nơi xuất hiện để truy vết được nguồn gốc.

2. **Trích siêu dữ liệu từ đường dẫn.** ``USACM11/white-front-model.jpg`` cho ta
   base code ``USACM11``, màu ``white``, mặt ``front``. Ba trường này dùng để chia
   tập theo nhóm ở bước sau.

3. **Sinh nhãn thô bằng pipeline AI** (BiRefNet + tinh chỉnh biên + kiểm định
   chất lượng). Ảnh nền trắng sạch nên phần lớn sẽ đạt READY; công việc còn lại
   của người gán nhãn là *duyệt* chứ không phải *vẽ*.

4. **Xuất bộ dữ liệu chuẩn** kèm manifest, overlay để review, và thống kê verdict.

Nhãn ở đây là *nhãn thô do máy đề xuất*, chưa phải nhãn cuối. Bước duyệt của
người nằm ở ``scripts/review_labels.py``; chỉ sau khi duyệt xong bộ nhãn mới
được coi là ground truth và đem chia tập.

    python scripts/build_mug_dataset.py
    python scripts/build_mug_dataset.py --source "D:/TTTN/mockupgenerator/src/main/resources/mask" --limit 30
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Console Windows mặc định là cp1252 và sẽ ném UnicodeEncodeError khi in tiếng
# Việt. Ép UTF-8 trước mọi lệnh print; errors="replace" để không bao giờ vì log
# mà làm hỏng cả lượt chạy.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.pipeline import ProcessOptions, process_image  # noqa: E402
from app.postprocess import refine  # noqa: E402
from app.segmentation import registry  # noqa: E402
from app.storage import JobStore  # noqa: E402

DEFAULT_SOURCE = Path("D:/TTTN/mockupgenerator/src/main/resources/mask")

# Tên file trong repo có dạng "<màu>-<mặt>-model.jpg"; màu có thể nhiều từ
# ("light-blue"), nên bắt mặt trước rồi phần còn lại là màu.
NAME_RE = re.compile(r"^(?P<color>[A-Za-z][A-Za-z0-9&_-]*)-(?P<side>front|back)-model$",
                     re.IGNORECASE)

# Chuẩn hoá tên màu về một dạng duy nhất (khoá là tên đã bỏ hết dấu gạch).
COLOR_ALIASES = {
    "lightblue": "light-blue",
    "lightyellow": "light-yellow",
    "lightgreen": "light-green",
    "blackwhite": "black-white",
    "black&white": "black-white",
}


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# Vài base chứa asset ghép sẵn tên kiểu "1-Model-Mask.jpg" - đó là ảnh cốc đã
# dán sẵn khối mask trắng lên, không phải ảnh chụp phôi. Chúng lọt vào vì
# Windows so khớp tên không phân biệt hoa thường ("Model" khớp "*model*"), và
# nhãn sinh ra cho chúng là rác. Chỉ nhận file đúng mẫu "<màu>-<mặt>-model".
def parse_meta(path: Path) -> dict | None:
    """base code = tên thư mục, màu + mặt = tên file. None nếu không phải ảnh phôi."""
    m = NAME_RE.match(path.stem)
    if not m:
        return None
    # Nguồn ghi màu không nhất quán: "lightblue" và "light-blue" là một.
    color = m.group("color").lower().replace("_", "-")
    color = COLOR_ALIASES.get(color.replace("-", ""), color)
    return {"base_code": path.parent.name, "color": color, "side": m.group("side").lower()}


def collect(source: Path) -> tuple[list[dict], dict]:
    """Liệt kê + khử trùng lặp. Trả về (danh sách ảnh duy nhất, thống kê)."""
    files = sorted(p for p in source.rglob("*model*.jp*g") if p.is_file())
    by_hash: dict[str, dict] = {}
    aliases: dict[str, list[str]] = defaultdict(list)

    skipped: list[str] = []
    for path in files:
        meta = parse_meta(path)
        if meta is None:
            skipped.append(str(path.relative_to(source)).replace("\\", "/"))
            continue
        digest = file_md5(path)
        rel = str(path.relative_to(source)).replace("\\", "/")
        aliases[digest].append(rel)
        if digest in by_hash:
            continue
        by_hash[digest] = {"path": path, "md5": digest, **meta}

    for digest, entry in by_hash.items():
        entry["used_by_bases"] = sorted({a.split("/")[0] for a in aliases[digest]})
        entry["occurrences"] = len(aliases[digest])

    stats = {
        "files_scanned": len(files),
        "not_product_photos": len(skipped),
        "not_product_examples": skipped[:8],
        "unique_images": len(by_hash),
        "duplicate_files_removed": len(files) - len(skipped) - len(by_hash),
    }
    return list(by_hash.values()), stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help="thư mục mask của Mockup Generator")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "mugs")
    ap.add_argument("--limit", type=int, default=0, help="chỉ xử lý N ảnh đầu (để thử nhanh)")
    ap.add_argument("--category", default="drinkware")
    ap.add_argument("--no-overlay", action="store_true", help="bỏ overlay review cho nhanh")
    args = ap.parse_args()

    if not args.source.is_dir():
        print(f"Không thấy thư mục nguồn: {args.source}")
        return 1

    print(f"Quét {args.source} ...")
    items, stats = collect(args.source)
    print(f"  {stats['files_scanned']} file quét được")
    print(f"  -{stats['not_product_photos']:4d} không phải ảnh phôi (asset ghép sẵn), ví dụ: "
          f"{', '.join(stats['not_product_examples'][:3])}")
    print(f"  -{stats['duplicate_files_removed']:4d} bản sao trùng nội dung")
    print(f"  ={stats['unique_images']:4d} ảnh duy nhất đưa vào bộ dữ liệu")

    if args.limit:
        items = items[: args.limit]
        print(f"  giới hạn còn {len(items)} ảnh")

    out = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels_auto").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    print(f"\nNạp mô hình trên {settings.resolved_device()} ...")
    warm = registry.warmup()
    print(f"  primary={warm.get('primary')}  cross_check={warm.get('cross_check')}")

    store = JobStore.create("mugds_")
    rows: list[dict] = []
    verdicts: Counter = Counter()
    t0 = time.perf_counter()

    for i, item in enumerate(items, start=1):
        src: Path = item["path"]
        # Tên chuẩn hoá, tự mô tả: 001_USACM11_white_front.jpg
        name = f"{i:03d}_{item['base_code']}_{item['color']}_{item['side']}"
        img_dst = out / "images" / f"{name}.jpg"
        shutil.copy2(src, img_dst)

        result = process_image(
            store, source_name=name, path=str(img_dst),
            options=ProcessOptions(
                category=args.category,
                emit_shadow_maps=False, emit_displacement=False,
                emit_trimap=False, emit_cutout=False,
                emit_overlay=not args.no_overlay,
            ),
            slug=name,
        )
        verdicts[result.verdict or "ERROR"] += 1

        mask_src = store.path_for(f"{name}_mask.png")
        if mask_src.exists():
            shutil.copy2(mask_src, out / "labels_auto" / f"{name}.png")
        ov_src = store.path_for(f"{name}_overlay.jpg")
        if ov_src.exists():
            shutil.copy2(ov_src, out / "overlays" / f"{name}.jpg")

        m = result.metrics
        rows.append({
            "id": name,
            "image": f"images/{name}.jpg",
            "label_auto": f"labels_auto/{name}.png",
            "overlay": f"overlays/{name}.jpg",
            "base_code": item["base_code"],
            "color": item["color"],
            "side": item["side"],
            # Nhóm dùng để chia tập: mọi màu của cùng một base + mặt có cùng
            # hình dáng, nên phải nằm cùng một tập, nếu không là rò rỉ dữ liệu.
            "group": f"{item['base_code']}_{item['side']}",
            "width": result.width or "",
            "height": result.height or "",
            "verdict": result.verdict or "",
            "confidence": result.confidence if result.confidence is not None else "",
            "coverage": m.coverage if m else "",
            "ensemble_iou": (m.ensemble_iou if (m and m.ensemble_iou is not None) else ""),
            "holes": m.holes if m else "",
            "solidity": m.solidity if m else "",
            "latency_ms": round(result.timings_ms.get("total", 0.0), 1),
            "occurrences": item["occurrences"],
            "used_by_bases": " ".join(item["used_by_bases"]),
            "source_path": str(src).replace("\\", "/"),
            "md5": item["md5"],
            "reason": (result.reasons[0] if result.reasons else ""),
            # Người duyệt điền: accept | fix | reject  (rỗng = chưa duyệt)
            "review_status": "",
            "reviewer_note": "",
        })

        if i % 10 == 0 or i == len(items):
            print(f"  [{i:3d}/{len(items)}] {name:44s} {result.verdict:6s} "
                  f"{result.confidence if result.confidence is not None else 0:.3f}")

    wall = time.perf_counter() - t0

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": str(args.source),
        "collected": stats,
        "processed": len(rows),
        "category_forced": args.category,
        "verdicts": dict(verdicts),
        "auto_accept_rate": round(verdicts.get("READY", 0) / max(1, len(rows)), 4),
        "groups": len({r["group"] for r in rows}),
        "base_codes": len({r["base_code"] for r in rows}),
        "colors": sorted({r["color"] for r in rows}),
        "wall_s": round(wall, 1),
        "sec_per_image": round(wall / max(1, len(rows)), 2),
        "model": {"primary": warm.get("primary"), "cross_check": warm.get("cross_check"),
                  "device": settings.resolved_device()},
    }
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                            encoding="utf-8")

    print(f"\n=== kết quả ===")
    print(f"  ảnh xử lý        {len(rows)}")
    print(f"  nhóm (base+mặt)  {summary['groups']}")
    print(f"  base code        {summary['base_codes']}")
    print(f"  màu              {len(summary['colors'])}: {', '.join(summary['colors'])}")
    for v, n in verdicts.most_common():
        print(f"  {v:8s}         {n:3d}  ({n / len(rows) * 100:.1f}%)")
    print(f"  thời gian        {wall:.0f}s  ({summary['sec_per_image']}s/ảnh)")
    print(f"\n  -> {out}")
    print(f"  Bước tiếp: python scripts/review_labels.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
