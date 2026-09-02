"""Chia bộ dữ liệu cốc sứ thành train / val / test.

Điểm mấu chốt: **chia theo nhóm, không chia ngẫu nhiên theo ảnh.**

Cùng một phôi cốc được chụp lại cho nhiều màu. Ảnh ``USACM11/white-front`` và
``USACM11/black-front`` là *cùng một hình dáng*, chỉ khác màu men. Nếu chia ngẫu
nhiên, bản trắng rơi vào train còn bản đen rơi vào test, mô hình chỉ việc nhớ
lại hình dáng đã thấy và điểm IoU trên test sẽ đẹp một cách giả tạo. Nhóm ở đây
là ``(base_code, side)``: mọi màu của cùng một base và cùng một mặt luôn nằm
trọn trong một tập.

Chia có phân tầng thô theo kích thước nhóm để ba tập không lệch nhau quá nhiều,
và có seed cố định để tái lập được.

Chỉ những ảnh đã được người duyệt (``review_status`` = accept hoặc fix) mới được
đưa vào; ảnh ``reject`` bị loại, ảnh chưa duyệt làm script dừng lại - dữ liệu
chưa xong thì không chia.

    python scripts/split_dataset.py
    python scripts/split_dataset.py --ratios 0.7 0.15 0.15 --seed 42
    python scripts/split_dataset.py --allow-unreviewed     # chỉ để thử nhanh
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "mugs"


def load_rows(dataset: Path) -> list[dict]:
    with (dataset / "manifest.csv").open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def assign(groups: list[tuple[str, int]], ratios: tuple[float, float, float],
           seed: int) -> dict[str, str]:
    """Gán từng nhóm vào một tập, cố cho tổng số ảnh sát tỉ lệ mong muốn.

    Duyệt nhóm từ lớn xuống nhỏ và luôn đưa nhóm kế tiếp vào tập đang *thiếu*
    so với hạn mức của nó. Cách này ổn định hơn nhiều so với xáo trộn thuần tuý
    khi số nhóm ít - với ~35 nhóm, một cú xáo xấu có thể làm test chỉ còn 5% dữ
    liệu.
    """
    rng = random.Random(seed)
    order = sorted(groups, key=lambda g: (-g[1], g[0]))
    # Xáo nhẹ trong từng bậc kích thước để seed vẫn có tác dụng mà không phá vỡ
    # thứ tự lớn-trước.
    buckets: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for name, n in order:
        buckets[n].append((name, n))
    order = []
    for size in sorted(buckets, reverse=True):
        chunk = buckets[size]
        rng.shuffle(chunk)
        order.extend(chunk)

    total = sum(n for _, n in groups)
    quota = {"train": ratios[0] * total, "val": ratios[1] * total, "test": ratios[2] * total}
    filled = {"train": 0, "val": 0, "test": 0}
    out: dict[str, str] = {}
    for name, n in order:
        # thiếu hụt tương đối lớn nhất thì được nhận trước
        split = max(quota, key=lambda s: (quota[s] - filled[s]) / max(quota[s], 1e-6))
        out[name] = split
        filled[split] += n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="cho phép chia khi chưa duyệt hết (chỉ để thử)")
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        print(f"Tổng tỉ lệ phải bằng 1.0, đang là {sum(args.ratios)}")
        return 1

    rows = load_rows(args.dataset)
    reviewed = [r for r in rows if (r.get("review_status") or "").strip()]
    todo = len(rows) - len(reviewed)
    rejected = [r for r in rows if r.get("review_status") == "reject"]
    usable = [r for r in rows if r.get("review_status") in ("accept", "fix")]

    print(f"Bộ dữ liệu: {args.dataset}")
    print(f"  tổng ảnh        {len(rows)}")
    print(f"  đã duyệt        {len(reviewed)}   (chưa duyệt {todo})")
    print(f"  loại bỏ         {len(rejected)}")
    print(f"  dùng được       {len(usable)}")

    if todo and not args.allow_unreviewed:
        print(f"\nCòn {todo} ảnh chưa duyệt. Duyệt nốt ở /review rồi chạy lại, "
              f"hoặc dùng --allow-unreviewed để thử.")
        return 2
    if args.allow_unreviewed and todo:
        print("  [cảnh báo] đang chia cả ảnh chưa duyệt - kết quả CHƯA dùng để báo cáo được")
        usable = [r for r in rows if r.get("review_status") != "reject"]

    if not usable:
        print("Không có ảnh nào dùng được.")
        return 1

    sizes = Counter(r["group"] for r in usable)
    mapping = assign(list(sizes.items()), tuple(args.ratios), args.seed)

    for r in usable:
        r["split"] = mapping[r["group"]]

    # ---- ghi kết quả ------------------------------------------------------
    out_dir = args.dataset / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        sel = [r for r in usable if r["split"] == split]
        lines = [f"{r['image']}\t{r.get('label_final') or r['label_auto']}" for r in sel]
        (out_dir / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (args.dataset / "manifest.csv").open(encoding="utf-8-sig", newline="") as fh:
        fields = list(csv.DictReader(fh).fieldnames or [])
    if "split" not in fields:
        fields.append("split")
    by_id = {r["id"]: r for r in usable}
    for r in rows:
        r["split"] = by_id[r["id"]]["split"] if r["id"] in by_id else ""
    tmp = (args.dataset / "manifest.csv").with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(args.dataset / "manifest.csv")

    # ---- báo cáo ----------------------------------------------------------
    print(f"\n{'tập':6s} {'ảnh':>5s} {'%':>6s} {'nhóm':>5s} {'base':>5s}  màu")
    summary = {}
    for split in ("train", "val", "test"):
        sel = [r for r in usable if r["split"] == split]
        colors = sorted({r["color"] for r in sel})
        pct = len(sel) / len(usable) * 100
        print(f"{split:6s} {len(sel):5d} {pct:5.1f}% {len({r['group'] for r in sel}):5d} "
              f"{len({r['base_code'] for r in sel}):5d}  {', '.join(colors[:6])}"
              f"{'...' if len(colors) > 6 else ''}")
        summary[split] = {
            "images": len(sel), "share": round(len(sel) / len(usable), 4),
            "groups": sorted({r["group"] for r in sel}),
            "base_codes": sorted({r["base_code"] for r in sel}),
            "colors": colors,
            "sides": dict(Counter(r["side"] for r in sel)),
        }

    # Kiểm tra rò rỉ: không nhóm nào được nằm ở hai tập.
    seen: dict[str, str] = {}
    leaks = []
    for r in usable:
        if seen.setdefault(r["group"], r["split"]) != r["split"]:
            leaks.append(r["group"])
    print(f"\nkiểm tra rò rỉ nhóm: {'ĐẠT - không nhóm nào nằm ở 2 tập' if not leaks else 'LỖI ' + str(set(leaks))}")

    payload = {
        "dataset": str(args.dataset),
        "policy": "group split by (base_code, side); mọi màu của cùng base+mặt nằm cùng một tập",
        "ratios_requested": list(args.ratios),
        "seed": args.seed,
        "total_images": len(rows),
        "rejected": len(rejected),
        "usable": len(usable),
        "splits": summary,
        "group_leakage": leaks,
    }
    (args.dataset / "split_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {out_dir}\\{{train,val,test}}.txt")
    print(f"  -> {args.dataset}\\split_summary.json")
    return 0 if not leaks else 1


if __name__ == "__main__":
    raise SystemExit(main())
