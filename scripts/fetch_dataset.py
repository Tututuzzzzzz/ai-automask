"""Build a *real-photo* test set from Wikimedia Commons.

Why Commons: every file carries machine-readable licence metadata, and the API
is stable and unauthenticated. We record the licence, author and source page for
each download into ``manifest.csv``, so the test set is auditable - which the
competition rules require of any third-party asset.

Two modes:

  # curated: download the exact files listed in data/sources.csv (reproducible)
  python scripts/fetch_dataset.py --curated

  # discover: pull fresh candidates from Commons categories (for expanding the set)
  python scripts/fetch_dataset.py --discover --per-category 12 --out data/candidates

``--curated`` is what CI and the demo use: a fixed list, same bytes every time.
``--discover`` is the tool used to *build* that list.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://commons.wikimedia.org/w/api.php"
UA = "ai-automask/1.0 (POD auto-masking hackathon; dataset builder)"

# Categories that actually contain product-like photographs of POD bases.
DISCOVER_CATEGORIES = {
    "apparel": ["T-shirts", "Tote bags", "Hoodies", "Polo shirts", "Sweatshirts"],
    "drinkware": ["Coffee mugs", "Mugs", "Drinking glasses", "Vacuum flasks", "Water bottles"],
    "wall_art": ["Empty picture frames", "Picture frames", "Stretched canvas"],
    "accessory": ["Baseball caps", "Mouse pads", "Cushions", "Aprons"],
}


def _api(params: dict) -> dict:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _meta_value(extmeta: dict, key: str, default: str = "") -> str:
    raw = (extmeta.get(key) or {}).get("value", default)
    # extmetadata values are HTML fragments; strip tags for a clean CSV.
    import re

    return re.sub(r"<[^>]+>", "", str(raw)).strip()


def _info_for_titles(titles: list[str], width: int) -> list[dict]:
    out: list[dict] = []
    for chunk_start in range(0, len(titles), 20):     # API caps at 50, stay polite
        chunk = titles[chunk_start:chunk_start + 20]
        data = _api({
            "action": "query", "format": "json", "titles": "|".join(chunk),
            "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime",
            "iiurlwidth": str(width),
        })
        for page in (data.get("query", {}).get("pages", {}) or {}).values():
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            ii = infos[0]
            em = ii.get("extmetadata", {}) or {}
            out.append({
                "title": page.get("title", "").replace("File:", ""),
                "url": ii.get("thumburl") or ii.get("url"),
                "full_url": ii.get("url"),
                "width": ii.get("thumbwidth") or ii.get("width"),
                "height": ii.get("thumbheight") or ii.get("height"),
                "mime": ii.get("mime", ""),
                "license": _meta_value(em, "LicenseShortName", "unknown"),
                "author": _meta_value(em, "Artist"),
                "page": ii.get("descriptionurl", ""),
            })
        time.sleep(0.8)
    return out


def discover(per_category: int, width: int) -> list[dict]:
    rows: list[dict] = []
    for category, cats in DISCOVER_CATEGORIES.items():
        for cat in cats:
            try:
                data = _api({
                    "action": "query", "format": "json", "generator": "categorymembers",
                    "gcmtitle": f"Category:{cat}", "gcmtype": "file",
                    "gcmlimit": str(per_category),
                    "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime",
                    "iiurlwidth": str(width),
                })
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] {cat}: {type(exc).__name__}: {exc}")
                time.sleep(2.0)
                continue
            pages = (data.get("query", {}) or {}).get("pages", {}) or {}
            kept = 0
            for page in pages.values():
                infos = page.get("imageinfo") or []
                if not infos:
                    continue
                ii = infos[0]
                if ii.get("mime") not in ("image/jpeg", "image/png"):
                    continue
                if min(ii.get("width", 0), ii.get("height", 0)) < 600:
                    continue
                em = ii.get("extmetadata", {}) or {}
                rows.append({
                    "category": category,
                    "commons_category": cat,
                    "title": page.get("title", "").replace("File:", ""),
                    "url": ii.get("thumburl") or ii.get("url"),
                    "width": ii.get("thumbwidth") or ii.get("width"),
                    "height": ii.get("thumbheight") or ii.get("height"),
                    "license": _meta_value(em, "LicenseShortName", "unknown"),
                    "author": _meta_value(em, "Artist"),
                    "page": ii.get("descriptionurl", ""),
                })
                kept += 1
            print(f"  {cat:28s} -> {kept} candidates")
            time.sleep(1.2)
    return rows


def download(rows: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []
    for i, row in enumerate(rows, start=1):
        url = row.get("url")
        if not url:
            continue
        ext = ".png" if str(url).lower().endswith(".png") else ".jpg"
        safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in row["title"])[:56]
        name = f"{row.get('category', 'base')}_{i:02d}_{safe}{ext}"
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 4096:
            row["file"] = name
            saved.append(row)
            continue
        # Commons rate-limits image fetches aggressively (HTTP 429). Back off and
        # retry rather than dropping the file - a half-downloaded test set is
        # worse than a slow one.
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                if len(data) < 4096:
                    raise RuntimeError(f"suspiciously small payload ({len(data)} bytes)")
                dest.write_bytes(data)
                row["file"] = name
                saved.append(row)
                print(f"  [ok ] {name}  ({len(data)/1000:.0f} kB)")
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 3:
                    wait = 4 * (attempt + 1)
                    print(f"  [wait] {row['title'][:40]}: rate limited, retrying in {wait}s")
                    time.sleep(wait)
                    continue
                print(f"  [err] {row['title']}: HTTP {exc.code}")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  [err] {row['title']}: {type(exc).__name__}: {exc}")
                break
        time.sleep(1.0)
    return saved


def write_manifest(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields = ["file", "category", "title", "license", "author", "page", "width", "height", "url"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nmanifest -> {path}  ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", action="store_true", help="download the files listed in data/sources.csv")
    ap.add_argument("--discover", action="store_true", help="query Commons categories for new candidates")
    ap.add_argument("--per-category", type=int, default=10)
    ap.add_argument("--width", type=int, default=1400, help="thumbnail long-edge to request")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not (args.curated or args.discover):
        args.curated = True

    if args.discover:
        out = args.out or (ROOT / "data" / "candidates")
        print("Discovering Commons candidates...")
        rows = discover(args.per_category, args.width)
        print(f"\n{len(rows)} candidates; downloading to {out}")
        saved = download(rows, out)
        write_manifest(saved, out / "manifest.csv")
        return 0

    sources = ROOT / "data" / "sources.csv"
    if not sources.exists():
        print(f"missing {sources}; run with --discover first")
        return 1
    out = args.out or (ROOT / "data" / "real")
    with sources.open(encoding="utf-8-sig") as fh:
        # Strip the '#' commentary before the CSV reader sees it, otherwise the
        # first comment line is taken as the header row.
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    rows = [r for r in csv.DictReader(lines) if (r.get("title") or "").strip()]
    print(f"Resolving {len(rows)} curated Commons files...")
    infos = {i["title"]: i for i in _info_for_titles([f"File:{r['title']}" for r in rows], args.width)}
    merged: list[dict] = []
    for r in rows:
        info = infos.get(r["title"])
        if not info:
            print(f"  [err] not found on Commons: {r['title']}")
            continue
        merged.append({**info, "category": r.get("category", "auto")})
    saved = download(merged, out)
    write_manifest(saved, out / "manifest.csv")
    print("\nRun the pipeline over it with:\n"
          f"  python scripts/run_folder.py {out} --report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
