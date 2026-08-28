"""Download all model weights used by the AI Auto-Masking pipeline.

All models below are open-source with commercial-friendly licenses:
  * BiRefNet / BiRefNet_lite ...... MIT            (ZhengPeng7/BiRefNet)
  * U2-Net (u2net.onnx) .......... Apache-2.0      (xuebinqin/U-2-Net, weights re-hosted by rembg, MIT)
  * MobileSAM (optional) ......... Apache-2.0      (ChaoningZhang/MobileSAM)

Usage:
    python scripts/download_models.py            # primary + fallback
    python scripts/download_models.py --all      # + SAM refiner
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

DIRECT_DOWNLOADS = {
    # name: (url, filename, approx MB)
    "u2net": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        "u2net.onnx",
        176,
    ),
}
OPTIONAL_DOWNLOADS = {
    "mobile_sam": (
        "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        "mobile_sam.pt",
        40,
    ),
}

HF_REPOS = ["ZhengPeng7/BiRefNet_lite", "ZhengPeng7/BiRefNet"]


def _progress(done: int, block: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, done * block * 100 // total)
    sys.stdout.write(f"\r    {pct:3d}%")
    sys.stdout.flush()


def download_file(url: str, dest: Path, approx_mb: int) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  [skip] {dest.name} already present ({dest.stat().st_size/1e6:.0f} MB)")
        return True
    print(f"  [get ] {dest.name} (~{approx_mb} MB) from {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        tmp.replace(dest)
        print(f"\r    done ({dest.stat().st_size/1e6:.0f} MB)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\r    FAILED: {type(exc).__name__}: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def download_hf(repo: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  huggingface_hub not installed - skipping", repo)
        return False
    print(f"  [get ] HF snapshot {repo}")
    try:
        path = snapshot_download(
            repo_id=repo,
            allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.md"],
        )
        print(f"    done -> {path}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also fetch optional models (SAM)")
    ap.add_argument("--skip-hf", action="store_true", help="skip HuggingFace models")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    ok = True
    print("== ONNX / torch checkpoints ==")
    for _name, (url, fn, mb) in DIRECT_DOWNLOADS.items():
        ok &= download_file(url, MODELS / fn, mb)
    if args.all:
        for _name, (url, fn, mb) in OPTIONAL_DOWNLOADS.items():
            download_file(url, MODELS / fn, mb)

    if not args.skip_hf:
        print("== HuggingFace models ==")
        for repo in HF_REPOS:
            download_hf(repo)

    print("\nAll set." if ok else "\nSome downloads failed - pipeline will use available fallbacks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
