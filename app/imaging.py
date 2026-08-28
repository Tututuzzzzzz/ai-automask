"""Image I/O: decode, orient, fetch, encode.

Everything that could make the output geometry drift away from the input is
handled here, in one place, because "mask must match the source resolution
exactly" is a hard requirement of the brief:

* EXIF orientation is applied **once**, on load, and the resulting pixel
  dimensions become the contract for every artifact. (A phone-shot base with an
  Orientation=6 tag is the classic way a mask ends up rotated 90 degrees against
  the image a downstream engine loads.)
* CMYK / grayscale / palette / 16-bit inputs are normalised to 8-bit RGB.
* An input that already has an alpha channel is flattened onto neutral grey
  rather than black, so a pre-cut PNG does not gain a black halo.
* Nothing else in the codebase is allowed to resize the final mask.
"""
from __future__ import annotations

import io
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import settings

ImageFile_LOAD_TRUNCATED = True
Image.MAX_IMAGE_PIXELS = 300_000_000        # allow big product shots, block decompression bombs

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
USER_AGENT = "ai-automask/1.0 (+mockup-generator)"


class ImageLoadError(RuntimeError):
    pass


def decode_image(data: bytes) -> np.ndarray:
    """Bytes -> HxWx3 uint8 RGB, EXIF-corrected."""
    if not data:
        raise ImageLoadError("empty image payload")
    try:
        pil = Image.open(io.BytesIO(data))
        pil.load()
    except Exception as exc:  # noqa: BLE001
        raise ImageLoadError(f"cannot decode image: {type(exc).__name__}: {exc}") from exc

    try:
        pil = ImageOps.exif_transpose(pil)
    except Exception:  # noqa: BLE001 - malformed EXIF must not fail the request
        pass

    if pil.mode in ("RGBA", "LA", "PA"):
        # Flatten onto neutral grey: black would bias the guided filter and the
        # shadow decomposition along the existing edge.
        pil = pil.convert("RGBA")
        bg = Image.new("RGBA", pil.size, (128, 128, 128, 255))
        pil = Image.alpha_composite(bg, pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    arr = np.asarray(pil, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ImageLoadError(f"unexpected image shape {arr.shape}")

    h, w = arr.shape[:2]
    if max(h, w) > settings.max_side:
        raise ImageLoadError(
            f"image is {w}x{h}; the service is configured to reject anything over "
            f"{settings.max_side}px on the long edge (AUTOMASK_MAX_SIDE)"
        )
    if min(h, w) < 32:
        raise ImageLoadError(f"image is {w}x{h}; too small to be a product base")
    return arr


def load_path(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXT:
        raise ImageLoadError(f"unsupported extension {p.suffix} (allowed: {sorted(SUPPORTED_EXT)})")
    return decode_image(p.read_bytes())


def fetch_url(url: str, timeout: int | None = None) -> np.ndarray:
    """Download a direct image URL. Raises ImageLoadError with a useful message."""
    timeout = timeout or settings.fetch_timeout
    if not str(url).lower().startswith(("http://", "https://")):
        raise ImageLoadError(f"only http(s) URLs are accepted, got: {url}")
    req = urllib.request.Request(str(url), headers={"User-Agent": USER_AGENT, "Accept": "image/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            limit = settings.max_upload_mb * 1024 * 1024
            data = resp.read(limit + 1)
            if len(data) > limit:
                raise ImageLoadError(f"remote image exceeds {settings.max_upload_mb} MB")
    except ImageLoadError:
        raise
    except urllib.error.HTTPError as exc:
        raise ImageLoadError(f"HTTP {exc.code} fetching {url}") from exc
    except Exception as exc:  # noqa: BLE001
        raise ImageLoadError(f"cannot fetch {url}: {type(exc).__name__}: {exc}") from exc
    return decode_image(data)


# ------------------------------------------------------------------- encoding
def encode_png(array: np.ndarray, compression: int = 6) -> bytes:
    """Encode uint8 grayscale / RGB / RGBA to PNG bytes.

    Compression level 6 rather than 9: on a 4000 px mask level 9 costs ~3x the
    CPU for ~4% smaller files, which is the wrong trade for batch throughput.
    """
    if array.ndim == 3 and array.shape[2] == 4:
        buf = array[:, :, [2, 1, 0, 3]]
    elif array.ndim == 3 and array.shape[2] == 3:
        buf = array[:, :, ::-1]
    else:
        buf = array
    ok, enc = cv2.imencode(".png", buf, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return enc.tobytes()


def encode_jpeg(array: np.ndarray, quality: int = 88) -> bytes:
    buf = array[:, :, ::-1] if array.ndim == 3 else array
    ok, enc = cv2.imencode(".jpg", buf, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return enc.tobytes()


def thumbnail(array: np.ndarray, long_edge: int = 512) -> np.ndarray:
    h, w = array.shape[:2]
    if max(h, w) <= long_edge:
        return array
    s = long_edge / float(max(h, w))
    return cv2.resize(array, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
