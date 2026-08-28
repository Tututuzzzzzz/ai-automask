# Integration guide

How to put this in front of an existing Mockup Generator. Written for the backend
engineer who has to make the change, not for the demo.

---

## 1. The shape of the integration

The service is stateless per request and owns no product data. The only thing it
needs is a way to read the image; the only thing it returns is artifacts plus a
verdict. So the smallest useful integration is three calls:

```
your base-ingest worker                  automask                       your DB / CDN
        │                                    │                               │
        │  POST /v1/mask/url                 │                               │
        │  { image_url, category, sku }      │                               │
        ├───────────────────────────────────▶│                               │
        │                                    │ fetch, segment, QC            │
        │  { verdict, confidence,            │                               │
        │    artifacts: {...}, print_area }  │                               │
        │◀───────────────────────────────────┤                               │
        │                                    │                               │
        │  GET  /artifacts/<job>/<file>.png  │                               │
        ├───────────────────────────────────▶│                               │
        │  (PNG bytes)                       │                               │
        │◀───────────────────────────────────┤                               │
        │                                                                    │
        │  store mask + print_area against the SKU, or                       │
        │  enqueue for design review if verdict != READY                     │
        ├───────────────────────────────────────────────────────────────────▶│
```

Recommended handling per verdict:

| Verdict | Do this |
|---|---|
| `READY` | write the mask + print-area geometry against the SKU and publish the base |
| `REVIEW` | store the artifacts, create a design-queue task, attach `overlay` and `trimap` (they show a reviewer exactly where the AI was unsure), and surface `reasons` verbatim — they are written to be read by a human |
| `FAILED` | do not publish. `reasons[0]` distinguishes "no product found" from "confidence below the floor", which usually means the photo needs re-shooting |
| `status: "error"` | a transport/decode problem, not a model problem. Retry with backoff; `error` carries the cause |

## 2. Pick your ingest path

**A. Per-SKU, synchronous** — you already have a worker per base.

```bash
curl -sX POST http://automask:8000/v1/mask/url \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: $AUTOMASK_API_KEY' \
  -d '{"image_url":"https://cdn.example.com/bases/mug-11oz-white.jpg",
       "category":"drinkware","sku":"MUG-11OZ-WHT"}'
```

Latency is 2–4 s, so this is fine inside a job but not inside a web request.

**B. Nightly bulk** — you have a CSV out of the PIM.

```bash
curl -sX POST http://automask:8000/v1/mask/manifest \
  -F 'file=@bases-2026-08.csv' -F 'shadow_maps=true'
```

Any of `image_url` / `url` / `image` / `src` / `link` works as the URL column,
plus optional `category` and `sku`. Delimiter is sniffed; `Image URL` and
`image-url` both match; `T-Shirt`, `mug`, `canvas`, `Phone Case` all map onto the
taxonomy. A row without a URL becomes a warning in the response, not a failure.

**C. Large async batch** — 500 URLs, poll for progress.

```bash
JOB=$(curl -sX POST http://automask:8000/v1/mask/batch/async \
        -H 'Content-Type: application/json' \
        -d @batch.json | jq -r .job_id)
curl -s "http://automask:8000/v1/jobs/$JOB" | jq '{state, processed, total, summary}'
```

**D. Air-gapped / drop folder** — no HTTP at all.

```bash
python scripts/run_folder.py /mnt/incoming/bases --report --workers 2
```

## 3. Compositing with the returned layers

Order matters. Each layer was produced to sit at a specific point in the stack:

```
1. warp the design onto print_area.quad     ← perspective_matrix
2. displace with displacement_map           ← fold geometry, 128 = no shift
3. multiply by shadow_map                   ← creases darken the artwork
4. screen with highlight_map                ← sheen comes back on top
5. mask with alpha_mask                     ← nothing leaks off the product
```

`print_area.perspective_matrix` is a 3×3 homography mapping the **unit square**
`[0,1]²` onto the returned quad, in source-image pixels. So for a design canvas
of any size, normalise to the unit square first and the same matrix applies:

```python
import cv2
import numpy as np

M = np.array(result["print_area"]["perspective_matrix"], np.float64)
design = cv2.imread("artwork.png", cv2.IMREAD_UNCHANGED)      # any size
h, w = design.shape[:2]

# unit-square -> design pixels, then design -> product pixels
S = np.array([[1 / w, 0, 0], [0, 1 / h, 0], [0, 0, 1]], np.float64)
warped = cv2.warpPerspective(design, M @ S, (result["width"], result["height"]),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
```

### libvips / sharp (Node)

```js
const sharp = require('sharp');

async function renderMockup(basePath, artworkBuffer, artifacts) {
  const get = (url) => fetch(`http://automask:8000${url}`).then(r => r.arrayBuffer()).then(Buffer.from);
  const [mask, shadow, highlight] = await Promise.all([
    get(artifacts.alpha_mask), get(artifacts.shadow_map), get(artifacts.highlight_map),
  ]);

  return sharp(basePath)
    .composite([
      { input: artworkBuffer, blend: 'over' },      // already warped to the quad
      { input: shadow,        blend: 'multiply' },
      { input: highlight,     blend: 'screen' },
      { input: mask,          blend: 'dest-in' },   // alpha_mask is greyscale = the alpha
    ])
    .jpeg({ quality: 90 })
    .toBuffer();
}
```

### ImageMagick (batch worker)

```bash
magick base.jpg design_warped.png -compose over      -composite \
       shadow.png                 -compose multiply  -composite \
       highlight.png              -compose screen    -composite \
       mask.png -alpha off        -compose copy_opacity -composite \
       mockup.png
```

### Python / Pillow

```python
from PIL import Image, ImageChops

base = Image.open("base.jpg").convert("RGB")
art = Image.open("design_warped.png").convert("RGBA")
base.paste(art, (0, 0), art)
base = ImageChops.multiply(base, Image.open("shadow.png").convert("RGB"))
base = ImageChops.screen(base, Image.open("highlight.png").convert("RGB"))
out = base.convert("RGBA")
out.putalpha(Image.open("mask.png").convert("L"))
out.save("mockup.png")
```

## 4. Deployment

```yaml
# kubernetes, one GPU per pod
apiVersion: apps/v1
kind: Deployment
metadata: { name: automask }
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: automask
          image: ai-automask:1.0
          ports: [{ containerPort: 8000 }]
          env:
            - { name: AUTOMASK_DEVICE,         value: "cuda" }
            - { name: AUTOMASK_BATCH_WORKERS,  value: "2" }
            - { name: AUTOMASK_OUTPUT_DIR,     value: "/data/outputs" }
            - name: AUTOMASK_API_KEY
              valueFrom: { secretKeyRef: { name: automask, key: api-key } }
          resources:
            limits: { nvidia.com/gpu: 1, memory: 8Gi }
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 25
          volumeMounts: [{ name: outputs, mountPath: /data/outputs }]
```

Rules learned the hard way:

* **One uvicorn worker per GPU.** Two workers each load their own copy of the
  weights and then fight over VRAM. Scale with `replicas`, not `--workers`.
* **Readiness, not liveness, for warmup.** Weights load on a background thread;
  `/health` returns `"warming"` until they are resident and `"ok"` after. A
  liveness probe with a short `initialDelaySeconds` will kill the pod mid-load.
* **Mount `AUTOMASK_OUTPUT_DIR`.** Artifacts are files. Job folders older than
  `AUTOMASK_RETENTION_HOURS` (default 72) are purged at boot.
* **Move artifacts to object storage for anything real.** `app/storage.py` is the
  only module that knows where they live — return a presigned S3 URL from
  `JobStore._write` and nothing else changes. For a small integration,
  `return_base64: true` on `/v1/mask/url` inlines the PNGs and skips the second
  round trip entirely.

## 5. Tuning for your library

The defaults are tuned for "never publish a bad mask". Recalibrate on your own
data — the confidence score correlates with true IoU at ρ = 0.81, so moving the
threshold has a predictable effect:

| Goal | Change |
|---|---|
| More automation, accept occasional touch-ups | `AUTOMASK_READY_THRESHOLD=0.75` |
| Zero tolerance, more review | `AUTOMASK_READY_THRESHOLD=0.90` |
| Faster, mask only | `AUTOMASK_SHADOW_MAPS=0 AUTOMASK_DISPLACEMENT=0` (−47 % latency) |
| CPU-only host | `AUTOMASK_DEVICE=cpu AUTOMASK_BIREFNET_REPO=ZhengPeng7/BiRefNet_lite` |
| Throughput over per-image latency | `AUTOMASK_BATCH_WORKERS=4` |

Do **not** set `AUTOMASK_ENSEMBLE=0` if you act on `READY` automatically. It
saves 0.45 s and removes the signal that guarantees no false accepts.

To validate a threshold change on your own library, label 50 bases and run:

```bash
python scripts/eval_iou.py --folder /path/to/labelled   # needs ground_truth/<stem>_gt.png
```

The `by_verdict` block in the output is the table to read: `READY.min_iou` is the
worst mask you would have published.

## 6. Failure modes to handle on your side

| Symptom | Cause | Handling |
|---|---|---|
| `422` with "cannot fetch" | your CDN URL is private or expired | presign before calling, or use the upload endpoint |
| `422` with "cannot decode" | HEIC / AVIF / a truncated file | convert first; JPEG, PNG, WEBP, BMP, TIFF are supported |
| `413` | file over `AUTOMASK_MAX_UPLOAD_MB` (40) | raise the limit or downscale upstream |
| `422` with "long edge" | image over `AUTOMASK_MAX_SIDE` (6000 px) | raise the limit; memory grows with pixel count |
| `/health` stuck at `"warming"` | weights missing and it is retrying | run `scripts/download_models.py`; check `models` in the health payload for the per-backend error |
| `model_used: "grabcut"` | the neural backends failed to load | check `/health`; the QC verdicts will be honest about the drop in quality, but fix the deployment |
| Everything comes back `REVIEW` | threshold too high for this library, or the cross-check model is missing | inspect `qc_detail.terms` in the sidecar JSON — it shows which signal is dragging the score |
