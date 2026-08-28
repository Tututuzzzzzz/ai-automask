# AI Auto-Masking for Mockup Generator

Turns a raw photo of a blank product into everything a mockup engine needs —
a pixel-aligned **alpha mask**, a **print area**, optional **shadow / highlight /
displacement** layers — and, critically, decides for itself whether the result is
good enough to publish: **READY / REVIEW / FAILED**.

Built for the AI Hackathon brief *"Nghiên cứu, ứng dụng AI tự động sinh mặt nạ
(mask) cho base sản phẩm E-commerce & POD"*.

---

## The problem, stated as a number

Adding one product base today costs a designer **30–90 minutes** in Photoshop:
cut the background, separate the shadow and highlight layers, map the print
coordinates. At a thousand new bases a month that is **one to two full-time
designers doing work no one wants to do**, and it is the reason the base library
cannot keep up with the catalogue.

Segmentation models are not the hard part of fixing this — good open ones exist.
The hard part is that **a model always returns something**. Without a trustworthy
way to tell a good mask from a bad one, a designer still has to open all 1,000
files, and the bottleneck has not moved at all.

So this system is built around one claim, and the claim is measured:

> **Nothing defective is auto-published.** On the labelled test set, every mask
> the system marked READY scored **IoU ≥ 0.983** against ground truth (mean
> 0.9975), and **every** defective or unsuitable mask landed in the REVIEW queue.
> That turns 19 images of designer work into 5 images of designer review.

---

## Results at a glance

Measured by `scripts/eval_iou.py` on the 19-image ground-truth set, on a laptop
RTX 3050 (4 GB). Full detail and method: [docs/EVALUATION.md](docs/EVALUATION.md).

| Metric | Value |
|---|---|
| Mean IoU / median IoU | **0.927 / 0.9985** |
| Mean boundary F1 (±2 px) | **0.874** |
| Mean alpha MAE / trimap MAE | **0.028 / 0.088** |
| Images with IoU ≥ 0.95 | 79 % |
| **Automation rate (READY)** | **73.7 %** |
| **Mean IoU of READY masks** | **0.9975** (worst 0.983) |
| Mean IoU of REVIEW masks | 0.730 (worst 0.360) |
| Confidence ↔ IoU rank correlation | **0.81** |
| Product-category accuracy | **100 %** (19/19) |
| Mean end-to-end latency | **2.19 s** mask only, 4.09 s with every bonus layer |
| Batch throughput | **43.6 images/min** (2 workers), 49.6 (4 workers), one 4 GB GPU |

The set is deliberately adversarial: white-on-white ceramic, hair falling across
a garment, a mug on a cluttered desk, a canvas in perspective, a translucent
bottle, a poster clipped by the frame, heavy JPEG artefacts.

Rendered ground truth is what makes those numbers measurements rather than
impressions — but it is not proof of real-world behaviour, so the pipeline is
also run over real Creative-Commons product photographs
(`docs/EVALUATION.md` §7b): 3/5 READY, and both REVIEW flags raised for reasons
a human would agree with. That check is also what caught a real bug — the
hair-strand remover was deleting printed lettering off tote bags — which is now
fixed and regression-tested.

---

## Quickstart

```bash
# 1. install (pick the torch wheel for your hardware first - see requirements.txt)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. fetch weights (~600 MB: BiRefNet MIT + U2-Net Apache-2.0)
python scripts/download_models.py

# 3. run the service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** for the review dashboard, or
**http://localhost:8000/docs** for Swagger.

```bash
# or in Docker (weights baked into the image, works air-gapped)
docker compose up --build
```

Offline / CLI paths:

```bash
python scripts/make_dataset.py                      # generate the 19-image test set + ground truth
python scripts/run_folder.py data/samples --report  # process a folder, write report.html
python scripts/eval_iou.py                          # IoU / boundary-F1 / routing quality
python scripts/bench.py                             # latency + throughput sweep
python scripts/fetch_dataset.py --curated           # pull real CC photos (licences recorded)
python scripts/export_deliverable.py                # package test set + outputs
pytest -q                                           # 57 contract tests, no GPU needed
```

`make help` lists the same things as make targets.

---

## Architecture

```
                 ┌──────────────────────────────────────────────┐
  upload ───────▶│  FastAPI  (app/main.py)                      │
  image URL ────▶│  /v1/mask · /v1/mask/url · /v1/mask/batch     │
  CSV / JSON ───▶│  /v1/mask/manifest · /v1/mask/upload-batch    │
                 └───────────────────────┬──────────────────────┘
                                         ▼
                        ┌────────────────────────────────┐
                        │  pipeline.py  (one image)      │
                        └────────────────────────────────┘
   ┌──────────┬────────────┬──────────┬───────────┬─────────┬──────────┬─────────┐
   │ 1 decode │ 2 segment  │ 3 cross- │ 4 classify│5 refine │ 6 QUALITY│7 print  │
   │  + EXIF  │  BiRefNet  │   check  │  category │  edges  │   CHECK  │  area   │
   │          │  (MIT)     │  U2-Net  │           │         │          │         │
   └──────────┴────────────┴──────────┴───────────┴─────────┴────┬─────┴────┬────┘
                                                                 ▼          ▼
                                            ┌────────────────────────┐  ┌──────────┐
                                            │ READY  → auto-publish  │  │ 8 shadow │
                                            │ REVIEW → designer queue│  │ highlight│
                                            │ FAILED → rejected      │  │ displace │
                                            └────────────────────────┘  └──────────┘
                                                          │
                                       artifacts (PNG at source resolution)
                                       + sidecar JSON + batch report + CSV
```

### Stage by stage

**1 · Decode** (`app/imaging.py`) — EXIF orientation is applied exactly once, on
load, and the resulting pixel dimensions become the contract for every artifact.
CMYK / palette / 16-bit inputs normalise to 8-bit RGB; an input that already has
alpha flattens onto neutral grey rather than black, so a pre-cut PNG gains no
dark fringe. **Nothing downstream is allowed to resize the mask.**

**2 · Segment** (`app/segmentation/birefnet.py`) — BiRefNet, MIT-licensed and
state of the art on high-resolution dichotomous segmentation, which is exactly
the "pixel-perfect boundary" problem being graded. Two quality tricks: a
**two-pass ROI refinement** (pass 1 whole frame, pass 2 on a padded crop around
the detected product, feather-stitched — a garment occupying 40 % of a 4000 px
photo effectively gets 2.5× the sampling density on its own boundary), and
optional horizontal-flip TTA.

The backend is one implementation of a registry interface with three members —
BiRefNet → U2-Net (ONNX, Apache-2.0) → GrabCut (pure OpenCV, always available).
A missing weight file or an OOM demotes to the next one instead of failing the
request, and the QC stage honestly reports the lower quality that results.

**3 · Cross-check** — U2-Net runs as a second, architecturally independent
opinion, at ~0.3 s on CPU. It is not there for accuracy; it is the strongest
input to the QC score. Two unrelated models agreeing on a boundary is the closest
thing to ground truth available at inference time.

**4 · Classify** (`app/postprocess/classify.py`) — the category selects the
print-area solver, the plausible coverage band and the edge-hardness curve. When
the caller knows it (`category=drinkware`) we trust them; otherwise geometric and
photometric heuristics infer it in ~3 ms: mug handles punch exactly one interior
hole and produce hard specular streaks on a desaturated body; canvases are near
perfect quads at 0.99 solidity; worn apparel shows a shoulder-width bulge and
skin-toned pixels next to the mask. CLIP zero-shot is available behind
`AUTOMASK_CLIP=1` and fuses 60/40 with the geometry when enabled.

**5 · Refine** (`app/postprocess/refine.py`) — where boundary quality is actually
won, all at full source resolution:

* topology cleanup — drop specks as a *ratio* of the main blob, keep structural
  holes (mug handle, tote handles, arm gaps) and count them for QC;
* **cast-shadow suppression** — a studio drop shadow has a specific signature
  (low alpha, desaturated, darker than the backdrop but not as dark as a real
  product edge). Detected and removed, so the mockup does not composite artwork
  onto a shadow;
* **hair-strand suppression** — the brief names *"tóc model vắt ngang"*
  explicitly. A morphological black-hat finds thin dark structures inside the
  garment body, then six guards decide what to remove (interior only, clearly
  darker than the *local* garment tone, thin, long-and-stringy per component,
  within an 8% budget, and lying on a locally thick part of the product) so
  folds, dark panels, printed lettering and strap hardware all survive. Worth
  +0.07 boundary F1 on the apparel subset;
* **guided-filter matting inside the trimap band** — a locally-implemented
  He et al. guided filter (no opencv-contrib dependency) recovers the boundary
  detail lost when a 1024 px matte is upsampled to 4000 px. Band width scales
  with resolution;
* a **category-aware contrast curve** — a canvas must not ship a 20 px alpha
  gradient; hair and knit fringe must keep theirs.

**6 · Quality check** (`app/postprocess/quality.py`) — see below.

**7 · Print area** (`app/postprocess/printarea.py`) — not the same problem as the
mask. Per category: `quad` fits a 4-corner polygon to the silhouette so
perspective survives; `torso` restricts to a torso prior, subtracts skin and hair
pixels, then solves the **exact largest inscribed rectangle** (maximal-rectangle
DP) so the box can never overhang a sleeve; `cylinder` removes the handle side via
column occupancy and pulls the edges in to respect a cylinder's wrap-around
falloff. Output is a quad **plus the 3×3 homography** mapping a unit design canvas
onto it — directly consumable by libvips / ImageMagick / a canvas engine.

**8 · Lighting maps** (`app/postprocess/maps.py`, bonus track) — an
intrinsic-image split, `L = albedo × shading`, where albedo is an *alpha-weighted*
low-pass so the backdrop never bleeds into the estimate (that bleed is why naive
implementations show a bright halo at the product edge). `shadow = clamp(1 −
shading)` as a multiply layer, `highlight = clamp(shading − 1)` as a screen
layer, and the mid-frequency residual as a displacement map in the standard
128-is-neutral encoding.

---

## The Quality Check — the part that makes this deployable

No ground truth exists at inference time, so every signal is **no-reference**,
computed from the image and the mask alone. Weights in
`app/postprocess/quality.py`:

| Signal | w | What it catches |
|---|---|---|
| `edge_alignment` | .20 | Does the outline sit on a real photometric edge? Measured on the Sobel field against the image's own median gradient — scale-free, so it works on a white-on-white sweep *and* a lifestyle shot (a fixed Canny pair does not). |
| `ensemble` | .20 | Does an independent architecture agree, on region (IoU) and on edge placement (boundary F1)? |
| `boundary_contrast` | .14 | Is there a colour step *across* the outline, relative to the local noise floor? Catches a mask that has swallowed a neighbouring object — where it crosses that object, inside and outside look identical. |
| `sharpness` | .12 | Uncertain pixels ÷ boundary length = mean ramp width. 1–3 px is correct anti-aliasing; >12 px composites as a grey halo. |
| `bg_consistency` | .10 | Do the pixels just *outside* the mask look like the backdrop (robust Lab median/MAD from the frame border, chroma-matched so a drop shadow is not mistaken for a different material)? This is the only signal that catches a **sharp, well-aligned cut of the wrong thing** — the print inside a picture frame. |
| `topology` | .10 | Fragmentation, hole count vs. category expectation, solidity. |
| `coverage` | .07 | Plausible product-to-frame ratio for the category. |
| `border` | .05 | A product running off the frame cannot be a clean base. |
| `shape_prior` | .02 | Class-conditional silhouette sanity, thresholds calibrated from measured data, not guessed. |

Plus **hard vetoes**: whatever the aggregate score says, a mask never reads READY
when it is fragmented, clipped by the frame, appears to be a sub-region of a
larger object, has an atypical silhouette for its class, cuts through an object,
or when the cross-check model materially disagrees. Model disagreement turned out
to be the single sharpest separator on the labelled set — every defective mask
scored below 0.72 agreement, every correct one above 0.93 — and it does not share
the primary model's blind spots.

Every term, its weight and its contribution ships in the response and in the
sidecar JSON, so a verdict is auditable rather than a black-box number. That is
also what the dashboard's detail drawer renders.

**Why this is not a pass-through wrapper around a third-party API** (competition
rule 10.2): the quality decision is computed here, from nine independent signals,
one of which is a second model we run ourselves. Swap BiRefNet for remove.bg and
the QC layer is unchanged and still does its job.

---

## Inputs and outputs

### Four ways in

| Method | Endpoint | Notes |
|---|---|---|
| Upload 1 file | `POST /v1/mask` | multipart; `category`, `sku`, layer flags as form fields |
| Upload N files | `POST /v1/mask/upload-batch` | |
| Direct image URL | `POST /v1/mask/url` | **the integration path** — another service posts a URL, we fetch |
| URL list (JSON) | `POST /v1/mask/batch` · `/batch/async` | sync, or poll `/v1/jobs/{id}` |
| CSV / JSON manifest | `POST /v1/mask/manifest` | tolerant ingest: `image_url\|url\|image\|src`, `category`, `sku`; sniffs `,` `;` tab `\|`; normalises `Image URL` / `image-url`; maps `T-Shirt`→apparel, `mug`→drinkware… ; a bad row becomes a warning, not a 500 |
| Bundled sample set | `POST /v1/mask/samples` | runs `data/samples/` server-side — one click in the UI |

### What comes back

```jsonc
{
  "id": "001_mug-11oz", "source": "https://cdn/…/mug.jpg", "status": "ok",
  "verdict": "READY", "confidence": 0.9821,
  "reasons": ["Clean single-piece mask, edges align with the source image, models agree."],
  "category": "drinkware", "category_source": "metadata",
  "width": 2400, "height": 2400,
  "model_used": "birefnet", "models_tried": ["birefnet", "u2net"],
  "timings_ms": { "load": 41, "segment": 1043, "refine": 118, "qc": 63, "total": 2210 },
  "metrics": { "coverage": 0.203, "edge_sharpness": 0.94, "uncertain_ratio": 0.0058,
               "ensemble_iou": 0.972, "holes": 1, "solidity": 0.97, "border_contact": 0.0 },
  "print_area": { "kind": "cylinder", "quad": [[…]], "bbox": [x,y,w,h],
                  "perspective_matrix": [[…]], "confidence": 0.66 },
  "artifacts": { "alpha_mask": "/artifacts/<job>/001_mug_mask.png",
                 "cutout_rgba": "…_cutout.png", "overlay": "…_overlay.jpg",
                 "trimap": "…_trimap.png", "shadow_map": "…_shadow.png",
                 "highlight_map": "…_highlight.png", "displacement_map": "…_displacement.png" }
}
```

Artifacts, all at the **exact source resolution**:

| File | Format | Purpose |
|---|---|---|
| `*_mask.png` | 8-bit greyscale | **the deliverable** — feed to libvips / ImageMagick / canvas |
| `*_cutout.png` | RGBA | product on transparency, edge colour bled outward to kill white fringing |
| `*_overlay.jpg` | RGB | review view: checkerboard, verdict-tinted product, traced outline, print-area quad |
| `*_trimap.png` | 8-bit | where the AI was *unsure* — tells a designer exactly where to look |
| `*_shadow.png` / `*_highlight.png` | 8-bit | multiply / screen layers |
| `*_displacement.png` | 8-bit, 128 = neutral | fold geometry for a mesh warp |
| `*_meta.json` | JSON | full QC breakdown, print-area geometry, model provenance |

Batch also produces `batch_results.json`, `batch_results.csv` (one row per base,
importable into a sheet) and a self-contained `report.html` — automation-rate
KPIs, then the REVIEW queue first, then original-vs-mask-vs-overlay for every
image. It references artifacts relatively, so zipping the job folder keeps it
working.

---

## Integrating with the mockup generator

The composite recipe, in the order the layers were designed for:

```
1. warp the design onto print_area.quad     (perspective_matrix from the response)
2. displace it with displacement_map        (folds)
3. multiply by shadow_map                   (creases go dark)
4. screen with highlight_map                (sheen returns)
5. mask with alpha_mask                     (nothing leaks off the product)
```

libvips (Node), the fastest path for a web backend:

```js
const sharp = require('sharp');
const { alpha_mask, shadow_map, highlight_map } = maskResult.artifacts;
const [[a,b,c],[d,e,f],[g,h,i]] = maskResult.print_area.perspective_matrix;

const art = await sharp('design.png').resize(1024, 1024, { fit: 'fill' })
  .affine(/* … or use a perspective transform node with the matrix above … */)
  .toBuffer();

await sharp('base.jpg')
  .composite([
    { input: art,                  blend: 'over' },
    { input: await fetch(shadow_map).then(r => r.buffer()),    blend: 'multiply' },
    { input: await fetch(highlight_map).then(r => r.buffer()), blend: 'screen' },
    { input: await fetch(alpha_mask).then(r => r.buffer()),    blend: 'dest-in' },
  ])
  .toFile('mockup.jpg');
```

ImageMagick, for a batch worker:

```bash
magick base.jpg design_warped.png -compose over -composite \
       shadow.png    -compose multiply -composite \
       highlight.png -compose screen   -composite \
       mask.png      -alpha off -compose copy_opacity -composite mockup.png
```

Operational notes for a real deployment:

* **One uvicorn worker per GPU.** Two workers each load their own copy of the
  weights and fight over VRAM; scale out with replicas.
* `AUTOMASK_API_KEY` turns on `X-API-Key` on every `/v1` route; `/health` stays
  open for probes.
* `storage.py` is the only place that knows where artifacts live — returning a
  presigned S3 URL instead of a static path is a one-file change.
* Job folders older than `AUTOMASK_RETENTION_HOURS` (default 72) are purged at
  boot.
* Errors inside a batch are **rows**, not HTTP failures: one dead CDN URL out of
  500 must not force a re-run of the whole batch.

Every knob is an environment variable — see [.env.example](.env.example).

---

## Is this worth deploying? (business feasibility)

**Where the time goes today.** 30–90 min of skilled Photoshop work per base;
call it 45 min at a blended $12/h ≈ **$9 per base**, plus a 2–5 day queue that
blocks the catalogue.

**What this changes.** At the measured 73.7 % automation rate on a deliberately
hard set (real studio libraries are easier — most bases are a product on a clean
sweep, which the system handles at IoU > 0.99), 1,000 bases/month becomes:

| | Manual today | With auto-masking |
|---|---|---|
| Designer-hours / 1,000 bases | ~750 h | ~66 h (263 REVIEW items × ~15 min touch-up) |
| Wall-clock for 1,000 bases | 4–6 weeks | **20–42 min of GPU time** (measured: 2,976 masks/h mask-only, 1,430/h with all layers) + review as capacity allows |
| Cost / 1,000 bases | ~$9,000 | ~$790 labour + well under $1 of compute |
| Marginal cost per base | ~$9 | **fractions of a cent** — and the measurements above are from a *4 GB laptop* GPU |

**Where the savings really come from** — it is not only the labour. It is that a
base library stops being a scheduling constraint. New SKUs go live the day the
photos land, seasonal drops stop needing a design sprint, and a re-shoot is a
re-run rather than a re-do.

**What it costs to run.** The measured 1,430–2,976 masks/hour comes from a
*4 GB laptop* GPU, so a modest cloud instance is comfortable and one GPU covers
a thousand bases a month many times over. Turning off the bonus layers halves
the per-image cost when the mockup engine does not use them (measured: −47 %).
CPU-only works with `BiRefNet_lite` if no GPU is available. Everything is
stateless behind the job store, so it scales horizontally without coordination.

**Honest risks.**

* The REVIEW bucket needs a human in the loop; the value proposition is
  *reduction*, not elimination, of designer work. Budget for a reviewer.
* The 26 % review rate here is set by an adversarial test set. Recalibrate
  `AUTOMASK_READY_THRESHOLD` on your own library — the confidence↔IoU rank
  correlation of 0.81 is what makes that recalibration meaningful rather than
  arbitrary.
* Semantic ambiguity is a genuine limit: nothing in the image tells the model
  whether the frame around a print is part of the product. Feed the category in
  from your PIM when you have it (`category_source: "metadata"`) — that is free
  accuracy.
* Any accuracy claim on a *rendered* test set is an upper bound on realism.
  Re-run `scripts/eval_iou.py` against 50 hand-labelled real bases before
  committing to an SLA.

---

## Repo layout

```
app/
  main.py               FastAPI service, 15 endpoints, Swagger
  pipeline.py           the per-image orchestration
  batch.py              CSV/JSON ingest, thread pool, automation summary
  imaging.py            decode / EXIF / fetch / encode - the resolution contract
  storage.py            artifact store (swap for S3 here)
  report.py             self-contained HTML batch report
  config.py             env-driven settings + the product taxonomy
  schemas.py            pydantic contracts (drives the Swagger docs)
  segmentation/         base.py · birefnet.py · u2net.py · grabcut.py
                        sam_refiner.py (optional) · registry.py
  postprocess/          refine.py · quality.py · printarea.py
                        maps.py · classify.py
  static/               the review dashboard (no framework, no CDN)
deliverables/
  sample_run/           the submitted test set + the system's output for it
                        (masks, overlays, sidecar JSON, batch report, CSV)
scripts/
  download_models.py    fetch weights
  make_dataset.py       render the 19-base test set + ground truth
  fetch_dataset.py      pull real photos from Wikimedia Commons (licence-recorded)
  run_folder.py         offline batch over a folder
  eval_iou.py           IoU / boundary-F1 / routing quality / correlation
  bench.py              latency + throughput sweep
  export_deliverable.py package the test set + outputs for submission
tests/                  57 tests: resolution contract, EXIF, ingest, QC, API
docs/
  EVALUATION.md         method, per-image results, ablation, benchmarks
  INTEGRATION.md        how to wire this into an existing mockup generator
  SLIDES.md             the 14-slide deck as speaker notes
  deck.html             the built deck, self-contained (open in a browser)
  DEMO_SCRIPT.md        shot-by-shot script for the 3-5 minute demo video
```

## Submission deliverables

| Asked for | Where it is |
|---|---|
| Repository with source + model weights | this repo; `python scripts/download_models.py` fetches the weights (licence-checked, not redistributed) |
| README describing architecture, workflow, business feasibility | this file (~10 min read) |
| Demo video walkthrough | script and shot list in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) |
| Test dataset of ≥ 15 diverse bases + system output | [`deliverables/sample_run/`](deliverables/sample_run/) — 19 bases, their masks, overlays, sidecar JSON, bonus layers for four showcases, batch report and CSV. Regenerate with `python scripts/export_deliverable.py` |
| Presentation slides with performance evaluation | [docs/deck.html](docs/deck.html) (built, self-contained) · notes in [docs/SLIDES.md](docs/SLIDES.md) · hosted copy: https://claude.ai/code/artifact/de9155a0-ef70-4953-8be4-87d163e8383b |
| UI/Dashboard or Swagger for judges to test | both: dashboard at `/`, Swagger at `/docs` |

## Licensing (competition rule 10.3)

Every default component is licensed for commercial use: **BiRefNet — MIT**,
**U2-Net — Apache-2.0**, OpenCV/PyTorch/transformers/timm/kornia — Apache-2.0 or
BSD, FastAPI/einops/onnxruntime/Pillow — MIT. Weights are downloaded, not
redistributed. The one AGPL-adjacent component (the `ultralytics` loader for the
optional MobileSAM refiner) is **disabled by default** with the reasoning written
into `app/segmentation/sam_refiner.py`. Full inventory: [LICENSE](LICENSE).

No manual intervention exists anywhere in the path (rule 10.1): the only
human-facing step is *reviewing* what the AI produced, and the review queue is
chosen by the AI.

## Roadmap

1. **Fine-tune on the customer's own base library.** BiRefNet is a generic
   saliency model; a few hundred labelled in-domain bases would mostly close the
   remaining gap, particularly on the frame/print ambiguity.
2. **Learn the QC scorer.** The nine signals are currently combined with hand-set
   weights. With a few hundred (mask, IoU) pairs from real review decisions, a
   gradient-boosted model on the same features would place the READY threshold
   optimally instead of conservatively.
3. **TensorRT / ONNX export of BiRefNet** — 2–3× on the same GPU, and the batch
   worker becomes GPU-bound rather than latency-bound.
4. **True displacement from photometric stereo** if the studio can shoot two
   flash positions; the current single-image estimate is a good approximation,
   not a measurement.
