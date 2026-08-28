# Presentation deck — AI Auto-Masking for Mockup Generator

14 slides, ~10 minutes. Every number here is produced by
`scripts/eval_iou.py` / `scripts/bench.py` and reproducible from the repo.

**The deck is built:** [`docs/deck.html`](deck.html) — open it in a browser, or
view the hosted copy at <https://claude.ai/code/artifact/de9155a0-ef70-4953-8be4-87d163e8383b>.
It is a single self-contained file (no build step, no CDN) with the
confidence-vs-IoU scatter, the stage-time split and the throughput chart drawn
live from the measured data. This file is the speaker script and source of truth
for its content.

---

## 1 — Title

**AI Auto-Masking for Mockup Generator**
Automatic alpha masks, print areas and self-assessed quality for E-commerce & POD bases

> One line to open with:
> *"Anyone can run a segmentation model. The hard part is knowing when to trust
> it — and that is what actually removes the designer from the loop."*

---

## 2 — The bottleneck, as a number

* 30–90 min of Photoshop per new base: cut background → split shadow/highlight → map print coordinates
* 1,000 new bases/month ⇒ **1–2 full-time designers**
* 2–5 day queue blocks the catalogue, not just the design team
* Human error is systematic: clipped sleeves, hair fringe, unnatural folds

**Visual:** the 4-step Photoshop workflow with a clock on each step.

---

## 3 — Why "just use SAM/U-Net" does not solve it

A model **always returns something**.

* No confidence signal ⇒ a designer still opens all 1,000 files ⇒ **the
  bottleneck has not moved**
* Worse: one bad mask reaching a live product page costs more than the review it
  skipped

> So the deliverable is not a mask. It is **a mask plus a trustworthy decision
> about that mask**.

**Visual:** two masks side by side, visually similar, one subtly broken. "Which
one would you ship? Now do it 1,000 times."

---

## 4 — What we built

```
upload / URL / CSV ─▶ decode ─▶ BiRefNet ─▶ cross-check ─▶ classify
                                                              │
       artifacts ◀── lighting maps ◀── print area ◀── QC ◀── refine
                                                    │
                            READY 🟢 / REVIEW 🟡 / FAILED 🔴
```

* FastAPI microservice + review dashboard, Docker-ready
* 4 ingest paths: file upload, direct URL, CSV/JSON manifest, drop folder
* Artifacts: alpha mask, RGBA cut-out, trimap, shadow, highlight, displacement,
  print-area quad + homography, sidecar JSON
* **Every artifact at the exact source resolution** — never resized

**Visual:** the architecture diagram from the README.

---

## 5 — Accuracy

| Metric | Value |
|---|---|
| Mean / median IoU | **0.927 / 0.9985** |
| Mean boundary F1 (±2 px) | 0.874 |
| Alpha MAE / trimap MAE | 0.028 / 0.088 |
| IoU ≥ 0.95 | 79 % of images |
| Category detection | 100 % (19/19) |

Test set: 19 rendered bases **with exact ground truth**, deliberately adversarial —
white-on-white ceramic, hair across a garment, mug on a cluttered desk, canvas in
perspective, translucent bottle, clipped poster, heavy JPEG.

**Visual:** 4 original/mask pairs — mug white-on-white, hair over tee, canvas in
perspective, tote with two handles.

---

## 6 — The slide that matters: routing quality

| Verdict | Share | Mean IoU | **Worst IoU** |
|---|---:|---:|---:|
| 🟢 READY — auto-published | **73.7 %** | **0.9975** | **0.983** |
| 🟡 REVIEW — to a designer | 26.3 % | 0.730 | 0.360 |
| 🔴 FAILED | 0 % | — | — |

* **Zero false accepts** — nothing defective was published
* **Zero missed defects** — all 5 bad masks went to REVIEW
* Confidence ↔ true IoU rank correlation **ρ = 0.81**

> 19 images of designer *work* became 5 images of designer *review*.

**Visual:** scatter plot, confidence (x) vs. true IoU (y), points coloured by
verdict — the READY cluster sits in the top-right corner alone.

---

## 7 — How the AI judges itself (9 no-reference signals)

No ground truth exists at inference time, so every signal comes from the image
and the mask alone:

| Signal | w | Catches |
|---|---|---|
| edge alignment (Sobel field) | .20 | outline not on a real edge |
| **ensemble agreement** | .20 | the primary model's blind spots |
| boundary contrast | .14 | mask cutting through an object |
| ramp sharpness | .12 | grey halo when composited |
| background consistency | .10 | **a sharp cut of the wrong thing** |
| topology / coverage / border / shape prior | .24 | fragmentation, inversion, clipping |

Plus **hard vetoes**: fragmented, frame-clipped, sub-region of a larger object,
atypical silhouette, or the cross-check model disagrees ⇒ never READY.

Every term ships in the response ⇒ the verdict is **auditable**, not a black box.

**Visual:** the dashboard's detail drawer with the signal bars.

---

## 8 — Ablation: what each signal bought

| Configuration | Automation | READY mean IoU | Worst READY | ρ |
|---|---:|---:|---:|---:|
| Baseline QC | 73.7 % | 0.963 | 0.743 | 0.45 |
| + bg-consistency, shape prior, strand suppression | 84.2 % | 0.969 | 0.769 | 0.57 |
| + boundary contrast, calibrated thresholds | 84.2 % | 0.969 | 0.769 | 0.57 |
| + ensemble disagreement as a veto | 73.7 % | 0.9975 | **0.983** | 0.69 |
| + gradient-field edge alignment | 73.7 % | 0.9975 | 0.983 | **0.81** |

**Automation rate is not the objective function.** We gave back 10 % of
automation to buy "worst published mask ≥ 0.98".

---

## 9 — Handling the cases the brief names

* **Hair across a garment** — black-hat detection of thin dark structures inside
  the mask body, then six guards (interior only, darker than the *local* tone,
  thin, long-and-stringy per component, an 8 % budget, locally thick product) so
  folds, dark panels, printed lettering and strap hardware survive.
  **+0.07 boundary F1 on apparel.**
* **Fabric folds** — kept, not smoothed: the fold *shading* becomes the
  displacement and shadow maps instead of contaminating the mask.
* **Cast shadow** — a studio drop shadow has a signature (low alpha, desaturated,
  darker than the backdrop, lighter than a real edge). Detected and removed.
* **Mug handle / tote handles** — structural holes are kept and *counted*; the
  count feeds QC, because 0 holes on a mug is as suspicious as 40 on a canvas.
* **Complex background** — cross-check disagreement flags what it cannot fix.

**Visual:** before/after of the hair case, and the shadow case.

---

## 10 — Performance

| | |
|---|---|
| Mask only | **2.19 s** mean, 2.65 s p95 |
| + all bonus layers | 4.09 s |
| Batch, 2 / 4 workers | **43.6 / 49.6 images/min** |
| Practical throughput | 1,430–2,976 masks/hour |
| Hardware | **RTX 3050 Laptop, 4 GB** — the weakest plausible target |

Time split (mask only): segment 46 %, cross-check 19 %, QC 16 %, refine 13 %.
Biggest lever measured: dropping the bonus layers, **−47 %** (it is PNG encoding,
not compute).

**Visual:** stacked bar of the stage split + the worker-scaling curve.

---

## 11 — Product & integration

* `POST /v1/mask/url` is the integration path: a service posts a URL, gets back a
  verdict and artifact URLs
* Swagger at `/docs`, review dashboard at `/`
* Print area returned as a **quad + 3×3 homography** ⇒ drops straight into
  libvips / ImageMagick / a canvas engine, no mockup-engine changes
* Batch report: automation-rate KPIs, REVIEW queue first, original-vs-mask
  overlays, CSV export, artifact zip
* Optional API key; artifact store is a one-file swap for S3

**Visual:** the dashboard grid with verdict chips, then the batch report header.

---

## 12 — Business case

| | Manual today | With auto-masking |
|---|---|---|
| Designer-hours / 1,000 bases | ~750 h | **~66 h** |
| Wall-clock | 4–6 weeks | **20–42 min GPU** + review as capacity allows |
| Cost / 1,000 bases | ~$9,000 | ~$790 labour + **under $1 compute** |

The bigger win is not the labour line: **the base library stops being a
scheduling constraint.** New SKUs go live the day the photos land; a re-shoot is
a re-run, not a re-do.

---

## 13 — Limits we are not hiding

1. **Semantic ambiguity is unsolvable from pixels** — nothing in a photo says
   whether the frame around a print is part of the product. We detect the
   ambiguity and ask. Fix: category from the PIM, or a fine-tune.
2. **Adjacent touching objects** can be absorbed. Detected, not yet corrected.
3. **Rendered ground truth** is an upper bound on realism. Validate on ~50
   hand-labelled real bases before an SLA.
4. **QC weights are hand-set** (calibrated against measured data, not intuition).
   A learned scorer over the same 9 features would place the threshold optimally.
5. **n = 19** supports the direction, not tight confidence intervals.

---

## 14 — Roadmap for the final round

1. **Fine-tune BiRefNet on the customer's own base library** — a few hundred
   labelled in-domain bases should close most of the remaining gap, especially
   the frame/print ambiguity.
2. **Learn the QC scorer** from real review decisions (same 9 features, GBM):
   turns a conservative threshold into an optimal one, directly lifting the
   automation rate.
3. **TensorRT / ONNX export** — 2–3× on the same GPU; note the stage split says
   the pipeline then becomes CPU-bound, so pair it with more batch workers.
4. **Auto-correct adjacent-object absorption** using the cross-check mask as a
   region prior instead of only as a QC signal.
5. **Photometric-stereo displacement** if the studio can shoot two flash
   positions — turns an estimate into a measurement.

**Closing line:** *"The mask is table stakes. What we are proposing is a base
pipeline that knows when to hand work back to a human — and on our numbers, it
handed back every single one of the bad ones."*
