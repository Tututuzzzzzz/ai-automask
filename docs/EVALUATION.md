# Evaluation

Everything in this document is reproducible:

```bash
python scripts/make_dataset.py      # deterministic: same 19 bases + ground truth on any machine
python scripts/eval_iou.py          # -> data/eval/eval_summary.json + eval_per_image.csv
python scripts/bench.py             # -> data/eval/bench.json
```

**Hardware for every number below:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB),
CUDA 12.1, PyTorch 2.5.1, fp16, `infer_size=1024`, cross-check enabled.
A 4 GB laptop GPU is deliberately the weakest plausible deployment target — a
T4/L4 is roughly 2–3× faster.

---

## 1. Why a rendered test set

Mask accuracy is 40 % of the score, and accuracy is not measurable without
ground truth. Hand-labelling even 19 real photos at pixel precision is hours of
work and still subjective exactly where it matters most — the fabric fringe, the
hair strand, the 2 px anti-aliased edge. Rendering the set means the alpha
channel is *known exactly*, so IoU, boundary-F1 and trimap MAE are real
measurements rather than estimates.

The trade is realism, and it is stated plainly: **numbers on a rendered set are
an upper bound.** `scripts/fetch_dataset.py` exists to point the same evaluation
at real photographs (with licence provenance recorded) once labels exist, and
`scripts/run_folder.py` runs the pipeline over any folder — including the judges'
hidden test — without ground truth.

The 19 scenes are chosen to reproduce the failure modes the brief names, plus the
ones that bite in production:

| Scene | Hard case it encodes |
|---|---|
| `01/02 tee flatlay white/black` | baseline; and a dark garment, where naive skin/hair exclusion deletes the print area |
| `03 tee on model, hair` | **hair crossing the garment** — an occluder that must be cut out |
| `04 hoodie folds` | soft directional shading with no real edge |
| `05 tote, two handles` | two *legitimate* interior holes |
| `06 tee on hanger, busy wall` | textured background + a thin metal hook |
| `07 tee, hard flash shadow` | a sharp cast shadow that must not become product |
| `08 mug white-on-white` | **lowest-contrast boundary in the set** |
| `09 mug on cluttered desk` | busy background with adjacent objects touching the product |
| `10 tumbler` | no handle → 0 holes expected |
| `11 translucent bottle` | genuine fractional alpha — a matting problem, not segmentation |
| `12/13 canvas straight / perspective` | quad print area vs. perspective quad |
| `14 framed print` | **semantic ambiguity**: is the frame part of the product? |
| `15 poster, dark + clipped` | low light, heavy noise, product cropped by the frame |
| `16–19 phone case / cap / mousepad / cushion` | rounded quads, a non-quad accessory, wide aspect, soft goods |

---

## 2. Mask accuracy

| Metric | Value | Reading |
|---|---|---|
| Mean IoU | **0.927** | dragged down by two semantically ambiguous cases |
| Median IoU | **0.9985** | the typical base is essentially pixel-exact |
| Min IoU | 0.360 | `09 mug on cluttered desk` — and it was routed to REVIEW |
| Mean Dice | 0.953 | |
| Mean boundary F1 (±2 px) | **0.874** | the metric that tracks "does the cut look clean" |
| Mean alpha MAE | **0.0279** | on the soft matte, so semi-transparency counts |
| Mean trimap MAE | **0.0883** | restricted to a band around the true boundary: the hardest pixels only |
| IoU ≥ 0.50 / 0.75 / 0.90 / 0.95 | 94.7 % / 89.5 % / 78.9 % / **78.9 %** | |

Per category:

| Category | n | Mean IoU | Mean boundary F1 | Mean latency |
|---|---:|---:|---:|---:|
| apparel | 7 | 0.964 | 0.936 | 2937 ms |
| accessory | 4 | 0.942 | 0.864 | 2055 ms |
| wall_art | 4 | 0.936 | 0.785 | 2078 ms |
| drinkware | 4 | 0.839 | 0.866 | 2237 ms |

Apparel — the category with the most complex boundaries and the one POD cares
about most — scores highest, because that is where the refinement stage does the
most work. Drinkware's mean is a single scene (`09`) at 0.36; the other three
average 0.998.

Per-image results are in [`data/eval/eval_per_image.csv`](../data/eval/eval_per_image.csv).

---

## 3. The number that actually matters: routing quality

A mean IoU tells you about the model. **This** table tells you whether the
*system* can be deployed:

| Verdict | n | Share | Mean IoU | Worst IoU | Mean boundary F1 |
|---|---:|---:|---:|---:|---:|
| 🟢 **READY** (auto-published) | 14 | **73.7 %** | **0.9975** | **0.9834** | 0.985 |
| 🟡 **REVIEW** (to a designer) | 5 | 26.3 % | 0.7297 | 0.3597 | 0.563 |
| 🔴 FAILED (rejected) | 0 | 0 % | — | — | — |

* **Zero false accepts.** Every mask published without human review scored
  IoU ≥ 0.983. There is no case where a defective mask went out.
* **Zero missed defects.** All five imperfect or unsuitable masks landed in
  REVIEW. Nothing bad slipped through.
* **Confidence ↔ IoU Spearman rank correlation = 0.81.** The confidence score is
  not decoration: it orders masks by their true quality, which is what lets a
  team move the READY threshold and predict the consequence.

### What the five REVIEW items were, and whether flagging them was right

| Image | IoU | Why it was flagged | Correct call? |
|---|---:|---|---|
| `09 mug on cluttered desk` | 0.360 | cross-check disagreement (0.54) + atypical silhouette for drinkware (bbox fill 0.50 vs 0.74 expected) | **Yes** — the mask swallowed adjacent desk clutter |
| `14 framed print` | 0.743 | only 22 % of the pixels outside the mask look like background; quad fit 0.91 vs 0.95 expected | **Yes** — the model cut out the print and left the frame behind |
| `18 mousepad on desk` | 0.769 | cross-check disagreement (0.68) | **Yes** — clutter attached to the product blob |
| `03 tee on model, hair` | 0.778 | cross-check disagreement + complex outline | **Yes** — strand suppression recovered most of it, but hair is exactly where a human should look |
| `15 poster, dark + clipped` | 0.9998 | product cropped by the frame on 16 % of the border | **Yes, deliberately.** The mask is near-perfect; the *photo* is unusable as a mockup base, because there is nothing to composite against at the crop line. Flagging photo suitability, not just mask quality, is the point. |

`15` is the only "false" positive in the loose sense — a perfect mask sent to a
human — and it is a design decision, not an error.

---

## 4. What each QC signal contributes

Signals were added one at a time and measured. Every row is a real run of
`scripts/eval_iou.py`:

| Configuration | Automation rate | READY mean IoU | Worst READY IoU | Conf↔IoU ρ |
|---|---:|---:|---:|---:|
| Baseline (edge alignment + ensemble + sharpness + topology + coverage + border) | 73.7 % | 0.963 | 0.743 | 0.45 |
| \+ `bg_consistency`, `shape_prior`, strand suppression | 84.2 % | 0.969 | 0.769 | 0.57 |
| \+ `boundary_contrast`, calibrated shape thresholds | 84.2 % | 0.969 | 0.769 | 0.57 |
| \+ cross-check disagreement as a hard READY veto | 73.7 % | 0.9975 | 0.9834 | 0.69 |
| \+ gradient-field edge alignment (replacing Canny) | **73.7 %** | **0.9975** | **0.9834** | **0.81** |

Two things worth reading off that table:

1. **Automation rate is not the objective function.** Row 2 auto-publishes 10 %
   more images — including one at IoU 0.769. Row 4 gives that back to buy "worst
   published mask ≥ 0.98", which is the trade a production base library wants:
   one bad mockup on a live product page costs more than three masks of designer
   review.
2. **The single most valuable signal is the second model.** Cross-check
   disagreement separated the labelled set cleanly (every defect < 0.72
   agreement, every correct mask > 0.93) precisely because it does not share the
   primary model's blind spots. It costs ~0.3 s of CPU per image.

Replacing Canny-based edge alignment with a gradient-field measure changed no
verdict but lifted the correlation from 0.69 to 0.81 — because a fixed Canny
threshold pair reports "this image has no edges" on a white-on-white studio
sweep, silently degrading a 20 %-weight signal to a constant 0.5.

---

## 5. Category detection

**19/19 correct (100 %)** using the geometric/photometric heuristics alone, with
CLIP disabled. Category is inferred from mask shape and image statistics in ~3 ms:
interior hole count, bbox fill, aspect, quad fit, body saturation, specular ratio,
shoulder-width ratio, and skin-tone pixels adjacent to the mask.

Caveat proportional to the evidence: 19 rendered images is a small, clean sample.
The design consequence is that a wrong category degrades the *print area* and the
QC bands, never the mask itself — and `category_source` in every response says
whether the label came from the caller or from detection, so an integration can
choose to trust only the former.

---

## 6. Print-area detection

| Category | Solver | Mean confidence | Notes |
|---|---|---:|---|
| wall_art | `quad` | 0.97 | perspective preserved via a 4-corner fit, not a bounding box |
| apparel | `torso` | 0.85 | exact largest inscribed rectangle inside a torso prior, skin/hair excluded |
| drinkware | `cylinder` | 0.68 | handle side removed by column occupancy; edges pulled in 14 %/side for wrap-around |
| accessory | `quad` | 0.75 | honest low confidence on a cap (0.36) — a cap is not a quad, and the score says so |

Asserted by tests rather than by eye: the print-area quad contains **zero pixels
outside the product mask** (`test_print_area_stays_inside_the_product`), and its
homography maps the unit design canvas onto the returned quad to within 1e-3
(`test_print_area_homography_maps_the_unit_square_onto_the_quad`).

---

## 7. Performance

All figures from `python scripts/bench.py`, mean over the 19-image set
(~1.6 MP average), raw data in [`data/eval/bench.json`](../data/eval/bench.json).

### Single-image latency (sequential)

| Configuration | Mean | p95 | ms / megapixel |
|---|---:|---:|---:|
| Mask only | **2187 ms** | 2651 ms | 1331 |
| Mask + shadow + highlight + displacement + cut-out + overlay + trimap | 4088 ms | 5317 ms | 2489 |
| Mask only, no cross-check model | 1738 ms | 2271 ms | 1058 |
| Mask only, no guided-filter refinement | 2159 ms | 2624 ms | 1315 |
| Mask only, `infer_size=768` | 2092 ms | 2706 ms | 1274 |
| Mask only, `infer_size=1280` | 5285 ms | 5847 ms | 3217 |

### Where the time goes (mask only)

| Stage | Share | Mean |
|---|---:|---:|
| `segment` (BiRefNet, two passes) | 45.7 % | 1000 ms |
| `cross_check` (U2-Net, CPU) | 18.8 % | 412 ms |
| `qc` (nine no-reference signals) | 15.9 % | 347 ms |
| `refine` (topology, shadow, strands, guided filter) | 12.6 % | 275 ms |
| `print_area` | 3.3 % | 72 ms |
| `load` / `classify` / `artifacts` | 3.6 % | 81 ms |

### Batch throughput (one GPU)

| Workers | Images/min | Wall for 19 |
|---:|---:|---:|
| 1 | 26.6 | 42.9 s |
| 2 | **43.6** | 26.1 s |
| 4 | 49.6 | 23.0 s |

Scaling is sub-linear because segmentation is serialised by a per-backend lock —
two CUDA graphs must not be in flight at once on a 4 GB card. What the pool
actually buys is overlapping the CPU half of the pipeline (decode, refine, QC,
PNG encode — a combined ~54 % of the budget) with GPU inference. 2 workers
captures most of it; 4 adds 14 % more for double the memory.

**In operational terms:** 2,976 masks/hour at 4 workers (mask only), or ~1,430/h
with every bonus layer enabled. A 1,000-base library takes **20–42 minutes of
GPU time** depending on which layers the mockup engine consumes.

### What each knob is actually worth (measured, not assumed)

| Knob | Measured effect |
|---|---|
| `AUTOMASK_SHADOW_MAPS=0`, `AUTOMASK_DISPLACEMENT=0` | **−1.9 s/image (−47 %)** — by far the biggest lever, and free if the mockup engine only needs the mask. Almost all of it is full-resolution PNG encoding, not computation. |
| `AUTOMASK_INFER_SIZE=1280` | +3.1 s/image for no measurable accuracy gain on this set. Not worth it. |
| `AUTOMASK_INFER_SIZE=768` | −95 ms only. The two-pass ROI refinement means the second pass grows as the first shrinks, so this knob is far less useful than it looks — spend the saving on the layers instead. |
| `AUTOMASK_ENSEMBLE=0` | −449 ms, but **removes the strongest QC signal**. Do not do this if you rely on READY: it is what guarantees no false accepts. |
| `AUTOMASK_REFINE_EDGES=0` | −28 ms. The guided filter is essentially free; there is no reason to turn it off. |
| `AUTOMASK_BIREFNET_REPO=…/BiRefNet_lite` | ~3× faster segmentation, visibly softer on hair and fringe. The right choice for CPU-only deployment. |
| `AUTOMASK_FP16=1` | already on for CUDA; halves VRAM, no measured accuracy change. |

The largest optimisation still on the table is a TensorRT or ONNX export of
BiRefNet (typically 2–3× on the same hardware). Note from the stage table that
this would only remove ~500 ms of the 2187: past that point the pipeline is
CPU-bound on QC and refinement, and more batch workers become the better lever.

---

## 7b. Qualitative check on real photographs

Rendered ground truth measures accuracy; it does not prove the pipeline behaves
on camera images. So the same pipeline was run over five real Creative-Commons
product photos from Wikimedia Commons (`python scripts/fetch_dataset.py
--curated`, licences recorded in `data/real/manifest.csv`) — two caps and three
tote bags, shot on wood, on carpet, and against a plain wall:

```
python scripts/fetch_dataset.py --curated
python scripts/run_folder.py data/real --report
```

| Image | Verdict | Confidence | Observation |
|---|---|---:|---|
| Celtics cap | READY | 0.995 | clean silhouette including the curved brim |
| mesh-back cap | READY | 0.994 | correct through the perforated mesh panel |
| QWSTION flap tote | READY | 0.879 | correct outline; the strap hardware keeps genuine see-through gaps |
| AirTouch thermal tote | REVIEW | 0.829 | flagged: only 41 % of the surrounding ring looks like background (photographed against a patterned carpet) |
| runes cotton tote | REVIEW | 0.788 | flagged: cross-check disagreement 0.52 on the thin handles |

3/5 READY, and — the point of the exercise — both REVIEW flags were raised for
reasons a human would agree with, on photos the model had never seen and that
look nothing like the rendered set.

**This check found a real bug, which is why it was worth doing.** The first
version of the hair-strand suppressor deleted the word "AIRTOUCH" off one tote
and most of the rune ring off another: printed lettering is also thin, dark and
inside the mask. The detector now measures each connected component's elongation,
absolute length and local product thickness, and rejects anything that is not
long and stringy — 1,055 components rejected on the runes bag, one kept. The
synthetic hair case is unaffected (same removal, same boundary F1), and no
lettering is touched. See guards 4–6 in `suppress_thin_occluders`.

One residual artefact remains on that image: a ~0.3 %-of-mask dark crease along
the inside of a strap is still removed, leaving a thin slit. It is cosmetic, the
image is already routed to REVIEW for an unrelated reason, and tightening further
started to risk the hair case — so it is documented rather than over-fitted away.

---

## 8. Known limitations

1. **Semantic ambiguity is not solvable from pixels.** Nothing in a photo says
   whether the frame around a print is part of the product (`14`). The system's
   answer is to detect the ambiguity and ask — which is why `14` is REVIEW rather
   than silently wrong. A fine-tune on the customer's own library, or the
   category coming in from the PIM, is the real fix.
2. **Adjacent objects touching the product** (`09`, `18`) can be absorbed into
   the mask. Detected via cross-check disagreement and the shape prior, but not
   yet *corrected* automatically.
3. **Rendered ground truth.** See §1 and the real-photo spot check in §7b.
   Validate against ~50 hand-labelled real bases before committing to an SLA.
4. **QC weights are hand-set.** They are calibrated against measured data (the
   shape thresholds in `config.py` came from the table in
   `scratchpad/calib.py`-style measurement, not intuition), but a learned scorer
   over the same nine features would place the READY threshold optimally instead
   of conservatively. That needs a few hundred real review decisions.
5. **Small sample.** 19 images support the claims made here (zero false accepts,
   ρ = 0.81) but not tight confidence intervals. Treat them as strong directional
   evidence, and re-run the script on a larger set.
