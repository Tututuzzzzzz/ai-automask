# Demo video script (3–5 minutes)

Target: **4:20**. Recorded at 1920×1080, browser at 100 % zoom, dark OS theme so
the dashboard's checkerboard reads clearly.

## Before you hit record

```bash
python scripts/download_models.py            # weights present
python scripts/make_dataset.py               # test set present
rm -rf outputs/*                             # clean job history in the UI
uvicorn app.main:app --port 8000              # wait for /health -> "ok"
```

Have ready, in tabs:
1. `http://localhost:8000` (dashboard)
2. `http://localhost:8000/docs` (Swagger)
3. a terminal in the repo root
4. a file manager on `data/samples/` for the drag-and-drop shot

---

## 0:00–0:25 — The problem (talking head or slide 2)

> "To preview a customer's design on a t-shirt, our mockup generator needs three
> things for every product base: a transparent alpha mask, a print area, and
> shadow layers. Today a designer builds those by hand in Photoshop — thirty to
> ninety minutes per base. At a thousand new bases a month, that is two
> full-time designers, and the base library is permanently behind the catalogue."

> "So we automated it. But the interesting part isn't the segmentation model."

---

## 0:25–0:50 — The real problem

**Show:** two masks side by side, one subtly broken (the framed print from the
test set is perfect for this: sharp, clean, and cutting out completely the wrong
object).

> "Any segmentation model always returns *something*. This mask is sharp and
> clean — and completely wrong: it cut out the print and left the frame behind.
> If the system can't tell the difference, a designer still has to open all
> thousand files, and we've saved nothing. So the thing we actually built is a
> pipeline that judges its own output."

---

## 0:50–1:40 — Single image, live

**Show:** dashboard → **Upload files** → drag `08_mug_white_on_white.png` →
category **Auto-detect** → *Generate masks*.

While it runs (~4 s), say:

> "This is the hardest case in drinkware: white ceramic on a white sweep. BiRefNet
> — MIT licensed, so it's commercially usable — produces the matte, then we
> refine the boundary with a guided filter at full resolution, and a second
> independent model cross-checks it."

**Show the card:** READY, confidence ~0.97, category auto-detected as
*drinkware*, ~2.3 s.

**Click the card** to open the drawer. Walk the artifacts:

> "Overlay for review — checkerboard behind the product so any leftover haze is
> obvious. The alpha mask itself, exactly 1100 × 1300, the same as the source.
> The trimap shows where the AI was *unsure*, which is where a designer should
> look first. And the shadow and highlight layers, split out automatically, so
> the artwork in the final mockup picks up the same lighting as the blank."

**Scroll to the print area:**

> "It also solved the print area — a cylinder solver that removes the handle side
> and pulls the edges in for the wrap-around, returned as a quad plus a 3×3
> homography. That drops straight into libvips or ImageMagick; no changes needed
> in the mockup engine."

---

## 1:40–2:20 — The hard cases (batch)

**Show:** the **Sample set** tab → *Run bundled sample set* (19 images, ~30 s).

While it runs:

> "Nineteen bases across four categories, all deliberately nasty: hair falling
> across a garment, a mug on a cluttered desk, a canvas in perspective, a
> translucent bottle, a poster clipped by the frame."

**When the KPI row appears**, point at it:

> "73.7 % automation rate. Fourteen bases published with no human involvement,
> five sent to review, and the review queue is sorted first — because that's the
> only work a person still owes."

**Click into `03_tee_on_model_hair`:**

> "Here's the case the brief calls out specifically — a model's hair across the
> shoulder. A saliency model returns the smooth garment outline *including* the
> pixels the hair covers, so the mockup would paint the design over the hair. We
> detect thin dark structures inside the garment with a morphological black-hat
> and cut them out. That's worth seven points of boundary F1 on apparel. The
> system still routes it to review — hair is exactly where a human should look."

**Click into `14_framed_print`:**

> "And this is that wrong-but-pretty mask from the start. Look at the reason: only
> 22 % of the pixels just *outside* the mask look like background. A correct
> product mask is surrounded by backdrop; a sub-region of a bigger object isn't.
> That check, plus the second model disagreeing, vetoes READY regardless of the
> aggregate score."

---

## 2:20–2:50 — The numbers

**Show:** the batch report (`Batch report` button), scroll the KPI header, then
cut to a terminal with `python scripts/eval_iou.py` output already on screen.

> "Because the test set is rendered, we have exact ground truth, so these are
> measurements, not impressions. Median IoU 0.9985. And the number that decides
> whether this is deployable: every mask marked READY scored IoU 0.983 or better
> — mean 0.9975. Every defective mask landed in review. Zero false accepts, zero
> missed defects, and confidence correlates with true IoU at 0.81."

---

## 2:50–3:30 — Integration

**Show:** Swagger at `/docs`, expand `POST /v1/mask/url`, then run in the terminal:

```bash
curl -sX POST localhost:8000/v1/mask/url -H 'Content-Type: application/json' \
  -d '{"image_url":"https://.../mug.jpg","category":"drinkware","sku":"MUG-11OZ"}' | jq
```

> "It's a microservice, not a notebook. Four ways in: file upload, a direct image
> URL for service-to-service, a CSV or JSON manifest from the PIM, or a drop
> folder for air-gapped runs. Everything is documented in Swagger, it ships with
> a Dockerfile that bakes the weights in so it works offline, and the whole thing
> is configured by environment variables."

**Show:** `docker compose up` scrolling briefly, or the CSV manifest upload.

---

## 3:30–4:00 — Performance

**Show:** `python scripts/bench.py --quick` output, or the bench table.

> "2.19 seconds per mask, 43 images a minute at two workers — on a four gigabyte
> laptop GPU, which is the weakest hardware anyone would deploy this on. A
> thousand bases is twenty to forty minutes of GPU time instead of four to six
> weeks of design queue."

---

## 4:00–4:20 — Close

> "What we're proposing isn't a mask generator. It's a base pipeline that knows
> when to hand work back to a human — and on our numbers it handed back every
> single one of the bad ones. Repository, test set, ground truth and the
> evaluation scripts are all in the repo, so every number in this video can be
> re-run in two commands."

---

## Shot list / B-roll

| Time | Shot |
|---|---|
| 0:25 | side-by-side of the good and the wrong-but-sharp mask |
| 1:00 | drag-and-drop into the dropzone (real cursor, real file) |
| 1:20 | drawer scroll: overlay → mask → trimap → shadow → displacement |
| 1:35 | print-area quad drawn over the mug in the overlay |
| 1:50 | KPI row appearing as the batch finishes |
| 2:05 | the hair case, zoomed on the shoulder |
| 2:30 | terminal: eval_iou.py routing-quality block |
| 3:00 | Swagger `/docs` expanded |
| 3:40 | bench.py stage-split output |

## Recording notes

* **Do not cut the processing wait.** A real 4 s latency on screen is more
  convincing than a jump cut, and the brief is graded on speed.
* Say "READY / REVIEW / FAILED" out loud at least twice — it is the mechanism
  everything else hangs off.
* Have `/health` visible once, briefly: it proves the model actually loaded and
  the run is live, not pre-baked.
* If a live model download or GPU hiccup is a risk, do one dry run immediately
  before recording so everything is warm and cached.
