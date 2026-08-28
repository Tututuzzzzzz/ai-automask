"""Self-contained HTML batch report.

Written into the job folder alongside the artifacts and referencing them with
*relative* paths, so the report works three ways: served by the API, opened
straight off disk, or zipped and emailed to whoever signs off the base library.

Content is chosen for the review workflow, in this order:
  1. the automation rate (the number that justifies the project)
  2. the REVIEW queue, first, because that is the only work a human still owes
  3. side-by-side original / mask / overlay for every base
"""
from __future__ import annotations

import html
import os
from datetime import datetime, timezone

from .schemas import BatchSummary, MaskResult

_VERDICT_ORDER = {"REVIEW": 0, "FAILED": 1, "READY": 2}

_CSS = """
:root{--bg:#0f1115;--card:#181b22;--line:#272b35;--fg:#e7e9ee;--dim:#9aa1af;
--ready:#2ecc71;--review:#f39c12;--failed:#e74c3c;--accent:#4c8dff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:24px;margin:0 0 4px}
.sub{color:var(--dim);margin-bottom:26px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:14px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:26px;font-weight:650;letter-spacing:-.5px}
.kpi .l{color:var(--dim);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.bar{display:flex;height:12px;border-radius:99px;overflow:hidden;background:#222630;margin:18px 0 30px}
.bar i{display:block;height:100%}
table{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:34px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:18px}
.card h3{margin:0 0 2px;font-size:15px;word-break:break-all}
.meta{color:var(--dim);font-size:12px;margin-bottom:12px}
.trio{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
figure{margin:0}
figure img{width:100%;border-radius:9px;display:block;background:
linear-gradient(45deg,#2a2e37 25%,transparent 25%,transparent 75%,#2a2e37 75%),
linear-gradient(45deg,#2a2e37 25%,#20242c 25%,#20242c 75%,#2a2e37 75%);
background-size:18px 18px;background-position:0 0,9px 9px}
figcaption{color:var(--dim);font-size:11.5px;margin-top:5px;text-transform:uppercase;letter-spacing:.05em}
.tag{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;font-weight:700;letter-spacing:.04em}
.READY{background:rgba(46,204,113,.16);color:var(--ready)}
.REVIEW{background:rgba(243,156,18,.16);color:var(--review)}
.FAILED{background:rgba(231,76,60,.16);color:var(--failed)}
ul.reasons{margin:10px 0 0;padding-left:20px;color:var(--dim);font-size:12.5px}
.metrics{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.metrics span{background:#20242c;border:1px solid var(--line);border-radius:7px;
padding:3px 8px;font-size:11.5px;color:var(--dim)}
.metrics b{color:var(--fg);font-weight:600}
a{color:var(--accent)}
footer{color:var(--dim);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
"""


def _rel(url: str | None) -> str | None:
    """Turn '/artifacts/<job>/<file>' into '<file>' so the report is portable."""
    if not url:
        return None
    if url.startswith("data:"):
        return url
    return os.path.basename(url)


def _tag(verdict: str | None) -> str:
    v = verdict or "FAILED"
    return f'<span class="tag {html.escape(v)}">{html.escape(v)}</span>'


def _metrics_row(r: MaskResult) -> str:
    m = r.metrics
    if not m:
        return ""
    bits = [
        ("confidence", f"{(r.confidence or 0):.3f}"),
        ("coverage", f"{m.coverage*100:.1f}%"),
        ("edge", f"{m.edge_sharpness:.2f}"),
        ("solidity", f"{m.solidity:.2f}"),
        ("holes", str(m.holes)),
    ]
    if m.ensemble_iou is not None:
        bits.insert(3, ("ensemble IoU", f"{m.ensemble_iou:.3f}"))
    if r.print_area and r.print_area.kind != "none":
        bits.append(("print area", f"{r.print_area.kind} ({r.print_area.confidence:.2f})"))
    bits.append(("latency", f"{r.timings_ms.get('total', 0):.0f} ms"))
    return '<div class="metrics">' + "".join(
        f"<span>{html.escape(k)} <b>{html.escape(v)}</b></span>" for k, v in bits
    ) + "</div>"


def _card(r: MaskResult) -> str:
    src = html.escape(r.source)
    head = (f'<h3>{_tag(r.verdict)} &nbsp;{html.escape(r.id)}</h3>'
            f'<div class="meta">{src} &middot; {r.width or "?"}x{r.height or "?"} px '
            f'&middot; {html.escape(r.category or "?")} ({html.escape(r.category_source or "-")}) '
            f'&middot; model {html.escape(r.model_used or "-")}</div>')

    if r.status == "error":
        return (f'<div class="card">{head}<ul class="reasons"><li>'
                f'{html.escape(r.error or "unknown error")}</li></ul></div>')

    figs = []
    for label, url in (
        ("Overlay (review view)", _rel(r.artifacts.overlay)),
        ("Alpha mask", _rel(r.artifacts.alpha_mask)),
        ("Cut-out RGBA", _rel(r.artifacts.cutout_rgba)),
        ("Shadow map", _rel(r.artifacts.shadow_map)),
        ("Highlight map", _rel(r.artifacts.highlight_map)),
        ("Displacement", _rel(r.artifacts.displacement_map)),
    ):
        if url:
            figs.append(f'<figure><img loading="lazy" src="{html.escape(url)}" alt="{html.escape(label)}">'
                        f'<figcaption>{html.escape(label)}</figcaption></figure>')

    reasons = ""
    if r.reasons:
        reasons = "<ul class=\"reasons\">" + "".join(
            f"<li>{html.escape(x)}</li>" for x in r.reasons) + "</ul>"

    return (f'<div class="card">{head}<div class="trio">{"".join(figs)}</div>'
            f'{_metrics_row(r)}{reasons}</div>')


def render_report(job_id: str, results: list[MaskResult], summary: BatchSummary,
                  warnings: list[str] | None = None, meta: dict | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = max(summary.total, 1)
    pct = lambda n: f"{n / total * 100:.1f}%"  # noqa: E731

    kpis = [
        ("Automation rate", f"{summary.automation_rate*100:.1f}%", "READY, no human touch"),
        ("Bases processed", str(summary.total), ""),
        ("Ready", str(summary.ready), pct(summary.ready)),
        ("Needs review", str(summary.review), pct(summary.review)),
        ("Rejected", str(summary.failed), pct(summary.failed)),
        ("Mean confidence", f"{summary.mean_confidence:.3f}", ""),
        ("Mean latency", f"{summary.mean_latency_ms:.0f} ms", "per image"),
        ("Throughput", f"{summary.throughput_img_per_min:.1f}/min", f"wall {summary.total_wall_ms/1000:.1f}s"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="v">{html.escape(v)}</div>'
        f'<div class="l">{html.escape(label)}</div>'
        + (f'<div class="l" style="text-transform:none">{html.escape(note)}</div>' if note else "")
        + "</div>"
        for label, v, note in kpis
    )

    bar = (f'<div class="bar">'
           f'<i style="width:{summary.ready/total*100:.2f}%;background:var(--ready)"></i>'
           f'<i style="width:{summary.review/total*100:.2f}%;background:var(--review)"></i>'
           f'<i style="width:{summary.failed/total*100:.2f}%;background:var(--failed)"></i>'
           f'</div>')

    rows = "".join(
        f"<tr><td>{html.escape(r.id)}</td><td>{_tag(r.verdict)}</td>"
        f"<td>{(r.confidence or 0):.3f}</td><td>{html.escape(r.category or '-')}</td>"
        f"<td>{r.width or '?'}x{r.height or '?'}</td>"
        f"<td>{r.timings_ms.get('total', 0):.0f} ms</td>"
        f"<td>{html.escape((r.reasons[0] if r.reasons else '') [:110])}</td></tr>"
        for r in sorted(results, key=lambda x: (_VERDICT_ORDER.get(x.verdict or 'FAILED', 3),
                                                x.confidence or 0))
    )

    warn_html = ""
    if warnings:
        warn_html = ('<div class="card"><h3>Ingest warnings</h3><ul class="reasons">'
                     + "".join(f"<li>{html.escape(w)}</li>" for w in warnings) + "</ul></div>")

    meta_html = ""
    if meta:
        meta_html = ('<div class="card"><h3>Run configuration</h3><div class="metrics">'
                     + "".join(f"<span>{html.escape(str(k))} <b>{html.escape(str(v))}</b></span>"
                               for k, v in meta.items())
                     + "</div></div>")

    # REVIEW first: it is the only queue that still costs a human time.
    cards = "".join(_card(r) for r in sorted(
        results, key=lambda x: (_VERDICT_ORDER.get(x.verdict or "FAILED", 3), -(x.confidence or 0))))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Auto-Masking report {html.escape(job_id)}</title>
<style>{_CSS}</style></head><body><div class="wrap">
<h1>AI Auto-Masking &mdash; batch report</h1>
<div class="sub">Job <code>{html.escape(job_id)}</code> &middot; {stamp}</div>
<div class="kpis">{kpi_html}</div>
{bar}
{meta_html}{warn_html}
<h2 style="font-size:17px;margin:0 0 10px">Result table</h2>
<table><thead><tr><th>ID</th><th>Verdict</th><th>Conf.</th><th>Category</th>
<th>Resolution</th><th>Latency</th><th>Primary QC note</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2 style="font-size:17px;margin:0 0 10px">Visual review &mdash; review queue first</h2>
{cards}
<footer>Generated by ai-automask. Masks are exported at the source resolution;
artifacts in this folder are referenced relatively, so this report stays valid
after the job folder is zipped and moved.</footer>
</div></body></html>"""
