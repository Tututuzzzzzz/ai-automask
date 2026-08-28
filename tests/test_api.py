"""API contract tests, driven through FastAPI's TestClient.

These cover the surface a partner backend actually integrates against: response
shape, status codes, auth, and the artifact URLs. They run against the GrabCut
backend so CI needs neither weights nor a GPU - the point is the contract, not
the mask quality.

Run with:  pytest -q tests/test_api.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Must be set before app.config is imported, since Settings reads the env once.
os.environ.setdefault("AUTOMASK_PRIMARY_MODEL", "grabcut")
os.environ.setdefault("AUTOMASK_ENSEMBLE", "0")
os.environ.setdefault("AUTOMASK_SHADOW_MAPS", "0")
os.environ.setdefault("AUTOMASK_DISPLACEMENT", "0")

from fastapi.testclient import TestClient  # noqa: E402

from app.imaging import encode_png  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Context-manager form so the lifespan (model warmup thread) actually runs.
    with TestClient(app) as c:
        yield c


def base_png(w: int = 300, h: int = 380) -> bytes:
    img = np.full((h, w, 3), 238, np.uint8)
    cv2.rectangle(img, (int(w * 0.2), int(h * 0.15)), (int(w * 0.8), int(h * 0.85)),
                  (55, 80, 190), -1, cv2.LINE_AA)
    return encode_png(cv2.GaussianBlur(img, (0, 0), 0.8))


# ------------------------------------------------------------------------- ops
def test_health_reports_models_and_settings(client):
    r = client.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] in {"ok", "warming", "degraded"}
    assert d["device"] in {"cpu", "cuda"}
    assert "backends" in d["models"]
    assert {"primary_model", "ready_threshold", "categories"} <= set(d["settings"])


def test_categories_endpoint_matches_the_taxonomy(client):
    d = client.get("/v1/categories").json()
    assert {"auto", "apparel", "drinkware", "wall_art", "accessory"} <= set(d)
    for cfg in d.values():
        assert "label" in cfg and "print_area" in cfg


def test_openapi_schema_documents_every_public_endpoint(client):
    """Swagger is a graded deliverable, so an undocumented route is a bug."""
    spec = client.get("/openapi.json").json()
    for path in ("/v1/mask", "/v1/mask/url", "/v1/mask/batch", "/v1/mask/manifest",
                 "/v1/mask/upload-batch", "/v1/jobs/{job_id}", "/health"):
        assert path in spec["paths"], f"{path} missing from the OpenAPI schema"
    assert spec["info"]["title"]


def test_dashboard_and_static_assets_are_served(client):
    assert client.get("/").status_code == 200
    assert "AI Auto-Masking" in client.get("/").text
    assert client.get("/ui/ui.css").status_code == 200
    assert client.get("/ui/ui.js").status_code == 200


# ------------------------------------------------------------------ single image
def test_single_upload_returns_a_complete_result(client):
    r = client.post("/v1/mask",
                    files={"file": ("tee.png", base_png(), "image/png")},
                    data={"category": "apparel"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "ok"
    assert d["verdict"] in {"READY", "REVIEW", "FAILED"}
    assert 0.0 <= d["confidence"] <= 1.0
    assert (d["width"], d["height"]) == (300, 380)
    assert d["category"] == "apparel"
    assert d["category_source"] == "metadata"
    assert d["artifacts"]["alpha_mask"].startswith("/artifacts/")
    assert d["metrics"] and d["reasons"]
    assert d["timings_ms"]["total"] > 0

    # The artifact URL in the response must actually resolve, and the served PNG
    # must carry the source resolution.
    got = client.get(d["artifacts"]["alpha_mask"])
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/png"
    mask = cv2.imdecode(np.frombuffer(got.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert mask.shape[:2] == (380, 300)


def test_auto_category_is_labelled_as_detected(client):
    d = client.post("/v1/mask", files={"file": ("x.png", base_png(), "image/png")},
                    data={"category": "auto"}).json()
    assert d["category_source"] == "auto-detected"
    assert d["category"] in {"apparel", "drinkware", "wall_art", "accessory"}


def test_undecodable_upload_returns_422(client):
    r = client.post("/v1/mask", files={"file": ("bad.png", b"nonsense", "image/png")})
    assert r.status_code == 422
    assert "decode" in r.json()["detail"].lower()


def test_empty_upload_returns_400(client):
    r = client.post("/v1/mask", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_url_endpoint_rejects_an_unreachable_host(client):
    r = client.post("/v1/mask/url", json={"image_url": "http://127.0.0.1:1/x.png"})
    assert r.status_code == 422


def test_url_endpoint_validates_the_url_field(client):
    r = client.post("/v1/mask/url", json={"image_url": "not-a-url"})
    assert r.status_code == 422


# -------------------------------------------------------------------- batching
def test_upload_batch_returns_a_summary_and_a_report(client):
    files = [("files", (f"b{i}.png", base_png(240 + i * 10, 300), "image/png")) for i in range(3)]
    r = client.post("/v1/mask/upload-batch", files=files, data={"category": "auto"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["summary"]["total"] == 3
    assert len(d["results"]) == 3
    assert 0.0 <= d["summary"]["automation_rate"] <= 1.0
    assert d["summary"]["throughput_img_per_min"] > 0

    report = client.get(d["report_url"])
    assert report.status_code == 200
    assert "automation rate" in report.text.lower()
    # Artifacts must be reachable through the mount the served report rewrites to.
    assert '/artifacts/' in report.text

    csv = client.get(f"/v1/jobs/{d['job_id']}/results.csv")
    assert csv.status_code == 200
    assert csv.text.splitlines()[0].startswith("id,source,status,verdict")
    assert len(csv.text.strip().splitlines()) == 4

    zip_resp = client.get(f"/v1/jobs/{d['job_id']}/download")
    assert zip_resp.status_code == 200
    assert zip_resp.content[:2] == b"PK"


def test_manifest_ingest_reports_bad_rows_but_still_processes(client):
    # One dead URL and one blank row: the batch must complete with an error row,
    # not fail wholesale.
    csv = ("sku,Image URL,category\n"
           "A,http://127.0.0.1:1/one.png,mug\n"
           "B,,canvas\n")
    r = client.post("/v1/mask/manifest",
                    files={"file": ("m.csv", csv.encode(), "text/csv")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["summary"]["total"] == 1
    assert d["results"][0]["status"] == "error"
    assert d["results"][0]["verdict"] == "FAILED"


def test_manifest_with_no_usable_rows_returns_422(client):
    r = client.post("/v1/mask/manifest",
                    files={"file": ("m.csv", b"foo,bar\n1,2\n", "text/csv")})
    assert r.status_code == 422


def test_async_batch_returns_a_pollable_job(client):
    r = client.post("/v1/mask/batch/async",
                    json={"items": [{"image_url": "http://127.0.0.1:1/a.png"}]})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    # TestClient runs background tasks before returning, so the job is already done.
    status = client.get(f"/v1/jobs/{job_id}").json()
    assert status["job_id"] == job_id
    assert status["state"] in {"queued", "running", "done"}
    assert status["total"] == 1


def test_batch_request_rejects_an_empty_item_list(client):
    assert client.post("/v1/mask/batch", json={"items": []}).status_code == 422


def test_jobs_listing_includes_recent_work(client):
    jobs = client.get("/v1/jobs?limit=5").json()
    assert isinstance(jobs, list) and jobs
    assert {"job_id", "state", "processed", "total"} <= set(jobs[0])


def test_unknown_job_returns_404(client):
    assert client.get("/v1/jobs/does_not_exist").status_code == 404
    assert client.get("/v1/jobs/does_not_exist/report").status_code == 404


# ------------------------------------------------------------------- artifacts
def test_artifact_path_traversal_is_blocked(client):
    d = client.post("/v1/mask", files={"file": ("x.png", base_png(), "image/png")}).json()
    job_id = d["artifacts"]["alpha_mask"].split("/")[2]
    r = client.get(f"/artifacts/{job_id}/../../../app/main.py")
    assert r.status_code in {400, 404}


def test_missing_artifact_returns_404(client):
    d = client.post("/v1/mask", files={"file": ("x.png", base_png(), "image/png")}).json()
    job_id = d["artifacts"]["alpha_mask"].split("/")[2]
    assert client.get(f"/artifacts/{job_id}/nope.png").status_code == 404


# ------------------------------------------------------------------------ auth
def test_api_key_is_enforced_when_configured(client):
    from app.config import settings

    settings.api_key = "secret-key"
    try:
        assert client.post("/v1/mask/url",
                           json={"image_url": "http://127.0.0.1:1/a.png"}).status_code == 401
        r = client.post("/v1/mask/url", json={"image_url": "http://127.0.0.1:1/a.png"},
                        headers={"X-API-Key": "secret-key"})
        assert r.status_code == 422          # past auth, fails on the dead URL
        # /health stays open so orchestrators can probe it without a credential.
        assert client.get("/health").status_code == 200
    finally:
        settings.api_key = None
