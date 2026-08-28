# Convenience targets. Everything here is a plain script - nothing depends on make.
.PHONY: help install models serve dev dataset run eval bench test lint docker clean deliverable

help:
	@echo "install     - pip install -r requirements.txt (install torch first, see requirements.txt)"
	@echo "models      - download BiRefNet + U2-Net weights"
	@echo "serve       - run the API + dashboard on :8000"
	@echo "dev         - same, with auto-reload"
	@echo "dataset     - render the 19-base test set + ground truth"
	@echo "run         - process data/samples offline and write report.html"
	@echo "eval        - IoU / boundary-F1 / routing quality against ground truth"
	@echo "bench       - latency + throughput sweep"
	@echo "test        - pytest (no GPU or weights needed)"
	@echo "deliverable - package the test set + outputs into deliverables/"
	@echo "docker      - build and run the container"

install:
	python -m pip install -r requirements.txt

models:
	python scripts/download_models.py

serve:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

dataset:
	python scripts/make_dataset.py

run:
	python scripts/run_folder.py data/samples --report

eval:
	python scripts/eval_iou.py

bench:
	python scripts/bench.py

test:
	AUTOMASK_PRIMARY_MODEL=grabcut AUTOMASK_ENSEMBLE=0 python -m pytest -q

deliverable:
	python scripts/export_deliverable.py

docker:
	docker compose up --build

clean:
	rm -rf outputs/* data/eval __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
