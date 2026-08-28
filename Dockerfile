# ---------------------------------------------------------------------------
# AI Auto-Masking microservice
#
# Two-stage build: weights are baked into the image at build time so a cold
# container does not download ~600 MB on its first request (and so the service
# works in an air-gapped cluster). The runtime stage carries no build tools.
#
#   docker build -t ai-automask:1.0 .
#   docker run --rm -p 8000:8000 ai-automask:1.0                 # CPU
#   docker run --rm --gpus all -p 8000:8000 ai-automask:1.0      # GPU
#
# For a GPU image, swap the base to nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# and install the cu121 torch wheels instead of the CPU ones (see requirements.txt).
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS weights

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1

RUN pip install --no-cache-dir huggingface-hub==0.34.4

WORKDIR /build
COPY scripts/download_models.py scripts/download_models.py
# Fetches BiRefNet (MIT) into HF_HOME and u2net.onnx (Apache-2.0) into models/.
RUN python scripts/download_models.py


# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    USE_TF=0 \
    OMP_NUM_THREADS=4 \
    AUTOMASK_DEVICE=auto \
    AUTOMASK_OUTPUT_DIR=/data/outputs

# libgl/libglib: OpenCV needs them even in headless builds for a few codecs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
      torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --from=weights /opt/hf /opt/hf
COPY --from=weights /build/models /app/models
COPY app app
COPY scripts scripts

RUN useradd -m -u 10001 automask \
 && mkdir -p /data/outputs \
 && chown -R automask:automask /app /data /opt/hf
USER automask

EXPOSE 8000
VOLUME ["/data/outputs"]

# The model warms up on a background thread, so the container reports healthy
# immediately and /health returns "warming" until the weights are resident.
HEALTHCHECK --interval=20s --timeout=5s --start-period=25s --retries=4 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "75"]
