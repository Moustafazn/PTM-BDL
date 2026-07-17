# ============================================================================
# Dockerfile — PTM-Driven Multimodal Self-Attention for Drug Resistance
# ============================================================================
# Reproducible environment for Nature Methods reviewers.
#
# Build:
#   docker build -t ptm-bdl .
#
# Run full pipeline:
#   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results ptm-bdl make all
#
# Run specific step:
#   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results ptm-bdl python scripts/step11_train.py
#
# Interactive shell:
#   docker run --rm -it -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results ptm-bdl bash
#
# GPU support (NVIDIA):
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime -t ptm-bdl-gpu .
#   docker run --rm --gpus all -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results ptm-bdl-gpu make all
# ============================================================================

ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

LABEL maintainer="Moustafa Zein"
LABEL description="Multimodal Self-Attention with PTM Biological Dynamics Layer for Drug Response Prediction"
LABEL org.opencontainers.image.source="https://github.com/Moustafazn/PTM-BDL-Framework"

# ── System Dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Working Directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Python Dependencies (cached layer — rebuild only when deps change) ───────
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# ── Project Source ───────────────────────────────────────────────────────────
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY CITATION.cff .
COPY Makefile .
COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY docs/ docs/

# Install project in editable mode (no deps — already installed above)
RUN pip install --no-cache-dir --no-deps -e .

# ── Data & Results Volumes ───────────────────────────────────────────────────
# Mount these at runtime:
#   -v $(pwd)/data:/app/data       (input data, ~1 GB)
#   -v $(pwd)/results:/app/results (output results)
VOLUME ["/app/data", "/app/results"]

# ── Default Command ──────────────────────────────────────────────────────────
CMD ["python", "-c", "import torch; import transformers; print('PTM-BDL environment ready.'); print(f'PyTorch {torch.__version__}'); print(f'Transformers {transformers.__version__}')"]
