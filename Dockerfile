# ==============================================================================
# Stage 1: CPU Base (Default, lightweight ~1.2 GB)
# ==============================================================================
FROM python:3.14-slim AS base-cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    DEVICE=cpu

# Install minimal OS dependencies for building C/C++ wheels if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install PyTorch CPU wheel specifically for a small image footprint
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Install timesfm-serve package and requirements
COPY .python-version pyproject.toml README.md /app/
RUN uv pip install --system --no-cache -e .

COPY timesfm_serve /app/timesfm_serve

# Create non-root user and pre-create Hugging Face cache dir
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser /app
USER appuser

EXPOSE 8088

HEALTHCHECK --interval=20s --timeout=10s --retries=3 --start-period=45s \
  CMD curl -sf http://localhost:8088/health || exit 1

ENTRYPOINT ["uvicorn", "timesfm_serve.main:app", "--host", "0.0.0.0", "--port", "8088"]

# ==============================================================================
# Stage 2: GPU Base (CUDA enabled for high-throughput inference)
# ==============================================================================
FROM python:3.14-slim AS base-gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    DEVICE=cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install PyTorch CUDA wheel
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cu124 || \
    uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cu121 || \
    uv pip install --system --no-cache torch

WORKDIR /app

COPY .python-version pyproject.toml README.md /app/
RUN uv pip install --system --no-cache -e .

COPY timesfm_serve /app/timesfm_serve

RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /home/appuser /app
USER appuser

EXPOSE 8088

HEALTHCHECK --interval=20s --timeout=10s --retries=3 --start-period=45s \
  CMD curl -sf http://localhost:8088/health || exit 1

ENTRYPOINT ["uvicorn", "timesfm_serve.main:app", "--host", "0.0.0.0", "--port", "8088"]
