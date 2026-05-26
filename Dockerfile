# ── HybridRAG API — Dockerfile ───────────────────────────────────────────────
# Multi-stage build: keeps the final image lean by separating dependency
# installation from the runtime layer.
#
# Build:  docker build -t hybridrag .
# Run:    docker compose up

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# System deps needed to compile psycopg2-binary and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source — .env is intentionally excluded (secrets must be injected
# at runtime via docker-compose env_file or environment variables, never baked in)
COPY src/ src/
COPY scripts/ scripts/

# BGE model downloads to ~/.cache/huggingface on first run.
# Mount a volume in docker-compose to persist it across restarts.
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

# Uvicorn with 2 workers — enough for a demo/portfolio workload.
# Increase workers for production.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
