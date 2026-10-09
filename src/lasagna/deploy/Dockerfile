# ============================================================================
# RAG Knowledge Service — Multi-stage Docker Build
#
# Bakes in embedding + reranker models at build time to eliminate cold-start
# downloads.  Supports both API-only and Celery worker modes via CMD override.
#
# Build:
#   docker build -t rag-knowledge -f src/lasagna/deploy/Dockerfile .
#
# Run API server:
#   docker run -p 8000:8000 --env-file .env rag-knowledge
#
# Run Celery worker:
#   docker run --env-file .env rag-knowledge celery \
#       -A lasagna.infrastructure.tasks.celery_app worker \
#       --loglevel=info --concurrency=4
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Download embedding + reranker models (cached in builder layer)
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS models

# Install build tools needed by native extensions (zlib-state, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir sentence-transformers>=5.1.2 FlagEmbedding>=1.3.0

# Pre-download default models — hard-fail to catch network issues at build time
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-m3'); \
SentenceTransformer('mixedbread-ai/mxbai-rerank-large-v1'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Pre-download fallback embedding models (soft-fail — not required for default config)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')" || true
# Lightweight model for stress tests (22M params, 384 dim, ~80MB — very fast on CPU)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

# ---------------------------------------------------------------------------
# Stage 2: Production image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS production

LABEL maintainer="Lasagna contributors"
LABEL description="RAG Knowledge Service — SOTA retrieval with multi-tenant support"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    libc6-dev \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash rag
WORKDIR /app

# Copy pre-downloaded models from builder into rag user's home.
# Models must be in rag's home because /root/ is 700 (inaccessible to non-root).
COPY --from=models /root/.cache/huggingface /home/rag/.cache/huggingface
RUN chown -R rag:rag /home/rag/.cache

# Copy project files needed for pip install. LICENSE and NOTICE go into the
# built package's metadata (pyproject's license-files), and Apache-2.0 section
# 4(d) requires NOTICE to travel with every redistribution; the README is the
# package's long description.
COPY pyproject.toml LICENSE NOTICE README.md ./
COPY src/lasagna/ ./src/lasagna/

# Install Python dependencies. `production` is what the production settings turn
# on (Redis, Celery, Prometheus, OpenTelemetry, PostgreSQL); the others are the
# backends and embedder the compose overlays select.
# Pin setuptools<78 so that pkg_resources remains available (required by milvus-lite)
RUN pip install --no-cache-dir "setuptools<78" && \
    pip install --no-cache-dir ".[production,pgvector,opensearch,flag-embedding]" \
    "gunicorn>=22.0.0"

# nltk's sentence tokenizer data. Nothing downloads it at runtime, so bake it in
# where nltk looks by default.
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab

# Models already baked in from Stage 1 (COPY --from=models above)

# Environment variable mappings for new config options
ENV RAG_EMBEDDING_MODEL="BAAI/bge-m3" \
    RAG_EMBEDDING_BACKEND="sentence-transformers" \
    RAG_RERANKER_MODEL="mixedbread-ai/mxbai-rerank-large-v1" \
    RAG_ENABLE_CONTEXT_REORDERING="true" \
    RAG_ENABLE_RAG_FUSION="false" \
    RAG_ENABLE_CRAG="false" \
    RAG_ENABLE_CONTEXTUAL_RETRIEVAL="false" \
    RAG_DOCLING_ENABLED="false"

# Create data directories writable by rag user (for volume mounts)
RUN mkdir -p /data/milvus && chown -R rag:rag /data

# Prometheus multiprocess directory (for multi-worker gunicorn)
RUN mkdir -p /tmp/prometheus_multiproc && chown rag:rag /tmp/prometheus_multiproc

# Switch to non-root user
USER rag

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

# Default: run API server with gunicorn + uvicorn workers
# WEB_CONCURRENCY controls worker count (default 4; set to 1 for Milvus Lite)
# Clear PROMETHEUS_MULTIPROC_DIR on startup to prevent stale metrics from prior runs
CMD rm -rf /tmp/prometheus_multiproc/* && \
    gunicorn \
    "lasagna.api.rest:create_app()" \
    --bind 0.0.0.0:8000 \
    --workers "${WEB_CONCURRENCY:-4}" \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --graceful-timeout 30 \
    --access-logfile -
