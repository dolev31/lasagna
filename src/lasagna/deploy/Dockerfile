# ============================================================================
# Lasagna: Multi-stage Docker Build
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

# Quoted: unquoted, the shell reads `>=5.1.2` as a redirect to a file named
# "=5.1.2", so pip installed whatever version it liked.
RUN pip install --no-cache-dir "sentence-transformers>=5.1.2" "FlagEmbedding>=1.3.0"

# Models baked into the image, space-separated. The defaults match the default
# config; a lighter image (the CI stack uses MiniLM) passes, for example,
#   --build-arg EMBEDDING_MODELS=sentence-transformers/all-MiniLM-L6-v2 \
#   --build-arg RERANKER_MODELS=cross-encoder/ms-marco-MiniLM-L-6-v2 \
#   --build-arg OPTIONAL_MODELS=
ARG EMBEDDING_MODELS="BAAI/bge-m3"
ARG RERANKER_MODELS="mixedbread-ai/mxbai-rerank-large-v1 cross-encoder/ms-marco-MiniLM-L-6-v2"
ARG OPTIONAL_MODELS="BAAI/bge-large-en-v1.5 sentence-transformers/all-MiniLM-L6-v2"

# Required models fail the build, so a network problem shows up here rather
# than as a download on the first request.
RUN python -c "\
import sys; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
[SentenceTransformer(m) for m in sys.argv[1].split()]; \
[CrossEncoder(m) for m in sys.argv[2].split()]" "$EMBEDDING_MODELS" "$RERANKER_MODELS"

# Fallback models, not needed by the default config.
RUN for m in $OPTIONAL_MODELS; do \
      python -c "import sys; from sentence_transformers import SentenceTransformer; SentenceTransformer(sys.argv[1])" "$m" || true; \
    done

# ---------------------------------------------------------------------------
# Stage 2: Production image
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS production

LABEL maintainer="Lasagna contributors"
LABEL description="Lasagna REST API, Celery worker and MCP server"

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
# on (Redis, Celery, Prometheus, OpenTelemetry, PostgreSQL); `pgvector`,
# `opensearch` and `flag-embedding` are backends and an embedder the compose
# overlays select. Milvus and Milvus Lite ship with the core package.
#
# VECTOR_BACKEND names the store the image talks to. qdrant, weaviate and
# pinecone need the `vector-backends` extra; their compose fragments set this
# argument, and `all` installs every client. The build fails if the client
# does not import, rather than the api failing its readiness check later.
# Pin setuptools<78 so that pkg_resources remains available (required by milvus-lite)
ARG VECTOR_BACKEND=milvus
RUN case "$VECTOR_BACKEND" in \
      milvus|milvus-lite|pgvector|opensearch) extra=""; check="" ;; \
      qdrant) extra=",vector-backends"; check="qdrant_client" ;; \
      weaviate) extra=",vector-backends"; check="weaviate" ;; \
      pinecone) extra=",vector-backends"; check="pinecone" ;; \
      all) extra=",vector-backends"; check="qdrant_client weaviate pinecone" ;; \
      *) echo "Unknown VECTOR_BACKEND=$VECTOR_BACKEND. Use one of: milvus milvus-lite pgvector opensearch qdrant weaviate pinecone all" >&2; exit 1 ;; \
    esac && \
    pip install --no-cache-dir "setuptools<78" && \
    pip install --no-cache-dir ".[production,pgvector,opensearch,flag-embedding$extra]" \
    "gunicorn>=22.0.0" && \
    for module in pymilvus psycopg opensearchpy $check; do python -c "import $module" || exit 1; done
# Read by `make validate`, which refuses an image built for another backend.
LABEL org.lasagna.vector-backend="$VECTOR_BACKEND"

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

# Data directories writable by the rag user. A new named volume copies the
# ownership of the directory it is mounted on, so /data/uploads must exist here
# or the shared uploads volume would be root-owned and unwritable.
RUN mkdir -p /data/milvus /data/uploads && chown -R rag:rag /data

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
#
# --preload imports the app once in the master and forks the workers from it,
# so torch and the package are loaded once and shared, and an import error stops
# the master instead of crash-looping every worker. It is safe because creating
# the app starts no thread and opens no connection; the backend is loaded per
# worker, in the app's lifespan, after the fork.
#
# --forwarded-allow-ips: the proxy addresses whose X-Forwarded-For is trusted.
# FORWARDED_ALLOW_IPS is gunicorn's own name, RAG_FORWARDED_ALLOW_IPS the one
# `lasagna serve` reads; unset trusts only loopback.
CMD rm -rf /tmp/prometheus_multiproc/* && \
    gunicorn \
    "lasagna.api.rest:create_app()" \
    --preload \
    --bind 0.0.0.0:8000 \
    --workers "${WEB_CONCURRENCY:-4}" \
    --worker-class uvicorn.workers.UvicornWorker \
    --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-${RAG_FORWARDED_ALLOW_IPS:-127.0.0.1}}" \
    --timeout 120 \
    --graceful-timeout 30 \
    --access-logfile -
