# RAG Stress Tests

End-to-end stress tests that validate the RAG Knowledge Service under load across all supported vector backends.

## Quick Start

```bash
# Prerequisites: Docker running, .env configured
cd src/lasagna/deploy
cp .env.example .env
# Edit .env: set RAG_JWT_SECRET (generate with: openssl rand -hex 32)

# Build the Docker image (one-time, downloads embedding models)
make build

# Run stress tests against a single backend (~6 min)
cd /path/to/lasagna
make -C src/lasagna/deploy stress-test

# Run against all backends (~27 min)
make -C src/lasagna/deploy stress-test-all
```

Or run directly with pytest:

```bash
# Single backend
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2 \
RAG_EMBEDDING_DIM=384 \
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
STRESS_BACKEND_ONLY=pgvector \
python -m pytest tests/stress/ -m stress -v

# All backends (run sequentially to avoid port conflicts)
for backend in pgvector qdrant weaviate milvus-lite; do
  RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2 \
  RAG_EMBEDDING_DIM=384 \
  RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
  STRESS_BACKEND_ONLY=$backend \
  python -m pytest tests/stress/ -m stress -v \
    2>&1 | tee /tmp/stress-$backend.log
done
```

## What Gets Tested

| Module | Tests | What it validates |
|--------|-------|-------------------|
| `test_infrastructure_baseline` | 4 | Startup, health checks, backend detection, Prometheus metrics |
| `test_concurrent_uploads` | 3 | Multi-tenant upload, upload+search overlap, large documents |
| `test_search_under_load` | 5 | Latency p50/p95, citations, routing tiers, result correctness, degradation |
| `test_rate_limiting` | 4 | Burst capacity, Retry-After headers, multi-tenant fairness, recovery |
| `test_cache_effectiveness` | 3 | Cache warmup, hit rate, tenant isolation |
| `test_tenant_isolation` | 3 | Document isolation, search isolation, deletion isolation |
| `test_rag_quality_under_load` | 4 | IR metrics baseline, IR under load, feature comparison, routing quality |
| `test_endurance` | 2 | Sustained load (latency drift), memory leak indicators |

**Total: 28 tests per backend, 140 across all 5 backends.**

## Supported Backends

| Backend | ARM64 | x86 | Notes |
|---------|-------|-----|-------|
| pgvector | Yes | Yes | Reuses PostgreSQL — simplest deployment |
| qdrant | Yes | Yes | Separate container |
| weaviate | Yes | Yes | Separate container, slower startup |
| milvus-lite | Yes | — | Embedded single-process, no celery |
| milvus | — | Yes | Full Milvus, requires x86 Docker |
| opensearch | Yes | Yes | Separate container, native BM25 + k-NN hybrid |

Architecture is auto-detected. On ARM64, milvus-lite is used instead of milvus.

## Environment Variables

### Model Configuration (Docker container)

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model. Use `all-MiniLM-L6-v2` for fast tests |
| `RAG_EMBEDDING_DIM` | `1024` | Must match model output dim. Set to `384` for MiniLM |
| `RAG_RERANKER_MODEL` | `mixedbread-ai/mxbai-rerank-large-v1` | Reranker. Use `cross-encoder/ms-marco-MiniLM-L-6-v2` for fast tests |

### Scale Configuration (test process)

| Variable | Default | Description |
|----------|---------|-------------|
| `STRESS_BACKEND_ONLY` | _(all)_ | Run against a single backend only |
| `STRESS_NUM_TENANTS` | `50` | Number of concurrent tenant sessions |
| `STRESS_DOCS_PER_TENANT` | `10` | Documents uploaded per tenant |
| `STRESS_CONCURRENT_SEARCHES` | `100` | Concurrent search requests |
| `STRESS_SEARCH_ROUNDS` | `5` | Search iterations per test |
| `STRESS_ENDURANCE_MINUTES` | `10` | Duration of sustained load tests |

### Recommended Profiles

| Profile | Tenants | Docs | Searches | Endurance | Duration | Use case |
|---------|---------|------|----------|-----------|----------|----------|
| **Smoke** | 5 | 3 | 10 | 1 min | ~5 min | CI, quick validation |
| **Standard** | 10 | 5 | 20 | 1 min | ~6 min | **Recommended default** |
| **Production** | 50 | 10 | 100 | 5 min | ~15 min | Pre-release validation |

Example (smoke profile):
```bash
STRESS_NUM_TENANTS=5 STRESS_DOCS_PER_TENANT=3 \
STRESS_CONCURRENT_SEARCHES=10 STRESS_ENDURANCE_MINUTES=1 \
STRESS_BACKEND_ONLY=pgvector \
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2 RAG_EMBEDDING_DIM=384 \
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
python -m pytest tests/stress/ -m stress -v
```

## How It Works

1. **conftest.py** discovers available backends based on architecture
2. **DockerComposeManager** starts the compose stack (API + Redis + PostgreSQL + vector DB)
3. **loaded_backend** fixture uploads test documents and waits for readiness
4. Tests run against the live API using **StressAPIClient** (async HTTP)
5. Docker is session-scoped — starts once, reuses across all 8 test modules
6. When testing multiple backends, Docker tears down and restarts between backends

## Troubleshooting

### Tests are slow
Use lightweight models (see Quick Start). The default models (`bge-m3` + `mxbai-rerank-large`) are 10-20x slower on CPU.

### Embedding dimension mismatch (milvus-lite)
Set `RAG_EMBEDDING_DIM=384` when using `all-MiniLM-L6-v2`. The default (1024) matches `bge-m3`.

### Docker image not found
Build it first: `make -C src/lasagna/deploy build`

### Port conflicts
Stop any existing stacks: `make -C src/lasagna/deploy down-v`

### Production scale failures (50+ tenants)
The API runs 4 gunicorn workers by default. At 50+ tenants with 100 concurrent searches, you may see 503s. This is a capacity limit, not a bug. Use the Standard profile (10 tenants) for reliable validation.
