# Deployment & Operations

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [Multi-Tenant SaaS](#multi-tenant-saas)
3. [Internal Architecture (Contributor Guide)](#internal-architecture-contributor-guide)
4. [Troubleshooting](#troubleshooting)
5. [Support](#support)

---

## Docker Deployment

The Knowledge SDK ships with production-ready Docker Compose configurations for deploying the RAG service with all supporting infrastructure.

### Quick Start

```bash
cd src/lasagna/deploy

# Start core services (API + Redis + PostgreSQL)
docker compose up -d

# With Celery background workers
docker compose --profile celery up -d

# With MCP server
docker compose --profile mcp up -d

# With distributed tracing (Jaeger)
docker compose --profile tracing up -d

# All profiles
docker compose --profile celery --profile mcp --profile tracing up -d
```

### docker-compose.yml Services

| Service | Port | Description | Profile |
|---------|------|-------------|---------|
| `api` | 8000 | FastAPI + Gunicorn (RAG REST API) | Always |
| `redis` | 6379 | Cache + message broker | Always |
| `postgres` | 5432 | Metadata database (PostgreSQL 16) | Always |
| `celery-worker` | -- | Background document processing | `celery` |
| `mcp` | 8005 | Standalone MCP server | `mcp` |
| `jaeger` | 16686/4317 | Distributed tracing UI | `tracing` |

### Environment Variable Configuration

All RAG settings can be configured via `RAG_*` environment variables (12-factor compatible):

```yaml
# docker-compose.yml snippet (environment section)
environment:
  # Database
  RAG_DATABASE_BACKEND: postgresql
  RAG_DATABASE_DSN: postgresql://rag:rag@postgres:5432/rag

  # Cache
  RAG_CACHE_BACKEND: redis
  RAG_REDIS_URL: redis://redis:6379

  # Multi-tenancy
  RAG_TENANT_ISOLATION: partition
  RAG_QUOTA_ENABLED: "true"
  RAG_RATE_LIMITING_ENABLED: "true"

  # Background processing
  RAG_CELERY_ENABLED: "true"
  RAG_CELERY_BROKER_URL: redis://redis:6379/0
  RAG_CELERY_RESULT_BACKEND: redis://redis:6379/1

  # Observability
  RAG_ENABLE_PROMETHEUS: "true"
  RAG_ENABLE_TRACING: "false"
  RAG_TRACING_ENDPOINT: http://jaeger:4317

  # Connection pooling (>= gunicorn workers x 2)
  RAG_CONNECTION_POOL_SIZE: "10"

  # Security
  RAG_JWT_SECRET: "<required — generate with: openssl rand -hex 32>"

  # Vector backend (override for pgvector, Milvus, etc.)
  # MILVUS_URI: http://milvus:19530
  # PGVECTOR_DSN: postgresql://rag:rag@postgres:5432/rag
```

### Backend Fragments

Combine the base compose file with backend-specific fragments:

```bash
# Use PgVector as the vector backend
docker compose -f docker-compose.yml -f docker-compose.pgvector.yml up -d

# Use ARM64-optimized images
docker compose -f docker-compose.yml -f docker-compose.arm64.yml up -d
```

### Health Checks

All services include health checks. The API server exposes `/health` and `/ready` endpoints:

```bash
# Check API health
curl http://localhost:8000/health

# Check readiness (includes backend connectivity)
curl http://localhost:8000/ready
```

---

## Multi-Tenant SaaS

The Knowledge SDK provides production-grade multi-tenancy support with partition isolation, per-tenant quotas, adaptive rate limiting, and a PostgreSQL backend for concurrent multi-worker deployments.

### Partition Isolation

Two isolation modes are available:

| Mode | Config Value | Description |
|------|-------------|-------------|
| **Metadata filter** | `none` (default) | All tenants share one collection; queries filter by `session_id` in metadata. Simple, zero-config. |
| **Partition** | `partition` | Native backend partitions (Milvus partitions, Pinecone namespaces) with metadata filter as defense-in-depth. Stronger isolation, better performance at scale. |

```python
from lasagna.sdk import KnowledgeConfig

# Production: partition isolation
config = KnowledgeConfig(
    tenant_isolation="partition",  # Native backend partitions
)

# Or via environment variable
# RAG_TENANT_ISOLATION=partition
```

### Per-Tenant Quotas

Resource quotas prevent "noisy neighbor" problems. When enabled, the system enforces limits on documents, storage, chunks, and API call rates per tenant.

```python
from lasagna.core.quota import (
    get_quota_manager,
    TenantQuota,
    QuotaExceededError,
)

# Quota manager is auto-configured from RAG config
qm = get_quota_manager()

# Default tenant limits
# - max_documents: 1,000
# - max_storage_bytes: 5 GB
# - max_chunks: 100,000
# - max_searches_per_minute: 600
# - max_uploads_per_hour: 200

# Override quota for a specific tenant
qm.set_tenant_quota("tenant_enterprise", TenantQuota(
    max_documents=10_000,
    max_storage_bytes=50 * 1024 * 1024 * 1024,  # 50 GB
    max_chunks=1_000_000,
    max_searches_per_minute=3000,
    max_uploads_per_hour=1000,
))

# Check quota before operations (raises QuotaExceededError)
try:
    qm.check_upload(session_id="tenant_abc", file_size_bytes=5_000_000)
    # ... perform upload ...
    qm.record_upload(session_id="tenant_abc", file_size_bytes=5_000_000, chunks=42)
except QuotaExceededError as e:
    print(f"Quota exceeded: {e.resource} (limit={e.limit}, current={e.current})")

# Get usage stats for a tenant
usage = qm.get_usage("tenant_abc")
print(f"Documents: {usage['documents']}/{usage['documents_limit']}")
print(f"Storage: {usage['storage_bytes']}/{usage['storage_limit']}")
```

**Enable via config or environment:**

```python
config = KnowledgeConfig(
    quota_enabled=True,
)
# Or: RAG_QUOTA_ENABLED=true
```

When Redis is configured (`cache_backend="redis"`), quotas use `RedisQuotaManager` for cross-worker consistency with atomic Lua scripts. Otherwise, in-memory counters are used (suitable for single-worker deployments).

### Adaptive Rate Limiting

The rate limiter uses a **token-bucket algorithm** that adapts refill rate based on system load. This ensures fair distribution under heavy traffic while maximizing throughput when idle.

Features:

- **Priority tiers**: CRITICAL (never throttled), HIGH, NORMAL, LOW (throttled first)
- **Load-aware**: Burst capacity increases when system is idle, decreases under load
- **Per-tenant**: Each tenant has their own token bucket

```python
from lasagna.core.rate_limiter import get_rate_limiter

rl = get_rate_limiter()

# Check if a request is allowed
allowed, retry_after = rl.try_acquire("session_abc", "/api/v1/search")
if not allowed:
    # Return HTTP 429 with Retry-After header
    return Response(status_code=429, headers={"Retry-After": str(retry_after)})
```

**Configuration:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `rate_limiting_enabled` | bool | False | Enable rate limiting |
| `rate_limiting_strategy` | str | `adaptive` | `adaptive` or `fixed` |
| `rate_limiting_rpm` | float | 60.0 | Base requests per minute per tenant |
| `rate_limiting_burst` | float | 10.0 | Max burst token capacity |
| `rate_limiting_burst_multiplier` | float | 2.0 | Capacity multiplier when idle |
| `rate_limiting_overload_threshold` | float | 0.8 | Load score that triggers throttling |
| `rate_limiting_idle_threshold` | float | 0.3 | Load score below which bursting is allowed |

**Enable via environment:**

```bash
RAG_RATE_LIMITING_ENABLED=true
RAG_RATE_LIMITING_RPM=120
RAG_RATE_LIMITING_STRATEGY=adaptive
```

### PostgreSQL Backend

For production multi-worker deployments, switch the metadata database from SQLite to PostgreSQL. This enables concurrent writes from multiple API workers and Celery background processors.

```python
config = KnowledgeConfig(
    database_backend="postgresql",
    database_dsn="postgresql://user:pass@localhost:5432/rag",
    database_pool_min=2,       # Min connection pool size
    database_pool_max=10,      # Max connection pool size
)
```

**Environment variables:**

```bash
RAG_DATABASE_BACKEND=postgresql
RAG_DATABASE_DSN=postgresql://user:pass@localhost:5432/rag
RAG_DATABASE_POOL_MIN=2
RAG_DATABASE_POOL_MAX=10
```

The PostgreSQL backend supports both `psycopg` (v3, recommended) and `psycopg2` as drivers, with automatic fallback.

### Production Architecture

A typical production deployment combines all multi-tenant features:

```text
                    Load Balancer
                         |
              +----------+----------+
              |          |          |
           API #1     API #2     API #3    (Gunicorn workers)
              |          |          |
              +---+------+----+----+
                  |           |
           +-----+-----+  +--+--+
           | PostgreSQL |  | Redis |
           | (metadata) |  | (cache, |
           +-----+-----+  | broker)|
                  |        +--+--+
           +------+------+    |
           | Vector DB   | Celery Workers
           | (Milvus /   |   (background
           |  PgVector)  |    processing)
           +-------------+

Per-tenant features:
  - Partition isolation (native backend partitions)
  - Resource quotas (Redis-backed, cross-worker)
  - Adaptive rate limiting (token bucket per tenant)
  - Session-scoped configuration
```

```python
# Full production configuration
config = KnowledgeConfig(
    # Multi-tenancy
    tenant_isolation="partition",
    quota_enabled=True,
    rate_limiting_enabled=True,
    rate_limiting_strategy="adaptive",

    # Database
    database_backend="postgresql",
    database_dsn="postgresql://rag:rag@postgres:5432/rag",
    database_pool_max=10,

    # Cache
    cache_backend="redis",
    redis_url="redis://redis:6379",

    # Background processing
    celery_enabled=True,
    celery_broker_url="redis://redis:6379/0",

    # Observability
    enable_prometheus=True,
    enable_tracing=True,
    tracing_endpoint="http://jaeger:4317",

    # Connection pooling
    connection_pool_size=10,
)
```

---

## Internal Architecture (Contributor Guide)

This section documents the internal structure of the knowledge package for contributors and maintainers. SDK consumers do not need to understand these details — the `KnowledgeClient` API is the intended public interface.

### RAGBackend Decomposition

The `RAGBackend` class (`api/rag_backend.py`) is a **facade** that delegates to four specialized managers:

```text
                     RAGBackend (facade, ~580 lines)
                          │
            ┌─────────────┼─────────────┐
            │             │             │
   CollectionManager  DocumentPipeline  SearchOrchestrator
   (517 lines)        (708 lines)      (1029 lines)
            │             │             │
            │         uses CM for    uses CM + AutoMerger
            │         resolve_backend  (172 lines)
```

| Module | File | Responsibility |
| ------ | ---- | -------------- |
| `_validation.py` | Input sanitization | Session ID, filename, Milvus string, document ID validation |
| `_auto_merger.py` | Hierarchical merge | Score-gap analysis for parent-chunk replacement |
| `_collection_manager.py` | Backend lifecycle | Vector backend resolution, collection CRUD, schema migration, per-session switching |
| `_document_pipeline.py` | Document lifecycle | Upload, text extraction, chunking, embedding, deletion, metadata queries |
| `_search_orchestrator.py` | Search pipeline | Query classification, HyDE, dense search, hybrid fusion, reranking, MMR, auto-merge |

All public methods on `RAGBackend` are thin one-liner delegations. Underscore-prefixed files are internal implementation — the SDK never imports from them directly.

### Session Config Manager

`KnowledgeConfigManager` (`core/session_config/`) uses the **mixin pattern** to decompose 1,700+ lines into focused modules:

| Mixin | Responsibility |
| ----- | -------------- |
| `_data_models.py` | `RAGProfile`, `KnowledgeConfigState`, constants |
| `_profiles.py` | Profile CRUD, presets (speed/standard/balanced/max_quality) |
| `_session_state.py` | Session get/patch/reset, import/export, RAG toggle, document focus |
| `_persistence.py` | File I/O, legacy migration, TTL cleanup |
| `_citations_store.py` | Cross-process citation sharing |
| `_backend_config.py` | Per-session vector backend selection |

### SDK Coupling

The SDK is cleanly decoupled from internal managers. It only uses:

- `get_rag_backend()` — singleton accessor
- `get_knowledge_config_manager()` — config singleton
- Public `RAGBackend` methods (upload, search, delete, etc.)
- `AgenticRAG` from `agentic/`

No SDK code imports underscore-prefixed modules or accesses private attributes.

---

## Troubleshooting

### Common Issues

**Import Error: langchain_core not found**
```bash
pip install langchain-core
```

**Upload fails with "file not found"**
```python
# Use absolute path
from pathlib import Path
result = await client.upload(Path("./doc.pdf").resolve())
```

**Search returns no results**
```python
# 1. Check documents exist and are indexed
docs = await client.list_documents()
print(f"Documents: {len(docs)}")
for d in docs:
    print(f"  {d.filename}: status={d.status}, chunks={d.chunks_count}")

# 2. Lower the score threshold (default may filter out low-confidence matches)
results = await client.search_documents(query, min_score=0.0)

# 3. Check session isolation — documents are scoped to session_id
print(f"Current session: {client.session_id}")

# 4. Try a broader query (exact phrases may not match embedding space)
results = await client.search_documents("general topic keywords")

# 5. If using hybrid search, ensure BM25 index is built
# (documents uploaded before hybrid was enabled won't have BM25 entries)
```

**Session isolation not working**
```python
# Ensure session_id is set correctly
print(f"Current session: {client.session_id}")
```

### Debug Mode

```python
import logging
logging.getLogger("lasagna").setLevel(logging.DEBUG)
```

### Performance Tips

1. **Use presets**: `config_for_speed()` for faster operations
2. **Disable unused features**: Set `enable_hyde=False` if not needed
3. **Batch operations**: Use `upload_batch()` for multiple files
4. **Focus searches**: Use `set_focused_documents()` for scoped queries
5. **Cache context**: Store `get_context()` results when query is stable

---

## Support

- GitHub Issues: https://github.com/dolev31/lasagna/issues
- Documentation: see `docs/` in this repository

---

## See Also

- [Configuration](configuration.md)
- [Backends](backends.md)
