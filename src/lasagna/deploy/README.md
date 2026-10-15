# Lasagna — Deployment Guide

Deploy Lasagna as a retrieval service with document ingestion, hybrid search, reranking, and multi-tenant support. One command to start; swap vector backends without duplicating config.

## Architecture

```text
                           ┌────────────────────────────────┐
                           │         Load Balancer           │
                           └───────────────┬────────────────┘
                                           │
                           ┌───────────────▼────────────────┐
                           │  API Server (Gunicorn+Uvicorn)  │
                           │  POST /api/v1/search            │
                           │  POST /api/v1/documents         │
                           │  GET  /metrics (Prometheus)     │
                           └──┬────────────┬────────────┬───┘
                              │            │            │
                ┌─────────────▼──┐  ┌──────▼──────┐  ┌──▼──────────────┐
                │   Vector DB    │  │    Redis     │  │   PostgreSQL    │
                │  (pluggable)   │  │ Cache+Broker │  │   Metadata DB   │
                └──────▲─────────┘  └──────▲──────┘  └──▲──────────────┘
                       │                   │            │
                       └───────────────────┼────────────┘
                                           │
                           ┌───────────────▼────────────────┐
                           │   Celery Workers (optional)     │
                           │   Background doc processing     │
                           └────────────────────────────────┘

  Optional:  MCP Server (profile: mcp)    Jaeger Tracing (profile: tracing)
```

## Quick Start

```bash
cd src/lasagna/deploy

# 1. Create your .env
cp .env.example .env

# 2. Generate and set the two required secrets
sed -i.bak "s/^RAG_JWT_SECRET=$/RAG_JWT_SECRET=$(openssl rand -hex 32)/" .env
sed -i.bak "s/^POSTGRES_PASSWORD=$/POSTGRES_PASSWORD=$(openssl rand -hex 16)/" .env

# 3. Start the stack (default: Milvus server, Celery worker and beat, MCP)
make up

# 4. Verify
make health
```

To use a different vector backend, edit `VECTOR_BACKEND` in `.env` before `make up`.
`make validate` (run by `make up`) checks the backend, both secrets, and that
`milvus-lite` is not combined with the `celery` or `mcp` profiles: Milvus Lite
is embedded in the api process, so no other container can open it.

Every line of `.env` is also passed into the api, worker, beat and mcp
containers, so LLM API keys and `RAG_*` overrides set there take effect.

**Common backends:**

| Default | Self-hosted alternative | Simplest infra | API only, one process |
| --- | --- | --- | --- |
| `VECTOR_BACKEND=milvus` | `VECTOR_BACKEND=qdrant` | `VECTOR_BACKEND=pgvector` | `VECTOR_BACKEND=milvus-lite` with `COMPOSE_PROFILES=` |
| Milvus server container | Dedicated vector DB | Reuses existing Postgres | No extra container, no Celery |

## Vector Backend Selection

| Backend | Best For | ARM64 | Extra Container | Cloud Option |
| --- | --- | --- | --- | --- |
| **milvus-lite** | API only, single process (no `celery` or `mcp` profile) | Yes | No (embedded) | — |
| **milvus** | Default, self-hosted | Yes | Yes | Zilliz Cloud |
| **pgvector** | Simple deployments, existing Postgres | Yes | No (reuses Postgres) | Any managed PG |
| **qdrant** | Production, high-performance search | Yes | Yes | Qdrant Cloud |
| **weaviate** | Multi-modal, graph-like queries | Yes | Yes | Weaviate Cloud |
| **pinecone** | Fully managed, zero-ops | N/A | No (cloud-only) | Pinecone |

## Docker Compose

### Base Services (always start)

- **api** — FastAPI + Gunicorn (port 8000)
- **redis** — Cache + Celery message broker (port 6379, localhost-bound)
- **postgres** — Metadata database (port 5432, localhost-bound)

### Optional Profiles

Controlled by `COMPOSE_PROFILES` in `.env` (default: `celery,mcp`):

| Profile | Service | Purpose | Enabled by default? |
| --- | --- | --- | --- |
| `celery` | celery-worker, celery-beat | Background document processing; periodic GC | Yes |
| `mcp` | mcp | Standalone MCP server (port 8005) | Yes |
| `tracing` | jaeger | Distributed tracing UI (port 16686) | No |

To run API-only, set `COMPOSE_PROFILES=` (empty). For full stack: `COMPOSE_PROFILES=celery,mcp,tracing`

### Backend Fragment Pattern

The stack is assembled from two compose files:

```bash
# Via Makefile (recommended):
VECTOR_BACKEND=pgvector make up

# Explicit:
docker compose -f docker-compose.yml -f backends/pgvector.yml up -d
```

Each `backends/*.yml` file adds the vector DB service (if needed) and injects `RAG_VECTOR_BACKEND` + connection config into the api, celery-worker, celery-beat and mcp services.

### Shared uploads

The api stores each upload at `/data/uploads/<document_id>/<file>` and hands
that path to the Celery worker, which opens it. The `uploads-data` volume is
mounted at `/data/uploads` on the api, celery-worker and mcp services, and
`RAG_UPLOAD_DIR=/data/uploads` is set for all of them, so the path resolves in
every container. Metadata and session configuration are in Postgres.

### Health checks

Every long-running service has one, so `docker compose ... up -d --wait`
returns when the stack can serve or names the service that cannot:

| Service | Check |
| --- | --- |
| api | `GET /api/v1/health/ready` (vector store, Redis and Postgres answer) |
| celery-worker | `celery inspect ping` against this worker |
| celery-beat | the beat schedule file exists |
| mcp | `GET /health` on port 8005 |

### MCP tools

The mcp service serves the `safe` tool set, which leaves out the six tools that
delete data or reset configuration. Set `LASAGNA_MCP_TOOLS=all` in `.env` to
serve them. `upload_document` over MCP reads server-side paths only inside
`RAG_MCP_UPLOAD_ROOT`, which is unset here; upload through the REST API instead.
Pointing it at `/data/uploads` would let any MCP client ingest another
session's stored files.

## Kubernetes

### Structure

```text
k8s/
├── base/                  # Backend-agnostic resources
│   ├── kustomization.yaml
│   ├── namespace.yaml     # rag-knowledge namespace
│   ├── configmap.yaml     # Non-sensitive config
│   ├── secrets.yaml       # Sensitive values (CHANGE_ME placeholders)
│   ├── deployment-api.yaml
│   ├── deployment-celery.yaml
│   ├── service.yaml       # ClusterIP on port 80
│   └── hpa.yaml           # Autoscaling for API + Celery
└── overlays/              # One per vector backend
    ├── milvus/
    ├── pgvector/
    ├── qdrant/
    ├── weaviate/
    └── pinecone/
```

### Deploy

```bash
# Choose your backend overlay:
kubectl apply -k k8s/overlays/pgvector/

# Or with kustomize directly:
kustomize build k8s/overlays/qdrant/ | kubectl apply -f -
```

### Before Production

1. **Replace all `CHANGE_ME` values** in `secrets.yaml` (or use sealed-secrets / external-secrets-operator)
2. **Generate a real JWT secret**: `openssl rand -hex 32`
3. **Adjust resource limits** in `deployment-api.yaml` based on your embedding model size
4. **Point to external databases** (Postgres, Redis, vector DB) instead of in-cluster instances
5. **Configure the tracing endpoint** in `configmap.yaml` to your OTel collector

### Autoscaling

| Workload | Min | Max | Scale-up trigger |
| --- | --- | --- | --- |
| API | 2 | 10 | CPU > 70% or Memory > 80% |
| Celery | 1 | 8 | CPU > 75% |

## Configuration

### Priority Chain (highest to lowest)

```text
Session overrides  →  RAG_* env vars  →  TOML file  →  Code defaults
```

### Key Environment Variables

| Category | Variable | Default | Notes |
| --- | --- | --- | --- |
| **Required** | `RAG_JWT_SECRET` | (none) | Generate: `openssl rand -hex 32` |
| **Backend** | `RAG_VECTOR_BACKEND` | — | Set by backend fragment |
| **Infra** | `RAG_DATABASE_BACKEND` | `sqlite` | `postgresql` in compose/k8s |
| **Infra** | `RAG_CACHE_BACKEND` | `memory` | `redis` in compose/k8s |
| **Infra** | `RAG_CONNECTION_POOL_SIZE` | `10` | >= workers x 2 |
| **Embedding** | `RAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | k8s uses `bge-large-en-v1.5` |
| **Search** | `RAG_DEFAULT_TOP_K` | `20` | Production: `75` |
| **Search** | `RAG_RERANKER_MODEL` | `mixedbread-ai/mxbai-rerank-large-v1` | |
| **Features** | `RAG_ENABLE_HYBRID_SEARCH` | `true` | |
| **Features** | `RAG_ENABLE_RERANKING` | `true` | |
| **Observe** | `RAG_ENABLE_PROMETHEUS` | `false` | `true` in compose/k8s |
| **Observe** | `RAG_ENABLE_TRACING` | `false` | Enable with Jaeger profile |
| **Network** | `RAG_FORWARDED_ALLOW_IPS` | (none) | Proxy addresses whose `X-Forwarded-For` `lasagna serve` and the image's gunicorn trust. Unset trusts none (the image: loopback only) |
| **Network** | `FORWARDED_ALLOW_IPS` | `127.0.0.1` | gunicorn's own name for the same setting; wins over `RAG_FORWARDED_ALLOW_IPS` in the image |
| **Network** | `RAG_ALLOWED_HOSTS` | (none) | Extra `Host` values the embedded `/mcp` accepts, e.g. a service name |

See [.env.example](.env.example) for the complete reference.

### Production Config

For optimized production defaults, add to your `.env`:

```bash
RAG_SETTINGS_FILE=settings.lasagna_production.toml
```

This enables: larger connection pools, adaptive rate limiting, production embedding model, broader recall (top_k=75) with aggressive reranking.

### Runtime Configuration

- **REST API**: `PATCH /api/v1/config` — change search params per-session
- **Configurator UI**: Web interface for all 91 parameters (enable with `RAG_CONFIGURATOR=true`)
- **SDK**: `KnowledgeConfig` presets — `config_for_speed()`, `config_for_balanced()`, `config_for_max_quality()`

## Makefile Reference

| Command | Description |
| --- | --- |
| `make up` | Validate config and start the stack |
| `make down` | Stop services (keep volumes) |
| `make down-v` | Stop services and delete volumes |
| `make build` | Rebuild the Docker image |
| `make logs` | Follow logs (`SVC=celery-worker` to filter) |
| `make ps` | Show running containers |
| `make health` | Check API health endpoint |
| `make validate` | Check VECTOR_BACKEND, RAG_JWT_SECRET, POSTGRES_PASSWORD, and refuse milvus-lite with the celery or mcp profile |
| `make shell` | Open bash in the API container |
| `make test` | Run deployment tests (offline) |
| `make stress-test` | Stress test current backend (~6 min) |
| `make stress-test-all` | Stress test all backends (~27 min) |
| `make help` | Show all available commands |

## Monitoring & Observability

### Prometheus

- Endpoint: `GET /metrics`
- Multi-worker support: `PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc`
- K8s scrape annotations are preconfigured on the API deployment

### Tracing

- Protocol: OpenTelemetry (OTLP gRPC)
- Enable: `RAG_ENABLE_TRACING=true` + Jaeger profile (`COMPOSE_PROFILES=tracing`)
- Jaeger UI: `http://localhost:16686`

### Health Endpoints

| Endpoint | Purpose | Used by |
| --- | --- | --- |
| `GET /api/v1/health/live` | Process alive? | k8s liveness probe, Docker HEALTHCHECK |
| `GET /api/v1/health/ready` | Can serve traffic? | k8s readiness probe |
| `GET /api/v1/health` | Full status + dependencies | Makefile `health` target, operators |

## Security Checklist

- [ ] Set `RAG_JWT_SECRET` to a random 32-byte hex value
- [ ] Set `POSTGRES_PASSWORD` to a random value (required; there is no default)
- [ ] Set `WEAVIATE_ANONYMOUS_ACCESS=false` for production Weaviate deployments
- [ ] Use sealed-secrets or external-secrets-operator for k8s instead of plaintext `secrets.yaml`
- [ ] All vector DB ports are bound to `127.0.0.1` (default) — change only if external access is needed
- [ ] Redis and Postgres ports are bound to `127.0.0.1` (default)
- [ ] Set `RAG_SETTINGS_FILE=settings.lasagna_production.toml` for hardened defaults

## Stress Testing

```bash
# Test current backend (~6 min)
make stress-test

# Test all local backends (~27 min): pgvector, qdrant, weaviate, milvus-lite
make stress-test-all
```

Uses lightweight models (`all-MiniLM-L6-v2`, 384d) for fast CPU execution. Configurable via env:

| Variable | Default | Description |
| --- | --- | --- |
| `STRESS_NUM_TENANTS` | 10 | Concurrent tenants |
| `STRESS_DOCS_PER_TENANT` | 5 | Documents per tenant |
| `STRESS_CONCURRENT_SEARCHES` | 20 | Parallel search queries |
| `STRESS_ENDURANCE_MINUTES` | 1 | Duration of endurance phase |

Results are logged to `/tmp/stress-<backend>.log`.
