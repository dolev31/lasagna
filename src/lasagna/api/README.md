# API Layer — RAG Knowledge Base Facade & REST API

Public interface for the RAG Knowledge system: a thread-safe singleton facade (`RAGBackend`) backed by four specialized managers, exposed through a FastAPI REST API with 38 endpoints, middleware chain, stateless JWT sessions, and optional embedded MCP server.

---

## Module Structure

### Core API (7 files)

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports: `RAGBackend`, `get_rag_backend`, `initialize`, `reset_instance`, `close_backend` |
| `rag_backend.py` | `RAGBackend` singleton facade — delegates to 4 managers below |
| `_collection_manager.py` | Vector backend lifecycle, per-session resolution, bounded LRU (max 500), schema migration |
| `_document_pipeline.py` | Upload → process → chunk → embed → index; delete with `keep_file` support; `get_document_by_id` O(1) lookup |
| `_search_orchestrator.py` | Full SOTA search pipeline: routing → cache → HyDE → embed → dense → hybrid fusion → rerank → MMR → auto-merge → cite |
| `_auto_merger.py` | Hierarchical chunk merging (LlamaIndex port) — replaces sibling child chunks with parent content |
| `_validation.py` | Input validation: session IDs, document IDs, filename sanitization, Milvus string escaping |

### REST Layer (14 files)

| File | Purpose |
|------|---------|
| `rest/app.py` | FastAPI factory, CORS, middleware chain, lifespan, Prometheus `/metrics`, MCP embed, configurator |
| `rest/dependencies.py` | `get_backend`, `get_session_id`, `get_user_id` — FastAPI dependency injection |
| `rest/models.py` | Pydantic request/response models for all endpoints |
| `rest/session_token.py` | JWT create/decode, `extract_session_id` shared logic |
| `rest/routes/__init__.py` | `APIRouter(prefix="/api/v1")` — includes 7 route modules |
| `rest/routes/health.py` | 4 endpoints: liveness, readiness, full health, stats |
| `rest/routes/documents.py` | 9 endpoints: upload, list, get, delete, chunks, status, reindex, **archive, restore** |
| `rest/routes/search.py` | 4 endpoints: search, search with citations, multi-query, agentic |
| `rest/routes/citations.py` | 2 endpoints: list citations, get citation by ID (two-tier lookup) |
| `rest/routes/config.py` | 4 endpoints: get config, list profiles, patch config, apply profile |
| `rest/routes/sessions.py` | 1 endpoint: create session token (POST) |
| `rest/routes/gc_admin.py` | **4 endpoints** under `/api/v1/admin/`: run GC, detect orphans, repair orphans, bulk-delete session (all require `X-Admin-Token`) |
| `rest/routes/knowledge_config.py` | 8 endpoints at `/api/config/knowledge/` for configurator UI |
| `rest/routes/reindex.py` | 1 endpoint at `/api/knowledge/reindex` for bulk reindex |
| `rest/DESIGN.md` | Existing design doc: identity model, middleware flow, security |

---

## Quick Start

**Programmatic**

```python
from lasagna.api import RAGBackend

backend = RAGBackend()
doc = backend.upload_document("report.pdf", session_id="user123")
results = backend.search("quarterly revenue", session_id="user123")
```

**REST API**

```bash
uvicorn lasagna.api.rest.app:app --host 0.0.0.0 --port 8000
```

**Docker**

```bash
cd src/lasagna/deploy
docker compose up
```

---

## Integration Points

The API layer can be accessed through 4 paths:

**1. RAGBackend direct** — Python code calls the facade directly.

```python
from lasagna.api import RAGBackend

backend = RAGBackend()
results = backend.search("query", session_id="sess_123")
```

**2. SDK Client** — High-level SDK wrapping RAGBackend or remote REST API.

```python
from lasagna.sdk import KnowledgeClient

async with KnowledgeClient() as client:
    results = await client.search("query")
```

**3. MCP Server** — `search_knowledge` MCP tool delegates to RAGBackend.

**4. REST API** — 31 HTTP endpoints at `/api/v1/` (versioned) and `/api/config/knowledge/` + `/api/knowledge/` (non-versioned).

```bash
# Create session
curl -X POST http://localhost:8000/api/v1/sessions

# Upload document
curl -X POST http://localhost:8000/api/v1/documents \
  -H "X-Session-Token: <token>" \
  -F "file=@report.pdf"

# Search
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-Session-Token: <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "quarterly revenue"}'
```

---

## Endpoint Summary

| Group | Prefix | Count | Auth |
|-------|--------|-------|------|
| Health & Stats | `/api/v1/health/*`, `/api/v1/stats` | 4 | None |
| Sessions | `/api/v1/sessions` | 1 | None |
| Documents | `/api/v1/documents` | 7 | Session token / header |
| Search | `/api/v1/search` | 4 | Session token / header |
| Citations | `/api/v1/citations` | 2 | Session token / header |
| Config | `/api/v1/config` | 4 | Session token / header |
| Configurator | `/api/config/knowledge` | 8 | Session token / header |
| Reindex | `/api/knowledge/reindex` | 1 | Session token / header |
| **Total** | | **31** | |

---

## Deep Dive Documentation

| Document | Content |
|----------|---------|
| [docs/architecture.md](docs/architecture.md) | Request flows, dependency graph, threading model, design decisions |
| [docs/endpoints.md](docs/endpoints.md) | Complete 31-endpoint REST API reference with curl examples |
| [docs/config.md](docs/config.md) | Environment variables, session tokens, middleware, validation constraints |
| [rest/DESIGN.md](rest/DESIGN.md) | Session identity flow, middleware design, security model |

---

## Cross-References

- Agentic RAG: [agentic/README.md](../agentic/README.md) — multi-hop retrieval with quality gates
- SDK: [sdk/README.md](../sdk/README.md) — client SDK documentation index
- Core config: `settings.lasagna.toml` — all configuration keys
