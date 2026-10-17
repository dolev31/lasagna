# Architecture: API Layer for Lasagna

---

## Package Overview

The `api/` package is the interface layer the SDK, the CLI, the MCP server and the REST server share. It exposes a thread-safe singleton facade (`RAGBackend`) that delegates to four specialized managers, and a FastAPI REST API with 41 endpoints, a middleware chain, and stateless session tokens.

Components:

- **RAGBackend**: thread-safe singleton facade with double-check locking, delegating to 4 managers
- **4 specialized managers**: `CollectionManager`, `DocumentPipeline`, `SearchOrchestrator`, `AutoMerger`
- **FastAPI REST API**: 41 endpoints across 10 route modules, 4 custom middleware layers plus CORS, JWT session tokens
- **Document lifecycle**: soft-delete (`archive`), restore, bulk session delete, APScheduler-driven GC

---

## Request Flow

```
Client
  |
  |  HTTPS
  v
Request-ID MW          Attach X-Request-ID (generated, or echoed when well-formed)
  |
  v
Body-Size-Limit MW     413 before a body over the route's limit is read
  |
  v
Rate-Limit MW          Per-tenant adaptive token bucket
  |                    Exempt: /, /metrics, and paths under /health,
  |                            /api/v1/health, /docs, /openapi.json, /redoc
  |                    429 + Retry-After on rejection
  v
Timing MW              X-Response-Time header
  |                    Prometheus histogram + load estimator
  |                    Same exempt paths
  v
CORSMiddleware         (allow_origins from RAG_CORS_ORIGINS; none by default)
  |
  v
FastAPI Route Handler
  |
  |  Depends(...)
  v
Dependencies
  |- get_backend()     RAGBackend singleton (503 if not ready)
  |- get_session_id()  Token > X-Session-ID > query param > "default"
  |- get_user_id()     Bearer JWT > token > X-User-ID > anon_<uuid>
  |
  v
RAGBackend (singleton facade)
  |
  |- CollectionManager      Vector backend lifecycle
  |- DocumentPipeline       Upload / process / delete
  |- SearchOrchestrator     Search pipeline
  |- AutoMerger             Hierarchical chunk merging
  |
  v
Core Layer (config, schema, database, session_config, quota, rate_limiter)
```

---

## Internal Module Dependencies

```
api/__init__.py
 └── rag_backend.py (RAGBackend facade, singleton)
      ├── _collection_manager.py  (vector backend lifecycle, LRU tracking)
      ├── _document_pipeline.py   (upload, process, delete, get_documents)
      ├── _search_orchestrator.py  (search pipeline: route → cache → HyDE →
      │                             embed → dense → hybrid → rerank → MMR →
      │                             auto-merge → cite)
      └── _auto_merger.py          (hierarchical chunk merging)

api/rest/
 ├── app.py            (FastAPI factory, middleware, lifespan, Prometheus,
 │                      MCP embed, configurator mount)
 ├── dependencies.py   (get_backend, get_session_id, get_user_id)
 ├── models.py         (Pydantic request/response models)
 ├── session_token.py  (JWT create/decode, extract_session_id)
 └── routes/
      ├── __init__.py        APIRouter prefix=/api/v1, includes 8 modules
      ├── health.py          4 endpoints: liveness, readiness, full health, stats
      ├── documents.py      10 endpoints: upload, batch upload, list, get,
      │                                   delete, chunks, status, reindex,
      │                                   archive, restore
      ├── search.py          4 endpoints: search, citations, multi-query, agentic
      ├── citations.py       2 endpoints: list, get by ID
      ├── config.py          4 endpoints: get, list profiles, patch, apply profile
      ├── sessions.py        1 endpoint:  POST /sessions (identity handshake)
      ├── pipelines.py       1 endpoint:  GET /pipelines
      ├── gc_admin.py        6 endpoints at /api/v1/admin/ (X-Admin-Token)
      ├── knowledge_config.py  8 endpoints at /api/config/knowledge/
      └── reindex.py           1 endpoint  at /api/knowledge/reindex
```

---

## Search Pipeline Flow

The `SearchOrchestrator` runs the retrieval pipeline, with each stage on or off as the session config sets it. The routing tier determines depth of processing.

```
Query
  |
  v
Route (FAST / STANDARD / COMPREHENSIVE)
  |
  v
Cache Check              Fingerprint-isolated per session
  |--- hit -------> Return cached results
  |--- miss
  v
HyDE Expansion           When enable_hyde is on
  |
  v
Embed Query              The configured embedder, with its model card's query prefix
  |
  v
Dense Vector Search      Top-K nearest neighbors from vector backend
  |                      (a retriever component, when one is named,
  |                      replaces this step and the fusion below)
  v
Hybrid Fusion            BM25 + dense via weighted Reciprocal Rank Fusion
  |
  v
Cross-Encoder Reranking  Score-based re-ordering
  |
  v
MMR Diversity Filtering  Remove near-duplicate results
  |
  v
Auto-Merge               Hierarchical parent chunk replacement
  |                      (when enough sibling children are retrieved)
  v
Citation Extraction      Stable hash-based citation tracking
  |
  v
Cache Put                Store results for future queries
  |
  v
Return Results
```

For the **COMPREHENSIVE** tier, `SearchOrchestrator` delegates to `AgenticRAG` for multi-hop retrieval with CRAG grading, Self-RAG quality gates, and query decomposition. In that tier, HyDE follows `agentic_hyde_strategy` (`off`, `always`, `first_hop`, `adaptive`). See [`agentic/docs/architecture.md`](../../agentic/docs/architecture.md) for the graph pipeline.

---

## Upload Pipeline Flow

The `DocumentPipeline` handles document ingestion, from validation to the `ready` status. By default the steps after Store File run on a background thread pool (`async_processing`), or on a Celery worker when `celery_enabled` is on, and the upload returns with status `processing`.

```
Validate File
  |  Check size limits and the allowed file extensions
  v
Quota Reserve
  |  TOCTOU-safe: check_and_reserve_upload (atomic)
  v
Store File
  |  Write to storage backend (local, S3, or an fsspec filesystem such as GCS)
  v
Extract Text
  |  Docling processor (preferred) or legacy extractor + OCR fallback
  v
Chunking
  |  The configured chunker (default: semantic), structure-preserving
  v
Embed Chunks
  |  The configured embedding model (default BAAI/bge-m3, 1024 dim), batched
  v
Vector Insert
  |  The configured vector backend (Milvus, Qdrant, Pinecone, Weaviate,
  |  pgvector or OpenSearch), into the model's own collection
  |  rag_documents_v2__<model>__<dim>
  v
BM25 Index Update
  |  Inverted index for hybrid search
  v
Quota Finalize
  |  finalize_upload_reservation (commit or rollback)
  v
Update Document Status --> ready
```

---

## RAGBackend Singleton Pattern

Thread-safe singleton using `__new__` + `__init__` with double-check locking. Both methods acquire the lock to prevent concurrent initialization when two threads receive the same instance from `__new__` and then both enter `__init__`.

```python
_instance: Optional[RAGBackend] = None
_instance_lock = threading.Lock()

class RAGBackend:
    def __new__(cls):
        global _instance
        if _instance is None:
            with _instance_lock:
                if _instance is None:           # double-check
                    _instance = super().__new__(cls)
                    _instance._initialized = False
        return _instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        with _instance_lock:
            if self._initialized:               # double-check
                return
            self._initialized = True
            self._do_init()                     # heavy: GPU, vector connections
```

- `get_rag_backend()`: creates or returns the existing instance.
- `close_backend()`: releases GPU models, vector connections, and caches.

---

## Middleware Chain

Registered in `create_app()`. FastAPI middleware executes outermost-first (last registered runs first):

```
Registration order in code:                     Execution order (outermost-first):

app.add_middleware(CORSMiddleware)                Request-ID       (1st to run)
app.middleware(_timing_middleware)                Body-Size-Limit  (2nd)
app.middleware(_rate_limit_middleware)            Rate-Limit       (3rd)
app.add_middleware(BodySizeLimitMiddleware)       Timing           (4th)
app.middleware(_request_id_middleware)            CORS             (5th)
                                                  Handler          (innermost)
```

| # | Middleware | Responsibility |
|---|-----------|---------------|
| 1 | **Request-ID** | Attaches the `X-Request-ID` header (a client's value when it is well-formed, else a generated one) and puts it in every log line of the request. |
| 2 | **Body-Size-Limit** | Refuses with `413` a declared `Content-Length` over the route's limit without reading the body, and counts a chunked body as it arrives (`api/rest/limits.py`). |
| 3 | **Rate-Limit** | Per-tenant adaptive token bucket with load-based scaling. Extracts `session_id` via shared helper. Caches decoded session token on `request.state` for zero-cost reuse by dependencies. Returns `429` + `Retry-After` when exceeded. |
| 4 | **Timing** | Records `X-Response-Time` header, feeds Prometheus `rag.http.request_duration_ms` histogram and rate-limiter load estimator. |
| 5 | **CORS** | Allows the origins in `RAG_CORS_ORIGINS`; with it unset, cross-origin requests are denied. |
| 6 | **Handler** | FastAPI route execution. |

Exempt paths (rate-limit + timing): `/` and `/metrics`, and every path under `/health`, `/api/v1/health`, `/docs`, `/openapi.json`, `/redoc`. `/api/v1/sessions` is deliberately not exempt, so token minting is rate-limited.

Rate limiting runs before timing so rejected requests do not skew latency metrics.

---

## Session Identity Model

Stateless JWT tokens signed with `RAG_JWT_SECRET` (HS256).

```
Token claims:
{
  "sub": "<user_id>",
  "sid": "<session_id>",
  "typ": "session",
  "iat": <unix_timestamp>,
  "exp": <unix_timestamp>,
  "iss": "lasagna",
  "prn": "<principal>"        (only with RAG_REQUIRE_AUTH on)
}
```

Tokens with the issuer `rag-knowledge` are still accepted.

| Property | Value |
|----------|-------|
| Algorithm | HS256 |
| Default TTL | 604800s (7 days), `RAG_SESSION_TOKEN_TTL_SECONDS` |
| Secret source | `RAG_JWT_SECRET` env var |
| Dev fallback | Ephemeral in-process secret (tokens stop working on restart); with `RAG_REQUIRE_AUTH` on, `create_app()` refuses to build without a real secret |
| Max session_id length | 128 characters |
| Revocation | Rotate `RAG_JWT_SECRET` (no per-token revocation) |

Identity resolution priority:

| `get_user_id()` | `get_session_id()` |
|------------------|--------------------|
| 1. JWT Bearer `sub` (upstream auth) | 1. `X-Session-Token` decoded `sid` |
| 2. `X-Session-Token` decoded `sub` | 2. `X-Session-ID` header |
| 3. `X-User-ID` header (opt-in) | 3. `session_id` query param |
| 4. Auto-assign `anon_<uuid>` | 4. `"default"` |

With `RAG_REQUIRE_AUTH` on, `get_session_id()` accepts only step 1: an unsigned header or query value, or the `default` fallback (unless `RAG_ALLOW_ANONYMOUS_SESSION` is set), is refused with 401.

Cross-reference: see [`rest/DESIGN.md`](../rest/DESIGN.md) for session flow diagrams.

---

## Threading Model

| Context | Mechanism |
|---------|-----------|
| RAGBackend | Singleton with double-check locking; safe for concurrent async handlers. |
| SearchOrchestrator | Thread-local storage (`threading.local()`) for routing decisions; prevents cross-request metadata leaks. |
| Async route handlers | Use `run_in_threadpool` for all sync backend calls. |
| `SearchOrchestrator.async_search()` | Uses `run_in_executor` to offload sync search to thread pool. |
| Reindex (batch) handler | Uses `asyncio.to_thread` to avoid blocking the event loop during heavy I/O. |
| CollectionManager | Per-collection init lock prevents duplicate schema creation under concurrent requests. |

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Singleton facade | `RAGBackend` with double-check locking | Single init of heavy resources (GPU models, vector connections, caches). |
| 4 specialized managers | SRP decomposition | Each class is focused on one domain: collections, documents, search, merging. |
| Versioned API | `/api/v1/` prefix | Forward compatibility for breaking changes. |
| Non-versioned configurator | `/api/config/knowledge/` | Matches configurator UI path conventions (frontend hardcodes paths). |
| Stateless sessions | JWT tokens | No session storage needed; horizontal scaling without shared state. |
| Two-tier citation lookup | Session store then DB fallback | Preserves API contract (`content` + `highlight`) for recent searches while supporting persistence. |
| Reindex keeps the old copy | New chunks go in under new ids; the old ones are removed after | A reindex that fails (model not downloaded, quota, extraction) leaves the document searchable. |
| Middleware-cached token decode | `request.state.decoded_session_token` | Single HMAC verification per request; dependencies reuse at zero cost. |
| Exempt paths for rate-limit + timing | Tuple of prefixes checked per request | Health probes and docs must never be throttled or skew metrics. |
| Thread-local routing | `threading.local()` in SearchOrchestrator | Concurrent requests cannot overwrite each other's routing tier metadata. |
