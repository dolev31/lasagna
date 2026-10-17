# REST API Endpoint Reference

Lasagna exposes 41 HTTP endpoints across two route groups: 32 versioned endpoints under `/api/v1/` (including 2 document lifecycle and 6 admin routes) and 9 non-versioned endpoints for the configurator UI and bulk reindexing. All endpoints accept and return JSON, except the two document upload routes, which take multipart form data. Session identity is resolved from headers or tokens on every request.

`tests/snapshots/openapi.json` records the full schema of every route, and `GET /openapi.json` on a running server returns the same document. When this page and that schema disagree, the schema is right. The interactive page at `/docs` loads the Swagger UI scripts and styles from cdn.jsdelivr.net, as FastAPI does, so it needs a network connection; offline, `/openapi.json` and this page have the same reference.

---

## Versioned Routes (`/api/v1/`): 32 endpoints

### Health & Stats

| # | Method | Path | Description | Response Model |
|---|--------|------|-------------|----------------|
| 1 | `GET` | `/api/v1/health/live` | Liveness: the process is up. 200, no dependency checked. | `LivenessResponse` |
| 2 | `GET` | `/api/v1/health/ready` | Readiness: 503 naming each failing component. | `HealthResponse` |
| 3 | `GET` | `/api/v1/health` | Detailed status: the same component states as readiness, always 200. | `HealthResponse` |
| 4 | `GET` | `/api/v1/stats` | System statistics (model, collection, feature flags). | `StatsResponse` |

---

#### `GET /api/v1/health/live`

Liveness: the process is up. Returns 200 and checks no dependency, so a slow or missing dependency never gets the process restarted.

**Response:** `LivenessResponse`

```json
{"status": "alive"}
```

---

#### `GET /api/v1/health/ready`

Readiness: can this instance serve? Checks that the embedding model of the session requests without one use (`lasagna serve -s`, else `default`) is loaded or loadable from disk, then pings the vector store, the metadata database (SQLite or Postgres) and, when this instance uses it, Redis, each with a 2 s timeout. Each component is `up`, `down` or `degraded`. Returns 503 with the same response body when any is `down`, naming each in `failing`. Redis is the one component with a fallback (the search cache and rate limits keep to process memory, uploads are processed in the server instead of by Celery), so Redis down is `degraded`, listed in `degraded`, and answered with 200.

**Response:** `HealthResponse`

```json
{
  "status": "healthy",
  "checks": {
    "embedding_model": {"status": "up", "latency_ms": 0.01, "detail": "BAAI/bge-m3 (dim=1024)"},
    "vector_backend": {"status": "up", "latency_ms": 0.8, "detail": "milvus"},
    "database":       {"status": "up", "latency_ms": 2.1, "detail": "sqlite"}
  },
  "failing": [],
  "degraded": [],
  "require_auth": false,
  "version": "0.1.0"
}
```

`version` is the installed package's version (`lasagna.__version__`). `require_auth` reports whether `RAG_REQUIRE_AUTH` is on. `status` is `"unhealthy"` when a component is down (`failing` lists it, for example `["embedding_model"]`), `"degraded"` when one is degraded and none is down, else `"healthy"`.

| Status Code | Condition |
|-------------|-----------|
| 200 | No component is down (some may be degraded) |
| 503 | A component is down or did not answer within 2 s |

```bash
curl http://localhost:8000/api/v1/health/ready
```

---

#### `GET /api/v1/health`

Detailed status: the same checks, component states, `status`, `failing` and `degraded` as `/ready`, always with 200. Use it for dashboards and monitoring, and `/ready` for orchestrator probes. The MCP `get_system_status` tool reports the same states.

---

#### `GET /api/v1/stats`

With `RAG_REQUIRE_AUTH=true`, this needs a session token like every other data route (401 without one): it names the embedding model, collection and feature flags. The liveness and readiness probes stay open.

**Response:** `StatsResponse`

```json
{
  "embedding_model": "BAAI/bge-m3",
  "embedding_dim": 1024,
  "collection_name": "rag_documents_v2__baai_bge_m3__1024",
  "vector_backend": null,
  "features": {"hybrid_search": true, "reranking": true, "mmr": true, "hyde": false, "ocr": false},
  "cache": {}
}
```

Each embedding model gets its own collection, named `rag_documents_v2__<model>__<dim>`. `vector_backend` and `cache` are declared in `StatsResponse`, but the backend's statistics do not fill them, so they come back `null` and `{}`. `GET /api/v1/health` names the vector backend.

---

### Sessions

| # | Method | Path | Description | Response Model | Status |
|---|--------|------|-------------|----------------|--------|
| 5 | `POST` | `/api/v1/sessions` | Create a signed session token. | `SessionResponse` | 201 |

---

#### `POST /api/v1/sessions`

Creates a self-contained JWT session token encoding `(session_id, user_id)`. The server decodes it on each request without a database lookup. Pass the token in the `X-Session-Token` header on later requests.

**Request:** `SessionRequest` (the body is optional)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | No | Desired session ID. Generated (`sess_<hex>`) if omitted. Max 128 chars. Naming one needs `X-Admin-Token`. |
| `user_id` | string | No | Desired user ID. Generated (`user_<hex>`) if omitted. Max 128 chars. Naming one needs `X-Admin-Token`. |

**Headers:**

| Header | When it is needed |
|--------|-------------------|
| `X-API-Key` | With `RAG_REQUIRE_AUTH=true`: one of the comma-separated keys in `RAG_API_KEYS` (each at least 32 characters). The key's principal is written into the token, and rate limits are counted per principal. |
| `X-Admin-Token` | To name a `session_id` or `user_id`; it must match `RAG_ADMIN_TOKEN`. It also stands in for `X-API-Key` when auth is on. |

Sessions are the isolation boundary, so a requested identity without a valid admin token is refused (403) rather than silently replaced with a generated one.

**Response:** `SessionResponse`

```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "session_id": "sess_a1b2c3d4e5f6",
  "user_id": "user_f6e5d4c3b2a1",
  "expires_at": "2026-03-12T10:30:00+00:00"
}
```

| Status Code | Condition |
|-------------|-----------|
| 201 | Token created |
| 401 | `RAG_REQUIRE_AUTH` is on and `X-API-Key` is missing or wrong |
| 403 | `session_id` or `user_id` named without a valid `X-Admin-Token` |
| 503 | `RAG_REQUIRE_AUTH` is on and `RAG_API_KEYS` lists no usable key |

```bash
# Create a session with generated IDs (auth off: no credentials needed)
curl -X POST http://localhost:8000/api/v1/sessions

# With RAG_REQUIRE_AUTH=true
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "X-API-Key: $LASAGNA_API_KEY"

# Create a session with explicit IDs (needs the admin token)
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: $RAG_ADMIN_TOKEN" \
  -d '{"session_id": "my-session", "user_id": "alice"}'
```

`$LASAGNA_API_KEY` stands for one of the keys you put in `RAG_API_KEYS`; the server reads no variable of that name.

---

### Documents

| # | Method | Path | Description | Response Model | Status |
|---|--------|------|-------------|----------------|--------|
| 6 | `POST` | `/api/v1/documents` | Upload and index a document. | `UploadResponse` | 201 |
| 7 | `POST` | `/api/v1/documents/batch` | Upload several documents in one request. | `BatchUploadResponse` | 201 |
| 8 | `GET` | `/api/v1/documents` | List all session documents. | `DocumentListResponse` | 200 |
| 9 | `GET` | `/api/v1/documents/{document_id}` | Get document metadata. | `DocumentResponse` | 200 |
| 10 | `DELETE` | `/api/v1/documents/{document_id}` | Hard-delete document and all chunks/vectors (irreversible). | `DeleteResponse` | 200 |
| 11 | `GET` | `/api/v1/documents/{document_id}/chunks` | Get document chunks (paginated). | `List[ChunkResponse]` | 200 |
| 12 | `GET` | `/api/v1/documents/{document_id}/status` | Poll processing status. | `DocumentStatusResponse` | 200 |
| 13 | `POST` | `/api/v1/documents/{document_id}/reindex` | Re-index with current settings. | `UploadResponse` | 200 |
| 14 | `POST` | `/api/v1/documents/{document_id}/archive` | Soft-delete a document (recoverable within grace window). | `DocumentResponse` | 200 |
| 15 | `POST` | `/api/v1/documents/{document_id}/restore` | Restore a soft-deleted document within the grace window. | `DocumentResponse` | 200 |

---

#### `POST /api/v1/documents`

Upload and index a document. Accepts multipart form data with a `file` field and an optional `document_id` field. Supported file types: PDF, DOCX, TXT, MD, CSV, JSON, HTML, XML. The document is chunked, embedded, and indexed into the vector store.

By default (`async_processing` on, Celery off) the file is stored and handed to a background thread pool, and the response comes back before indexing finishes, with `status` `"processing"` and `chunks_count` `0`. Poll `GET /api/v1/documents/{document_id}/status` until it reads `ready`. With `async_processing` off the document is indexed in the request.

`document_id` is an idempotency key: re-uploading with an id the session already owns replaces that document instead of creating a duplicate. The replacement is indexed in the request (the response reads `ready`); the old copy answers searches until the new chunks are stored, and then its chunks and file are removed and the quota counts the document once. A replacement that fails leaves the old copy as it was. An id that belongs to another session is refused with 404.

Without a `document_id`, a file whose bytes the session already holds in the collection it searches now (ready, or still within the time processing may take) is not uploaded again: the response describes that document and names it in `duplicate_of`. A copy in another embedding model's collection, a failed one, or one stuck processing does not count, so after a model switch uploading the file again indexes it for the new model. Two identical uploads at the same time index one copy. Send `allow_duplicates=true` to index another copy.

**Request:** Multipart form (`file`, optional `document_id`, optional `allow_duplicates`)

**Response:** `UploadResponse`

```json
{
  "document_id": "doc_abc123",
  "filename": "report.pdf",
  "file_type": "pdf",
  "file_size": 1048576,
  "status": "processing",
  "chunks_count": 0,
  "duplicate_of": null
}
```

| Status Code | Condition |
|-------------|-----------|
| 201 | Document accepted (indexed, or queued for indexing) |
| 400 | Invalid file type or validation error |
| 404 | `document_id` belongs to another session |
| 413 | File exceeds `max_file_size` (50 MB by default) |
| 429 | Quota exceeded |
| 503 | The processing pool is full (`Retry-After: 5`) |
| 500 | Internal server error |

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "X-Session-Token: eyJhbGciOi..." \
  -F "file=@report.pdf"
```

---

#### `POST /api/v1/documents/batch`

Upload several documents in one request: multipart form data with one or more `files` fields. Files are processed concurrently, up to `max_concurrent_uploads` (default 5) at a time, each through the same pipeline as the single upload. Without Celery, a file that finds every processing thread busy waits for one (at most `processing_timeout`, 300 s), so a batch larger than the pool is processed whole and the response returns once every file is accepted. A batch may hold at most `RAG_MAX_BATCH_FILES` files (default 20, 400 above it) and `RAG_MAX_BATCH_BYTES` in total (default 200 MiB, 413 before the body is read). Duplicates are handled as for the single upload, including two identical files in one batch; the optional `allow_duplicates=true` form field indexes them anyway.

**Response:** `BatchUploadResponse`: `total`, `succeeded`, `failed`, and one `results` entry per file (`document_id`, `filename`, `status`, `file_type`, `file_size`, `chunks_count`, `error`, `duplicate_of`).

```bash
curl -X POST http://localhost:8000/api/v1/documents/batch \
  -H "X-Session-Token: eyJhbGciOi..." \
  -F "files=@report.pdf" -F "files=@notes.md"
```

---

#### `GET /api/v1/documents`

List all documents in the current session. The optional `exclude_ids` query parameter takes a comma-separated list of document IDs to leave out.

**Response:** `DocumentListResponse`

```json
{
  "documents": [
    {
      "document_id": "doc_abc123",
      "session_id": "sess_a1b2c3",
      "filename": "report.pdf",
      "file_type": "pdf",
      "file_size": 1048576,
      "status": "ready",
      "chunks_count": 42,
      "upload_date": "2026-03-05T08:00:00Z",
      "error_message": null
    }
  ],
  "total": 1
}
```

---

#### `GET /api/v1/documents/{document_id}`

Get a single document's metadata.

**Response:** `DocumentResponse` (same shape as elements in `DocumentListResponse.documents`)

| Status Code | Condition |
|-------------|-----------|
| 200 | Document found |
| 404 | Document not found |

---

#### `DELETE /api/v1/documents/{document_id}`

Delete a document and all its chunks, vectors, and citations.

**Response:** `DeleteResponse`

```json
{"success": true, "document_id": "doc_abc123", "message": "Document doc_abc123 deleted"}
```

| Status Code | Condition |
|-------------|-----------|
| 200 | Deleted |
| 404 | Document not found |

---

#### `GET /api/v1/documents/{document_id}/chunks`

Get chunks from a document with pagination.

**Query Parameters:**

| Param | Type | Default | Constraints |
|-------|------|---------|-------------|
| `offset` | int | `0` | >= 0 |
| `limit` | int | `50` | 1 to 500 |

**Response:** `List[ChunkResponse]`

```json
[
  {
    "chunk_id": "chk_001",
    "content": "The quarterly revenue increased...",
    "document_id": "doc_abc123",
    "chunk_index": 0,
    "page_number": 1,
    "metadata": {}
  }
]
```

---

#### `GET /api/v1/documents/{document_id}/status`

Poll the processing status of a document, to see when a background upload has finished indexing.

**Response:** `DocumentStatusResponse`

```json
{
  "document_id": "doc_abc123",
  "status": "ready",
  "chunks_count": 42,
  "error_message": null
}
```

Status values: `uploading`, `processing`, `indexing`, `ready`, `error`, `soft_deleted`

| Status Code | Condition |
|-------------|-----------|
| 200 | Document found |
| 404 | Document not found |

---

#### `POST /api/v1/documents/{document_id}/reindex`

Re-index an existing document with the session's current settings. Useful after config changes (e.g. switching embedding model or chunking strategy). The stored file is processed in the request; the document keeps answering searches from its old chunks until the new ones are stored, then the old chunks are removed and the document is recorded in the current model's collection. A reindex that fails at any step leaves the document as it was, and the error says so.

**Response:** `UploadResponse`

| Status Code | Condition |
|-------------|-----------|
| 200 | Re-indexed |
| 400 | The settings refuse the file (type, size) or the embedding size is unknown |
| 404 | Document not found |
| 409 | The document is still processing, or is archived |
| 410 | Source file no longer available in storage |
| 429 | The new chunks would exceed the session's chunk quota |
| 503 | The session's embedding model is not downloaded |
| 500 | Internal server error |

```bash
curl -X POST http://localhost:8000/api/v1/documents/doc_abc123/reindex \
  -H "X-Session-Token: eyJhbGciOi..."
```

---

### Document Lifecycle

| # | Method | Path | Description | Auth |
|---|--------|------|-------------|------|
| 14 | `POST` | `/api/v1/documents/{document_id}/archive` | Soft-delete: moves the document to `soft_deleted` status. Vectors and the stored file stay, and quota is still counted, until GC hard-deletes it. | Session token |
| 15 | `POST` | `/api/v1/documents/{document_id}/restore` | Restore a soft-deleted document to `ready` within the grace window (`soft_delete_grace_hours`, default 24). Returns 409 if the grace window expired. | Session token |

---

### Admin: GC and Lifecycle (requires `X-Admin-Token`)

All endpoints below require the `X-Admin-Token` header matching the `RAG_ADMIN_TOKEN` environment variable (401 when it does not match).
If `RAG_ADMIN_TOKEN` is not set, is a placeholder, or is shorter than 32 characters, these endpoints return `503 Service Unavailable`.

| # | Method | Path | Description |
|---|--------|------|-------------|
| 16 | `POST` | `/api/v1/admin/gc/run` | Run a full 7-phase GC cycle (`?session_id=` limits it to one session). Takes the database advisory lock, and returns at once if another GC run holds it. |
| 17 | `GET` | `/api/v1/admin/gc/orphans` | Read-only orphan scan: lists documents in DB-only or vector-only state. Does NOT modify data. |
| 18 | `POST` | `/api/v1/admin/gc/repair` | Detect and repair orphans. `?dry_run=true` (default) previews changes; `?dry_run=false` applies. |
| 19 | `GET` | `/api/v1/admin/gc/policy` | The lifecycle policy in force (grace window, TTL, GC intervals, safety thresholds). |
| 20 | `PATCH` | `/api/v1/admin/gc/policy` | Change policy fields for this process only; a restart reverts to `[rag.lifecycle]` in `settings.lasagna.toml` and the `RAG_GC_*` variables. Unknown fields are rejected with 422. |
| 21 | `DELETE` | `/api/v1/admin/sessions/{session_id}` | Hard-delete ALL documents in a session. Irreversible. |

---

### Search

| # | Method | Path | Description | Response Model |
|---|--------|------|-------------|----------------|
| 22 | `POST` | `/api/v1/search` | Search with the session's retrieval pipeline. | `SearchResponse` |
| 23 | `POST` | `/api/v1/search/citations` | Search with citation tracking and LLM context. | `SearchWithCitationsResponse` |
| 24 | `POST` | `/api/v1/search/multi-query` | RAG Fusion multi-query expansion search. | `SearchResponse` |
| 25 | `POST` | `/api/v1/search/agentic` | Agentic multi-hop retrieval with iterative search. | `SearchWithCitationsResponse` |

All search endpoints accept the same `SearchRequest` body.

---

#### `SearchRequest` (shared request body)

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `query` | string | (required) | 1 to 10,000 chars | Search query |
| `top_k` | int | `5` | 1 to 100 | Number of results to return |
| `min_score` | float | `null` | | Drop results whose `score` is below this. `null`: no threshold, as in the CLI, the MCP server and the Python SDK |
| `document_ids` | list[string] | `null` | | Filter to specific documents |
| `use_hyde` | bool | `true` | | Enable HyDE query expansion |
| `use_reranking` | bool | `true` | | Enable cross-encoder reranking |
| `routing_tier` | string | `null` | `fast`, `standard`, `comprehensive` | Force a specific routing tier |

---

#### `POST /api/v1/search`

Search with the session's retrieval pipeline. Each stage runs as the session config sets it: hybrid dense and BM25 search, cross-encoder reranking, MMR diversity, HyDE query expansion, adaptive-k retrieval, hierarchical auto-merge. A stage that failed and fell back is named in `degraded_stages`, so an empty list means every configured stage ran. `documents_elsewhere` means some of the session's documents are in another embedding model's collection, so this search could not find them: reindex them with the session's model (`PUT /api/config/knowledge/active` with `reindex: true`), or switch back to the profile that indexed them. A session with no documents (a misspelled `X-Session-ID`, for one) is not searched: the answer is `200` with no results and a `note` naming the sessions that have documents, or how to add one.

`score` is absolute, so it can be compared across queries: with reranking on
(every profile but `speed`) it is the cross-encoder's probability that the chunk
answers the query; with reranking off it is the retrieval score, (cosine
similarity + 1) / 2 for dense search (`speed`) and the fused rank score for hybrid
search. `rank` is the position, from 1.

A server never downloads a model inside a request: when the session's embedding
model is not on disk (for example an unconfigured session, which uses the
settings file's BGE-M3), the search answers `503` with the command that fixes
it, and `/api/v1/health/ready` reports the served session's model as `down`.
`RAG_ALLOW_MODEL_DOWNLOAD_IN_REQUESTS=true` lets the server download instead.

**Response:** `SearchResponse`

```json
{
  "results": [
    {
      "chunk": {
        "chunk_id": "chk_001",
        "content": "Revenue grew 15% year-over-year...",
        "document_id": "doc_abc123",
        "chunk_index": 3,
        "page_number": 2,
        "metadata": {}
      },
      "score": 0.92,
      "rank": 1,
      "document_id": "doc_abc123",
      "filename": "report.pdf"
    }
  ],
  "query": "quarterly revenue growth",
  "total_results": 5,
  "routing": {
    "tier": "standard",
    "strategy": "rule_based",
    "confidence": 0.85,
    "rationale": "Medium complexity factual query",
    "query_type": "factual",
    "estimated_complexity": "moderate"
  },
  "degraded_stages": []
}
```

| Status Code | Condition |
|-------------|-----------|
| 200 | Success |
| 400 | Invalid query or validation error |
| 429 | Quota exceeded |
| 503 | The session's focused documents could not be read |
| 504 | The search ran past `RAG_SEARCH_TIMEOUT` (default 120 s) |
| 500 | Internal server error |

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: eyJhbGciOi..." \
  -d '{"query": "quarterly revenue growth", "top_k": 10}'
```

---

#### `POST /api/v1/search/citations`

Search with citation tracking. Returns search results, stable citations with highlight extraction, and a pre-formatted context string ready for injection into LLM prompts.

**Response:** `SearchWithCitationsResponse`

```json
{
  "results": [...],
  "citations": [
    {
      "citation_id": "cit_abc123",
      "document_id": "doc_abc123",
      "document_name": "report.pdf",
      "chunk_id": "chk_001",
      "chunk_index": 3,
      "page_number": 2,
      "score": 0.92,
      "content": "Revenue grew 15% year-over-year...",
      "highlight": "Revenue grew 15%"
    }
  ],
  "context": "<knowledge_base>\n[1] Revenue grew 15%...\n</knowledge_base>",
  "query": "quarterly revenue growth",
  "total_results": 5,
  "routing": null,
  "degraded_stages": []
}
```

```bash
curl -X POST http://localhost:8000/api/v1/search/citations \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: eyJhbGciOi..." \
  -d '{"query": "What factors affected Q3 performance?"}'
```

---

#### `POST /api/v1/search/multi-query`

RAG Fusion multi-query expansion. Generates `n_variations` alternative query phrasings with an LLM, searches each, and fuses the results with reciprocal rank fusion, weighting the original query twice. It needs `enable_query_expansion` on in the session config; with it off (or `n_variations` 0) the request runs as a plain `POST /api/v1/search`. The `routing_tier` field in `SearchRequest` is ignored.

**Query Parameters:**

| Param | Type | Default |
|-------|------|---------|
| `n_variations` | int | `3` |

**Response:** `SearchResponse`

---

#### `POST /api/v1/search/agentic`

Agentic multi-hop retrieval with iterative search. Uses an LLM agent to decompose complex queries, perform multiple retrieval rounds, and synthesize results with citations. Requires `use_agentic_rag` to be enabled in session config. The `routing_tier` field in `SearchRequest` is ignored. When no LLM is configured, it answers 503 with what to set.

**Response:** `SearchWithCitationsResponse`

Note: the `results` array is empty for agentic search. Results come back in `citations` and the pre-formatted `context` string.

```bash
curl -X POST http://localhost:8000/api/v1/search/agentic \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: eyJhbGciOi..." \
  -d '{"query": "Compare revenue trends across all quarters", "top_k": 15}'
```

---

### Citations

| # | Method | Path | Description | Response Model |
|---|--------|------|-------------|----------------|
| 26 | `GET` | `/api/v1/citations` | List session citations from last search. | `CitationListResponse` |
| 27 | `GET` | `/api/v1/citations/{citation_id}` | Get a specific citation by ID. | `CitationResponse` |

---

#### `GET /api/v1/citations`

List all citations for the current session from the most recent search.

**Response:** `CitationListResponse`

```json
{
  "citations": [
    {
      "citation_id": "cit_abc123",
      "document_id": "doc_abc123",
      "document_name": "report.pdf",
      "chunk_id": "chk_001",
      "chunk_index": 3,
      "page_number": 2,
      "score": 0.92,
      "content": "Revenue grew 15% year-over-year...",
      "highlight": "Revenue grew 15%"
    }
  ],
  "session_id": "sess_a1b2c3",
  "total": 1
}
```

---

#### `GET /api/v1/citations/{citation_id}`

Get a specific citation by ID. Uses a two-tier lookup: first checks the in-memory session store (preserves `content` and `highlight` from the last search), then falls back to the persistent database (stores `content_preview`, no `highlight`).

**Response:** `CitationResponse`

| Status Code | Condition |
|-------------|-----------|
| 200 | Citation found |
| 404 | Citation not found in session store or database |

---

### Config

| # | Method | Path | Description | Response Model |
|---|--------|------|-------------|----------------|
| 28 | `GET` | `/api/v1/config` | Get current session configuration. | `ConfigResponse` |
| 29 | `GET` | `/api/v1/config/profiles` | List available profiles. | `ProfileListResponse` |
| 30 | `PATCH` | `/api/v1/config` | Patch configuration keys. | `ConfigResponse` |
| 31 | `PUT` | `/api/v1/config/profiles/{profile_name}` | Apply a named profile. | `ConfigResponse` |

---

#### `GET /api/v1/config`

Get the current RAG configuration for this session. The response has 60 fields: the 57 patchable keys (see `PATCH /api/v1/config`), `embedding_model`, `embedding_dim`, and `warnings`. An excerpt:

**Response:** `ConfigResponse`

```json
{
  "embedding_model": "BAAI/bge-m3",
  "embedding_dim": 1024,
  "chunking_strategy": "semantic",
  "enable_hybrid_search": true,
  "enable_reranking": true,
  "enable_mmr": true,
  "enable_hyde": false,
  "enable_hierarchical": true,
  "default_top_k": 20,
  "rerank_top_k": 5,
  "similarity_threshold": 0.3,
  "hybrid_alpha": 0.7,
  "mmr_lambda": 0.5,
  "use_agentic_rag": true,
  "enable_prometheus": false,
  "enable_tracing": false,
  "enable_query_routing": true,
  "routing_strategy": "rule_based",
  "routing_default_tier": "standard",
  "agentic_pipeline": "legacy_react",
  "use_structured_graph": false,
  "enable_hallucination_check": false,
  "enable_answer_quality_check": false,
  "max_quality_check_attempts": 2,
  "coverage_scorer_mode": "keyword",
  "final_search_mode": "merge",
  "warnings": []
}
```

---

#### `GET /api/v1/config/profiles`

List available configuration profiles and the currently active profile.

**Response:** `ProfileListResponse`

```json
{
  "profiles": [
    {"name": "speed",       "description": "..."},
    {"name": "standard",    "description": "..."},
    {"name": "balanced",    "description": "..."},
    {"name": "max_quality", "description": "..."}
  ],
  "active": "standard"
}
```

Each profile carries a one-line description. `active` is `null` when no profile has been applied to the session.

---

#### `PATCH /api/v1/config`

Patch session-scoped RAG configuration. Only non-null keys are updated. Unknown keys are rejected with 422. A change that would put a different embedding size into the session's existing collection is refused with 409.

**Request:** `ConfigPatchRequest` (extra="forbid")

`ConfigPatchRequest` accepts 57 keys. The table lists the retrieval and agentic ones; the rest cover chunking (`chunking_strategy`, `chunk_size`, `chunk_overlap`, ...), Docling and OCR, document awareness, context compression and reordering, CRAG, RAG Fusion, `sparse_method`, `rrf_k` and `vector_backend`. `ConfigPatchRequest` in `src/lasagna/api/rest/models.py` has every key with its constraint.

| Field | Type | Constraints |
|-------|------|-------------|
| `enable_hybrid_search` | bool | |
| `enable_reranking` | bool | |
| `enable_mmr` | bool | |
| `enable_hyde` | bool | |
| `enable_hierarchical` | bool | |
| `use_agentic_rag` | bool | |
| `enable_prometheus` | bool | |
| `enable_tracing` | bool | |
| `default_top_k` | int | 1 to 100 |
| `rerank_top_k` | int | 1 to 100 |
| `similarity_threshold` | float | -1.0 to 1.0 |
| `hybrid_alpha` | float | 0.0 to 1.0 |
| `mmr_lambda` | float | 0.0 to 1.0 |
| `enable_query_classification` | bool | |
| `enable_query_routing` | bool | |
| `routing_strategy` | string | `rule_based` or `adaptive` |
| `routing_default_tier` | string | `fast`, `standard`, or `comprehensive` |
| `routing_confidence_threshold` | float | 0.0 to 1.0 |
| `agentic_pipeline` | string | `legacy_react`, `default`, or any registered pipeline name (1 to 100 chars) |
| `use_structured_graph` | bool | |
| `enable_hallucination_check` | bool | |
| `enable_answer_quality_check` | bool | |
| `max_quality_check_attempts` | int | 1 to 10 |
| `coverage_scorer_mode` | string | `keyword` or `llm` |
| `final_search_mode` | string | `merge` or `replace` |

**Response:** `ConfigResponse` (full config after patch)

```bash
curl -X PATCH http://localhost:8000/api/v1/config \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: eyJhbGciOi..." \
  -d '{"enable_hyde": true, "similarity_threshold": 0.5}'
```

---

#### `PUT /api/v1/config/profiles/{profile_name}`

Apply a preset configuration profile for this session.

**Path Parameters:** `profile_name`: one of `speed`, `standard`, `balanced`, `max_quality`

**Response:** `ConfigResponse` (full config after profile applied)

| Status Code | Condition |
|-------------|-----------|
| 200 | Profile applied |
| 404 | Profile name not found |
| 409 | The profile's embedding size does not fit the session's existing collection |

```bash
curl -X PUT http://localhost:8000/api/v1/config/profiles/max_quality \
  -H "X-Session-Token: eyJhbGciOi..."
```

---

### Pipelines

| # | Method | Path | Description | Response Model |
|---|--------|------|-------------|----------------|
| 32 | `GET` | `/api/v1/pipelines` | List all registered agentic pipelines. | `PipelineListResponse` |

---

#### `GET /api/v1/pipelines`

List all registered agentic pipelines. With `RAG_REQUIRE_AUTH=true`, this needs a session token (401 without one), since a pipeline name can be a tenant's own.

**Response:** `PipelineListResponse`

```json
{
  "pipelines": [
    {
      "name": "default",
      "description": "Built-in structured RAG graph (13 nodes, CRAG, quality gates)",
      "class_name": "BuiltinStructuredPipeline"
    },
    {
      "name": "legacy_react",
      "description": "Legacy ReAct agent with monolithic search tool",
      "class_name": "LegacyReActPipeline"
    }
  ],
  "active": "legacy_react",
  "total": 2
}
```

| Field | Type | Description |
|-------|------|-------------|
| `pipelines` | array | All registered pipelines |
| `pipelines[].name` | string | Registry key |
| `pipelines[].description` | string | Human-readable description |
| `pipelines[].class_name` | string | Implementing class name (e.g. `"BuiltinStructuredPipeline"`). Shows `"deferred"` if the factory has not been loaded yet. Use `name` to identify/select a pipeline. |
| `active` | string | Currently active pipeline name from config. May reference a pipeline not in the list if config is stale or the package was uninstalled. |
| `total` | integer | Total registered pipelines |

---

## Non-Versioned Routes: 9 endpoints

### Configurator (`/api/config/knowledge`)

These endpoints serve the configurator frontend UI. They provide full profile CRUD, session-scoped configuration, and import/export. All return `KnowledgeConfigFullResponse`.

**Deprecated for other callers:** they are unversioned because the configurator calls them by these paths, and they will move under `/api/v1` in 0.2.0. A program should read and change a session's settings with `GET` and `PATCH /api/v1/config`.

With `RAG_CONFIGURATOR=true`, the UI's own routes are mounted too: `GET /configurator/api/schema` and `POST /configurator/api/validate`. With `RAG_REQUIRE_AUTH=true`, they need a session token like the routes below (401 without one).

| # | Method | Path | Description |
|---|--------|------|-------------|
| 33 | `GET` | `/api/config/knowledge` | Get full config state (profiles, overrides, RAG status). |
| 34 | `PUT` | `/api/config/knowledge/active` | Set active profile with optional config overrides. |
| 35 | `POST` | `/api/config/knowledge/profile` | Save or update a named profile. |
| 36 | `POST` | `/api/config/knowledge/profile/delete` | Delete a custom profile (built-ins protected). |
| 37 | `POST` | `/api/config/knowledge/profile/rename` | Rename a custom profile (built-ins protected). |
| 38 | `POST` | `/api/config/knowledge/reset` | Reset config to defaults. |
| 39 | `GET` | `/api/config/knowledge/export` | Export full session config as JSON. |
| 40 | `POST` | `/api/config/knowledge/import` | Import config from JSON (validated depth/size). |

---

#### `KnowledgeConfigFullResponse` (shared response)

```json
{
  "schema_version": 1,
  "defaults_source": "settings.lasagna.toml",
  "profiles": [
    {"name": "standard", "description": "...", "config": {"enable_reranking": true}}
  ],
  "active_profile": "standard",
  "session_overrides": {},
  "rag_enabled": true
}
```

---

#### `GET /api/config/knowledge`

The endpoint the configurator UI calls on load. Returns the profiles this caller may see, the active profile, session overrides, and RAG status.

---

#### `PUT /api/config/knowledge/active`

Set the active profile and optionally apply config overrides.

**Request:** `SetActiveProfileRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `active_profile` | string | Yes | Profile name (1 to 100 chars) |
| `config` | object | No | Additional config overrides to apply |

---

#### `POST /api/config/knowledge/profile`

Save or update a named profile globally.

**Request:** `SaveProfileRequest`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `profile_name` | string | Yes | | Profile name (1 to 100 chars) |
| `config` | object | No | `{}` | Profile configuration keys |
| `set_active` | bool | No | `true` | Activate this profile after saving |

---

#### `POST /api/config/knowledge/profile/delete`

Delete a custom profile. Built-in profiles are protected and cannot be deleted.

**Request:** `DeleteProfileRequest`

| Field | Type | Required |
|-------|------|----------|
| `profile_name` | string | Yes |

---

#### `POST /api/config/knowledge/profile/rename`

Rename a custom profile. Built-in profiles are protected.

**Request:** `RenameProfileRequest`

| Field | Type | Required |
|-------|------|----------|
| `old_name` | string | Yes |
| `new_name` | string | Yes |

---

#### `POST /api/config/knowledge/reset`

Reset configuration to defaults.

**Request:** `ResetConfigRequest`

| Field | Type | Default | Values |
|-------|------|---------|--------|
| `scope` | string | `"active"` | `"active"` clears session overrides only; `"all"` resets profiles and the active profile too |

---

#### `GET /api/config/knowledge/export`

Export the full session configuration as JSON. Identical response shape to `GET /api/config/knowledge`.

---

#### `POST /api/config/knowledge/import`

Import configuration from JSON. Validates nesting depth (max 5) and profile size (max 10,000 bytes per profile).

**Request:** `ImportConfigRequest` (extra="forbid")

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `schema_version` | int | 1 to 100 | Config schema version |
| `profiles` | list[object] | Max 50 profiles, 10KB each, depth 5 | Profile definitions |
| `active_profile` | string | 1 to 100 chars | Profile to activate |
| `session_overrides` | object | Max 10KB, depth 5 | Session-scoped overrides |
| `rag_enabled` | bool | | Enable/disable RAG |

---

### Reindex (`/api/knowledge`)

| # | Method | Path | Description | Response Model |
|---|--------|------|-------------|----------------|
| 41 | `POST` | `/api/knowledge/reindex` | Bulk reindex all session documents. | `ReindexResponse` |

---

#### `POST /api/knowledge/reindex`

Re-index all documents in the current session with current settings, one at a time, as the single-document reindex does. A document whose reindex fails keeps its previous index; the response is then `partial`, and `errors` names each such document with the reason.

**Request:** `ReindexRequest`

| Field | Type | Note |
|-------|------|------|
| `session_id` | string | **Deprecated.** Ignored: the session comes from the identity headers. Will be removed in a future version. |

**Response:** `ReindexResponse`

```json
{
  "status": "completed",
  "documents_reindexed": 3,
  "message": null,
  "errors": null
}
```

Partial failure example:

```json
{
  "status": "partial",
  "documents_reindexed": 2,
  "message": null,
  "errors": ["doc_xyz789: source file no longer available in storage"]
}
```

```bash
curl -X POST http://localhost:8000/api/knowledge/reindex \
  -H "Content-Type: application/json" \
  -H "X-Session-Token: eyJhbGciOi..." \
  -d '{}'
```

---

## Authentication

Session identity is resolved on every request from headers and tokens. No login endpoint is required: identity is stateless.

With `RAG_REQUIRE_AUTH=true`, only a verified `X-Session-Token` names a session. An unsigned `X-Session-ID` header or `session_id` query parameter is refused with 401, and so is the `default` fallback unless `RAG_ALLOW_ANONYMOUS_SESSION=true`. Tokens come from `POST /api/v1/sessions`, which then needs an `X-API-Key` from `RAG_API_KEYS`. The health probes stay open. With auth off, the server refuses to start on a non-loopback address unless `RAG_BIND_IS_PROTECTED=true` says something outside the process restricts access.

These security variables are read under their `RAG_` names only; the `LASAGNA_` aliases apply to settings and data paths.

### Session Identity Resolution

Resolved in priority order (first match wins):

| Priority | Source | Mechanism |
|----------|--------|-----------|
| 1 | `X-Session-Token` header | JWT decoded `sid` claim |
| 2 | `X-Session-ID` header | Direct value |
| 3 | `session_id` query parameter | Direct value |
| 4 | (fallback) | `"default"` |

Session IDs are capped at 128 characters. Exceeding this returns 400. An `X-Session-Token` that is present but invalid or expired is refused with 401 rather than falling through to the next source.

### User Identity Resolution

| Priority | Source | Mechanism |
|----------|--------|-----------|
| 1 | `Authorization: Bearer <jwt>` | Decoded `sub` claim (upstream auth) |
| 2 | `X-Session-Token` header | Decoded `sub` claim |
| 3 | `X-User-ID` header | Direct value (requires `RAG_ALLOW_HEADER_IDENTITY=true`) |
| 4 | (fallback) | Auto-assigned `anon_<uuid>` |

### Session Token JWT Claims

Tokens created via `POST /api/v1/sessions` contain:

| Claim | Description |
|-------|-------------|
| `sub` | User ID |
| `sid` | Session ID |
| `iat` | Issued-at timestamp |
| `exp` | Expiry timestamp (default: 7 days, configurable via `RAG_SESSION_TOKEN_TTL_SECONDS`) |
| `iss` | `"lasagna"` (tokens issued as `"rag-knowledge"` are still accepted) |
| `typ` | `"session"` |
| `prn` | The principal that minted it (an API key's digest, or `admin`); present only with `RAG_REQUIRE_AUTH` on. A token whose key is no longer in `RAG_API_KEYS` is refused with 401 |

Tokens are signed with `RAG_JWT_SECRET` (env var). If unset, an ephemeral secret is generated per process, so tokens stop working on restart. With `RAG_REQUIRE_AUTH` on, an unset or placeholder secret is a startup error.

---

## Error Responses

Errors raised by the routes return `ErrorResponse`, which in practice carries only `detail` (no route sets `error_code`):

```json
{"detail": "Description of the error"}
```

Request validation failures (422) use FastAPI's own shape, where `detail` is a list of `{loc, msg, type}` objects. Every response carries an `X-Request-ID` header, which the server log records.

### Status Code Reference

| Code | Meaning | Common Triggers |
|------|---------|-----------------|
| 400 | Validation error | Invalid input, `session_id` exceeds 128 chars, malformed query, too many files in a batch |
| 401 | Unauthorized | Invalid or expired `X-Session-Token`; with `RAG_REQUIRE_AUTH` on, a missing credential or `X-API-Key`; `X-User-ID` sent without `RAG_ALLOW_HEADER_IDENTITY=true`; wrong `X-Admin-Token` |
| 403 | Forbidden | `POST /api/v1/sessions` naming an identity without `X-Admin-Token` |
| 404 | Not found | Document, citation, or profile not found; `document_id` owned by another session |
| 409 | Conflict | Restore after the grace window; an embedding size that does not fit the session's collection |
| 410 | Gone | Source file no longer available (reindex) |
| 413 | Payload too large | File exceeds `max_file_size`, batch exceeds `RAG_MAX_BATCH_BYTES`, body over the route's limit |
| 422 | Validation error | Pydantic model validation failure, unknown config keys |
| 429 | Rate limited / quota exceeded | `Retry-After` header included for rate limits |
| 500 | Internal server error | Unhandled backend exception |
| 503 | Service unavailable | Backend not initialized (startup), dependencies down (readiness), processing pool full, admin token not configured, no LLM for agentic search |
| 504 | Gateway timeout | Search ran past `RAG_SEARCH_TIMEOUT` |

---

## Response Models Reference

| Model | Key Fields |
|-------|------------|
| `LivenessResponse` | `status` |
| `HealthResponse` | `status`, `checks: {name: {status, latency_ms, detail}}`, `failing`, `require_auth`, `version` |
| `StatsResponse` | `embedding_model`, `embedding_dim`, `collection_name`, `vector_backend`, `features`, `cache` |
| `SessionResponse` | `token`, `session_id`, `user_id`, `expires_at` |
| `UploadResponse` | `document_id`, `filename`, `file_type`, `file_size`, `status`, `chunks_count` |
| `BatchUploadResponse` | `total`, `succeeded`, `failed`, `results: [{document_id, filename, status, file_type, file_size, chunks_count, error}]` |
| `DocumentResponse` | `document_id`, `session_id`, `filename`, `file_type`, `file_size`, `status`, `chunks_count`, `upload_date`, `error_message` |
| `DocumentListResponse` | `documents: [DocumentResponse]`, `total` |
| `DeleteResponse` | `success`, `document_id`, `message` |
| `DocumentStatusResponse` | `document_id`, `status`, `chunks_count`, `error_message` |
| `ChunkResponse` | `chunk_id`, `content`, `document_id`, `chunk_index`, `page_number`, `metadata` |
| `SearchResponse` | `results: [SearchResultResponse]`, `query`, `total_results`, `routing?`, `degraded_stages`, `note?` |
| `SearchResultResponse` | `chunk: ChunkResponse`, `score`, `rank`, `document_id`, `filename` |
| `SearchWithCitationsResponse` | `results`, `citations: [CitationResponse]`, `context`, `query`, `total_results`, `routing?`, `degraded_stages`, `note?` |
| `CitationResponse` | `citation_id`, `document_id`, `document_name`, `chunk_id`, `chunk_index`, `page_number`, `score`, `content`, `highlight`, `highlights`, `highlight_ranges` |
| `CitationListResponse` | `citations: [CitationResponse]`, `session_id`, `total` |
| `ConfigResponse` | 60 fields: the 57 keys `PATCH /api/v1/config` accepts, plus `embedding_model`, `embedding_dim`, `warnings` |
| `ProfileListResponse` | `profiles: [{name, description}]`, `active` |
| `KnowledgeConfigFullResponse` | `schema_version`, `defaults_source`, `profiles`, `active_profile`, `session_overrides`, `rag_enabled` |
| `ReindexResponse` | `status`, `documents_reindexed`, `message`, `errors` |
| `ErrorResponse` | `detail`, `error_code` |
| `RoutingInfo` | `tier`, `strategy`, `confidence`, `rationale`, `query_type`, `estimated_complexity` |
