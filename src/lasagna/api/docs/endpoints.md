# REST API Endpoint Reference

The RAG Knowledge Base exposes 38 HTTP endpoints across two route groups: 28 versioned endpoints under `/api/v1/` (including 2 document lifecycle + 4 admin GC routes) and 10 non-versioned endpoints for the configurator UI and bulk operations. All endpoints accept and return JSON (except document upload, which uses multipart form data). Session identity is resolved from headers/tokens on every request.

---

## Versioned Routes (`/api/v1/`) -- 28 endpoints

### Health & Stats

| # | Method | Path | Description | Response Model |
|---|--------|------|-------------|----------------|
| 1 | `GET` | `/api/v1/health/live` | Liveness probe. Always returns 200. | `LivenessResponse` |
| 2 | `GET` | `/api/v1/health/ready` | Readiness probe. Returns 503 if dependencies are down. | `HealthResponse` |
| 3 | `GET` | `/api/v1/health` | Full health check. Always returns 200. | `HealthResponse` |
| 4 | `GET` | `/api/v1/stats` | System statistics (model, features, cache). | `StatsResponse` |

---

#### `GET /api/v1/health/live`

Kubernetes liveness probe. Always returns 200 unless the process is deadlocked.

**Response:** `LivenessResponse`

```json
{"status": "alive"}
```

---

#### `GET /api/v1/health/ready`

Readiness probe. Checks embedding model, vector backend, and metadata database. Returns 503 with the same response body when any critical dependency is down.

**Response:** `HealthResponse`

```json
{
  "status": "healthy",
  "checks": {
    "embedding_model": {"status": "up", "latency_ms": 1.2, "detail": "all-MiniLM-L6-v2 (dim=384)"},
    "vector_backend": {"status": "up", "latency_ms": 0.8, "detail": "milvus"},
    "database":       {"status": "up", "latency_ms": 2.1, "detail": "sqlite"}
  },
  "version": "1.0.0"
}
```

| Status Code | Condition |
|-------------|-----------|
| 200 | All dependencies healthy |
| 503 | One or more dependencies down |

```bash
curl http://localhost:8000/api/v1/health/ready
```

---

#### `GET /api/v1/health`

Full health check -- same dependency checks as `/ready` but always returns 200 regardless of dependency status. Use for dashboards and monitoring.

---

#### `GET /api/v1/stats`

**Response:** `StatsResponse`

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "collection_name": "rag_default",
  "vector_backend": "milvus",
  "features": {"hybrid_search": true, "reranking": true, "hyde": true},
  "cache": {"hit_rate": 0.72, "size": 128}
}
```

---

### Sessions

| # | Method | Path | Description | Response Model | Status |
|---|--------|------|-------------|----------------|--------|
| 5 | `POST` | `/api/v1/sessions` | Create a signed session token. | `SessionResponse` | 201 |

---

#### `POST /api/v1/sessions`

Creates a self-contained JWT session token encoding `(session_id, user_id)`. The server decodes it on each request without database lookup. Pass the token via `X-Session-Token` header on subsequent requests.

**Request:** `SessionRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | string | No | Desired session ID. Auto-generated (`sess_<hex>`) if omitted. Max 128 chars. |
| `user_id` | string | No | Desired user ID. Auto-generated (`user_<hex>`) if omitted. Max 128 chars. |

**Response:** `SessionResponse`

```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "session_id": "sess_a1b2c3d4e5f6",
  "user_id": "user_f6e5d4c3b2a1",
  "expires_at": "2026-03-12T10:30:00+00:00"
}
```

```bash
# Create session with auto-generated IDs
curl -X POST http://localhost:8000/api/v1/sessions

# Create session with explicit IDs
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my-session", "user_id": "alice"}'
```

---

### Documents

| # | Method | Path | Description | Response Model | Status |
|---|--------|------|-------------|----------------|--------|
| 6 | `POST` | `/api/v1/documents` | Upload and index a document. | `UploadResponse` | 201 |
| 7 | `GET` | `/api/v1/documents` | List all session documents. | `DocumentListResponse` | 200 |
| 8 | `GET` | `/api/v1/documents/{document_id}` | Get document metadata. | `DocumentResponse` | 200 |
| 9 | `DELETE` | `/api/v1/documents/{document_id}` | Hard-delete document and all chunks/vectors (irreversible). | `DeleteResponse` | 200 |
| 10 | `GET` | `/api/v1/documents/{document_id}/chunks` | Get document chunks (paginated). | `List[ChunkResponse]` | 200 |
| 11 | `GET` | `/api/v1/documents/{document_id}/status` | Poll processing status. | `DocumentStatusResponse` | 200 |
| 12 | `POST` | `/api/v1/documents/{document_id}/reindex` | Re-index with current settings. | `UploadResponse` | 200 |
| 13 | `POST` | `/api/v1/documents/{document_id}/archive` | Soft-delete a document (recoverable within grace window). | `DocumentResponse` | 200 |
| 14 | `POST` | `/api/v1/documents/{document_id}/restore` | Restore a soft-deleted document within the grace window. | `DocumentResponse` | 200 |

---

#### `POST /api/v1/documents`

Upload and index a document. Accepts multipart form data with a single `file` field. Supported file types: PDF, DOCX, TXT, MD, CSV, JSON, HTML, XML. The document is chunked, embedded, and indexed into the vector store.

**Request:** Multipart form (`file` field)

**Response:** `UploadResponse`

```json
{
  "document_id": "doc_abc123",
  "filename": "report.pdf",
  "file_type": "pdf",
  "file_size": 1048576,
  "status": "ready",
  "chunks_count": 42
}
```

| Status Code | Condition |
|-------------|-----------|
| 201 | Document uploaded and indexed |
| 400 | Invalid file type or validation error |
| 413 | File exceeds `max_file_size` |
| 429 | Quota exceeded |
| 500 | Internal server error |

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "X-Session-Token: eyJhbGciOi..." \
  -F "file=@report.pdf"
```

---

#### `GET /api/v1/documents`

List all documents in the current session.

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
| `limit` | int | `50` | 1--500 |

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

Poll the processing status of a document. Useful after async uploads to check when processing is complete.

**Response:** `DocumentStatusResponse`

```json
{
  "document_id": "doc_abc123",
  "status": "ready",
  "chunks_count": 42,
  "error_message": null
}
```

Status values: `processing`, `ready`, `error`

| Status Code | Condition |
|-------------|-----------|
| 200 | Document found |
| 404 | Document not found |

---

#### `POST /api/v1/documents/{document_id}/reindex`

Re-index an existing document. Deletes existing chunks and vectors, then re-processes the source file with current settings. Useful after config changes (e.g. switching embedding model or chunking strategy).

**Response:** `UploadResponse`

| Status Code | Condition |
|-------------|-----------|
| 200 | Re-indexed |
| 404 | Document not found |
| 410 | Source file no longer available in storage |
| 500 | Internal server error |

```bash
curl -X POST http://localhost:8000/api/v1/documents/doc_abc123/reindex \
  -H "X-Session-Token: eyJhbGciOi..."
```

---

### Document Lifecycle

| # | Method | Path | Description | Auth |
|---|--------|------|-------------|------|
| 13 | `POST` | `/api/v1/documents/{document_id}/archive` | Soft-delete — moves to `SOFT_DELETED` status; quota still counted until GC hard-deletes. | Session token |
| 14 | `POST` | `/api/v1/documents/{document_id}/restore` | Restore a soft-deleted document within the grace window (default 24h). Returns 409 if grace window expired. | Session token |

---

### Admin — GC & Lifecycle (requires `X-Admin-Token`)

All endpoints below require the `X-Admin-Token` header matching `RAG_ADMIN_TOKEN` env var.
If `RAG_ADMIN_TOKEN` is not set, these endpoints return `503 Service Unavailable`.

| # | Method | Path | Description |
|---|--------|------|-------------|
| 15 | `POST` | `/api/v1/admin/gc/run` | Trigger a full 6-phase GC cycle. Acquires advisory lock — returns immediately if lock is held. |
| 16 | `GET` | `/api/v1/admin/gc/orphans` | Read-only orphan scan: lists documents in DB-only or vector-only state. Does NOT modify data. |
| 17 | `POST` | `/api/v1/admin/gc/repair` | Detect and repair orphans. `?dry_run=true` (default) previews changes; `?dry_run=false` applies. |
| 18 | `DELETE` | `/api/v1/admin/sessions/{session_id}` | Hard-delete ALL documents in a session. Irreversible. |

---

### Search

| # | Method | Path | Description | Response Model |
|---|--------|------|-------------|----------------|
| 13 | `POST` | `/api/v1/search` | Semantic search with full SOTA pipeline. | `SearchResponse` |
| 14 | `POST` | `/api/v1/search/citations` | Search with citation tracking and LLM context. | `SearchWithCitationsResponse` |
| 15 | `POST` | `/api/v1/search/multi-query` | RAG Fusion multi-query expansion search. | `SearchResponse` |
| 16 | `POST` | `/api/v1/search/agentic` | Agentic multi-hop retrieval with iterative search. | `SearchWithCitationsResponse` |

All search endpoints accept the same `SearchRequest` body.

---

#### `SearchRequest` (shared request body)

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `query` | string | (required) | 1--10,000 chars | Search query |
| `top_k` | int | `5` | 1--100 | Number of results to return |
| `min_score` | float | `-100.0` | -- | Minimum score threshold |
| `document_ids` | list[string] | `null` | -- | Filter to specific documents |
| `use_hyde` | bool | `true` | -- | Enable HyDE query expansion |
| `use_reranking` | bool | `true` | -- | Enable cross-encoder reranking |
| `routing_tier` | string | `null` | `fast`, `standard`, `comprehensive` | Force a specific routing tier |

---

#### `POST /api/v1/search`

Semantic search with the full SOTA pipeline: hybrid dense+BM25, cross-encoder reranking, MMR diversity, HyDE query expansion, adaptive-k retrieval, hierarchical auto-merge.

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
  }
}
```

| Status Code | Condition |
|-------------|-----------|
| 200 | Success |
| 400 | Invalid query or validation error |
| 429 | Quota exceeded |
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
  "routing": null
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

RAG Fusion multi-query expansion. Generates `n_variations` alternative query phrasings, searches each independently, and fuses results via reciprocal rank fusion for higher recall. The `routing_tier` field in `SearchRequest` is ignored.

**Query Parameters:**

| Param | Type | Default |
|-------|------|---------|
| `n_variations` | int | `3` |

**Response:** `SearchResponse`

---

#### `POST /api/v1/search/agentic`

Agentic multi-hop retrieval with iterative search. Uses an LLM agent to decompose complex queries, perform multiple retrieval rounds, and synthesize results with citations. Requires `use_agentic_rag` to be enabled in session config. The `routing_tier` field in `SearchRequest` is ignored.

**Response:** `SearchWithCitationsResponse`

Note: the `results` array is empty for agentic search -- results are returned via `citations` and the pre-formatted `context` string.

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
| 17 | `GET` | `/api/v1/citations` | List session citations from last search. | `CitationListResponse` |
| 18 | `GET` | `/api/v1/citations/{citation_id}` | Get a specific citation by ID. | `CitationResponse` |

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
| 19 | `GET` | `/api/v1/config` | Get current session configuration. | `ConfigResponse` |
| 20 | `GET` | `/api/v1/config/profiles` | List available profiles. | `ProfileListResponse` |
| 21 | `PATCH` | `/api/v1/config` | Patch configuration keys. | `ConfigResponse` |
| 22 | `PUT` | `/api/v1/config/profiles/{profile_name}` | Apply a named profile. | `ConfigResponse` |

---

#### `GET /api/v1/config`

Get the current RAG configuration for this session.

**Response:** `ConfigResponse`

```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "chunking_strategy": "semantic",
  "enable_hybrid_search": true,
  "enable_reranking": true,
  "enable_mmr": true,
  "enable_hyde": true,
  "enable_hierarchical": false,
  "default_top_k": 5,
  "rerank_top_k": 20,
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
  "final_search_mode": "merge"
}
```

---

#### `GET /api/v1/config/profiles`

List available configuration profiles and the currently active profile.

**Response:** `ProfileListResponse`

```json
{
  "profiles": [
    {"name": "speed",       "description": "Pure vector search -- fastest, no reranking/hybrid/HyDE"},
    {"name": "standard",    "description": "Core SOTA features with lightweight embedding model"},
    {"name": "balanced",    "description": "Best quality-speed tradeoff with BGE-large embeddings"},
    {"name": "max_quality", "description": "All features enabled -- maximum recall and precision"}
  ],
  "active": "standard"
}
```

---

#### `PATCH /api/v1/config`

Patch session-scoped RAG configuration. Only non-null keys are updated. Unknown keys are rejected with 422.

**Request:** `ConfigPatchRequest` (extra="forbid")

| Field | Type | Constraints |
|-------|------|-------------|
| `enable_hybrid_search` | bool | -- |
| `enable_reranking` | bool | -- |
| `enable_mmr` | bool | -- |
| `enable_hyde` | bool | -- |
| `enable_hierarchical` | bool | -- |
| `use_agentic_rag` | bool | -- |
| `enable_prometheus` | bool | -- |
| `enable_tracing` | bool | -- |
| `default_top_k` | int | 1--100 |
| `rerank_top_k` | int | 1--100 |
| `similarity_threshold` | float | -1.0--1.0 |
| `hybrid_alpha` | float | 0.0--1.0 |
| `mmr_lambda` | float | 0.0--1.0 |
| `enable_query_routing` | bool | -- |
| `routing_strategy` | string | `rule_based` or `adaptive` |
| `routing_default_tier` | string | `fast`, `standard`, or `comprehensive` |
| `agentic_pipeline` | string | `legacy_react`, `default`, or any registered pipeline name |
| `use_structured_graph` | bool | -- |
| `enable_hallucination_check` | bool | -- |
| `enable_answer_quality_check` | bool | -- |
| `max_quality_check_attempts` | int | 1--10 |
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

**Path Parameters:** `profile_name` -- one of: `speed`, `standard`, `balanced`, `max_quality`

**Response:** `ConfigResponse` (full config after profile applied)

| Status Code | Condition |
|-------------|-----------|
| 200 | Profile applied |
| 404 | Profile name not found |

```bash
curl -X PUT http://localhost:8000/api/v1/config/profiles/max_quality \
  -H "X-Session-Token: eyJhbGciOi..."
```

---

### Pipelines

| # | Method | Path | Description | Response Model |
|---|--------|------|-------------|----------------|
| 23 | `GET` | `/api/v1/pipelines` | List all registered agentic pipelines. | `PipelineListResponse` |

---

#### `GET /api/v1/pipelines`

List all registered agentic pipelines.

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

## Non-Versioned Routes -- 9 endpoints

### Configurator (`/api/config/knowledge`)

These endpoints serve the configurator frontend UI. They provide full profile CRUD, session-scoped configuration, and import/export. All return `KnowledgeConfigFullResponse`.

| # | Method | Path | Description |
|---|--------|------|-------------|
| 23 | `GET` | `/api/config/knowledge` | Get full config state (profiles, overrides, RAG status). |
| 24 | `PUT` | `/api/config/knowledge/active` | Set active profile with optional config overrides. |
| 25 | `POST` | `/api/config/knowledge/profile` | Save or update a named profile. |
| 26 | `POST` | `/api/config/knowledge/profile/delete` | Delete a custom profile (built-ins protected). |
| 27 | `POST` | `/api/config/knowledge/profile/rename` | Rename a custom profile (built-ins protected). |
| 28 | `POST` | `/api/config/knowledge/reset` | Reset config to defaults. |
| 29 | `GET` | `/api/config/knowledge/export` | Export full session config as JSON. |
| 30 | `POST` | `/api/config/knowledge/import` | Import config from JSON (validated depth/size). |

---

#### `KnowledgeConfigFullResponse` (shared response)

```json
{
  "schema_version": 1,
  "defaults_source": "settings.lasagna.toml",
  "profiles": [
    {"name": "standard", "description": "Core SOTA features", "config": {"enable_reranking": true}}
  ],
  "active_profile": "standard",
  "session_overrides": {},
  "rag_enabled": true
}
```

---

#### `GET /api/config/knowledge`

Primary endpoint the configurator UI calls on load. Returns all profiles, active profile, session overrides, and RAG status.

---

#### `PUT /api/config/knowledge/active`

Set the active profile and optionally apply config overrides.

**Request:** `SetActiveProfileRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `active_profile` | string | Yes | Profile name (1--100 chars) |
| `config` | object | No | Additional config overrides to apply |

---

#### `POST /api/config/knowledge/profile`

Save or update a named profile globally.

**Request:** `SaveProfileRequest`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `profile_name` | string | Yes | -- | Profile name (1--100 chars) |
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
| `scope` | string | `"active"` | `"active"` -- clears session overrides only; `"all"` -- resets profiles and active profile too |

---

#### `GET /api/config/knowledge/export`

Export the full session configuration as JSON. Identical response shape to `GET /api/config/knowledge`.

---

#### `POST /api/config/knowledge/import`

Import configuration from JSON. Validates nesting depth (max 5) and profile size (max 10,000 bytes per profile).

**Request:** `ImportConfigRequest` (extra="forbid")

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `schema_version` | int | 1--100 | Config schema version |
| `profiles` | list[object] | Max 50 profiles, 10KB each, depth 5 | Profile definitions |
| `active_profile` | string | 1--100 chars | Profile to activate |
| `session_overrides` | object | Max 10KB, depth 5 | Session-scoped overrides |
| `rag_enabled` | bool | -- | Enable/disable RAG |

---

### Reindex (`/api/knowledge`)

| # | Method | Path | Description | Response Model |
|---|--------|------|-------------|----------------|
| 31 | `POST` | `/api/knowledge/reindex` | Bulk reindex all session documents. | `ReindexResponse` |

---

#### `POST /api/knowledge/reindex`

Re-index all documents in the current session with current settings. Iterates over every document, deletes old chunks/vectors (preserving source files), and re-processes each file. Returns partial results if some documents fail.

**Request:** `ReindexRequest`

| Field | Type | Note |
|-------|------|------|
| `session_id` | string | **Deprecated.** Ignored -- session is extracted from auth headers. Will be removed in a future version. |

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
  "errors": ["doc_xyz789: source file no longer on disk"]
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

Session identity is resolved on every request through header/token inspection. No login endpoint is required -- identity is stateless.

### Session Identity Resolution

Resolved in priority order (first match wins):

| Priority | Source | Mechanism |
|----------|--------|-----------|
| 1 | `X-Session-Token` header | JWT decoded `sid` claim |
| 2 | `X-Session-ID` header | Direct value |
| 3 | `session_id` query parameter | Direct value |
| 4 | (fallback) | `"default"` |

Session IDs are capped at 128 characters. Exceeding this returns 400.

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
| `iss` | `"rag-knowledge"` |
| `typ` | `"session"` |

Tokens are signed with `RAG_JWT_SECRET` (env var). If unset, an ephemeral secret is generated per process (tokens lost on restart).

---

## Error Responses

All errors return `ErrorResponse`:

```json
{"detail": "Description of the error", "error_code": null}
```

### Status Code Reference

| Code | Meaning | Common Triggers |
|------|---------|-----------------|
| 400 | Validation error | Invalid input, `session_id` exceeds 128 chars, malformed query |
| 401 | Unauthorized | `X-User-ID` header sent without `RAG_ALLOW_HEADER_IDENTITY=true` |
| 404 | Not found | Document, citation, or profile not found |
| 410 | Gone | Source file no longer available (reindex) |
| 413 | Payload too large | File exceeds `max_file_size` |
| 422 | Validation error | Pydantic model validation failure, unknown config keys |
| 429 | Rate limited / quota exceeded | `Retry-After` header included for rate limits |
| 500 | Internal server error | Unhandled backend exception |
| 503 | Service unavailable | Backend not initialized (startup), dependencies down (readiness) |

---

## Response Models Reference

| Model | Key Fields |
|-------|------------|
| `LivenessResponse` | `status` |
| `HealthResponse` | `status`, `checks: {name: {status, latency_ms, detail}}`, `version` |
| `StatsResponse` | `embedding_model`, `embedding_dim`, `collection_name`, `vector_backend`, `features`, `cache` |
| `SessionResponse` | `token`, `session_id`, `user_id`, `expires_at` |
| `UploadResponse` | `document_id`, `filename`, `file_type`, `file_size`, `status`, `chunks_count` |
| `DocumentResponse` | `document_id`, `session_id`, `filename`, `file_type`, `file_size`, `status`, `chunks_count`, `upload_date`, `error_message` |
| `DocumentListResponse` | `documents: [DocumentResponse]`, `total` |
| `DeleteResponse` | `success`, `document_id`, `message` |
| `DocumentStatusResponse` | `document_id`, `status`, `chunks_count`, `error_message` |
| `ChunkResponse` | `chunk_id`, `content`, `document_id`, `chunk_index`, `page_number`, `metadata` |
| `SearchResponse` | `results: [SearchResultResponse]`, `query`, `total_results`, `routing?` |
| `SearchResultResponse` | `chunk: ChunkResponse`, `score`, `document_id`, `filename` |
| `SearchWithCitationsResponse` | `results`, `citations: [CitationResponse]`, `context`, `query`, `total_results`, `routing?` |
| `CitationResponse` | `citation_id`, `document_id`, `document_name`, `chunk_id`, `chunk_index`, `page_number`, `score`, `content`, `highlight` |
| `CitationListResponse` | `citations: [CitationResponse]`, `session_id`, `total` |
| `ConfigResponse` | 22 config fields (see PATCH /api/v1/config for full list) |
| `ProfileListResponse` | `profiles: [{name, description}]`, `active` |
| `KnowledgeConfigFullResponse` | `schema_version`, `defaults_source`, `profiles`, `active_profile`, `session_overrides`, `rag_enabled` |
| `ReindexResponse` | `status`, `documents_reindexed`, `message`, `errors` |
| `ErrorResponse` | `detail`, `error_code` |
| `RoutingInfo` | `tier`, `strategy`, `confidence`, `rationale`, `query_type`, `estimated_complexity` |
