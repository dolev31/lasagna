# RAG Knowledge — UI Integration Guide

A contract-safe reference for integrating RAG Knowledge features into any frontend or system. Organized by priority tier with SDK (Python) and REST (HTTP) examples for each feature.

---

## Table of Contents

- [How to Use This Guide](#how-to-use-this-guide)
- [Quick Start: RAG Chat in 15 Minutes](#quick-start-rag-chat-in-15-minutes)
- [Integration Patterns](#integration-patterns)
- **Tier 1 — Core**
  - [1. Document Upload & Management](#1-document-upload--management)
  - [2. Search & Context Retrieval](#2-search--context-retrieval)
  - [3. Session Management](#3-session-management)
- **Tier 2 — Enhanced**
  - [4. Citation Sidebar & Inline Sources](#4-citation-sidebar--inline-sources)
  - [5. Document Focus ("/" Picker)](#5-document-focus--picker)
  - [6. Document Awareness Panel](#6-document-awareness-panel)
  - [7. RAG Profile Switcher](#7-rag-profile-switcher)
  - [8. Configuration Patching](#8-configuration-patching)
- **Tier 3 — Advanced**
  - [9. Quality Indicators & Trust Badges](#9-quality-indicators--trust-badges)
  - [10. Coverage Feedback](#10-coverage-feedback)
  - [11. Configurator UI (Schema-Driven Settings)](#11-configurator-ui-schema-driven-settings)
  - [12. Agentic Multi-Hop Search](#12-agentic-multi-hop-search)
  - [13. MCP Integration](#13-mcp-integration)
  - [14. Pipeline Metrics (Architecture Notes)](#14-pipeline-metrics-architecture-notes)
  - [15. Web Search Fallback (Architecture Notes)](#15-web-search-fallback-architecture-notes)
- [Appendix A: REST API Reference](#appendix-a-rest-api-reference)
- [Appendix B: SDK Method Reference](#appendix-b-sdk-method-reference)
- [Appendix C: Configuration Parameters](#appendix-c-configuration-parameters)
- [Appendix D: Type Definitions & Enums](#appendix-d-type-definitions--enums)

---

## How to Use This Guide

### Three Tiers

| Tier | Label | When to Use |
|------|-------|-------------|
| **1** | Core | Every RAG UI needs these — document management, search, sessions |
| **2** | Enhanced | Recommended for good UX — citations, document focus, profiles |
| **3** | Advanced | Power users & admin — quality gates, configurator, agentic search |

Start with Tier 1. Add Tier 2 when you want a polished experience. Tier 3 is for specialized use cases.

### Ready-to-Use Code

For copy-paste React components, TypeScript types, and hooks for every feature in this guide, see the companion **[UI Components Reference](components.md)**. It includes a complete RAG chat app example you can run immediately.

### Integration Surfaces

| Surface | Language | Best For |
|---------|----------|----------|
| **Local SDK** | Python | Full features, no network overhead |
| **Remote SDK** (`from_url`) | Python | Lightweight `httpx` client, subset of features |
| **REST `/api/v1/`** | Any | Core CRUD + search, session tokens |
| **REST `/api/config/knowledge`** | Any | Full profile CRUD, import/export |
| **MCP** | Any (tool-based) | Claude Desktop, Cursor, tool-calling agents |

### Capability Matrix Legend

Each feature section includes a capability matrix showing availability per surface:
- ✅ = fully supported
- ⚠️ = partial (see notes)
- ❌ = not available

---

## Quick Start: RAG Chat in 15 Minutes

> **Import path:** All Python examples below use `knowledge.sdk` as the import base.
> Replace with your actual installed package path depending on how you've integrated
> the package (e.g., `your_app.knowledge.sdk`).

### SDK (Python)

```python
from knowledge.sdk import KnowledgeClient

async def main():
    client = KnowledgeClient()

    # 1. Upload a document
    result = await client.upload("./quarterly_report.pdf")
    print(f"Uploaded: {result.document_id} ({result.chunk_count} chunks)")

    # 2. Search with context for LLM injection
    context = await client.get_context("What is the quarterly revenue?")

    # 3. Use in your LLM prompt
    prompt = f"Based on this context:\n{context}\n\nAnswer: What is the quarterly revenue?"

    # 4. Search with citations for UI display
    results = await client.search_documents("revenue growth")
    for r in results:
        print(f"[{r.score:.2f}] {r.document.filename}: {r.chunk.content[:100]}...")

    await client.close()
```

### REST (HTTP)

```javascript
// 1. Create session
const session = await fetch("/api/v1/sessions", { method: "POST" });
const { token } = await session.json();
const headers = { "X-Session-Token": token };

// 2. Upload document
const form = new FormData();
form.append("file", fileInput.files[0]);
const upload = await fetch("/api/v1/documents", { method: "POST", headers, body: form });

// 3. Search with citations
const search = await fetch("/api/v1/search/citations", {
  method: "POST",
  headers: { ...headers, "Content-Type": "application/json" },
  body: JSON.stringify({ query: "What is the quarterly revenue?", top_k: 5 }),
});
const { results, citations, context } = await search.json();
```

### Minimal RAG Chat Wireframe

```
┌─────────────────────────────────────────────────────┐
│  📄 Documents           │  💬 Chat                  │
│  ┌───────────────────┐  │                           │
│  │ report.pdf   ✅   │  │  User: What was Q3 rev?  │
│  │ slides.pptx  ❌   │  │                           │
│  │ data.csv     ✅   │  │  Bot: Based on the report │
│  └───────────────────┘  │  revenue was $4.2M [1]    │
│                         │                           │
│  📎 Upload              │  ─────────────────────    │
│                         │  [1] report.pdf p.12      │
│                         │  "Q3 revenue reached..."  │
│                         │                           │
│                         │  ┌─────────────────────┐  │
│                         │  │ Ask a question...   │  │
│                         │  └─────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

---

## Integration Patterns

### Choosing Your Integration Surface

| Feature | Local SDK | Remote SDK | REST `/api/v1/` | REST `/api/config/knowledge` | MCP |
|---------|-----------|------------|------------------|------------------------------|-----|
| Upload documents | ✅ | ✅ | ✅ | — | ✅ |
| List/delete documents | ✅ | ✅ | ✅ | — | ✅ |
| Get document chunks | ✅ | ❌ (returns []) | ✅ | — | ✅ |
| Search | ✅ | ✅ | ✅ | — | ✅ |
| Search with citations | ✅ | ✅ | ✅ | — | ✅ |
| Agentic search (full) | ✅ (hop data) | ❌ (local deps required) | ⚠️ (no hop data) | — | ⚠️ (no hop data) |
| Get context for LLM | ✅ | ✅ | ✅ | — | ✅ |
| Document focus | ✅ | — | ⚠️ (via `document_ids`) | — | ✅ |
| Create session | ✅ | ✅ | ✅ | — | — (uses `session_id` param) |
| Delete session | ✅ | — | ❌ | — | ⚠️ (`clear_session` deletes docs only) |
| Get config | ✅ | — | ✅ | ✅ | ✅ (`get_rag_config`) |
| Patch config | ✅ | — | ✅ | — | ✅ (`patch_rag_config`) |
| Apply preset profile | ✅ | — | ✅ | ✅ | ✅ (`set_active_profile`) |
| Profile CRUD | — | — | — | ✅ | ✅ (save/delete/reset) |
| Import/export config | — | — | — | ✅ | — |
| List/get citations | ✅ | — | ✅ | — | — |
| Health check | — | — | ✅ | — | — |
| Configurator schema | — | — | — | — | — |

> **Configurator schema** is served at `/configurator/api/*` (not under `/api/v1/` or `/api/config/knowledge`). Requires `RAG_CONFIGURATOR=true` env var. See [§11](#11-configurator-ui-schema-driven-settings).

### Authentication & Sessions

**REST sessions** are created via `POST /api/v1/sessions` and return a JWT token:

```javascript
// Create session (both fields optional — auto-generated if omitted)
const res = await fetch("/api/v1/sessions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ session_id: "user_123", user_id: "user_123" }),
});
const { token, session_id, user_id, expires_at } = await res.json();

// Use on all subsequent requests
headers["X-Session-Token"] = token;
// Alternative: pass as query param: ?session_id=user_123
```

**SDK sessions** use the `session_id` parameter:

```python
client = KnowledgeClient(session_id="user_123")

# Switch sessions
client.session_id = "user_456"
```

**Important:** REST only supports session creation — there is no REST endpoint to delete or switch sessions. Use the SDK's `delete_session()` for cleanup.

### Error Handling

| HTTP Status | SDK Exception | Meaning |
|-------------|---------------|---------|
| 400 | `QueryError`, `ValueError` | Invalid query or parameters |
| 404 | `DocumentNotFoundError`, `CitationNotFoundError` | Resource not found |
| 413 | — | File too large |
| 422 | `ConfigurationError` | Validation error (config patch, profile) |
| 429 | `KnowledgeError` | Rate/quota limit exceeded (internal `QuotaExceededError` not exported in SDK exceptions) |
| 500 | `BackendError` | Internal server error |

```python
from knowledge.sdk.exceptions import (
    KnowledgeError, DocumentNotFoundError, SearchError, BackendError,
)

try:
    results = await client.search_documents("query")
except DocumentNotFoundError as e:
    show_error(f"Document not found: {e.document_id}")
except SearchError:
    show_error("Search failed — please try again")
except KnowledgeError as e:
    show_error(f"Knowledge error: {e}")
```

---

## TIER 1: Core

### 1. Document Upload & Management

**What:** Full document lifecycle — upload, list, get details, check status, view chunks, delete, reindex.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| Local SDK | ✅ | Full features |
| Remote SDK | ⚠️ | `get_document_chunks` returns `[]` |
| REST | ✅ | Full features |
| MCP | ✅ | Full features |

**Supported Formats (runtime-verified):**
`.txt`, `.md`, `.csv`, `.json`, `.xml`, `.html`, `.htm`, `.pdf`, `.docx`

OCR for scanned PDFs: set `enable_ocr=True`, engines: `easyocr` (default), `tesseract`, `paddleocr`.

#### SDK Examples

```python
# Upload
result = await client.upload("./report.pdf", metadata={"category": "finance"})
# result.document_id, result.chunk_count, result.status, result.processing_time_ms

# Upload with explicit document ID (for evaluation/benchmarks)
result = await client.upload("./report.pdf", document_id="eval_doc_001")

# List documents
docs = await client.list_documents()
for doc in docs:
    print(f"{doc.filename}: {doc.status.value}, {doc.chunk_count} chunks")

# Filter by status
ready_docs = await client.list_documents(status=DocumentStatus.READY)

# Get chunks (Local SDK only)
chunks = await client.documents.get_chunks("doc_123")

# Delete
success = await client.delete_document("doc_123")

# Delete all documents in session
count = await client.documents.delete_all(confirm=True)
```

#### REST Examples

```
POST   /api/v1/documents                    — upload (multipart/form-data)
GET    /api/v1/documents                    — list all documents
GET    /api/v1/documents/{id}               — get document details
GET    /api/v1/documents/{id}/status        — poll processing status
GET    /api/v1/documents/{id}/chunks        — get chunks (paginated: ?offset=0&limit=50)
DELETE /api/v1/documents/{id}               — delete document
POST   /api/v1/documents/{id}/reindex       — re-index with current settings
```

**Upload response:**

```json
{
  "document_id": "abc123",
  "filename": "report.pdf",
  "file_type": ".pdf",
  "file_size": 1048576,
  "status": "ready",
  "chunks_count": 42
}
```

**Status polling response:**

```json
{
  "document_id": "abc123",
  "status": "processing",
  "chunks_count": 0,
  "error_message": null
}
```

Status values: `pending` → `processing` → `indexing` → `ready` (or `failed`).

**Configuration:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `chunking_strategy` | enum | `semantic` | `fixed`, `semantic`, `recursive`, `sentence_window` |
| `chunk_size` | int | 800 | Target chunk size in characters |
| `chunk_overlap` | int | 0 | Overlap between chunks |
| `enable_ocr` | bool | false | Enable OCR for scanned documents |
| `ocr_engine` | enum | `easyocr` | `easyocr`, `tesseract`, `paddleocr` |
| `docling_enabled` | bool | false | Enable Docling advanced parsing |
| `enable_hierarchical` | bool | true | Parent-child chunk structure |

**UI Suggestion:** Document list panel with status badges (green=ready, yellow=processing, red=failed). Drag-and-drop upload area. Status polling with spinner until `ready`.

> **Code:** See [components.md § DocumentUploadPanel](components.md#31-documentuploadpanel) for a ready-to-use React component with drag-and-drop, progress bar, and status badges.

---

### 2. Search & Context Retrieval

**What:** Three search modes — raw search, LLM context string, and search with full citation tracking.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| Local SDK | ✅ | Full features including `AgenticSearchResults` with hop data |
| Remote SDK | ✅ | Standard search |
| REST | ✅ | Agentic endpoint returns context + citations only (no hop data) |
| MCP | ✅ | `search_knowledge` + `search_with_context` (returns formatted text, not SDK objects) |

#### SDK Examples

```python
# Basic search
results = await client.search_documents("What is the revenue?", top_k=5)
for r in results:
    print(f"[{r.score:.2f}] {r.document.filename}: {r.chunk.content[:100]}")

# Get context for LLM injection (recommended for RAG)
context = await client.get_context("key findings and recommendations", max_chunks=5)
prompt = f"Based on this context:\n{context}\n\nAnswer the question."

# Force a routing tier
results = await client.search_documents("quick lookup", routing_tier="fast")
results = await client.search_documents("complex analysis", routing_tier="comprehensive")

# Access routing info from search results
print(f"Search time: {results.search_time_ms}ms")
print(f"Total chunks searched: {results.total_chunks_searched}")

# Convert results to context
context = results.to_context(max_chunks=3)

# Get unique source documents
source_docs = results.get_documents()
```

#### REST Examples

```
POST /api/v1/search              — standard search
POST /api/v1/search/citations    — search with citation tracking + context
POST /api/v1/search/multi-query  — multi-query expansion search
POST /api/v1/search/agentic      — agentic multi-hop search
```

**Search request body:**

```json
{
  "query": "What is the revenue?",
  "top_k": 5,
  "min_score": -100.0,
  "document_ids": ["doc_123"],
  "use_hyde": true,
  "use_reranking": true,
  "routing_tier": "standard"
}
```

`routing_tier` values: `"fast"`, `"standard"`, `"comprehensive"` (or omit for auto-routing).

**Search response:**

```json
{
  "results": [
    {
      "chunk": {
        "chunk_id": "chunk_001",
        "content": "Revenue reached $4.2M...",
        "document_id": "doc_123",
        "chunk_index": 5,
        "page_number": 12,
        "metadata": {}
      },
      "score": 0.87,
      "document_id": "doc_123",
      "filename": "report.pdf"
    }
  ],
  "query": "What is the revenue?",
  "total_results": 5,
  "routing": {
    "tier": "standard",
    "strategy": "rule_based",
    "confidence": 0.85,
    "rationale": "Factual query with moderate complexity",
    "query_type": "FACTUAL",
    "estimated_complexity": "moderate"
  }
}
```

**Search with citations response** adds `citations` and `context` fields:

```json
{
  "results": [...],
  "citations": [
    {
      "citation_id": "a1b2c3d4",
      "document_id": "doc_123",
      "document_name": "report.pdf",
      "chunk_id": "chunk_001",
      "chunk_index": 5,
      "page_number": 12,
      "score": 0.87,
      "content": "Revenue reached $4.2M in Q3...",
      "highlight": "Revenue reached $4.2M"
    }
  ],
  "context": "[Source 1: report.pdf (page 12)]\nRevenue reached $4.2M...",
  "query": "What is the revenue?",
  "total_results": 5,
  "routing": { ... }
}
```

**Configuration:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_top_k` | int | 20 | Chunks to retrieve before reranking |
| `rerank_top_k` | int | 5 | Chunks to return after reranking |
| `similarity_threshold` | float | 0.3 | Minimum similarity score |
| `enable_hybrid_search` | bool | true | Combine vector + BM25 |
| `enable_reranking` | bool | true | Cross-encoder reranking |
| `enable_hyde` | bool | false | Hypothetical Document Embeddings |
| `enable_query_routing` | bool | true | Auto-route to fast/standard/comprehensive |
| `hybrid_alpha` | float | 0.7 | Dense vs sparse weight (1.0 = all dense) |

**UI Suggestion:** Search input with optional routing tier selector (Fast/Standard/Deep). Results as cards with score bar, content preview, source document, page number. Routing info as subtle badge ("FAST 50ms" or "COMPREHENSIVE 2.1s").

> **Code:** See [components.md § SearchResults](components.md#32-searchresults) for a ready-to-use React component with score bars and routing info.

---

### 3. Session Management

**What:** Session creation, scoped data isolation, session cleanup.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| Local SDK | ✅ | Create, switch, delete |
| Remote SDK | ✅ | Create, switch |
| REST | ⚠️ | Create only (`POST /api/v1/sessions`) — no delete/switch |
| MCP | ⚠️ | Each tool requires explicit `session_id`; `clear_session` deletes docs (not full session cleanup) |

#### SDK Examples

```python
# Create with session ID
client = KnowledgeClient(session_id="user_123")

# Switch session (clears focused documents)
client.session_id = "user_456"

# Delete session (irreversible — removes all documents, citations, cache)
result = await client.delete_session(confirm=True)
# result = {"documents_deleted": 5, "citations_deleted": 42, "cache_invalidated": True, "session_cleared": True}

# Context manager
async with KnowledgeClient(session_id="temp") as client:
    await client.upload("./doc.pdf")
    results = await client.search_documents("query")
# client.close() called automatically
```

#### REST Examples

```
POST /api/v1/sessions    — create session token
```

```json
// Request
{ "session_id": "user_123", "user_id": "user_123" }

// Response (201)
{
  "token": "eyJ...",
  "session_id": "user_123",
  "user_id": "user_123",
  "expires_at": "2026-03-09T12:00:00+00:00"
}
```

Pass the token via `X-Session-Token` header or `?session_id=user_123` query param.

**Important:** There is no REST endpoint to delete or switch sessions. For session cleanup, use the Local SDK's `delete_session()`.

---

## TIER 2: Enhanced

### 4. Citation Sidebar & Inline Sources

**What:** Stable citation IDs embedded in LLM responses as `[abc12345]` markers. Each citation maps to a source chunk with metadata.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| Local SDK | ✅ | `Citation` dataclass |
| REST | ✅ | `CitationResponse` model |
| MCP | ✅ | Via search tools |

Both SDK and REST use the same field shape:

| Field | Type | Description |
|-------|------|-------------|
| `citation_id` | str | Stable hash-based ID |
| `chunk_id` | str | Source chunk ID |
| `document_id` | str | Source document ID |
| `filename` / `document_name` | str | Source filename (SDK uses `filename`, REST uses `document_name`) |
| `page_number` | int? | Page number if available |
| `content_preview` / `content` | str? | Content preview (SDK uses `content_preview`, REST uses `content`) |
| `score` | float | Retrieval score (0-1) |
| `highlight` | str? | Highlighted text span |
| `hop_index` | int | Which search hop found this (SDK only) |

#### SDK Examples

```python
# Search with citations
results = await client.search_documents("revenue growth")
for citation in results.citations:
    print(f"[{citation.citation_id}] {citation.filename} p.{citation.page_number}")
    print(f"  Score: {citation.score:.2f}")
    print(f"  Preview: {citation.content_preview}")
    print(f"  Highlight: {citation.highlight}")

# Get citation by ID
citation = await client.citations.get("a1b2c3d4")

# List recent citations for session
all_citations = await client.citations.list_recent()

# Clear all citations
count = await client.citations.clear(confirm=True)
```

#### REST Examples

```
GET /api/v1/citations            — list session citations
GET /api/v1/citations/{id}       — get citation by ID
```

**Citation strategies** (configured via `chunk_retrieval_strategy`):

| Strategy | Description |
|----------|-------------|
| `adaptive` | Score-gap analysis selects natural cutoff (default) |
| `fixed` | Return exactly N chunks |
| `above_threshold` | Only chunks scoring above threshold |
| `agent_decides` | LLM audits which chunks support claims |

**Configuration:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_citations` | bool | true | Enable citation tracking |
| `citation_format` | enum | `stable_id` | `stable_id`, `index`, `markdown`, `none` |
| `verify_citations` | bool | true | Verify citations exist in DB |
| `show_page_numbers` | bool | true | Include page numbers |
| `citation_db_max_per_session` | int | 1000 | Max citations per session |
| `citation_db_max_age_hours` | int | 168 | Citation TTL (1 week) |

**UI Suggestion:** Inline `[1]` markers in chat messages, clickable to open a sidebar showing: source document, page number, content preview with highlight, confidence score. "View in document" button.

> **Code:** See [components.md § CitationBadge](components.md#33-citationbadge--preview) and [MessageWithCitations](components.md#34-messagewithcitations) for ready-to-use React components with hover previews and source modals.

---

### 5. Document Focus ("/" Picker)

**What:** Scope all searches to specific documents. Users select which documents to search in.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| Local SDK | ✅ | `set_focused_documents()` / `clear_focused_documents()` |
| REST | ⚠️ | Pass `document_ids` in search request body |
| MCP | ✅ | Via tool parameters |

#### SDK Examples

```python
# Focus on specific documents
client.set_focused_documents(["doc_123", "doc_456"])

# Searches now only look in focused documents
results = await client.search_documents("revenue")

# Check current focus
print(client.focused_document_ids)  # ["doc_123", "doc_456"]

# Clear focus (search all documents again)
client.clear_focused_documents()
```

#### REST Examples

Pass `document_ids` in the search request body:

```json
{
  "query": "revenue growth",
  "top_k": 5,
  "document_ids": ["doc_123", "doc_456"]
}
```

**UI Suggestion:** "/" trigger in chat input opens a dropdown of available documents. Selected documents appear as chips above the input. Badge showing "Searching in: report.pdf, data.csv".

> **Code:** See [components.md § DocumentPicker](components.md#36-documentpicker) for a ready-to-use React component with keyboard navigation and focused-document chips.

---

### 6. Document Awareness Panel

**What:** Show users what documents are available with previews/summaries before they search. The agentic pipeline automatically injects document awareness into agent prompts.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| Local SDK | ✅ | Via agentic pipeline + document list |
| REST | ⚠️ | Indirect — `GET /api/v1/documents` returns listing (no `preview_text`; use chunk endpoints for content) |

#### SDK Examples

```python
# List documents with metadata
docs = await client.list_documents()
for doc in docs:
    print(f"{doc.filename} ({doc.file_type}, {doc.file_size} bytes)")
    print(f"  Chunks: {doc.chunk_count}")

# To get content previews, fetch the first chunk per document (Local SDK only)
chunks = await client.documents.get_chunks(doc.document_id)
if chunks:
    print(f"  Preview: {chunks[0].content[:150]}...")
```

> **Note:** The SDK `Document` dataclass has a `preview_text` field, but it is populated
> only by the agentic pipeline's document awareness enrichment — not by `list_documents()`.
> The REST `DocumentResponse` does **not** include `preview_text` at all. To show
> document previews in a UI, fetch the first chunk via `GET /api/v1/documents/{id}/chunks?limit=1`.

**Enrichment Strategies** (configurable):

| Strategy | Description | Cost |
|----------|-------------|------|
| `first_tokens` | Extract first N tokens from document | Free |
| `llm_summary` | Generate LLM summary | API tokens |
| `hybrid` | Auto-select based on document size | Mixed |

**Configuration:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `document_awareness_enabled` | bool | true | Enable document awareness |
| `enrichment_strategy` | enum | `first_tokens` | `first_tokens`, `llm_summary`, `hybrid` |
| `first_tokens_count` | int | 150 | Tokens to extract for first_tokens strategy |
| `max_documents_in_context` | int | 10 | Maximum documents in context |

**UI Suggestion:** Side panel with document cards: filename, file type icon, size, upload date, snippet preview. Expanding a card shows chunk count and full preview.

---

### 7. RAG Profile Switcher

**What:** One-click quality/speed presets. 4 built-in profiles plus custom profile management.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| Local SDK | ✅ | `use_preset()`, `set_config()`, `get_config_dict()` |
| REST `/api/v1/config` | ✅ | List profiles, apply preset, patch config |
| REST `/api/config/knowledge` | ✅ | Full CRUD: save, rename, delete, import/export, reset |
| MCP | ✅ | `list_profiles`, `set_active_profile`, `get_active_profile`, `save_rag_profile`, `delete_rag_profile`, `reset_rag_config` |

**Built-in Presets:**

| Preset | Target Latency | Key Features |
|--------|---------------|--------------|
| `speed` | ~50-100ms | Pure vector search, no reranking/HyDE/hybrid |
| `standard` | ~200-400ms | Hybrid + reranking + MMR + BGE-M3 + Docling |
| `balanced` | ~300-600ms | Standard + hierarchical chunking + hallucination check |
| `max_quality` | ~1-3s | All features: HyDE, RAG Fusion, CRAG, quality gates |

#### SDK Examples

```python
# Apply a preset
client.use_preset("balanced")

# Available presets
# "speed", "standard", "balanced", "max_quality" (also "quality" as alias)
```

#### REST Examples — `/api/v1/config` surface

```
GET  /api/v1/config/profiles           — list profiles (name + description)
PUT  /api/v1/config/profiles/{name}    — apply preset (returns updated config)
```

#### REST Examples — `/api/config/knowledge` surface

```
GET    /api/config/knowledge               — full config state (profiles, overrides, rag_enabled)
PUT    /api/config/knowledge/active        — set active profile + optional overrides
POST   /api/config/knowledge/profile       — save custom profile
POST   /api/config/knowledge/profile/delete — delete custom profile
POST   /api/config/knowledge/profile/rename — rename custom profile
POST   /api/config/knowledge/reset         — reset to defaults (scope: "active" | "all")
GET    /api/config/knowledge/export        — export full config as JSON
POST   /api/config/knowledge/import        — import config from JSON
```

**Full config state response:**

```json
{
  "schema_version": 1,
  "defaults_source": "settings.lasagna.toml",
  "profiles": [
    { "name": "custom_research", "description": "Optimized for research", "config": { ... } }
  ],
  "active_profile": "balanced",
  "session_overrides": { "enable_hyde": true },
  "rag_enabled": true
}
```

**Note:** `rag_enabled` is readable in the response but there is no dedicated REST endpoint to toggle it. The `set_rag_enabled()` function exists only in the internal session config layer.

**UI Suggestion:** Dropdown or radio group with preset names and brief descriptions. "Custom" section for saving/loading profiles. Active profile badge visible during chat.

> **Code:** See [components.md § ProfileSwitcher](components.md#37-profileswitcher) for a ready-to-use React component.

---

### 8. Configuration Patching

**What:** Update individual RAG settings per session without changing the active profile.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| Local SDK | ✅ | `set_config(**kwargs)` |
| REST | ✅ | `PATCH /api/v1/config` |
| MCP | ✅ | `get_rag_config`, `patch_rag_config` |

#### SDK Examples

```python
# Patch individual settings
await client.set_config(
    enable_hyde=True,
    similarity_threshold=0.5,
    enable_reranking=False,
)

# Get current config as dict
config = client.get_config_dict()
print(config["enable_hyde"])  # True
```

#### REST Examples

```
PATCH /api/v1/config    — partial config update (returns full ConfigResponse)
GET   /api/v1/config    — get current config
```

```json
// PATCH request body (only non-null fields are applied)
{
  "enable_hyde": true,
  "similarity_threshold": 0.5,
  "routing_strategy": "adaptive"
}
```

**Patchable fields** (all optional, validated with ranges):

Feature toggles: `enable_hybrid_search`, `enable_reranking`, `enable_mmr`, `enable_hyde`, `enable_hierarchical`, `use_agentic_rag`, `enable_context_reordering`, `enable_rag_fusion`, `enable_crag`, `use_structured_graph`, `enable_hallucination_check`, `enable_answer_quality_check`, `enable_ocr`, `docling_enabled`

Numeric: `default_top_k` (1-100), `rerank_top_k` (1-100), `similarity_threshold` (-1.0 to 1.0), `hybrid_alpha` (0.0-1.0), `mmr_lambda` (0.0-1.0), `chunk_size` (50-100000), `chunk_overlap` (0-50000)

Enums: `routing_strategy` (`rule_based`|`adaptive`), `routing_default_tier` (`fast`|`standard`|`comprehensive`), `chunking_strategy` (`fixed`|`semantic`|`recursive`|`sentence_window`|`hierarchical`), `ocr_engine` (`easyocr`|`tesseract`|`paddleocr`)

**UI Suggestion:** Settings panel with toggle switches for key features and sliders for numeric params. Changes apply instantly per-session.

> **Code:** See [components.md § RAGConfigPanel](components.md#38-ragconfigpanel) for a ready-to-use React component with toggles and sliders.

---

## TIER 3: Advanced

### 9. Quality Indicators & Trust Badges

**What:** Quality gates that grade retrieval and generation quality. Available when using the structured graph pipeline.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| Local SDK | ✅ | Via structured graph pipeline internals |
| REST | ⚠️ | Partial — quality info influences the agentic response but is not directly exposed in the response schema |

**Quality Checks:**

| Check | Values | Config Key |
|-------|--------|------------|
| CRAG document grading | `correct`, `ambiguous`, `incorrect` | `enable_crag` |
| Hallucination check (ISSUP) | `supported`, `partially_supported`, `not_supported` | `enable_hallucination_check` |
| Answer quality (ISUSE) | `addresses_question`, `partially_addresses`, `does_not_address` | `enable_answer_quality_check` |

**Prerequisites:** `use_structured_graph=True` (or `agentic_pipeline="default"`)

```python
await client.set_config(
    use_structured_graph=True,
    enable_hallucination_check=True,
    enable_answer_quality_check=True,
    enable_crag=True,
)
```

**UI Suggestion:** Quality badge on each response: green checkmark (supported), yellow warning (partially), red flag (not supported). These are internal pipeline decisions that influence the final output quality — the pipeline auto-retries on poor grades.

---

### 10. Coverage Feedback

**What:** How well the query was answered, with suggestions for follow-up queries.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| Local SDK | ⚠️ | `HopData.coverage_score` per hop only |
| REST | ❌ | Not exposed in response schema |

The internal `CoverageScore` class tracks `coverage_ratio`, `uncovered_aspects`, and `suggested_followup`, but these are **not exposed** in the SDK's `AgenticSearchResults` type. Only `HopData.coverage_score` (a single `Optional[float]` per hop) is available.

```python
results = await client.agentic_search("What factors affected Q3?")
for hop in results.hops:
    print(f"Hop {hop.iteration}: {hop.query}")
    print(f"  Chunks retrieved: {hop.chunks_retrieved}")
    print(f"  Coverage score: {hop.coverage_score}")
    print(f"  Time: {hop.hop_time_ms}ms")
```

---

### 11. Configurator UI (Schema-Driven Settings)

**What:** Auto-generate a settings UI from 98 parameter definitions. The backend provides type, range, default, help text, tab grouping, and reindex flags.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| REST | ⚠️ | **Requires `RAG_CONFIGURATOR=true` env var** |

**Deployment prerequisite:** Set `RAG_CONFIGURATOR=true` environment variable. The configurator is mounted at `/configurator` only when this flag is set.

#### REST Endpoints

```
GET  /configurator/api/schema     — get all 98 parameter definitions
POST /configurator/api/validate   — validate config + detect reindex needs
```

**Schema response** (each parameter):

```json
{
  "key": "enable_hybrid_search",
  "type": "boolean",
  "label": "Smart Search",
  "description": "Combine vector and keyword search",
  "help_text": "Uses both semantic similarity and keyword matching...",
  "tab": "essential",
  "group": "Search Features",
  "default": true,
  "requires_reindex": false
}
```

For numeric parameters, additional fields: `min_value`, `max_value`, `step`.
For enum parameters: `enum_options` with `value`, `label`, `description` per option.
Parameters with `requires_reindex: true` need document reindexing after change.

**Validation request:**

```json
{
  "config": { "chunk_size": 500, "enable_hyde": true },
  "baseline": { "chunk_size": 800 }
}
```

Returns errors, warnings, and which changes require reindex.

**UI Suggestion:** Tabbed settings panel (Essential / Search / Processing / Advanced). Render widgets by type: toggles for booleans, sliders for ranges, dropdowns for enums. Yellow banner when changes require reindex. "Reindex Now" button.

---

### 12. Agentic Multi-Hop Search

**What:** Iterative search with query refinement for complex information needs.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| Local SDK | ✅ | Full `AgenticSearchResults` with per-hop data |
| Remote SDK | ❌ | `agentic_search()` requires local dependencies (`AgenticRAGWrapper`); not delegated to REST |
| REST | ⚠️ | Returns `SearchWithCitationsResponse` (context + citations, no hop data) |
| MCP | ⚠️ | `search_with_context` returns formatted context + citations, no `AgenticSearchResults` hop-level data |

> **Note:** `agentic_search()` always instantiates a local `AgenticRAGWrapper` — it does not delegate to the remote backend. Remote SDK users should use `POST /api/v1/search/agentic` directly via HTTP for agentic capabilities.

#### SDK Examples

```python
results = await client.agentic_search(
    "What factors affected Q3 performance and how do they compare to Q2?",
    max_iterations=3,
    document_ids=["doc_123"],
)

print(f"Total iterations: {results.total_iterations}")
print(f"Total chunks: {results.total_chunks_retrieved}")
print(f"Search time: {results.search_time_ms}ms")
print(f"Context:\n{results.final_context}")

# Per-hop data
for hop in results.hops:
    print(f"Hop {hop.iteration}: '{hop.query}' → {hop.chunks_retrieved} chunks")

# Get specific citation
citation = results.get_citation("a1b2c3d4")

# Refined answer (if available)
if results.refined_answer:
    print(results.refined_answer)
```

#### REST Example

```
POST /api/v1/search/agentic
```

Returns `SearchWithCitationsResponse` with `context`, `citations`, and `total_results`. The `results` array is empty (agentic search returns pre-formatted context). `routing_tier` in the request body is ignored — the agentic endpoint is the comprehensive tier by definition.

**Configuration:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_agentic_rag` | bool | true | Enable agentic search |
| `max_agent_iterations` | int | 3 | Maximum search hops |
| `use_structured_graph` | bool | false | Use structured StateGraph pipeline |
| `agentic_pipeline` | str? | null | Pipeline name: `"default"`, `"legacy_react"`, or custom |

**UI Suggestion:** "Deep research" mode toggle. When active, show progress: "Searching... Refining query... Synthesizing...". Display hop count after completion.

---

### 13. MCP Integration

**What:** First-class integration surface for tool-calling agents and AI IDEs.

**Capability Matrix:**

| Surface | Available | Notes |
|---------|-----------|-------|
| MCP Server | ✅ | Expose 29 RAG tools — embedded (`/mcp/sse`) or standalone (port 8005) |
| MCP Client | ✅ | Consume remote MCP servers |

**Server mode** — two deployment options:

1. **Embedded** in the REST API app: Set `RAG_EMBED_MCP=true` env var. MCP SSE endpoint mounted at `/mcp/sse`.
2. **Standalone**: Run `python -m lasagna.mcp --host 0.0.0.0 --port 8005`. Provides `/sse` (MCP transport), `/docs` (interactive API docs), `/health`.

**All 29 MCP tools:**

| Category | Tools |
|----------|-------|
| **Search** | `search`, `search_knowledge` *(deprecated)*, `search_with_context` *(deprecated)*, `get_document_context` |
| **Documents** | `upload_document`, `list_documents`, `delete_document`, `get_document_info`, `get_document_chunks`, `get_document_status`, `reindex_document` |
| **Lifecycle** | `archive_document`, `restore_document`, `bulk_delete_session`, `get_gc_status` |
| **Sessions** | `clear_session`, `get_system_status`, `rebuild_collection` |
| **Focus** | `set_focused_documents`, `get_focused_documents` |
| **Config** | `get_rag_config`, `patch_rag_config` |
| **Profiles** | `list_profiles`, `set_active_profile`, `get_active_profile`, `get_profile_stats`, `save_rag_profile`, `delete_rag_profile`, `reset_rag_config` |

> Each tool requires a `session_id` parameter for tenant isolation (except `get_system_status`).

**Client mode** — consume remote MCP servers from Python:

```python
from knowledge.sdk import MCPKnowledgeClient

# Connect to a remote MCP server
mcp_client = MCPKnowledgeClient(url="http://localhost:8005/sse")
```

**Use cases:** Claude Desktop, Cursor, or any MCP-compatible agent can call RAG tools directly.

---

### 14. Pipeline Metrics (Architecture Notes)

**Internal pipeline data — not directly exposed as integration contracts.**

The agentic pipeline tracks detailed metrics internally:
- `hop_count`, `latency_ms`, `token_reduction_pct`
- `input_tokens_before`, `input_tokens_after`
- `overall_grade`, `hallucination_check`, `answer_quality`

Only `HopData.coverage_score` is surfaced in the SDK types. For infrastructure monitoring, enable Prometheus metrics (`enable_prometheus=True`) which exposes standard `/metrics` endpoint.

---

### 15. Web Search Fallback (Architecture Notes)

**Internal pipeline behavior — not directly exposed as integration contracts.**

When CRAG grades retrieval results as "incorrect" (low quality), the pipeline automatically falls back to DuckDuckGo web search. Web results appear as citations alongside knowledge base results.

**Configuration:** `enable_crag=True` and `crag_web_search_fallback=True`.

---

## Appendix A: REST API Reference

### Surface 1: `/api/v1/` (Core API)

| Method | Path | Description | Response Model |
|--------|------|-------------|----------------|
| POST | `/api/v1/sessions` | Create session token | `SessionResponse` |
| POST | `/api/v1/documents` | Upload document (multipart) | `UploadResponse` |
| GET | `/api/v1/documents` | List documents | `DocumentListResponse` |
| GET | `/api/v1/documents/{id}` | Get document details | `DocumentResponse` |
| GET | `/api/v1/documents/{id}/status` | Poll processing status | `DocumentStatusResponse` |
| GET | `/api/v1/documents/{id}/chunks` | Get chunks (paginated) | `List[ChunkResponse]` |
| DELETE | `/api/v1/documents/{id}` | Delete document | `DeleteResponse` |
| POST | `/api/v1/documents/{id}/reindex` | Re-index document | `UploadResponse` |
| POST | `/api/v1/search` | Semantic search | `SearchResponse` |
| POST | `/api/v1/search/citations` | Search with citations | `SearchWithCitationsResponse` |
| POST | `/api/v1/search/multi-query` | Multi-query search | `SearchResponse` |
| POST | `/api/v1/search/agentic` | Agentic multi-hop search | `SearchWithCitationsResponse` |
| GET | `/api/v1/config` | Get config | `ConfigResponse` |
| PATCH | `/api/v1/config` | Patch config | `ConfigResponse` |
| GET | `/api/v1/config/profiles` | List profiles | `ProfileListResponse` |
| PUT | `/api/v1/config/profiles/{name}` | Apply preset | `ConfigResponse` |
| GET | `/api/v1/citations` | List citations | `CitationListResponse` |
| GET | `/api/v1/citations/{id}` | Get citation | `CitationResponse` |
| GET | `/api/v1/pipelines` | List pipelines | `PipelineListResponse` |
| GET | `/api/v1/health` | Health check | `HealthResponse` |
| GET | `/api/v1/health/live` | Liveness probe | `LivenessResponse` |
| GET | `/api/v1/health/ready` | Readiness probe | `HealthResponse` |
| GET | `/api/v1/stats` | System statistics | `StatsResponse` |

### Surface 2: `/api/config/knowledge` (Config Management)

| Method | Path | Description | Response Model |
|--------|------|-------------|----------------|
| GET | `/api/config/knowledge` | Full config state | `KnowledgeConfigFullResponse` |
| PUT | `/api/config/knowledge/active` | Set active profile | `KnowledgeConfigFullResponse` |
| POST | `/api/config/knowledge/profile` | Save custom profile | `KnowledgeConfigFullResponse` |
| POST | `/api/config/knowledge/profile/delete` | Delete custom profile | `KnowledgeConfigFullResponse` |
| POST | `/api/config/knowledge/profile/rename` | Rename custom profile | `KnowledgeConfigFullResponse` |
| POST | `/api/config/knowledge/reset` | Reset config (scope: active\|all) | `KnowledgeConfigFullResponse` |
| GET | `/api/config/knowledge/export` | Export config | `KnowledgeConfigFullResponse` |
| POST | `/api/config/knowledge/import` | Import config | `KnowledgeConfigFullResponse` |

### Surface 3: `/api/knowledge/` (Configurator Support)

| Method | Path | Description | Response Model |
|--------|------|-------------|----------------|
| POST | `/api/knowledge/reindex` | Re-index all session documents | `ReindexResponse` |

> Bulk reindex re-processes all documents in the current session with current config settings. Useful after config changes (e.g., switching embedding model or chunking strategy).

### Surface 4: `/configurator/` (Requires `RAG_CONFIGURATOR=true`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/configurator/api/schema` | Parameter schema (98 params) |
| POST | `/configurator/api/validate` | Validate config + reindex detection |

---

## Appendix B: SDK Method Reference

### KnowledgeClient

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `__init__` | `(config?, session_id?)` | — | Create client |
| `from_url` | `(url, api_key?, session_id?)` | `KnowledgeClient` | Connect to remote API |
| `upload` | `(file_path, filename?, metadata?, document_id?)` | `UploadResult` | Upload document |
| `search_documents` | `(query, top_k=5, min_score=0.0, document_ids?, routing_tier?)` | `SearchResults` | Search chunks |
| `get_context` | `(query, document_ids?, max_chunks=5, routing_tier?)` | `str` | Get LLM context |
| `list_documents` | `(status?)` | `List[Document]` | List documents |
| `delete_document` | `(document_id)` | `bool` | Delete document |
| `set_focused_documents` | `(document_ids)` | `None` | Focus searches |
| `clear_focused_documents` | `()` | `None` | Clear focus |
| `agentic_search` | `(query, max_iterations?, document_ids?)` | `AgenticSearchResults` | Multi-hop search |
| `set_config` | `(**kwargs)` | `None` | Update config |
| `use_preset` | `(preset)` | `None` | Apply preset |
| `get_config_dict` | `()` | `Dict` | Get config as dict |
| `delete_session` | `(confirm)` | `Dict` | Delete all session data |
| `close` | `()` | `None` | Cleanup resources |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `session_id` | `str` (read/write) | Current session ID |
| `config` | `KnowledgeConfig` (read) | Current config |
| `documents` | `DocumentManager` (read) | Document operations |
| `search` | `SearchManager` (read) | Search operations |
| `citations` | `CitationManager` (read) | Citation operations |
| `focused_document_ids` | `List[str]?` (read) | Currently focused docs |

---

## Appendix C: Configuration Parameters

All fields from `KnowledgeConfig`. Grouped by category.

### Feature Toggles

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_hybrid_search` | bool | true | Combine vector + BM25 |
| `enable_reranking` | bool | true | Cross-encoder reranking |
| `enable_mmr` | bool | true | Max Marginal Relevance diversity |
| `enable_hyde` | bool | false | Hypothetical Document Embeddings |
| `enable_ocr` | bool | false | OCR for scanned documents |
| `enable_query_expansion` | bool | false | Generate query variations |
| `enable_query_decomposition` | bool | false | Break queries into sub-questions |
| `enable_embedding_cache` | bool | true | Cache embeddings |
| `enable_search_cache` | bool | true | Cache search results |
| `enable_context_reordering` | bool | true | Lost-in-the-middle mitigation |
| `enable_contextual_retrieval` | bool | false | LLM context per chunk |
| `enable_rag_fusion` | bool | false | RAG Fusion multi-query |
| `enable_crag` | bool | false | Corrective RAG |
| `enable_hierarchical` | bool | true | Parent-child chunks |
| `enable_query_routing` | bool | true | Auto-route queries |
| `enable_query_classification` | bool | true | Classify query type |
| `enable_citations` | bool | true | Citation tracking |
| `document_awareness_enabled` | bool | true | Document awareness |
| `use_agentic_rag` | bool | true | Agentic multi-hop |
| `use_structured_graph` | bool | false | Structured StateGraph pipeline |
| `enable_hallucination_check` | bool | false | ISSUP grounding check |
| `enable_answer_quality_check` | bool | false | ISUSE quality check |

### Search Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_top_k` | int | 20 | Chunks before reranking |
| `rerank_top_k` | int | 5 | Chunks after reranking |
| `similarity_threshold` | float | 0.3 | Min similarity score |
| `hybrid_alpha` | float | 0.7 | Dense vs sparse weight |
| `mmr_lambda` | float | 0.5 | Diversity vs relevance |
| `mmr_score_normalization` | str | sigmoid | Score normalization method |

### Embedding

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `embedding_provider` | enum | `huggingface` | `huggingface`, `openai` |
| `embedding_model` | str | `BAAI/bge-m3` | Model name |
| `embedding_model_fallback` | str? | `BAAI/bge-large-en-v1.5` | Fallback model |
| `embedding_dim` | int | 1024 | Vector dimension |
| `normalize_embeddings` | bool | true | Normalize for cosine similarity |
| `embedding_backend` | str | `auto` | `auto`, `flag-embedding`, `sentence-transformers` |
| `batch_size` | int | 32 | Embedding batch size |

### Chunking

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `chunking_strategy` | enum | `semantic` | `fixed`, `semantic`, `recursive`, `sentence_window` |
| `chunk_size` | int | 800 | Target chunk size (chars) |
| `chunk_overlap` | int | 0 | Overlap between chunks |
| `semantic_threshold` | float | 0.75 | Semantic chunking threshold |
| `min_chunk_size` | int | 200 | Min chunk size |
| `max_chunk_size` | int | 1500 | Max chunk size |
| `parent_chunk_size` | int | 2000 | Hierarchical parent size |
| `child_chunk_size` | int | 500 | Hierarchical child size |
| `auto_merge_threshold` | float | 0.5 | Auto-merge ratio |
| `preserve_tables` | bool | true | Preserve table structure |
| `preserve_headers` | bool | true | Track section headers |

### Query Routing

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `routing_strategy` | str | `rule_based` | `rule_based`, `adaptive` |
| `routing_default_tier` | str | `standard` | `fast`, `standard`, `comprehensive` |
| `routing_confidence_threshold` | float | 0.5 | Escalation threshold |

### Agentic RAG

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_agent_iterations` | int | 3 | Max search hops |
| `chunk_refinement_method` | str | `auto` | `vanilla`, `refined`, `auto` |
| `agent_temperature` | float | 0.1 | Agent model temperature |
| `agentic_pipeline` | str? | null | Pipeline name |

### Citations

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `citation_format` | enum | `stable_id` | `stable_id`, `index`, `markdown`, `none` |
| `citation_id_length` | int | 8 | Hash length |
| `verify_citations` | bool | true | Verify in DB |
| `track_citations_db` | bool | true | Persist citations |
| `show_page_numbers` | bool | true | Include page numbers |
| `show_confidence_scores` | bool | false | Show scores |
| `citation_db_max_age_hours` | int | 168 | TTL (1 week) |
| `citation_db_max_per_session` | int | 1000 | Max per session |

### Document Awareness

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enrichment_strategy` | enum | `first_tokens` | `first_tokens`, `llm_summary`, `hybrid` |
| `first_tokens_count` | int | 150 | Tokens for first_tokens |
| `max_documents_in_context` | int | 10 | Max docs in context |

### Infrastructure

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `vector_backend` | enum | `milvus` | `milvus`, `pinecone`, `qdrant`, `weaviate`, `pgvector`, `memory` |
| `database_backend` | str | `sqlite` | `sqlite`, `postgresql` |
| `cache_backend` | str | `memory` | `memory`, `redis` |
| `storage_backend` | str | `local` | `local`, `s3`, `fsspec` |
| `tenant_isolation` | str | `none` | `none`, `partition` |
| `quota_enabled` | bool | false | Per-tenant quotas |
| `rate_limiting_enabled` | bool | false | Rate limiting |
| `enable_prometheus` | bool | false | Prometheus metrics |
| `enable_tracing` | bool | false | OpenTelemetry tracing |
| `celery_enabled` | bool | false | Background processing |
| `connection_pool_size` | int | 1 | Vector backend pool |

---

## Appendix D: Type Definitions & Enums

### Enums

| Enum | Values |
|------|--------|
| `ChunkingStrategy` | `fixed`, `semantic`, `recursive`, `sentence_window` |
| `EmbeddingProvider` | `huggingface`, `openai` |
| `OCREngine` | `easyocr`, `tesseract`, `paddleocr` |
| `HyDEStrategy` | `off`, `first_hop`, `adaptive`, `always` |
| `ChunkRetrievalStrategy` | `fixed`, `above_threshold`, `adaptive`, `agent_decides` |
| `EnrichmentStrategy` | `first_tokens`, `llm_summary`, `hybrid` |
| `CitationFormat` | `stable_id`, `index`, `markdown`, `none` |
| `CitationScoreMethod` | `minmax`, `sigmoid`, `clamp` |
| `VectorBackend` | `milvus`, `pinecone`, `qdrant`, `weaviate`, `pgvector`, `opensearch`, `memory` |
| `DocumentStatus` | `pending`, `processing`, `indexing`, `ready`, `failed` |

### SDK Dataclasses

**Document:** `document_id`, `filename`, `session_id`, `status`, `file_path?`, `file_type?`, `file_size?`, `chunk_count`, `created_at?`, `error_message?`, `preview_text?`, `metadata`

**DocumentChunk:** `chunk_id`, `document_id`, `content`, `chunk_index`, `page_number?`, `section_header?`, `line_start?`, `line_end?`, `metadata`

**Citation:** `citation_id`, `chunk_id`, `document_id`, `filename`, `page_number?`, `content_preview?`, `score`, `highlight?`, `hop_index`

**SearchResult:** `chunk`, `document`, `score`, `citation_id?`, `highlight?`, `rerank_score?`, `dense_score?`, `sparse_score?`

**SearchResults:** `results`, `query`, `total_chunks_searched`, `search_time_ms`, `citations`
- Methods: `__iter__`, `__len__`, `__getitem__`, `to_context(max_chunks)`, `get_documents()`

**UploadResult:** `document_id`, `filename`, `status`, `chunk_count`, `processing_time_ms`, `message`, `error?`
- Property: `success` (True if status is READY or INDEXING)

**HopData:** `iteration`, `query`, `chunks_retrieved`, `refined_query?`, `coverage_score?`, `hop_time_ms`

**AgenticSearchResults:** `final_context`, `citations`, `hops`, `total_iterations`, `total_chunks_retrieved`, `search_time_ms`, `results?`, `refined_answer?`
- Method: `get_citation(citation_id)`

**ChunkCiteRetrieval:** `strategy`, `fixed_size`, `threshold_score`, `max_chunks`, `min_chunks`

**DocumentAwareness:** `enabled`, `enrichment_strategy`, `first_tokens_count`, `include_filename`, `include_upload_date`, `include_file_type`, `summary_model`, `summary_max_tokens`, `summary_prompt`, `hybrid_llm_threshold_kb`, `max_documents_in_context`, `cache_snippets`, `snippet_cache_ttl`

### REST Response Models

**CitationResponse:** `citation_id`, `document_id`, `document_name`, `chunk_id`, `chunk_index`, `page_number?`, `score`, `content`, `highlight?`, `highlights?`, `highlight_ranges?`

**Note:** SDK `Citation` uses `filename` and `content_preview`. REST `CitationResponse` uses `document_name` and `content`. Both have `highlight: Optional[str]`. REST additionally has `highlights: Optional[List[str]]` and `highlight_ranges: Optional[List[List[int]]]` for multi-span highlighting (populated by the internal citation manager when multiple highlight spans are detected).

**Note:** `AgenticSearchResults` (SDK) includes `hops`, `total_iterations`, `total_chunks_retrieved`, `search_time_ms`, `refined_answer` which are NOT present in the REST agentic response (`SearchWithCitationsResponse`).

### Exception Hierarchy

```
KnowledgeError
├── DocumentError
│   ├── DocumentNotFoundError (document_id)
│   ├── UploadError (file_path, stage)
│   ├── ChunkingError
│   └── ExtractionError
├── SearchError
│   ├── QueryError (query)
│   ├── EmbeddingError
│   └── RetrievalError
├── CitationError (citation_id)
│   └── CitationNotFoundError
├── ConfigurationError (config_key)
│   └── ProfileNotFoundError (profile_name)
├── BackendError (backend_name, operation)
│   ├── BackendConnectionError
│   └── BackendTimeoutError
├── AgenticError (iteration)
│   └── MaxIterationsError
├── EvaluationError
│   └── JudgeError
└── IntegrationError (framework)
```

## See Also

- [UI Components](components.md)
- [User Guide](../user_guide.md)
- [Getting Started](../getting_started.md)
