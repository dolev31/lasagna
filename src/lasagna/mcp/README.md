# MCP Server — Lasagna

MCP server exposing the RAG knowledge base as 31 tools for semantic search, document management, runtime configuration, and document lifecycle management. Compatible with Claude Desktop, custom agents, and any MCP client.

Self-contained within the `knowledge` package — no dependency on the host agent framework.

## Architecture

```
                        MCP Clients
                   (Claude Desktop, agents)
                            │
                       SSE / STDIO
                            │
┌───────────────────────────┼───────────────────────────┐
│                    MCP Server                         │
│                                                       │
│   Search ──────┐    Documents    Config    Session    │
│   (4 tools)    │    (6 tools)   (9 tools) (3 tools)  │
│                │                                      │
│   System (3 tools)   Lifecycle (6 tools)              │
└────────────────┼──────────────────────────────────────┘
                 │
        ┌────────┼────────────────────────┐
        │        │                        │
        ▼        ▼                        ▼
┌──────────┐ ┌──────────────┐   ┌──────────────┐
│ Agentic  │ │  RAGBackend  │   │ Session      │
│ RAG      │ │  (singleton) │   │ Config Mgr   │
│ multi-hop│ │              │   │ (per-session) │
└────┬─────┘ └──────┬───────┘   └──────────────┘
     │              │
     ▼              ▼
┌────────────────────────────────────────────────┐
│              Infrastructure                     │
│  Vector DB    Metadata DB    Embedding Manager  │
│  (Milvus/     (SQLite/       (sentence-         │
│   Qdrant/      PostgreSQL)    transformers/      │
│   Weaviate/                   FlagEmbedding)     │
│   pgvector)                                      │
│  Cache (memory/Redis) · Rate Limiter             │
└────────────────────────────────────────────────┘
```

## Deployment

| Mode | Use Case | Command |
| ---- | -------- | ------- |
| **SSE** | Multi-client, behind a proxy | `lasagna-mcp --host 127.0.0.1 --port 8005` |
| **STDIO** | Claude Desktop, agent frameworks | `python -m lasagna.mcp --transport stdio` |
| **Embedded** | Single-process with REST API | Set env `RAG_EMBED_MCP=true` (mounts at `/mcp/sse`) |

## Tools

All tools that operate on data take a `session_id` parameter (scopes documents and config) and an optional `session_token` for multi-user auth.

### Search (4 tools)

| Tool | Description |
| ---- | ----------- |
| `search` | SOTA retrieval pipeline: hybrid search, reranking, MMR, HyDE, adaptive routing, agentic multi-hop. Returns LLM-ready context with inline citations + `<search_metadata>` JSON block. Retrieval config is session-driven. |
| `search_knowledge` | **Deprecated** alias for `search` — kept for one release cycle |
| `search_with_context` | **Deprecated** alias for `search` — kept for one release cycle |
| `get_document_context` | Document awareness for agent prompts — describes available docs and gives adaptive search guidance |

### Documents (6 tools)

| Tool | Description |
| ---- | ----------- |
| `upload_document` | Upload and index a file (PDF, DOCX, TXT, MD, CSV, JSON, HTML, XML) |
| `list_documents` | List all documents in a session with status, type, size, chunk count |
| `get_document_info` | Detailed document metadata (JSON) |
| `get_document_chunks` | Paginated chunk content with metadata |
| `get_document_status` | Poll indexing status for a document by ID |
| `delete_document` | Hard-delete a document and all its indexed chunks |

### Lifecycle (6 tools)

| Tool | Description |
| ---- | ----------- |
| `archive_document` | Soft-delete a document — recoverable during grace window, excluded from search immediately |
| `restore_document` | Restore a soft-deleted document to READY state (must be within grace window) |
| `bulk_delete_session` | Hard-delete all documents in a session; requires `confirm=True` |
| `get_gc_status` | Last GC run report — phases, counts, quota sync status |
| `get_lifecycle_policy` | Read the session's retention policy — grace window, quotas, GC schedule |
| `set_lifecycle_policy` | Change retention for the session; affects what archive and GC do |

### Session

| Tool | Description |
| ---- | ----------- |
| `set_focused_documents` | Restrict searches to specific document IDs (empty string to clear). An ID not in the session is refused and the focus is left unchanged |
| `get_focused_documents` | Get currently focused document IDs |
| `clear_session` | Delete all documents in a session (requires `confirm=True`) |

### Config Tools

| Tool | Description |
| ---- | ----------- |
| `get_rag_config` | Full resolved config for session (profile + overrides) |
| `patch_rag_config` | Apply runtime overrides — pass JSON like `{"enable_reranking": false}` |
| `reset_rag_config` | Reset to defaults. `scope="active"` clears overrides, `"all"` full reset |
| `list_profiles` | List built-in + custom profiles |
| `get_active_profile` | Current active profile name |
| `set_active_profile` | Switch session to a different profile |
| `save_rag_profile` | Create or update a named custom profile |
| `delete_rag_profile` | Delete a custom profile (built-in profiles are protected) |
| `get_profile_stats` | Profile and session counts |

### System

| Tool | Description |
| ---- | ----------- |
| `get_system_status` | Health check — vector DB connection, embeddings, features, storage |
| `rebuild_collection` | Drop and recreate vector collection (needed when embedding dim changes) |
| `reindex_document` | Re-process a document with current chunking/embedding settings. `force_rebuild=true` recreates the collection at the configured embedding size only while it is empty; a collection holding documents is never dropped from here |

## Configuration

### Precedence

When resolving a config value, the first source that defines it wins:

```text
1. Per-request params     →  top_k, use_hyde, routing_tier (tool arguments)
2. Session overrides      →  patch_rag_config (persists for session)
3. Active profile         →  set_active_profile (named config preset)
4. TOML defaults          →  settings.lasagna.toml (base)
```

### Built-in Profiles

| Profile | Description |
| ------- | ----------- |
| **default** | Settings from `settings.lasagna.toml` as-is |
| **speed** | Vector-only search, no reranking/hybrid/HyDE. MiniLM-L6-v2. Lowest latency |
| **standard** | BGE-M3, hybrid search, reranking, MMR, agentic RAG. Good balance |
| **balanced** | Standard + hierarchical chunking, structured graph, adaptive routing |
| **max_quality** | All features: HyDE, query expansion, RAG Fusion, CRAG, contextual retrieval |

### Configuration Surfaces

| Surface | Scope | Description |
| ------- | ----- | ----------- |
| `settings.lasagna.toml` | Global | Base defaults, loaded at startup |
| `RAG_SETTINGS_FILE` env | Global | Override TOML file path |
| CLI args | Server | `--host`, `--port`, `--transport`, `--lazy-init`, `--log-level`, `--config-file` |
| MCP config tools | Per-session | `patch_rag_config`, `set_active_profile`, etc. |
| REST API | Per-session | `/api/v1/config`, `/api/config/knowledge` |
| Configurator UI | Per-session | Interactive web UI at `/configurator` |
| SDK `KnowledgeConfig` | Programmatic | Python dataclass with preset factories |

### Config Categories (120+ parameters)

- **Embedding** — model, dim, backend, GPU, batch size, normalization
- **Chunking** — strategy (fixed/semantic/sentence_window), sizes, overlap, hierarchical
- **Search** — top_k, threshold, hybrid alpha, MMR lambda, query routing
- **Reranking** — model, batch size, custom reranker
- **Agentic RAG** — max iterations, temperature, HyDE strategy, refinement
- **Citations** — format, verification, page numbers, DB tracking
- **Document awareness** — enrichment strategy, token count, caching
- **Infrastructure** — cache TTLs, OCR engine, Prometheus, OpenTelemetry

## Quick Start

Start the server:

```bash
lasagna-mcp --host 127.0.0.1 --port 8005
```

> **Binding to `0.0.0.0` publishes the tool surface.** The server has no
> authentication of its own, so put a reverse proxy in front of it and let that
> terminate TLS and authenticate — do not expose it directly. The default bind
> is `127.0.0.1` deliberately.
>
> The `/docs` page's "try it" button is powered by an unauthenticated
> `POST /api/call` that can invoke **any** tool, including deletion. It is off
> unless you pass `--enable-docs-api`, and the server refuses to start if you
> combine that flag with a non-local `--host`.

Connect from Python:

```python
import asyncio
from lasagna.sdk.integrations import MCPKnowledgeClient

async def main():
    async with MCPKnowledgeClient("http://localhost:8005/sse") as client:
        result = await client.search("What is RAG?", session_id="demo")
        print(result)

asyncio.run(main())
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "lasagna": {
      "command": "lasagna-mcp",
      "args": ["--transport", "stdio"]
    }
  }
}
```

## HTTP Endpoints (SSE mode)

| Endpoint | Description |
| -------- | ----------- |
| `/` | Welcome page — server info, feature overview, links |
| `/docs` | Interactive docs — test any tool in-browser |
| `/health` | Health check (JSON) |
| `/sse` | MCP SSE transport endpoint |
| `/api/call` | Direct tool invocation for the docs UI (no authentication) |
