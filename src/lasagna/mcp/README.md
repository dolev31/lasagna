# Lasagna MCP server

MCP server exposing the RAG knowledge base as 29 tools for search, document management, runtime configuration, and document lifecycle management. Works with Claude Code, Claude Desktop, Cursor, and any MCP client.

By default it serves the **safe** set: every tool except the seven that delete data, reset configuration or change the server-wide retention policy, so a client lists 22. See [Tool sets](#tool-sets).

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
│   (2 tools)    │    (6 tools)   (9 tools) (3 tools)  │
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
| **STDIO** (default) | Claude Code, Claude Desktop, Cursor, agent frameworks | `lasagna-mcp` |
| **SSE** | Multi-client, behind a proxy | `lasagna-mcp --transport sse --host 127.0.0.1 --port 8005` |
| **Embedded** | Single-process with REST API | `lasagna serve --mcp -s <session>` (or `RAG_EMBED_MCP=true`), at `/mcp/sse`; `-s` is the session for calls that name none |

`--host` and `--port` apply only to `--transport sse`; passing them without it is an error rather than a stdio server waiting on input.

A local Milvus Lite store opens in one process at a time. While `lasagna serve`
runs, a `lasagna-mcp` tool call answers at once with the server's `/mcp/sse`
endpoint; start the server with `--mcp` and register that URL in the client.

## Tool sets

| Set | How to select | What it serves |
| --- | ------------- | -------------- |
| `safe` (default) | `--tools safe`, or `LASAGNA_MCP_TOOLS=safe` | Every tool except the seven marked *(all only)* below: 22 listed. |
| `all` | `--tools all`, or `LASAGNA_MCP_TOOLS=all` | All 29, including `delete_document`, `clear_session`, `bulk_delete_session`, `rebuild_collection`, `delete_rag_profile`, `reset_rag_config` and `set_lifecycle_policy`. |

`patch_rag_config` stays in `safe`: it changes only the caller's session, a change can be patched back, and one that would leave stored documents unsearchable is refused unless `force=true` (then `reindex_document` moves each document to the new model). `set_lifecycle_policy` is server-wide: a grace period of zero or a document TTL makes the next garbage collection delete other sessions' documents.

Under `safe` the seven are not registered at all, so neither a client nor the `/api/call` tester can call them. The embedded `/mcp` mount reads `LASAGNA_MCP_TOOLS` the same way.

**Which search tool:** use `search`.

The names, descriptions and input schemas of every tool are recorded in `tests/snapshots/mcp_tools.json` and follow the deprecation policy in `GOVERNANCE.md`.

## Tools

Tools that operate on data take an optional `session_id` (it scopes documents and config; without it a tool acts on the server's session, `RAG_SESSION_ID` or `lasagna serve --mcp -s <session>`) and an optional `session_token` for multi-user auth. No tool requires `session_id`.

### Search (2 tools)

| Tool | Description |
| ---- | ----------- |
| `search` | Runs the session's retrieval pipeline (hybrid search, reranking, MMR, HyDE, routing and agentic multi-hop, each as the session config sets it). Returns context with inline citations and a `<search_metadata>` JSON block. With no LLM configured, a session set to agentic retrieval runs its standard pipeline, and the metadata's `degraded_stages` names `agentic`. |
| `get_document_context` | Document awareness for agent prompts: describes available docs and gives adaptive search guidance |

### Documents (6 tools)

| Tool | Description |
| ---- | ----------- |
| `upload_document` | Upload and index a file (PDF, DOCX, TXT, MD, CSV, JSON, HTML, XML) |
| `list_documents` | List all documents in a session with status, type, size, chunk count |
| `get_document_info` | Detailed document metadata (JSON) |
| `get_document_chunks` | Paginated chunk content with metadata |
| `get_document_status` | Poll indexing status for a document by ID |
| `delete_document` | Hard-delete a document and all its indexed chunks *(all only)* |

### Lifecycle (6 tools)

| Tool | Description |
| ---- | ----------- |
| `archive_document` | Soft-delete a document: recoverable during the grace window, excluded from search immediately |
| `restore_document` | Restore a soft-deleted document to READY state (must be within grace window) |
| `bulk_delete_session` | Hard-delete all documents in a session; requires `confirm=True` *(all only)* |
| `get_gc_status` | Last GC run report: phases, counts, quota sync status |
| `get_lifecycle_policy` | Read the server's retention policy: grace window, quotas, GC schedule |
| `set_lifecycle_policy` | Change the server-wide retention policy, which decides what archive and GC delete *(all only)* |

### Session (3 tools)

| Tool | Description |
| ---- | ----------- |
| `set_focused_documents` | Restrict searches to specific document IDs (empty string to clear). An ID not in the session is refused and the focus is left unchanged |
| `get_focused_documents` | Get currently focused document IDs |
| `clear_session` | Delete all documents in a session (requires `confirm=True`) *(all only)* |

### Config (9 tools)

| Tool | Description |
| ---- | ----------- |
| `get_rag_config` | Full resolved config for session (profile + overrides) |
| `patch_rag_config` | Apply runtime overrides: pass JSON like `{"enable_reranking": false}`; returns each setting it changed, before and after, and a wrong type is refused naming the key and the type it takes |
| `reset_rag_config` | Reset to defaults. `scope="active"` clears overrides, `"all"` full reset *(all only)* |
| `list_profiles` | List built-in + custom profiles |
| `get_active_profile` | Current active profile name |
| `set_active_profile` | Switch session to a different profile; `reindex=true` reindexes its documents with the new profile first and switches only when all are reindexed (a failure changes nothing); `force=true` switches even when its documents were embedded with another model (reindex them after) |
| `save_rag_profile` | Create or update a named custom profile |
| `delete_rag_profile` | Delete a custom profile (built-in profiles are protected) *(all only)* |
| `get_profile_stats` | Profile and session counts |

### System (3 tools)

| Tool | Description |
| ---- | ----------- |
| `get_system_status` | Health check: vector DB connection, embeddings, features, storage |
| `rebuild_collection` | Drop and recreate vector collection (needed when embedding dim changes) *(all only)* |
| `reindex_document` | Re-process a document with current chunking/embedding settings; it keeps its old chunks until the new ones are stored, so a failed reindex changes nothing. `force_rebuild=true` recreates the collection at the configured embedding size only while it is empty; a collection holding documents is never dropped from here |

## Configuration

### Precedence

When resolving a config value, the first source that defines it wins:

1. Tool arguments: `top_k`, `use_hyde`, `routing_tier` and the like, for one call.
2. The session: `set_active_profile` and `patch_rag_config` (or `lasagna configure
   -s SESSION`), stored in the metadata database, so they persist across restarts.
3. The environment: `LASAGNA_<NAME>` or `RAG_<NAME>`, as the server process sees it.
4. The settings file: the packaged `settings.lasagna.toml`, or `--config-file`.

[Configuration](../../../docs/configuration.md#where-a-setting-comes-from) explains
the same order for every surface.

### Built-in Profiles

| Profile | Description |
| ------- | ----------- |
| **default** | Settings from `settings.lasagna.toml` as they are |
| **speed** | Vector-only search with all-MiniLM-L6-v2; no reranking, hybrid search, HyDE or LLM call |
| **standard** | BGE-M3, hybrid search, reranking, MMR, query routing with agentic RAG |
| **balanced** | Standard plus hierarchical chunking, the structured graph and adaptive routing |
| **max_quality** | Balanced plus HyDE, query expansion, RAG fusion, CRAG and contextual retrieval |

What each downloads and when it calls an LLM is in
[Configuration](../../../docs/configuration.md#profiles).

### Configuration Surfaces

| Surface | Scope | Description |
| ------- | ----- | ----------- |
| `settings.lasagna.toml` | Global | Base defaults, loaded at startup |
| `LASAGNA_SETTINGS_FILE` or `RAG_SETTINGS_FILE` env | Global | Override TOML file path |
| CLI args | Server | `--transport`, `--tools`, `--host`, `--port`, `--lazy-init`, `--eager-init`, `--log-level`, `--config-file`, `--db-dir`, `--upload-dir`, `--session-backend` |
| MCP config tools | Per-session | `patch_rag_config`, `set_active_profile`, etc. |
| REST API | Per-session | `/api/v1/config`, `/api/config/knowledge` |
| Configurator UI | Per-session | Interactive web UI at `/configurator` |
| SDK `KnowledgeConfig` | Programmatic | Python dataclass with preset factories |

### Settings

`patch_rag_config` takes any setting in
[the settings reference](../../../docs/reference/settings.md): embedding, chunking,
search, reranking, agentic RAG, citations, document awareness and infrastructure.

## Quick Start

A client that launches the server (Claude Code, Claude Desktop, Cursor) runs it
over stdio, the default; see [Claude Desktop](#claude-desktop). To serve several
clients from one process, start it on a port:

```bash
lasagna-mcp --transport sse --host 127.0.0.1 --port 8005
```

> **Binding to `0.0.0.0` publishes the tool surface.** Without
> `RAG_REQUIRE_AUTH=true` any client that reaches the port can call every tool
> in the served set, and the server refuses a non-loopback bind unless auth is
> on (or `RAG_BIND_IS_PROTECTED=true` says something else protects it). With
> auth on, tools need a `session_token` issued by `POST /api/v1/sessions`. Put
> a reverse proxy in front for TLS. The default bind is `127.0.0.1`.
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

Add to `claude_desktop_config.json` (Claude Code reads the same block from
`.mcp.json` in the project):

```json
{
  "mcpServers": {
    "lasagna": {
      "command": "lasagna-mcp",
      "env": { "RAG_SESSION_ID": "my-corpus" }
    }
  }
}
```

`RAG_SESSION_ID` is the session a tool call uses when it names none, so the
agent searches the documents uploaded with `lasagna upload FILE -s my-corpus`.
If `lasagna-mcp` is not on the client's `PATH`, give its absolute path inside
your virtualenv. Add `"args": ["--tools", "all"]` to serve the deletion tools.

## HTTP Endpoints (SSE mode)

| Endpoint | Description |
| -------- | ----------- |
| `/` | Welcome page: server info, feature overview, links |
| `/docs` | Interactive docs; the in-browser tool tester needs `--enable-docs-api` |
| `/health` | Health check (JSON) |
| `/sse` | MCP SSE transport endpoint |
| `/api/call` | Direct tool invocation for the docs UI (no authentication) |
