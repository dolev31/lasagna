# MCP server

The MCP server gives an agent search over your documents. It works with Claude
Code, Claude Desktop, Cursor and any other MCP client. It serves 29 tools for
search, documents, configuration and document lifecycle. By default a client
sees 22 of them.

## Start it

A client that launches the server itself (Claude Code, Claude Desktop, Cursor)
runs it over stdio, which is the default:

```bash
lasagna-mcp
```

To serve several clients from one process, start it on a port:

```bash
lasagna-mcp --transport sse --host 127.0.0.1 --port 8005
```

`--host` and `--port` apply only to `--transport sse`. Passing them without it is
an error.

| Mode | Use it for | Command |
|---|---|---|
| stdio (default) | A client that launches the server | `lasagna-mcp` |
| SSE | Several clients, or a server behind a proxy | `lasagna-mcp --transport sse --host 127.0.0.1 --port 8005` |
| Embedded | One process for REST and MCP | `lasagna serve --mcp -s <session>`, served at `/mcp/sse` |

With the embedded mode, `-s` names the session that calls use when they name
none. `RAG_EMBED_MCP=true` has the same effect as `--mcp`.

!!! note "One process at a time for a local store"
    A local Milvus Lite store opens in one process at a time. While
    `lasagna serve` runs, a `lasagna-mcp` call answers at once with the
    server's `/mcp/sse` endpoint. Start the server with `--mcp` and register
    that URL in the client.

## Connect a client

Add this to `claude_desktop_config.json`. Claude Code reads the same block from
`.mcp.json` in your project.

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

`RAG_SESSION_ID` is the session a tool call uses when it names none. The agent
then searches the documents you uploaded with `lasagna upload FILE -s my-corpus`.
If `lasagna-mcp` is not on the client's `PATH`, give its absolute path inside your
virtual environment.

To register a running server by URL instead:

```json
{
  "mcpServers": {
    "lasagna": { "type": "sse", "url": "http://127.0.0.1:8000/mcp/sse" }
  }
}
```

From Python:

```python
import asyncio
from lasagna.sdk.integrations import MCPKnowledgeClient


async def main():
    async with MCPKnowledgeClient("http://localhost:8005/sse") as client:
        result = await client.search("What is RAG?", session_id="demo")
        print(result)


asyncio.run(main())
```

## Choose a tool set

| Set | Select it with | What it serves |
|---|---|---|
| `safe` (default) | `--tools safe` or `LASAGNA_MCP_TOOLS=safe` | 22 tools. Every tool except the seven below. |
| `all` | `--tools all` or `LASAGNA_MCP_TOOLS=all` | All 29 tools. |

The seven tools that only `all` serves delete data, reset configuration or change
the server-wide retention policy: `delete_document`, `clear_session`,
`bulk_delete_session`, `rebuild_collection`, `delete_rag_profile`,
`reset_rag_config` and `set_lifecycle_policy`.

Under `safe`, these seven are not registered. No client can call them, and the
docs page cannot either. To serve them from a client config, add
`"args": ["--tools", "all"]` next to `"command"`.

`patch_rag_config` stays in `safe`. It changes only the caller's session, and
you can patch the change back. A change that would leave stored documents
unsearchable is refused unless you pass `force=true`.

## The tools

Tools that work on data take an optional `session_id`, which scopes documents and
configuration. Without it, a tool acts on the server's session. Multi-user setups
also pass `session_token`. No tool requires `session_id`.

| Group | Count | Tools |
|---|---|---|
| Search | 2 | `search`, `get_document_context` |
| Documents | 6 | `upload_document`, `list_documents`, `get_document_info`, `get_document_chunks`, `get_document_status`, `delete_document` (all only) |
| Lifecycle | 6 | `archive_document`, `restore_document`, `get_gc_status`, `get_lifecycle_policy`, `bulk_delete_session` (all only), `set_lifecycle_policy` (all only) |
| Session | 3 | `set_focused_documents`, `get_focused_documents`, `clear_session` (all only) |
| Config | 9 | `get_rag_config`, `patch_rag_config`, `list_profiles`, `get_active_profile`, `set_active_profile`, `save_rag_profile`, `get_profile_stats`, `reset_rag_config` (all only), `delete_rag_profile` (all only) |
| System | 3 | `get_system_status`, `reindex_document`, `rebuild_collection` (all only) |

Use `search` to retrieve. It runs the session's retrieval pipeline and returns
context with inline citations and a `<search_metadata>` JSON block. With no LLM
configured, a session set to agentic retrieval runs its standard pipeline, and
the metadata's `degraded_stages` names `agentic`.

The full description of every tool is in the
[MCP README](https://github.com/dolev31/lasagna/blob/main/src/lasagna/mcp/README.md).
Tool names, descriptions and input schemas follow the deprecation policy in the
[governance document](https://github.com/dolev31/lasagna/blob/main/GOVERNANCE.md).

## Settings that apply to a call

The first source that defines a value wins:

1. Tool arguments, such as `top_k` or `use_hyde`, for one call.
2. The session: `set_active_profile` and `patch_rag_config` (or
   `lasagna configure -s SESSION`). These persist across restarts.
3. The environment: `LASAGNA_<NAME>` or `RAG_<NAME>`.
4. The settings file: the packaged `settings.lasagna.toml`, or `--config-file`.

[Configuration](../configuration.md#where-a-setting-comes-from) explains the same
order for every surface.

## Keep it safe

!!! warning "Binding to 0.0.0.0 publishes the tool surface"
    Without `RAG_REQUIRE_AUTH=true`, any client that reaches the port can call
    every tool in the served set. The server refuses a non-loopback bind unless
    auth is on, or `RAG_BIND_IS_PROTECTED=true` says something else protects it.
    With auth on, tools need a `session_token` from `POST /api/v1/sessions`.
    Put a reverse proxy in front for TLS. The default bind is `127.0.0.1`.

The docs page at `/docs` has a "try it" button. It runs on an unauthenticated
`POST /api/call` that can invoke any tool, including deletion. It is off unless
you pass `--enable-docs-api`, and the server refuses to start if you combine that
flag with a non-local `--host`.

SSE mode also serves `/` (a welcome page), `/docs`, `/health` and `/sse`.
