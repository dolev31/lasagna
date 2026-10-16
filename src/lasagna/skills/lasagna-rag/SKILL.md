---
name: lasagna-rag
description: Use when an agent or application needs retrieval over its own documents: ingesting files, searching them, or wiring Lasagna's MCP server into Claude Code, Claude Desktop, Cursor or any MCP client. Covers the fastest path from zero to a working answer.
---

# Give an agent retrieval over a corpus

Two ways in. Pick by whether the agent should call retrieval as a **tool** (MCP)
or the application should call it as a **library** (Python).

## Before anything: does it work here?

```bash
lasagna doctor
```

Reports version, backend, credentials, model cache and anything that failed to
load quietly. Read it before debugging anything else: most "it hangs" reports
are the first-run model download, which this names explicitly.

## Path A: MCP (the agent calls retrieval as a tool)

```bash
pip install lasagna-rag
lasagna configure --profile speed -s my-corpus   # 91 MB model, no API key
for f in ./docs/*.md; do lasagna upload "$f" -s my-corpus; done
```

Then register the server. **Claude Code** (`.mcp.json` in the project):

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

**Claude Desktop**: the same block inside `claude_desktop_config.json`.
`lasagna-mcp` speaks stdio, which is how both launch it; `--transport sse`
serves a port instead.

Tools the agent gets: 31 tools, in six groups.

| Group | Tools |
|---|---|
| Search | `search`, `search_knowledge`, `search_with_context`, `get_document_context` |
| Documents | `upload_document`, `list_documents`, `delete_document`, `get_document_status`, `get_document_info`, `get_document_chunks` |
| Lifecycle | `archive_document`, `restore_document`, `bulk_delete_session`, `get_gc_status`, `get_lifecycle_policy`, `set_lifecycle_policy` |
| Session | `set_focused_documents`, `get_focused_documents`, `clear_session` |
| Config | `get_rag_config`, `patch_rag_config`, `reset_rag_config`, `save_rag_profile`, `list_profiles`, `get_active_profile`, `set_active_profile`, `delete_rag_profile`, `get_profile_stats` |
| System | `get_system_status`, `rebuild_collection`, `reindex_document` |

By default the server runs with `--tools safe`, which leaves out the seven tools
that delete data, reset configuration or change the server-wide retention
policy: `delete_document`, `clear_session`, `bulk_delete_session`,
`rebuild_collection`, `delete_rag_profile`, `reset_rag_config` and
`set_lifecycle_policy`. Add `"args": ["--tools", "all"]` to the server's entry (or set
`LASAGNA_MCP_TOOLS=all`) to serve them.

Use `search` for every search. `search_knowledge` and `search_with_context` are
deprecated aliases of it: they still answer, but are listed only under
`--tools all`. Use `get_document_context` to learn which documents exist before
searching. `set_focused_documents` narrows every subsequent search to a chosen
set, which is cheaper and more precise than filtering afterwards.

Prefer `archive_document` over `delete_document` when a human might want it
back: archive is recoverable during the grace window and drops out of search
immediately.

## Path B: Python (the application calls retrieval directly)

```python
from lasagna.sdk import KnowledgeClient

client = KnowledgeClient(session_id="my-corpus")
doc = await client.upload("report.pdf")
await client.documents.wait_for_processing(doc.document_id)  # upload returns once queued
results = await client.search_documents("what were Q3 revenues?", top_k=5)
for r in results.results:
    print(r.score, r.chunk.content[:200])
await client.close()
```

Sessions are the isolation boundary: two `session_id`s never see each other's
documents. Use one per tenant, per user, or per experiment.

## Choosing a profile

| Profile | What it does | Needs an LLM key |
|---|---|---|
| `speed` | dense vector search only, 91 MB model | no |
| `standard` | hybrid search, reranking, MMR and agentic retrieval, BGE-M3 (3.1 GB) | yes |
| `balanced` | `standard` plus hierarchical chunking, the structured agentic graph and adaptive routing | yes |
| `max_quality` | `balanced` plus HyDE, query expansion, RAG fusion, CRAG and contextual retrieval | yes |

```bash
lasagna configure --profile standard -s my-corpus
```

A profile applies to the session named with `-s`, and wins over the settings
file and environment for that session (`docs/configuration.md`). Each embedding
model keeps its own collection, so after switching to a profile with another
model, upload the documents again to search them with it.

## When results are bad

Work in this order; the first two explain most cases:

1. `lasagna status -s my-corpus`: are the documents actually ingested, and is
   the chunk count what you expect? Zero chunks means ingestion failed quietly.
2. `lasagna search "..." -s my-corpus -k 20`: if the right chunk appears at
   rank 15, this is a ranking problem (enable reranking); if it never appears,
   it is a retrieval or chunking problem.
3. Only then change the profile. Changing several things at once means you will
   not know which one helped.

## Rules

- **Never** put an API key in `.mcp.json` or any committed file. Use the
  environment.
- The first search downloads a model. On the default profile that is about
  4.5 GB and 7 to 40 minutes. Say so before a user thinks it has hung.
- **Retrieval itself needs no LLM key, but the shipped default can call one.**
  `use_agentic_rag = true` by default, and the router sends a query it judges
  complex to the agentic path, which calls an LLM; without a key that search
  fails with "No LLMProvider configured". Run
  `lasagna configure --profile speed` (which turns it off) or set
  `use_agentic_rag = false` for keyless retrieval.

## If something is broken

Use the `lasagna-support` skill. It captures diagnostics and files a report with
enough detail to be actionable.
