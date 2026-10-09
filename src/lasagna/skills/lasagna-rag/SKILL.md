---
name: lasagna-rag
description: Use when an agent or application needs retrieval over its own documents — ingesting files, searching them, or wiring Lasagna's MCP server into Claude Code, Claude Desktop, Cursor or any MCP client. Covers the fastest path from zero to a working answer.
---

# Give an agent retrieval over a corpus

Two ways in. Pick by whether the agent should call retrieval as a **tool** (MCP)
or the application should call it as a **library** (Python).

## Before anything: does it work here?

```bash
lasagna doctor
```

Reports version, backend, credentials, model cache and anything that failed to
load quietly. Read it before debugging anything else — most "it hangs" reports
are the first-run model download, which this names explicitly.

## Path A — MCP (the agent calls retrieval as a tool)

```bash
pip install lasagna-rag
lasagna configure --profile speed -s my-corpus   # 80 MB model; skip for the 5 GB default
for f in ./docs/*.md; do lasagna upload "$f" -s my-corpus; done
```

Then register the server. **Claude Code** (`.mcp.json` in the project):

```json
{
  "mcpServers": {
    "lasagna": {
      "command": "lasagna-mcp",
      "args": ["--transport", "stdio"],
      "env": { "RAG_SESSION_ID": "my-corpus" }
    }
  }
}
```

**Claude Desktop** — same block inside `claude_desktop_config.json`.

Tools the agent gets: 31 tools, in six groups.

| Group | Tools |
|---|---|
| Search | `search`, `search_knowledge`, `search_with_context`, `get_document_context` |
| Documents | `upload_document`, `list_documents`, `delete_document`, `get_document_status`, `get_document_info`, `get_document_chunks` |
| Lifecycle | `archive_document`, `restore_document`, `bulk_delete_session`, `get_gc_status`, `get_lifecycle_policy`, `set_lifecycle_policy` |
| Session | `set_focused_documents`, `get_focused_documents`, `clear_session` |
| Config | `get_rag_config`, `patch_rag_config`, `reset_rag_config`, `save_rag_profile`, `list_profiles`, `get_active_profile`, `set_active_profile`, `delete_rag_profile`, `get_profile_stats` |
| System | `get_system_status`, `rebuild_collection`, `reindex_document` |

Start with `search`. Use `search_with_context` when the agent needs surrounding
text rather than isolated chunks, and `get_document_context` when it already
knows which document matters. `set_focused_documents` narrows every subsequent
search to a chosen set — cheaper and more precise than filtering afterwards.

Prefer `archive_document` over `delete_document` when a human might want it
back: archive is recoverable during the grace window and drops out of search
immediately.

## Path B — Python (the application calls retrieval directly)

```python
from lasagna.sdk import KnowledgeClient

client = KnowledgeClient(session_id="my-corpus")
await client.upload("report.pdf")
results = await client.search("what were Q3 revenues?", top_k=5)
for r in results:
    print(r.score, r.content[:200])
await client.close()
```

Sessions are the isolation boundary: two `session_id`s never see each other's
documents. Use one per tenant, per user, or per experiment.

## Choosing a profile

| Profile | What it does | When |
|---|---|---|
| `speed` | pure vector search, 80 MB model | demos, CI, first run |
| `standard` | + reranking + hybrid search | most production use |
| `max_quality` | + agentic retrieval | when accuracy beats latency |

```bash
lasagna configure --profile standard
```

## When results are bad

Work in this order — the first two explain most cases:

1. `lasagna status -s my-corpus` — are the documents actually ingested, and is
   the chunk count what you expect? Zero chunks means ingestion failed quietly.
2. `lasagna search "..." -s my-corpus -k 20` — if the right chunk appears at
   rank 15, this is a ranking problem (enable reranking); if it never appears,
   it is a retrieval or chunking problem.
3. Only then change the profile. Changing several things at once means you will
   not know which one helped.

## Rules

- **Never** put an API key in `.mcp.json` or any committed file. Use the
  environment.
- The first search downloads a model. On the default profile that is ~5 GB and
  11–26 minutes. Say so before a user thinks it has hung.
- **Retrieval itself needs no LLM key, but the shipped default does.**
  `use_agentic_rag = true` by default, and the agentic path calls an LLM — so
  a keyless `search` fails with "No LLMProvider configured". Run
  `lasagna configure --profile speed` (which disables it) or set
  `use_agentic_rag = false` for keyless retrieval.

## If something is broken

Use the `lasagna-support` skill. It captures diagnostics and files a report with
enough detail to be actionable.
