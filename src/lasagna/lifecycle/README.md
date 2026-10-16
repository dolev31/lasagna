# Document lifecycle and garbage collection

This package lets a deleted document be restored for a while, expires
documents after a set number of days, cleans up on a schedule, and finds
documents whose metadata row and vectors have come apart.

- **Archive and restore.** Archiving a document (a soft delete) hides it from
  search and listings but keeps its vectors and file. Within the grace window
  (24 hours by default) a restore brings it back without re-ingesting.
- **Expiry.** With `document_ttl_days` set, each upload gets an `expires_at`
  date, and garbage collection archives it once that date passes.
- **Garbage collection.** A six-phase cycle removes archived documents past
  their grace window, archives expired and failed documents, drops expired
  citations and idle session settings, and rebuilds the quota counters from
  the database.
- **Orphan detection.** Compares the metadata database with the vector
  collection and reports documents that exist in only one of them.
- **Whole-session delete.** Deletes every document of a session in one call.

`RAGBackend` builds a `DocumentLifecycleManager` and starts the
`LifecycleScheduler` when it starts; there is nothing to wire up yourself.

## States

```text
UPLOADING -> PROCESSING -> INDEXING -> READY
any state -> ERROR
any state -> SOFT_DELETED          archive, or expiry (phase 2)
ERROR -> SOFT_DELETED              error retention (phase 3)
SOFT_DELETED -> READY              restore, within the grace window
SOFT_DELETED -> removed            garbage collection phase 1, after the grace window
any state -> removed               DELETE /api/v1/documents/{id}, at once
```

Archiving sets `status = soft_deleted` and `deleted_at`. Restoring sets the
status to `ready` and fails with `DocumentExpiredError` once the grace window
has passed.

## Garbage collection

`GarbageCollector.run_full_cycle()` runs the phases in order. Each writes one
row to the `gc_audit_log` table. Given a `session_id`, phases 0 to 3 are limited
to that session; phases 4 to 6 always cover everything.

| Phase (name in reports) | Selects | Does |
| --- | --- | --- |
| 0 `phase0_fail_stale_processing` | documents still `uploading`, `processing` or `indexing` after 12 x `processing_timeout` (at least 1 h) | Marks them `error`, with the reason: the worker processing them stopped (out of memory, a crash, or Celery's 600 s limit). Also runs when a process builds its backend (the lifecycle scheduler starts) |
| 1 `phase1_hard_delete_soft_deleted` | archived documents whose `deleted_at` is older than `soft_delete_grace_hours` | Deletes each through the document pipeline: vectors, file, row, quota |
| 2 `phase2_expire_by_ttl` | documents whose `expires_at` has passed | Archives them. Skipped when `document_ttl_days` is unset or 0 |
| 3 `phase3_cleanup_error_documents` | `error` documents uploaded more than `error_retention_days` ago | Archives them. Skipped when `error_retention_days` is 0 |
| 4 `phase4_cleanup_citations` | citations past their expiry | Deletes them |
| 5 `phase5_cleanup_idle_sessions` | session settings rows not used for `session_idle_days` | Deletes them; documents are untouched |
| 6 `phase6_reconcile_quotas` | every session | Overwrites the quota counters with the counts in the database, whether or not quotas are enforced |

**Safety threshold.** If phase 1 would delete more than
`safety_threshold_percent` (default 0.25) of all documents in the database, it
deletes none and marks its result `safety_aborted`. The other phases still run.

**One cycle at a time.** A cycle takes a lock in the `gc_lock` table (held for
at most 3 hours) and releases it when done. A cycle that finds the lock held
returns at once with `aborted: true`. The orphan scan APScheduler runs takes
the same lock. Processes that share the metadata database therefore never run two
cycles at once, though each process that schedules its own cycles still runs
them.

## Scheduling

At startup `LifecycleScheduler` picks one of two schedulers:

- **Celery Beat**, when Celery is installed (the `production` extra) and its
  broker answers. Lasagna then schedules nothing itself. Beat sends
  `knowledge.gc.run_full_cycle` every `gc_cycle_hours` and
  `knowledge.gc.orphan_scan` (detection only) every `orphan_scan_hours`, read
  from the policy when Beat starts. Run a worker and one Beat process:

  ```bash
  celery -A lasagna.infrastructure.tasks.celery_app worker --loglevel=info
  celery -A lasagna.infrastructure.tasks.celery_app beat --loglevel=info
  ```

  The compose stack's `celery` profile runs both.
- **APScheduler** in the same process, otherwise. It needs the `lifecycle`
  extra (`pip install "lasagna-rag[lifecycle]"`); without it a warning is
  logged and garbage collection runs only when you trigger it. It runs a cycle
  every `gc_cycle_hours` and an orphan scan, which only logs what it finds,
  every `orphan_scan_hours`.

Both count intervals from when they start, not from a time of day.

## Policy

`LifecyclePolicy.load()` (also `LifecyclePolicy.from_env()`) reads the
`[rag.lifecycle]` section of `settings.lasagna.toml`, then lets the variables
below override it. These variables have only the `RAG_` form.

| Field | Default | Environment | Meaning |
| --- | --- | --- | --- |
| `soft_delete_grace_hours` | 24 | `RAG_GC_GRACE_HOURS` | Hours an archived document can be restored |
| `document_ttl_days` | unset | `RAG_GC_TTL_DAYS` | Days from upload to expiry; unset or 0 turns expiry off |
| `error_retention_days` | 7 | `RAG_GC_ERROR_RETENTION_DAYS` | Days before an `error` document is archived; 0 turns this off |
| `session_idle_days` | 30 | `RAG_GC_SESSION_IDLE_DAYS` | Days before an unused session's settings are deleted |
| `gc_cycle_hours` | 24 | `RAG_GC_CYCLE_HOURS` | Hours between cycles |
| `orphan_scan_hours` | 168 | `RAG_GC_ORPHAN_SCAN_HOURS` | Hours between orphan scans |
| `safety_threshold_percent` | 0.25 | `RAG_GC_SAFETY_THRESHOLD` | Fraction of all documents above which phase 1 deletes nothing |
| `batch_size` | 100 | `RAG_GC_BATCH_SIZE` | Accepted and reported; no phase reads it |
| `reconcile_quota_on_startup` | true | `RAG_GC_RECONCILE_QUOTA` | Accepted and reported; nothing reconciles quotas at startup (phase 6 does it each cycle) |

```python
from lasagna.lifecycle import LifecyclePolicy

policy = LifecyclePolicy.load()
policy = LifecyclePolicy(soft_delete_grace_hours=48, document_ttl_days=90, safety_threshold_percent=0.1)
```

`PATCH /api/v1/admin/gc/policy` and the `set_lifecycle_policy` MCP tool change
the running process's policy until it restarts; with APScheduler they also
reschedule the jobs, while Celery Beat keeps its intervals until it restarts.
An upload's `expires_at` comes from the configured policy (settings file and
environment), not from one changed at runtime.

## Orphan detection

`OrphanDetector.detect()` compares the metadata database with the distinct
`document_id` values in the collection of the server's default configuration,
read through the backend's `list_document_ids()`, at most 50,000 of them.

When that scan fails or stops at the limit, the report sets `incomplete` and
`incomplete_reason` and lists no orphans, because a document the scan did not
reach is not missing its vectors. Repair then changes nothing and returns
`aborted: true` with the reason. For a larger collection, scan one session at
a time (`?session_id=...`).

| Kind | Meaning | `repair(dry_run=False)` |
| --- | --- | --- |
| `db_only` | A `ready` row with no vectors, for example after a crash during ingestion. A row whose `collection_name` names another collection is not counted | Archives the row, so phase 1 removes it after the grace window |
| `vector_only` | Vectors that no row owns, in any status: the document was hard-deleted (phase 1, or `DELETE` on a document) or never recorded | Deletes the vectors |

A document still uploading, processing or indexing, one in `error` and an
archived one all own their vectors, so a repair keeps them. Repair reads the
rows again before deleting, so a document saved after the scan keeps its
vectors, and when the rows cannot be read no vectors are reported as orphans.
Files in storage with no row are not detected, because storage backends cannot
list their files.

## REST API

The admin routes need `RAG_ADMIN_TOKEN` (32 characters or more, not a
placeholder; there is no `LASAGNA_` form) in the server's environment and the
same value in the `X-Admin-Token` header. Without a usable token they return
503.

```bash
export RAG_ADMIN_TOKEN=$(openssl rand -hex 32)
```

| Method | Path | Does |
| --- | --- | --- |
| `POST` | `/api/v1/admin/gc/run` | Runs a cycle (optional `?session_id=`) and returns each phase's counts |
| `GET` | `/api/v1/admin/gc/orphans` | Detects orphans; changes nothing |
| `POST` | `/api/v1/admin/gc/repair` | Detects and repairs; `dry_run` defaults to `true` |
| `GET` | `/api/v1/admin/gc/policy` | Returns the running policy |
| `PATCH` | `/api/v1/admin/gc/policy` | Changes policy fields until restart; unknown fields are refused with 422 |
| `DELETE` | `/api/v1/admin/sessions/{session_id}` | Deletes every document of a session, archived ones included |

The document routes use the ordinary session authentication:

| Method | Path | Does |
| --- | --- | --- |
| `POST` | `/api/v1/documents/{id}/archive` | Archives the document |
| `POST` | `/api/v1/documents/{id}/restore` | Restores it within the grace window |
| `DELETE` | `/api/v1/documents/{id}` | Deletes it at once |

```bash
H="X-Admin-Token: $RAG_ADMIN_TOKEN"
curl -H "$H" http://localhost:8000/api/v1/admin/gc/orphans
curl -X POST -H "$H" "http://localhost:8000/api/v1/admin/gc/repair?dry_run=true"
curl -X POST -H "$H" "http://localhost:8000/api/v1/admin/gc/repair?dry_run=false"
curl -X POST -H "$H" http://localhost:8000/api/v1/admin/gc/run
curl -X DELETE -H "$H" http://localhost:8000/api/v1/admin/sessions/session_abc123
```

If phase 1 reports `"safety_aborted": true`, find out why so many documents are
archived before you raise the threshold:

```bash
curl -X PATCH -H "$H" -H "Content-Type: application/json" \
  -d '{"safety_threshold_percent": 0.5}' http://localhost:8000/api/v1/admin/gc/policy
curl -X POST -H "$H" http://localhost:8000/api/v1/admin/gc/run
```

The audit log is in the metadata database (`~/.lasagna/dbs/rag.db` with the
default SQLite backend):

```sql
SELECT run_id, phase, docs_scanned, docs_cleaned, bytes_freed, errors, started_at
FROM gc_audit_log ORDER BY started_at DESC LIMIT 30;
```

## SDK

```python
from lasagna.sdk import KnowledgeClient

async with KnowledgeClient(session_id="demo") as client:
    doc = await client.documents.archive("doc_abc123")
    print(doc.status)                       # soft_deleted
    doc = await client.documents.restore("doc_abc123")
    print(doc.status)                       # ready

    deleted = await client.documents.bulk_delete(["doc_1", "doc_2"])   # number deleted
    result = await client.documents.delete_session(confirm=True)
    print(result["deleted"], result["bytes_freed"])
```

The same calls work on `KnowledgeClient.from_url(...)`. There, `delete_session`
calls the admin route and sends `RAG_ADMIN_TOKEN` from the client's environment.

## MCP tools

| Tool | Does |
| --- | --- |
| `archive_document(document_id, session_id)` | Archives a document |
| `restore_document(document_id, session_id)` | Restores it within the grace window |
| `get_gc_status(session_id)` | Returns the 20 most recent audit log rows (for the whole instance) |
| `get_lifecycle_policy()` | Returns the running policy |
| `set_lifecycle_policy(session_id, policy_updates)` | Changes policy fields (a JSON object) until restart |
| `bulk_delete_session(session_id, confirm=True)` | Deletes every document of the session |

`bulk_delete_session` is left out of the default `safe` tool set; start the
server with `--tools all` (or `LASAGNA_MCP_TOOLS=all`) to register it. On a
networked transport with `RAG_REQUIRE_AUTH` set, the four tools that act on the
whole instance (`get_gc_status`, `get_lifecycle_policy`, `set_lifecycle_policy`,
`bulk_delete_session`) also take `admin_token`.

## Quota

| Event | Quota |
| --- | --- |
| Upload | Reserved at upload |
| Archive | Unchanged: an archived document still counts |
| Phase 1, `DELETE /documents/{id}`, session delete | Released, per document, by the pipeline's delete |
| Phase 6 | Counters overwritten with the database's counts |

## Vector backends

The orphan detector calls `list_document_ids(collection_name, session_id, limit=..., offset=...)`.
The default in `VectorStoreAdapter` scans every row with `iter_all()`, which
pages through `query()`. PgVector and OpenSearch override `list_document_ids`,
and Milvus overrides `iter_all` with a native cursor. A backend that can list distinct document
ids natively should override it too (see
[the vector backends page](../vector_backends/README.md#add-a-backend)).

Tests: `tests/test_lifecycle.py`, `tests/test_gc.py`, `tests/test_orphan_detector.py`.
