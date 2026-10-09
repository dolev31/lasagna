# Document Lifecycle & Garbage Collector

Production-grade document lifecycle management for the RAG knowledge package. Provides
**recoverable deletion**, **automatic TTL expiry**, **cross-store consistency**, **scheduled GC**,
and **bulk session operations** — all with quota accuracy guarantees and multi-process safety.

## Soft-delete & restore

A deleted document enters a **grace window** (default 24 h) before it is permanently removed.
During this window it is hidden from search results but can be fully restored with a single API
call — no re-ingestion needed.  This protects against accidental deletions that would otherwise
require uploading and re-processing the original file.

## TTL-based expiry

Documents can be given an expiry date at upload time (`document_ttl_days`).  When the date
passes, GC automatically soft-deletes them so they enter the grace window before being purged.
This is useful for compliance requirements (e.g. "retain for 90 days") or for temporary knowledge
bases such as per-meeting or per-sprint document sets that should not accumulate indefinitely.

## Orphan detection

A two-store system (metadata DB + vector store) can become inconsistent after partial failures:
a crash mid-ingestion leaves a DB record with no vectors; a failed hard-delete leaves vectors
with no DB record.  The orphan detector compares document IDs across both stores and classifies
each discrepancy so it can be repaired automatically or reviewed first via a dry-run report.

## Scheduled GC

Without automated cleanup, error documents, expired documents, and idle sessions accumulate
silently and inflate storage costs and quota usage.  A 6-phase GC cycle runs on a configurable
schedule (daily by default) and handles every cleanup category in a single pass, with an advisory
lock that prevents duplicate runs across multiple API workers.

## Bulk session operations

Deleting a session that contains hundreds of documents would otherwise require one API call per
document.  The bulk session delete removes every document, its vectors, and its citations in a
single operation with accurate per-document quota accounting — useful for multi-tenant
deployments where a tenant's entire data set must be wiped cleanly.

## Quota reconciliation

In-memory quota counters can drift from the source-of-truth DB over time (e.g. after a crash,
a rolled-back transaction, or a manual DB edit).  Every GC cycle overwrites the in-memory
counters from the DB, so storage limits remain accurate without requiring a process restart.

---

## 1. State Machine

```text
UPLOADING ──→ PROCESSING ──→ INDEXING ──→ READY
                   │                        │
                 ERROR             POST /archive  (soft delete)
                   │                        │
          (error_retention_days)      SOFT_DELETED ──→ (grace expires) ──→ [hard-deleted]
                   │
          soft-delete + Phase 3
```

### Transitions

| From | To | Trigger | Initiator |
|------|----|---------|-----------|
| READY | SOFT_DELETED | `POST /documents/{id}/archive` | User / SDK / MCP |
| SOFT_DELETED | READY | `POST /documents/{id}/restore` (within grace window) | User / SDK / MCP |
| SOFT_DELETED | *(purged)* | GC Phase 1 (grace window expired) | Scheduler |
| ERROR | SOFT_DELETED | GC Phase 3 (`error_retention_days` exceeded) | Scheduler |
| READY | *(TTL expiry)* | GC Phase 2 (`expires_at` in past) | Scheduler |
| Any | *(purged)* | `DELETE /documents/{id}` | User (immediate, irreversible) |

---

## 3. GC Phases

| Phase | Name | Trigger Condition | Action | Default Frequency |
|-------|------|-------------------|--------|-------------------|
| 1 | hard_delete_soft_deleted | `deleted_at < now() - grace_hours` | Hard-delete via `pipeline.delete_document()` | Daily |
| 2 | expire_by_ttl | `expires_at < now()` | Soft-delete (enters grace window) | Daily |
| 3 | cleanup_error_documents | ERROR status, `upload_date < now() - error_retention_days` | Soft-delete | Daily |
| 4 | cleanup_expired_citations | (existing logic) | Delete expired citation records | Daily |
| 5 | cleanup_idle_sessions | (existing logic) | Clear session config/cache for idle sessions | Daily |
| 6 | reconcile_quotas | Every GC cycle — unconditional DB-to-memory counter sync | Overwrite in-memory QuotaManager counters from authoritative DB values (works even when quota enforcement is disabled) | Daily |

All phases:
- Run in sequence within a single GC cycle
- Write per-phase results to the `gc_audit_log` DB table
- Check the **safety threshold** (default 25%) before Phase 1 — if >25% of all docs
  would be hard-deleted, the phase is aborted to prevent runaway deletes

---

## 4. Orphan Detection

### Algorithm

```text
db_ids  = set of document_id from metadata DB  (status != soft_deleted)
vec_ids = set of document_id from vector store  (via list_document_ids())

db_only  = db_ids - vec_ids   ← DB record, no vectors  (failed ingestion)
vec_only = vec_ids - db_ids   ← Vectors, no DB record  (leaked vectors)
```

### Actions

| Orphan Type | Meaning | Repair Action |
|-------------|---------|---------------|
| `db_only` | Ingestion failed mid-way | Soft-delete the DB record (vectors were never written) |
| `vec_only` | DB delete succeeded but vector delete failed | Delete matching vectors from the backend |

### Limitations

- **Pre-migration docs**: Documents without a `collection_name` field (ingested before this
  migration) are skipped with a warning.
- **Pinecone**: `list_document_ids()` uses a paginated fetch (O(N) scan) because Pinecone has
  no native aggregation API.  For very large indexes, this can be slow.
- **File orphans** (files in storage with no DB record): Out of scope for now.
  The storage backend has no `list_files()` API.

---

## 5. Scheduler Safety

### Celery Beat (production)

In production, periodic GC tasks are driven by Celery Beat:

```bash
# Start worker
celery -A lasagna.infrastructure.tasks.celery_app worker --loglevel=info

# Start beat scheduler (one instance per cluster)
celery -A lasagna.infrastructure.tasks.celery_app beat --loglevel=info
```

Beat schedule (configured in `celery_app.py`):

| Task | Schedule |
|------|----------|
| `knowledge.gc.run_full_cycle` | Daily at 03:00 UTC |
| `knowledge.gc.orphan_scan` | Weekly, Sunday at 04:00 UTC |

### APScheduler (single-node / dev fallback)

When no Celery broker is reachable, `LifecycleScheduler` falls back to APScheduler:

```python
from lasagna.lifecycle import LifecycleScheduler, DocumentLifecycleManager
from lasagna.lifecycle.policy import LifecyclePolicy

policy = LifecyclePolicy.from_env()
# lifecycle_manager is a DocumentLifecycleManager instance
scheduler = LifecycleScheduler(gc=lifecycle_manager._gc, policy=policy)
scheduler.start()  # Auto-detects Celery or falls back to APScheduler
```

### Multi-process Safety (DB Advisory Lock)

Both schedulers guard against duplicate GC runs in multi-worker deployments via the
`gc_lock` DB table:

1. Before each GC cycle, acquire a row in `gc_lock` (owner_id = UUID, expires_at = now + TTL)
2. If another process holds the lock (non-expired row), skip the run silently
3. On completion, release the lock

This ensures exactly one GC run per cycle even with multiple Gunicorn workers.

---

## 6. Configuration Reference

Policy is loaded with a three-tier priority chain (highest wins):

1. **Environment variables** (`RAG_GC_*`)
2. **`[rag.lifecycle]` section** of `settings.lasagna.toml`
3. **Dataclass defaults**

| Field | Default | Env Var | Description |
|-------|---------|---------|-------------|
| `soft_delete_grace_hours` | 24 | `RAG_GC_GRACE_HOURS` | Hours before GC hard-deletes soft-deleted docs |
| `document_ttl_days` | `None` | `RAG_GC_TTL_DAYS` | Auto-expire documents after N days (`None` or `0` = disabled; env `RAG_GC_TTL_DAYS=0` also disables) |
| `error_retention_days` | 7 | `RAG_GC_ERROR_RETENTION_DAYS` | Days before ERROR docs are soft-deleted |
| `session_idle_days` | 30 | `RAG_GC_SESSION_IDLE_DAYS` | Days before idle sessions are cleaned |
| `gc_cycle_hours` | 24 | `RAG_GC_CYCLE_HOURS` | Full GC cycle interval (APScheduler **and** Celery Beat) |
| `orphan_scan_hours` | 168 | `RAG_GC_ORPHAN_SCAN_HOURS` | Orphan scan interval (APScheduler **and** Celery Beat) |
| `safety_threshold_percent` | 0.25 | `RAG_GC_SAFETY_THRESHOLD` | Abort if >25% of docs would be hard-deleted in one cycle |
| `batch_size` | 100 | `RAG_GC_BATCH_SIZE` | Docs per DB/vector batch in GC phases |
| `reconcile_quota_on_startup` | `True` | `RAG_GC_RECONCILE_QUOTA` | Rebuild quota counters from DB at startup |

```python
from lasagna.lifecycle import LifecyclePolicy

# Programmatic — explicit values override everything
policy = LifecyclePolicy(
    soft_delete_grace_hours=48,   # 2-day grace window
    document_ttl_days=90,         # Auto-expire after 90 days
    safety_threshold_percent=0.1, # More conservative: abort if >10%
)

# Recommended: load from TOML + env overrides
policy = LifecyclePolicy.load()   # reads [rag.lifecycle] TOML, then RAG_GC_* env vars

# Backward-compatible alias
policy = LifecyclePolicy.from_env()  # same as load()
```

To change the policy at runtime without a restart, use `PATCH /api/v1/admin/gc/policy`
or the `set_lifecycle_policy` MCP tool (see §7 below).

---

## 7. Admin API

All admin endpoints require the `X-Admin-Token` header. Configure via:

```bash
export RAG_ADMIN_TOKEN=my-secret-admin-token
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/admin/gc/run` | Trigger a full GC cycle |
| `GET` | `/api/v1/admin/gc/orphans` | Detect orphans (read-only); includes `scan_limited` flag |
| `POST` | `/api/v1/admin/gc/repair?dry_run=false` | Repair orphans |
| `GET` | `/api/v1/admin/gc/policy` | Return current lifecycle policy |
| `PATCH` | `/api/v1/admin/gc/policy` | Update policy fields at runtime (not persisted) |
| `DELETE` | `/api/v1/admin/sessions/{session_id}` | Bulk-delete entire session |

### Document Lifecycle Endpoints (no admin token needed)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/documents/{id}/archive` | Soft-delete a document |
| `POST` | `/api/v1/documents/{id}/restore` | Restore a soft-deleted document |
| `DELETE` | `/api/v1/documents/{id}` | Hard-delete (unchanged, non-breaking) |

---

## 8. SDK Usage

```python
from lasagna.sdk import KnowledgeClient

async with KnowledgeClient.from_url("http://localhost:8000") as client:
    # Soft-delete (recoverable)
    doc = await client.documents.archive("doc_abc123")
    print(doc.status)  # DocumentStatus.SOFT_DELETED

    # Restore within grace window
    doc = await client.documents.restore("doc_abc123")
    print(doc.status)  # DocumentStatus.READY

    # Bulk delete specific documents
    deleted = await client.documents.bulk_delete(["doc_1", "doc_2", "doc_3"])

    # Delete entire session efficiently
    result = await client.documents.delete_session(confirm=True)
    print(f"Deleted {result['deleted']} docs, freed {result['bytes_freed']} bytes")
```

---

## 9. MCP Tools

Six lifecycle MCP tools are registered in the server:

| Tool | Description |
|------|-------------|
| `archive_document(document_id, session_id)` | Soft-delete a document |
| `restore_document(document_id, session_id)` | Restore within grace window |
| `bulk_delete_session(session_id, confirm=True)` | Wipe entire session |
| `get_gc_status(session_id)` | View recent GC audit log entries |
| `get_lifecycle_policy()` | Return the current lifecycle policy |
| `set_lifecycle_policy(policy_updates)` | Update policy fields at runtime (not persisted) |

---

## 10. Quota Semantics

| Event | Quota Impact |
|-------|-------------|
| Upload | Quota reserved immediately |
| Soft-delete | **No change** — quota still consumed during grace window |
| GC Phase 1 (hard-delete) | **Quota freed** — `record_delete(bytes, chunks)` called |
| `DELETE /documents/{id}` (hard-delete) | Quota freed immediately |
| `DELETE /admin/sessions/{id}` | Quota freed per-document (pipeline deletes each doc individually) |

This ensures quota is freed exactly once (on hard delete) without risk of double-decrement.

---

## 11. Operational Runbook

### Check orphan / consistency status (read-only)

```bash
# Read-only orphan scan — does NOT modify any data
curl -H "X-Admin-Token: $RAG_ADMIN_TOKEN" \
  http://localhost:8000/api/v1/admin/gc/orphans
# Or via MCP:
get_gc_status(session_id="any_session")
```

### Manually trigger GC

```bash
# POST triggers a full GC cycle (destructive, acquires advisory lock)
curl -X POST -H "X-Admin-Token: $RAG_ADMIN_TOKEN" \
  "http://localhost:8000/api/v1/admin/gc/run"
```

### Diagnose orphans

```bash
# Detect (read-only)
curl -H "X-Admin-Token: $RAG_ADMIN_TOKEN" \
  "http://localhost:8000/api/v1/admin/gc/orphans"

# Repair (preview first, then apply)
curl -X POST -H "X-Admin-Token: $RAG_ADMIN_TOKEN" \
  "http://localhost:8000/api/v1/admin/gc/repair?dry_run=true"

curl -X POST -H "X-Admin-Token: $RAG_ADMIN_TOKEN" \
  "http://localhost:8000/api/v1/admin/gc/repair?dry_run=false"
```

### Bulk-delete a session

```bash
curl -X DELETE -H "X-Admin-Token: $RAG_ADMIN_TOKEN" \
  "http://localhost:8000/api/v1/admin/sessions/session_abc123"
```

### Interpret the audit log

The `gc_audit_log` table has one row per GC phase per run:

```sql
SELECT run_id, phase, docs_scanned, docs_cleaned, bytes_freed, errors, started_at
FROM gc_audit_log
ORDER BY started_at DESC
LIMIT 30;
```

### Safety threshold triggered

If you see `aborted=true` in a GC report, the safety threshold fired. This means the GC
detected that >25% of docs (or >25% of the session's docs) would be hard-deleted. Investigate
why before re-running with a higher threshold:

```python
from lasagna.lifecycle import LifecyclePolicy
policy = LifecyclePolicy(safety_threshold_percent=0.5)
backend._lifecycle_manager._gc._policy = policy
backend.run_gc()
```

---

## 12. Extension Guide — New Vector Backend

When implementing a new vector backend, override `list_document_ids()` for efficient
orphan detection. The base class provides a fallback via `query()` but it may be
inefficient for large collections.

```python
class MyNewBackend(VectorStoreAdapter):
    # ...plus the abstract methods every backend implements (see vector_backends/README.md)

    def list_document_ids(
        self,
        collection_name: str,
        session_id: Optional[str] = None,
        *,
        limit: int = 10_000,
        offset: int = 0,
    ) -> List[str]:
        """Return distinct document_ids from the vector store."""
        # Use your backend's native aggregation/distinct API:
        # e.g. SQL: SELECT DISTINCT document_id FROM ... LIMIT ? OFFSET ?
        # e.g. aggregation: {"aggs": {"doc_ids": {"terms": {"field": "document_id", "size": limit}}}}
        ...
```
