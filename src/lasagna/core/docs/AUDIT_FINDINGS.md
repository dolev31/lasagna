# Core Package Audit Findings

## Parity Contract

### Required Behaviors (both backends MUST exhibit identically)

1. `store_citations()` with N > limit must retain only the most recent `max_citations` per session (per-session scope, default 5000 in hot path)
2. `store_citations()` on conflict (same `citation_id`) must update ALL mutable fields, not a subset
3. `get_stats()` must return shared keys: `documents.{total,sessions,by_status}`, `citations.{total,expired_pending_cleanup,sessions}`, `schema_version`
4. All 16 `DatabaseBackend` Protocol methods must produce identical outputs for identical inputs
5. Schema DDL for shared tables (`rag_documents`, `citations`, `parent_chunks`, `schema_meta`) must be semantically equivalent

### Intended Differences (documented, NOT bugs)

- PostgreSQL has extra tables: `rag_profiles`, `rag_sessions`, `rag_quotas` (distributed session persistence)
- PostgreSQL has 10 extra methods not in Protocol: `save_profile`, `get_profiles`, `delete_profile`, `get_profile`, `save_session`, `get_session`, `get_active_sessions`, `cleanup_expired_sessions`, `get_user_quota`, `save_user_quota`
- Type mappings: `REAL`<->`DOUBLE PRECISION`, `INTEGER`<->`BIGINT`, `TIMESTAMP`<->`TIMESTAMPTZ`, `TEXT '{}'`<->`JSONB '{}'::jsonb`
- Connection strategy: per-call (SQLite) vs pool (PostgreSQL)
- `get_stats()` backend-specific keys: `db_path`/`schema_version` (SQLite), `backend`/`dsn` (PG)

### API Stability Policy

- `__all__` in `core/__init__.py` = stable public API (38 symbols)
- Symbols in submodules but NOT in `__all__` = semi-public (import from submodule directly)
- Semi-public: `DatabaseBackend`, `get_rag_database`, `QuotaManager`, `get_quota_manager`, `RateLimiter`, `get_rate_limiter`, `TenantManager`, `get_tenant_manager`, `ComponentTestHarness`

---

## Findings

| ID | Severity | File:Line | Category | Description | Impact | Fix Status |
|----|----------|-----------|----------|-------------|--------|------------|
| F01 | P0 | `database_pg.py:431-480` | parity | PG `store_citations()` does NOT call `enforce_citation_limit` — SQLite does (atomically, max=5000) | Unbounded citation growth in PG deployments | **FIXED** (CTE-based eviction + composite index) |
| F02 | P0 | `_citations_store.py:32-35` | security | `_get_citations_file_path()` only replaces `/` and `\` — no collision handling, no path containment check | Path traversal risk (low), filename collision risk (medium) | **FIXED** (regex allowlist + hash suffix + length cap + containment check) |
| F03 | P1 | `database_pg.py:454-456` | parity | PG `store_citations` ON CONFLICT updates only 4 fields (`session_id`, `query_id`, `score`, `expires_at`) — SQLite replaces entire row (15 fields) | Stale citation data on re-insert in PG | **FIXED** (all 14 mutable fields now updated) |
| F04 | P1 | `schema.py:264-280` | schema-drift | `CREATE_DOCUMENTS_TABLE` in `schema.py` has stricter constraints (`NOT NULL` vs nullable) and is missing `vector_backend_name`/`storage_backend_name` columns | Misleading constant; consumers get wrong schema | **FIXED** (synced with database.py canonical DDL) |
| F05 | P1 | `quota.py:284-297` | thread-safety | `get_usage()` reads `TenantUsage` fields and calls `searches_last_minute()` (which mutates `_search_timestamps`) outside `self._lock` | Data race on stats endpoint; benign under CPython GIL but not portable | **FIXED** (wrapped in `with self._lock` using `_get_usage_locked`) |
| F06 | P1 | `_persistence_db.py:32-44` | thread-safety | `_get_pg_db()` uses `hasattr`-based singleton with NO lock — race on first concurrent call | Double initialization of PG connection pool | **FIXED** (added `_pg_db_lock` with double-check locking) |
| F07 | P1 | `database.py` + `database_pg.py` | scalability | No composite index `(session_id, created_at DESC, id DESC)` on citations — citation limit enforcement requires separate index scans | Slow citation eviction on large tables | **FIXED** (composite index added to both backends) |
| F08 | P2 | `quota.py:47-61` | scalability | `TenantUsage._search_timestamps` and `_upload_timestamps` are unbounded `list` — grow until pruned by `searches_last_minute()`/`uploads_last_hour()` | Memory growth under sustained load without reads | **FIXED** (replaced with `deque(maxlen=1200/400)` + popleft pruning) |
| F09 | P2 | `database_pg.py` | resource-mgmt | No `close()` method on `PostgresRAGDatabase` — psycopg3 pool never cleanly shut down | Connection leak on singleton reset/testing | **FIXED** (added idempotent `close()`, integrated with `reset_rag_database()`) |
| F10 | P2 | `schema.py:127-147` | dead-code | `RAGQueryRequest` and `RAGQueryResponse` classes have zero external consumers | Unused code | **FIXED** (removed) |
| F11 | P2 | `database_pg.py:707-720` vs `database.py:956-969` | parity | `get_stats()` returns different metadata keys: SQLite has `db_path`/`schema_version`, PG has `backend`/`dsn` — no shared `schema_version` in PG | Minor inconsistency for admin APIs | **FIXED** (added `schema_version` to PG `get_stats()`) |

---

## Feedback Round 2 Findings

| ID | Severity | File:Line | Category | Description | Impact | Fix Status |
|----|----------|-----------|----------|-------------|--------|------------|
| F12 | P0 | `database_pg.py` (store_citations) | parity | CTE-based citation eviction used `OFFSET 5000` + `DELETE WHERE < cutoff`, keeping 5001 rows instead of 5000 (off-by-one) | Per-session citation cap off by one in PG | **FIXED** (changed `<` to `<=` in CTE DELETE) |
| F13 | P0 | `quota.py` (RedisQuotaManager) | distributed-consistency | `check_and_reserve_upload()`, `finalize_upload_reservation()`, `release_upload_reservation()` not overridden in `RedisQuotaManager` — falls through to in-memory base class | Redis-mode quota enforcement broken for distributed upload flow | **FIXED** (added all 3 methods with Redis-backed ZADD/pipeline) |
| F14 | P1 | `quota.py:release_upload_reservation` | correctness | `release_upload_reservation()` decremented doc/storage counters but did not remove the upload timestamp appended by `check_and_reserve_upload()` | Failed uploads still count toward `max_uploads_per_hour` | **FIXED** (added `usage._upload_timestamps.pop()`) |
| F15 | P1 | `quota.py:_get_usage_locked` | scalability | Hardcoded `deque(maxlen=1200/400)` can under-enforce higher custom quotas (e.g., `max_searches_per_minute=2000` needs maxlen >= 4000) | Timestamp eviction before time-window check for high-quota tenants | **FIXED** (maxlen now `max(1200, quota.max_searches_per_minute * 2)` / `max(400, quota.max_uploads_per_hour * 2)`) |
| F16 | P1 | `tests/test_backend_parity.py` | test-quality | Parity test fixture was pinned to `params=["sqlite"]` only, all test methods used `sqlite_db` directly — PG parity never exercised | Misleading "21 parity tests passing" — only SQLite tested | **FIXED** (parametrized `backend` fixture with `[sqlite, pg]`, PG skipped when no DSN; all tests use `backend`) |

---

## Feedback Round 3 Findings

| ID | Severity | File:Line | Category | Description | Impact | Fix Status |
|----|----------|-----------|----------|-------------|--------|------------|
| F17 | P0 | `quota.py` (RedisQuotaManager.check_and_reserve_upload) | distributed-consistency | Redis check-and-reserve was non-atomic: separate GET/CHECK reads then pipeline INCR/ZADD. Two concurrent workers both pass checks before either reserves. | Quota bypass under concurrent uploads | **FIXED** (single Lua script `_LUA_CHECK_AND_RESERVE_UPLOAD` executes all checks+reserve atomically in Redis) |
| F18 | P1 | `quota.py` (check_and_reserve_upload / release_upload_reservation) | correctness | Reservation rollback removes "latest timestamp" not "this reservation's timestamp". In-memory `pop()` / Redis `ZREVRANGEBYSCORE` removes newest entry regardless of which reservation failed. Out-of-order failures corrupt hourly counters. | Incorrect upload rate tracking when concurrent uploads complete out of order | **FIXED** (tagged reservations: `check_and_reserve_upload()` returns reservation_id, `release_upload_reservation()` accepts it. In-memory: dict keyed by ID. Redis: ZADD with ID as member, ZREM by ID. Caller `_document_pipeline.py` updated.) |
| F19 | P2 | `quota.py:set_tenant_quota` | scalability | Dynamic deque sizing only applied on first `TenantUsage` creation. If `set_tenant_quota()` raises limits after usage tracking exists, existing search deque stays at old maxlen. | Under-tracking for sessions with upgraded quotas | **FIXED** (`set_tenant_quota()` now resizes existing search deque with `deque(old_data, maxlen=new_maxlen)`) |
| F20 | P2 | `tests/test_backend_parity.py` | test-quality | No test pushes `store_citations()` beyond 5000 to validate citation cap retention behavior | Off-by-one fix (F12) not stress-tested | **FIXED** (added `test_store_citations_enforces_5000_cap`: inserts 5050 in batches, asserts exactly 5000 remain) |

---

## Feedback Round 4 Findings

| ID | Severity | File:Line | Category | Description | Impact | Fix Status |
|----|----------|-----------|----------|-------------|--------|------------|
| F21 | P1 | `quota.py` (release_upload_reservation) | correctness | `release_upload_reservation()` decremented doc/storage counters even when `reservation_id` was unknown, empty, or already released. The "fallback: remove newest" path could release the wrong reservation entirely. | Quota counter drift (undercount) from duplicate/invalid releases; wrong reservation released on out-of-order calls | **FIXED** (no-op + `logger.warning` for unknown/empty reservation_id in both in-memory `QuotaManager` and `RedisQuotaManager`. Redis uses atomic `_LUA_RELEASE_RESERVATION` Lua script: ZREM checks existence before DECRBY. No fallback path.) |
| F22 | P1 | `_document_pipeline.py` + `document_tasks.py` | correctness | Async/Celery upload path used non-atomic `check_upload()` preflight then later `record_upload()` — TOCTOU gap allowed quota bypass under concurrent uploads. Reservation lifecycle not propagated to Celery task. | Quota bypass when multiple uploads dispatch concurrently via Celery | **FIXED** (`check_and_reserve_upload()` called before Celery dispatch branch; `reservation_id` + `reserved_file_size` forwarded to `dispatch_processing()` → `process_document_task.delay()`. Task finalizes on success, releases on terminal failure, holds on retry.) |
| F23 | P2 | `tests/` | test-quality | No targeted tests for reservation ID tagging, no-op release semantics, double-release idempotency, deque resize, or async dispatch forwarding | Reservation correctness fixes (F18, F21) not regression-tested | **FIXED** (added `tests/test_quota_reservations.py`: 12 tests across 5 classes — `TestReservationTagging`, `TestReleaseNoOp`, `TestReservationQuotaEnforcement`, `TestDequeResize`, `TestAsyncDispatchReservation`) |

---

## Feedback Round 5 Findings

| ID | Severity | File:Line | Category | Description | Impact | Fix Status |
|----|----------|-----------|----------|-------------|--------|------------|
| F24 | P1 | `_document_pipeline.py` (upload_document) | correctness | `check_and_reserve_upload()` ran after `_store_file()` + `_save_document_metadata()`. A race-induced reservation failure left orphaned stored files and DB metadata rows. No rollback on pre-processing I/O failures after reservation. | Orphaned files/metadata when quota exhausted between validation and storage; leaked reserved counters when `_store_file` or `_save_document_metadata` throws | **FIXED** (moved `check_and_reserve_upload()` before `_store_file()`/`_save_document_metadata()`; wrapped both in `try/except` with `release_upload_reservation()` rollback on failure) |
| F25 | P2 | `tests/` | test-quality | No tests for worker-path lifecycle (finalize on success, release on terminal failure, hold on retry) or upload pipeline reservation ordering (reserve-before-I/O, rollback on storage/metadata failure) | Worker and pipeline reservation correctness not regression-tested | **FIXED** (added `TestWorkerReservationLifecycle` — 4 tests: success+reservation→finalize, success−reservation→record_upload, terminal failure+reservation→release, terminal failure−reservation→skip; added `TestUploadPipelineReservationOrdering` — 4 tests: reservation failure prevents storage, store failure releases, metadata failure releases, source-level ordering assertion) |

---

## Feedback Round 6 Findings

| ID | Severity | File:Line | Category | Description | Impact | Fix Status |
|----|----------|-----------|----------|-------------|--------|------------|
| F26 | P1 | `_document_pipeline.py` (upload_document) | correctness | On `_save_document_metadata()` failure, the already-stored file (from `_store_file`) was not cleaned up — only quota was released. | Orphaned stored files accumulate when metadata writes fail | **FIXED** (split `try/except`: `_store_file` failure releases quota only; `_save_document_metadata` failure calls `self.storage.delete(stored_file_path)` best-effort then releases quota) |
| F27 | P1 | `_document_pipeline.py` (upload_document, sync path) | correctness | If `_update_document_status()` threw during sync processing failure, `release_upload_reservation()` was skipped entirely | Leaked reserved quota counters on cascading errors | **FIXED** (wrapped status update in `try/except` with `finally: release_upload_reservation(...)` to guarantee release regardless of status update outcome) |
| F28 | P1 | `document_tasks.py` + `tests/` | testability / test-quality | `process_document_task` body only existed when Celery was installed (`if shared_task is not None`), making it untestable without Celery. Worker lifecycle tests simulated branches manually instead of executing real code paths. Pipeline ordering tests similarly did not call `upload_document()`. | Worker and pipeline correctness was not actually regression-tested by the tests claiming to test them | **FIXED** (extracted `_process_document_impl()` + `TaskRetryContext` from Celery wrapper — testable without Celery. `TestWorkerReservationLifecycle` now calls `_process_document_impl()` directly with mocked deps for all 5 branches: success+reservation, success−reservation, terminal failure+reservation, terminal failure−reservation, retry. `TestUploadPipelineReservationOrdering` now calls `DocumentPipeline.upload_document()` with mocked collaborators for all 4 scenarios: reservation failure prevents I/O, store failure releases, metadata failure deletes file + releases, call-ordering via intercepted mock side effects.) |
