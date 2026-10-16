# Database backend parity

## Parity Contract

### Required Behaviors (both backends MUST exhibit identically)

1. `store_citations()` with N > limit must retain only the most recent `max_citations` per session (per-session scope, default 5000 in hot path)
2. `store_citations()` on conflict (same `citation_id`) must update ALL mutable fields, not a subset
3. `get_stats()` must return shared keys: `documents.{total,sessions,by_status}`, `citations.{total,expired_pending_cleanup,sessions}`, `schema_version`
4. All 41 `DatabaseBackend` Protocol methods must produce identical outputs for identical inputs
5. Schema DDL for shared tables (`rag_documents`, `citations`, `parent_chunks`, `schema_meta`, `rag_profiles`, `rag_sessions`, `collection_meta`, `gc_audit_log`, `gc_lock`) must be semantically equivalent

### Intended Differences (documented, NOT bugs)

- PostgreSQL has one extra table: `rag_quotas` (quota storage shared across replicas)
- PostgreSQL has methods outside the Protocol that SQLite lacks: `get_profile`, `get_user_quota`, `save_user_quota`, `close` (pool shutdown). SQLite has `connection()` and `transaction()` context managers that PostgreSQL lacks.
- Type mappings: `REAL`<->`DOUBLE PRECISION`, `INTEGER`<->`BIGINT`, `TIMESTAMP`<->`TIMESTAMPTZ`, `TEXT '{}'`<->`JSONB '{}'::jsonb`
- Connection strategy: per-call (SQLite) vs pool (PostgreSQL)
- `get_stats()` backend-specific keys: `db_path`/`schema_version` (SQLite), `backend`/`dsn` (PG)

### API Stability Policy

- `__all__` in `core/__init__.py` = stable public API (33 symbols)
- Symbols in submodules but NOT in `__all__` = semi-public (import from submodule directly)
- Semi-public: `DatabaseBackend`, `get_rag_database`, `QuotaManager`, `get_quota_manager`, `RateLimiter`, `get_rate_limiter`, `TenantManager`, `get_tenant_manager`, `ComponentTestHarness`
