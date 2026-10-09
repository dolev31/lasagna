# Database backend parity

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
