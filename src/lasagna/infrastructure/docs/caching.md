# Caching: Deep Dive

## Two-Tier Cache Design

The infrastructure layer provides two complementary cache layers:

```
            High-Level                          Low-Level
    ┌───────────────────────┐          ┌───────────────────────┐
    │      SearchCache      │          │     CacheBackend      │
    │                       │          │                       │
    │  LRU + TTL            │          │  Key-Value store      │
    │  Session-aware        │          │  Protocol-based       │
    │  12-param cache key   │          │  InMemory or Redis    │
    │  Component fingerprint│          │                       │
    └───────────────────────┘          └───────────────────────┘
       Used by: api/_search_             Used by: available for
       orchestrator.py                   custom consumers
```

**SearchCache** is the primary consumer-facing cache. **CacheBackend** is a lower-level building block.

---

## CacheBackend (Low-Level)

### Protocol

```python
class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[bytes]: ...
    def set(self, key: str, value: bytes, ttl: int = 300) -> None: ...
    def delete(self, key: str) -> bool: ...
    def delete_pattern(self, pattern: str) -> int: ...
    def exists(self, key: str) -> bool: ...
    def flush(self) -> None: ...
```

### InMemoryCacheBackend

- Storage: `OrderedDict[str, (bytes, float)]`, LRU-ordered, `(value, expires_at)`
- Lock: `threading.Lock`
- Max size: 10,000 entries (default)
- Eviction: **expired-first, then LRU**: on `set()`, purges expired entries before evicting the least recently used
- TTL: `time.monotonic()` based (immune to clock skew)
- `exists()` delegates to `get()` (checks expiry + updates LRU)

### RedisCacheBackend

- Storage: Redis with key prefix (`"rag:"` default)
- Connection: `redis.Redis.from_url()` with connection pool (`max_connections=10`)
- TTL: Redis-native `SETEX`
- Pattern delete: `SCAN` + `DELETE` (no `KEYS` for safety)
- **Fail-open**: all operations catch exceptions, log warnings, and return `None`/`False`/`0`
- No `close()` method, which is acceptable for a singleton that lives as long as the process

### Implementation Symmetry

| Feature | InMemory | Redis |
| ------- | -------- | ----- |
| TTL | Lazy on access + expired-first eviction | Redis-native |
| LRU eviction | `OrderedDict.popitem(last=False)` | Not applicable (Redis handles) |
| Pattern delete | `fnmatch.fnmatch` | `SCAN` + `DELETE` |
| Flush | `dict.clear()` | `SCAN` prefix + `DELETE` |
| Thread safety | `threading.Lock` | Redis is inherently thread-safe |
| Fail-open | N/A (in-process) | Yes (all ops wrapped in try/except) |

---

## SearchCache (High-Level)

### Key Computation

The cache key is a SHA256 hash (truncated to 32 hex chars) of 12 parameters:

```python
key_parts = {
    "query": query,                        # The search text
    "session_id": session_id,              # Session isolation
    "document_ids": sorted(document_ids),  # Document filter
    "top_k": top_k,                        # Result count
    "use_hyde": use_hyde,                   # HyDE enabled
    "use_reranking": use_reranking,        # Reranking enabled
    "hybrid_alpha": round(alpha, 3),       # Dense/sparse weight
    "mmr_lambda": round(lambda_, 3),       # Diversity parameter
    "hyde_n_docs": hyde_n_docs,            # HyDE doc count (0 if disabled)
    "enable_hybrid_search": bool,          # BM25 fusion
    "enable_mmr": bool,                    # MMR diversity
    "component_fingerprint": str,          # Pipeline identity string
}
```

The `component_fingerprint` joins the identity of the active reranker, retriever override, query enhancer and embedder with `config.settings_fingerprint()`, a hash of the session's resolved config and any per-query overrides. Swapping a component or changing any session or per-query setting (a threshold, the reranker model, the candidate depth) therefore gives a different key, so one sweep arm is never served another arm's results. Process-wide base settings are not hashed: they are fixed for the life of the process, and so is this in-memory cache.

### Eviction Policy: LRU + TTL

```
get(key)
├── Lock acquired (RLock)
├── _maybe_cleanup(): periodic expired entry purge
├── Key exists?
│   ├── No → miss++, return None
│   └── Yes → expired?
│       ├── Yes → delete, miss++, expirations++, return None
│       └── No → move_to_end (LRU), hit++, return results
└── Lock released

put(key, results)
├── Lock acquired (RLock)
├── At capacity?
│   └── Yes → popitem(last=False): evict LRU entry
│       └── Clean up session→keys mapping
├── Store entry with expires_at = now + ttl
├── Update session→keys mapping
└── Lock released
```

### Session-Aware Invalidation

```python
# O(1) invalidation via session → keys mapping
cache._session_keys = {
    "session-123": {"key_abc", "key_def", ...},
    "session-456": {"key_ghi", ...},
}
```

`invalidate_session(session_id)`:
1. Look up all cache keys for the session in `_session_keys`
2. Delete each from `_cache`
3. Remove the session from `_session_keys`

This is O(k) where k = number of cached queries for that session, instead of O(n) scanning the entire cache.

### Invalidation Triggers

| Event | Invalidation | Called By |
| ----- | ------------ | --------- |
| Document uploaded (async) | `invalidate_session(session_id)` | `tasks/document_tasks.py` |
| Document uploaded (sync) | `invalidate_session(session_id)` | `api/_document_pipeline.py` |
| Document deleted | `invalidate_session(session_id)` | `api/_document_pipeline.py` |
| Session state cleared | `invalidate_session(session_id)` | `api/rag_backend.py` (`clear_session_state`) |
| Specific query re-search | `invalidate_query(query, session_id)` | Available but rarely used |
| Config or component change | Implicit (different `component_fingerprint` = different key) | Automatic |

### TTL Enforcement

- **Lazy**: on every `get()`, expired entries are returned as misses
- **Periodic**: `_maybe_cleanup()` runs every `cleanup_interval` seconds (default 60), scanning all entries and removing expired ones
- **Under low traffic**: expired entries persist until the next access or cleanup. This is acceptable: they consume memory but never return stale results.

### Statistics

```text
cache.stats() → {
    "size": 42,            # Current entries
    "max_size": 1000,      # Capacity
    "hits": 1523,          # Cache hits
    "misses": 312,         # Cache misses
    "hit_rate": 0.83,      # hits / (hits + misses)
    "evictions": 5,        # LRU evictions
    "expirations": 89,     # TTL expirations
    "ttl_seconds": 300,    # Configured TTL
}
```

### First-Config-Wins Behavior

`get_search_cache(config)` initializes with the first `config` and retains it. Subsequent calls with a different `config` (different `max_size` or `ttl`) log a warning but do not reconfigure. This prevents dropping a warm cache due to an accidental config divergence between callers.

To force reconfiguration: call `reset_search_cache()` first, then `get_search_cache(new_config)`.

---

## Memory Estimates

### SearchCache

Each `CachedSearchResult` contains:
- `results: List[Dict]`: serialized chunks (~1-5 KB per chunk, 5-20 chunks typical)
- `query_embedding: List[float]`: 1024 floats (~8 KB) with the default `BAAI/bge-m3`
- Metadata fields (~200 bytes)

Estimate per entry: **10-100 KB** (depends on result count and chunk size).

At `max_size=1000`: **10-100 MB** total.

### EmbeddingCache

See [embedding.md](embedding.md#memory-footprint): about 40 MB at default settings.
