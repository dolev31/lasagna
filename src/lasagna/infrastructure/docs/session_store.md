# Session Store -- Deep Dive

## Overview

The `SessionStore` protocol provides shared session state for the RAG system. Session state includes config overrides, focused documents, active profile, and RAG toggle — state that must be consistent when multiple API workers run behind a load balancer.

---

## Protocol

```python
@runtime_checkable
class SessionStore(Protocol):
    def load(self, session_id: str) -> Optional[Dict[str, Any]]: ...
    def save(self, session_id: str, state: Dict[str, Any]) -> None: ...
    def delete(self, session_id: str) -> bool: ...
    def list_sessions(self) -> List[str]: ...
    def cleanup_expired(self, max_age_seconds: float = 86400) -> int: ...
```

---

## Implementations

### InMemorySessionStore

- Storage: `Dict[str, Dict[str, Any]]`
- Thread safety: `threading.Lock` on all operations
- Expiration: explicit via `cleanup_expired()` — compares `last_accessed` timestamp
- Lifetime: lost on process restart (acceptable for single-user mode where `_persistence.py` handles file-based backup)

### RedisSessionStore

- Storage: Redis keys with prefix `"rag:session:"` and JSON-encoded values
- TTL: 86,400 seconds (24 hours) by default, refreshed on `load()`
- Connection: `redis.Redis.from_url(url, decode_responses=True)`
- Expiration: Redis-native TTL (`SETEX`, `EXPIRE`)
- `cleanup_expired()`: no-op (Redis handles it)
- **Fail-open**: all operations catch exceptions and log warnings

### Implementation Symmetry

| Feature | InMemory | Redis |
| ------- | -------- | ----- |
| `load()` | Dict lookup + touch `last_accessed` | `GET` + `EXPIRE` + touch `last_accessed` |
| `save()` | Dict insert + set `last_accessed` | `SETEX` with TTL |
| `delete()` | `dict.pop()` | `DELETE` |
| `list_sessions()` | `dict.keys()` | `SCAN` with prefix match |
| `cleanup_expired()` | Scan + delete by `last_accessed` | No-op (Redis TTL) |
| Thread safety | `threading.Lock` | Redis is inherently thread-safe |
| Failure mode | N/A | Logs warning, returns `None`/`False`/`[]` |

---

## Singleton Lifecycle

`get_session_store()` uses double-checked locking:

```python
_instance = None
_instance_lock = threading.Lock()

def get_session_store():
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                backend_type = rag_config.cache_backend  # "memory" or "redis"
                if backend_type == "redis":
                    _instance = RedisSessionStore(url=rag_config.redis_url)
                else:
                    _instance = InMemorySessionStore()
    return _instance
```

**Fixed-after-init**: no config-drift detection. To replace, use `configure_session_store(store)`.

**Fallback**: if any exception occurs during initialization (including Redis connection failure), falls back to `InMemorySessionStore`.

---

## Multi-Instance Deployment

```
                   ┌──────────┐
        ┌──────────┤  Redis   ├──────────┐
        │          └──────────┘          │
        │                                │
   ┌────┴────┐                      ┌────┴────┐
   │ Worker 1│                      │ Worker 2│
   │         │                      │         │
   │ Redis   │                      │ Redis   │
   │ Session │                      │ Session │
   │ Store   │                      │ Store   │
   └─────────┘                      └─────────┘
```

When `cache_backend = "redis"`:
- All workers connect to the same Redis instance
- Session state is consistent regardless of which worker handles the request
- `SETEX` ensures TTL-based expiry across all workers
- `SCAN` for listing avoids blocking `KEYS` command

---

## Consumers

| Module | Usage |
| ------ | ----- |
| `core/session_config/_persistence_db.py` | Load/save session config overrides |
| `mcp/server.py` | Track rebuild signatures across MCP tool calls |
| Tests (`test_agentic_rag_integration.py`) | Session state for integration tests |

---

## Configuration

| Option | Default | What it does |
| ------ | ------- | ------------ |
| `cache_backend` | `"memory"` | Selects InMemory or Redis implementation |
| `redis_url` | `"redis://localhost:6379"` | Redis connection URL |

The session store shares the `cache_backend` config with `CacheBackend`. Both switch to Redis together.
