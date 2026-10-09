# Architecture -- Infrastructure Layer

## Dependency Graph

```
                           ┌─────────────────────────────┐
                           │    External Services         │
                           │  Redis · GPU · LLM APIs ·    │
                           │  HuggingFace · Celery        │
                           └──────────┬──────────────────┘
                                      │
        ┌─────────────────────────────┼──────────────────────────────┐
        │               infrastructure/                              │
        │                                                            │
        │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
        │  │  embedding/  │  │    cache/     │  │  session_store   │ │
        │  │  manager.py  │  │  backend.py   │  │                  │ │
        │  │              │  │  search_      │  │  InMemory /      │ │
        │  │  BGE-M3 /    │  │  cache.py     │  │  Redis           │ │
        │  │  sentence-   │  │              │  │                  │ │
        │  │  transformers│  │  InMemory /   │  └──────────────────┘ │
        │  └──────────────┘  │  Redis        │                      │
        │                    └──────────────┘  ┌──────────────────┐ │
        │  ┌──────────────┐                    │  llm_manager.py  │ │
        │  │   tasks/     │                    │  (explicit       │ │
        │  │  celery_app  │                    │   import only)   │ │
        │  │  document_   │                    │  8 LLM platforms │ │
        │  │  tasks       │                    └──────────────────┘ │
        │  └──────────────┘                                         │
        └─────────────────────────────┬─────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │            Consumers                                       │
        │                                                            │
        │  api/           agentic/        core/          sdk/  mcp/ │
        │  rag_backend    agentic_rag     registry       client     │
        │  _search_orch.  nodes/retrieve  session_config             │
        │  _doc_pipeline  nodes/refine    base_components            │
        │  _collection_mgr                                           │
        └───────────────────────────────────────────────────────────┘
                                      │
                           ┌──────────┴──────────┐
                           │   core/config.py    │
                           │   (RAGConfig)       │
                           └─────────────────────┘
```

---

## Singleton Lifecycle

### Creation (all 5 singletons)

All singletons use double-checked locking:

```python
_instance = None
_lock = threading.Lock()

def get_singleton():
    global _instance
    if _instance is None:                    # Fast path (no lock)
        with _lock:
            if _instance is None:            # Double-check under lock
                _instance = create_instance()
    return _instance
```

**LLMManager** uses a class-level `__new__` variant of the same pattern.

### Config-Drift Reload

Only **EmbeddingManager** supports automatic config-drift detection and reload:

```
get_embedding_manager(config)
├── Instance exists?
│   ├── No  → create under lock → return
│   └── Yes → config signature matches?
│       ├── Yes → return existing
│       └── No  → acquire lock → double-check signature
│           ├── Still mismatched → create new manager → replace singleton
│           │                      → best-effort torch.cuda.empty_cache()
│           └── Already fixed by another thread → return existing
```

The old manager is NOT explicitly destroyed. In-flight requests holding a reference continue using it until they complete. Python GC reclaims it when all references are dropped.

### Reconfiguration Semantics

| Singleton | Behavior | Details |
| --------- | -------- | ------- |
| EmbeddingManager | Drift-reload | Compares 9-field signature; reloads model under lock |
| SearchCache | First-config-wins | Logs warning if different config passed after init |
| CacheBackend | Fixed-after-init | Uses `configure_cache_backend()` for explicit replacement |
| SessionStore | Fixed-after-init | Uses `configure_session_store()` for explicit replacement |
| LLMManager | Cache-per-key | Each model settings hash gets an LRU-cached entry; eviction at max capacity |

---

## Graceful Degradation Matrix

| Component | Failure | Behavior |
| --------- | ------- | -------- |
| Redis cache | Connection refused | `RedisCacheBackend` logs warning, returns `None` on get / no-op on set. Service continues without caching. |
| Redis session store | Connection refused | `get_session_store()` falls back to `InMemorySessionStore`. Warning logged. |
| GPU (CUDA) | Not available | Falls back to MPS (Apple Silicon) or CPU. Warning logged. |
| FlagEmbedding library | Not installed | Falls back to sentence-transformers. Warning logged. |
| Primary embedding model | Download fails | Falls back to `embedding_model_fallback`. If both fail: `RuntimeError`. |
| Celery broker | Not available | `dispatch_processing()` returns `False`. Caller processes synchronously. |
| Celery not installed | Import fails | `process_document_task = None`. Dispatch always returns `False`. |
| LLM platform env vars | Missing | `ValueError` with descriptive message (WatsonX, OpenRouter). |
| langchain_openai/ibm | Not installed | Only affects `llm_manager.py` (not imported from `__init__.py`). |

---

## Thread Safety Model

| Component | Lock Type | What It Protects |
| --------- | --------- | ---------------- |
| `get_embedding_manager()` | `threading.Lock` | Singleton creation + config-drift reload |
| `EmbeddingCache` | `threading.RLock` | LRU cache operations (allows re-entrant access) |
| `SearchCache` | `threading.RLock` | LRU+TTL cache + session-keys mapping |
| `get_search_cache()` | `threading.Lock` | Singleton creation |
| `get_cache_backend()` | `threading.Lock` | Singleton creation |
| `InMemoryCacheBackend` | `threading.Lock` | LRU cache operations |
| `get_session_store()` | `threading.Lock` | Singleton creation |
| `InMemorySessionStore` | `threading.Lock` | Session dict operations |
| `LLMManager.__new__()` | `threading.Lock` (class-level) | Singleton creation |
| `LLMManager.get_model()` | `threading.Lock` (instance `_cache_lock`) | Pre-instantiated check + model cache lookup + creation + LRU eviction |
| `LLMManager.set_llm()` / `clear_pre_instantiated_model()` | `threading.Lock` (instance `_cache_lock`) | Model source switching (pre-instantiated vs cache) |

**Note:** `_update_model_parameters()` mutates models in-place. For cached models, this is done under `_cache_lock`. For the pre-instantiated model (`set_llm()` path), a per-call clone is created via `model_copy(deep=True)` (Pydantic v2) or `copy.deepcopy()`, so parameter mutation does not leak across threads. The model cache capacity (`_max_models`) is configurable via `RAG_LLM_CACHE_MAX_SIZE` env var with validation — invalid or non-positive values fall back to the default (50) with a warning logged.
