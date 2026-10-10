# Infrastructure -- Technical Services Layer

Shared singletons for embedding, caching, session state, LLM client management, and background task dispatch. Bridges to external services (Redis, GPU/CUDA, 7+ LLM platforms, HuggingFace models, Celery).

---

## Module Structure

| File | Purpose |
| ---- | ------- |
| `__init__.py` | Public API surface (embedding, cache, session store, vector backends) |
| `llm_manager.py` | Singleton LLM instance cache across 8 platforms (self-contained, explicit import required) |
| `session_store.py` | `SessionStore` protocol + in-memory/Redis implementations |
| `cache/backend.py` | `CacheBackend` protocol + in-memory/Redis LRU implementations |
| `cache/search_cache.py` | TTL-based LRU search result cache with session-aware invalidation |
| `embedding/manager.py` | `EmbeddingManager` (BGE-M3 / sentence-transformers), `EmbeddingCache`, `EmbeddingManagerAdapter` (the `default` embedder), late chunking |
| `embedding/embedders.py` | The embedder components: `sentence_transformers`, `bge_m3`, `openai`, `cohere`, `voyage` |
| `tasks/celery_app.py` | Celery application configuration |
| `tasks/document_tasks.py` | Background document processing task + synchronous fallback |

---

## Singleton Inventory

| Singleton | Accessor | Reset / Configure | Config Keys | Reconfiguration |
| --------- | -------- | ----------------- | ----------- | --------------- |
| `EmbeddingManager` | `get_embedding_manager(config)` | `reset_embedding_manager()` | `embedding_model`, `embedding_backend`, `embedding_dim`, `use_gpu`, `enable_embedding_cache`, `embedding_cache_size`, `batch_size`, `normalize_embeddings` | **Drift-reload**: detects config signature change and reloads model under lock |
| `SearchCache` | `get_search_cache(config)` | `reset_search_cache()` | `search_cache_max_size`, `search_cache_ttl` | **First-config-wins**: logs warning if different config passed after init |
| `CacheBackend` | `get_cache_backend()` | `configure_cache_backend(backend)` | `cache_backend`, `redis_url` | **Fixed-after-init**: no drift detection |
| `SessionStore` | `get_session_store()` | `configure_session_store(store)` | `cache_backend`, `redis_url` | **Fixed-after-init**: no drift detection |
| `LLMManager` | `LLMManager()` (class singleton) | N/A (use `set_llm()` / `clear_pre_instantiated_model()`) | Platform-specific env vars | **Cache-per-key**: each unique model settings hash gets its own entry; LRU eviction at max capacity |

All singletons use double-checked locking for thread-safe creation.

---

## External Service Connections

| Service | Used By | Failure Handling |
| ------- | ------- | ---------------- |
| **Redis** | `RedisCacheBackend`, `RedisSessionStore` | Fail-open: operations log warnings and return `None`/`False`; service continues with degraded caching |
| **GPU (CUDA / MPS)** | `EmbeddingManager` | Falls back to CPU; logs device selection |
| **HuggingFace Hub** | `EmbeddingManager` (model download) | Falls back to `embedding_model_fallback`; raises `RuntimeError` if both fail |
| **LLM APIs** (OpenAI, Azure, Groq, WatsonX, Google, OpenRouter) | `LLMManager` | Raises on creation; no retry/fallback (caller handles) |
| **Celery + Redis broker** | `tasks/celery_app.py`, `tasks/document_tasks.py` | `dispatch_processing()` returns `False`; caller processes synchronously |

---

## Import-Time Dependencies

Most infrastructure modules are safe to import in any environment. **Exception:**

- `llm_manager.py` imports `langchain_openai` and `langchain_ibm` at module level. It is **NOT** exported from `infrastructure/__init__.py` to avoid breaking imports when those packages are not installed.

```python
# Safe (always works):
from lasagna.infrastructure import SearchCache, get_embedding_manager

# Requires langchain_openai + langchain_ibm:
from lasagna.infrastructure.llm_manager import LLMManager
```

---

## Embedder Components

The `embedder` stage has six registered components. A sweep arm that names one (`embedders: [...]`) gets a corpus embedded by it, in a collection of its own at its dimension, and its queries are encoded by the same component.

| Name | What it embeds with | Needs |
| ---- | ------------------- | ----- |
| `default` | The configured `embedding_model`, through the shared `EmbeddingManager` | core |
| `sentence_transformers` | Any sentence-transformers model named by `embedding_model`, with the query and document prefixes its model card prescribes | core |
| `bge_m3` | BGE-M3's dense vectors (Chen et al. 2024), through FlagEmbedding when installed and sentence-transformers otherwise. Its sparse and multi-vector outputs are available as methods but are not indexed | `lasagna-rag[flag-embedding]` for FlagEmbedding |
| `openai` | OpenAI `text-embedding-3-small` (default), `-3-large` or `ada-002`; `embedding_truncate_dim` becomes the API's `dimensions` | `OPENAI_API_KEY` |
| `cohere` | Cohere Embed v3 (`embed-english-v3.0` by default) or v4, with `search_query` / `search_document` input types | `pip install cohere`, `COHERE_API_KEY` or `CO_API_KEY` |
| `voyage` | Voyage AI `voyage-3` family, with `query` / `document` input types | `pip install voyageai`, `VOYAGE_API_KEY` |

An `embedding_model` that starts with the provider's model prefix (`text-embedding-`, `embed-`, `voyage-`) picks the API model; any other name gets the provider's default. A missing package or key raises naming it. Model-card prefixes are resolved from the model name, a specific family before a general one: E5's `query:`/`passage:`, the BGE v1.5 retrieval instruction, Nomic's `search_query:`/`search_document:`, the instruction templates of Qwen3-Embedding, gte-Qwen and E5-Mistral, and the query prefixes of mxbai and Arctic-Embed. BGE-M3, GTE and MiniLM get none.

**Late chunking** is an embedding-time setting, not a component: with `enable_late_chunking`, the whole document is embedded once and each chunk's vector is the mean of its tokens' embeddings, so it works with any chunker. See [docs/embedding.md](docs/embedding.md#late-chunking).

---

## Quick Config Reference

| Option | Default | What it does |
| ------ | ------- | ------------ |
| `embedding_model` | `"BAAI/bge-m3"` | Embedding model name (HuggingFace ID) |
| `embedding_backend` | `"auto"` | `"auto"` / `"flag-embedding"` / `"sentence-transformers"` |
| `embedding_truncate_dim` | `None` | Matryoshka truncation: keep this many dimensions and renormalize; the API embedders request that output size where the model supports one |
| `enable_late_chunking` | `false` | Embed the whole document once and pool each chunk's vector from its tokens |
| `late_chunking_overlap` | `512` | Tokens shared by consecutive windows when a document is longer than the model's window |
| `use_gpu` | `true` | Enable GPU acceleration for embeddings |
| `enable_embedding_cache` | `true` | LRU embedding cache |
| `embedding_cache_size` | `10000` | Max cached embeddings (~40 MB at 1024-dim float32) |
| `enable_search_cache` | `true` | TTL-based search result cache |
| `search_cache_max_size` | `1000` | Max cached search results |
| `search_cache_ttl` | `300` | Search cache TTL in seconds (5 min) |
| `cache_backend` | `"memory"` | `"memory"` or `"redis"` (also controls session store) |
| `redis_url` | `"redis://localhost:6379"` | Redis connection URL |
| `celery_enabled` | `false` | Enable Celery background task processing |

---

## Integration Points

The infrastructure layer is consumed through 4 paths:

**1. API Layer** (`api/rag_backend.py`, `api/_search_orchestrator.py`, `api/_document_pipeline.py`)

```python
from lasagna.infrastructure.embedding import get_embedding_manager
from lasagna.infrastructure.cache import get_search_cache
```

**2. Agentic RAG** (`agentic/agentic_rag.py`, `agentic/nodes/retrieve.py`)

```python
from lasagna.infrastructure.embedding.manager import get_embedding_manager
```

**3. Core Pipeline** (`core/registry.py`, `core/session_config/_persistence_db.py`)

```python
from lasagna.infrastructure.embedding.manager import EmbeddingManagerAdapter
from lasagna.infrastructure.session_store import get_session_store
```

**4. SDK / MCP** (`sdk/`, `mcp/`)

```python
from lasagna.infrastructure.session_store import get_session_store
```

---

## Deep Dive Documentation

| Document | Content |
| -------- | ------- |
| [docs/architecture.md](docs/architecture.md) | Dependency graph, singleton lifecycle, graceful degradation, thread safety model |
| [docs/embedding.md](docs/embedding.md) | EmbeddingManager internals, GPU detection, cache mechanics, adapter pattern |
| [docs/caching.md](docs/caching.md) | Two-tier cache design, key computation, invalidation paths, eviction policy |
| [docs/llm_manager.md](docs/llm_manager.md) | 8-platform support, env var reference, model cache, reasoning model handling |
| [docs/session_store.md](docs/session_store.md) | Protocol, dual-mode (memory/Redis), TTL, multi-instance deployment |
