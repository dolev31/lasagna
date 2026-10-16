# infrastructure: embedding, caches, session store, background tasks

Process-wide services the pipeline uses but that hold no retrieval logic: the embedding
manager and the `embedder` stage components, the search-result cache and its key-value
backend, the shared session store, and Celery dispatch for background document processing.
Each is a lazily built singleton with an explicit reset or configure function for tests.

Embedding and cache settings with their defaults: [docs/reference/settings.md](../../../docs/reference/settings.md).
Choosing an embedder in a sweep: [docs/guide.md](../../../docs/guide.md#8-measure-it-on-a-benchmark).

## Module map

| File | Responsibility |
| ---- | -------------- |
| `__init__.py` | Re-exports the embedding, cache, session store and vector backend APIs (not `LLMManager`) |
| `embedding/manager.py` | `EmbeddingManager` (model load, device choice, query and document prefixes, late chunking), `EmbeddingCache`, `get_embedding_manager`, `shared_embedding_manager`, `reset_embedding_manager`, `EmbeddingManagerAdapter` (the `default` embedder) |
| `embedding/embedders.py` | Embedder components: `sentence_transformers`, `bge_m3`, and the API embedders `openai`, `cohere`, `voyage` (`_APIEmbedder` base) |
| `cache/search_cache.py` | `SearchCache`: TTL and LRU cache of search results, keyed per session, invalidated on document changes |
| `cache/backend.py` | `CacheBackend` protocol, `InMemoryCacheBackend`, `RedisCacheBackend` |
| `session_store.py` | `SessionStore` protocol, `InMemorySessionStore`, `RedisSessionStore` |
| `tasks/celery_app.py` | Celery app configuration |
| `tasks/document_tasks.py` | Background processing task and `dispatch_processing()`, which returns `False` when the caller must process synchronously |
| `llm_manager.py` | `LLMManager`: an LRU cache of LangChain chat models (default 50, `RAG_LLM_CACHE_MAX_SIZE`). No other module in the package imports it; generation and judge models come from `lasagna.providers.llm_provider` |

Deeper pages: [docs/architecture.md](docs/architecture.md), [docs/embedding.md](docs/embedding.md),
[docs/caching.md](docs/caching.md), [docs/session_store.md](docs/session_store.md),
[docs/llm_manager.md](docs/llm_manager.md).

## Singletons

| Object | Accessor | Reset | When the config changes |
| ------ | -------- | ----- | ----------------------- |
| `EmbeddingManager` | `get_embedding_manager(config)` | `reset_embedding_manager()` | Reloads under `_embedding_lock` when the config signature (`_config_signature`) differs |
| `SearchCache` | `get_search_cache(config)` | `reset_search_cache()` | First config wins; a different one logs a warning |
| `CacheBackend` | `get_cache_backend()` | `configure_cache_backend(b)` | Fixed after first use (`cache_backend`, `redis_url`) |
| `SessionStore` | `get_session_store()` | `configure_session_store(s)` | Fixed after first use; falls back to in-memory if building it fails |
| `LLMManager` | `LLMManager()` | `set_llm()`, `clear_pre_instantiated_model()` | One cache entry per model-settings hash |

All are created with double-checked locking.

## Embedder components

The `embedder` stage: `default` (`EmbeddingManagerAdapter`, the configured `embedding_model`
through the shared manager), `sentence_transformers` and `bge_m3` (local, through
`shared_embedding_manager`), and `openai`, `cohere`, `voyage` (`_APIEmbedder` subclasses; each
needs its package and key and raises naming whichever is missing). An API embedder uses
`embedding_model` when it starts with one of the provider's model prefixes (`families`) and its
`default_model` otherwise. Query and document prefixes are resolved from the model name, as
each model card prescribes. Late chunking (`enable_late_chunking`,
`EmbeddingManager.encode_documents_late_chunking`) is an embedding-time setting, not a
component, so it works with any chunker.

## Design notes

**One model per signature, and arms never replace it.** `get_embedding_manager` swaps the
process singleton when the config signature changes; in-flight callers keep their reference
and the old model is collected later. `shared_embedding_manager` is for components that embed
with their own model (a sweep arm's embedder, the semantic chunker): it returns the singleton
when the signature matches and otherwise keeps up to four extra managers (`_SHARED_MANAGERS_MAX`)
beside it, so an arm naming another model never changes the model other searches use.

**Each embedder gets its own collection.** An arm that names an embedder is ingested into a
collection for that model and dimension, and its queries are encoded by the same component.
See [docs/configuration.md](../../../docs/configuration.md#one-collection-per-embedding-model).

**MPS needs one lock.** Model loads and inference batches run inside
`core.accelerator.device_guard`, because Metal aborts the process when two threads encode at
once. Take it innermost, around one load or one batch.

**Redis is fail-open.** `RedisCacheBackend` and `RedisSessionStore` log a warning and return
`None` or `False` when Redis is unreachable, so the service keeps running without a shared
cache. Celery dispatch is the same: `dispatch_processing()` reports `False` and the caller
processes the document inline.

**Licenses are announced on load.** `EmbeddingManager` calls
`core.model_licenses.warn_if_restricted` for the model it loads, once per model.

**Import cost.** `llm_manager.py` imports `langchain_openai` at module level, which is why the
package `__init__` does not export it.
