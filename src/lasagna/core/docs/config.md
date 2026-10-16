# RAGConfig Property Reference

## Overview

`RAGConfig` in `config.py` exposes **over 230 `@property` decorators** providing
type-safe access to the merged configuration. Properties resolve through a
4-level priority chain:

```
Session override  →  Environment variable (LASAGNA_* or RAG_*)  →  TOML setting  →  Default
```

## Resolution Mechanism

All properties delegate to `_get_knowledge_setting(*keys, default=value)`:

```python
@property
def vector_backend(self) -> str:
    return self._get_knowledge_setting(
        "rag", "vector_backend", default=self._config["vector_backend"]
    )
```

The method checks in order:
1. **Session overrides**: from `session_provider.get_resolved_config(session_id)`
2. **Environment variables**: `self._env_overrides[leaf_key]`, type-coerced
3. **TOML settings**: `self._knowledge_settings` (Dynaconf)
4. **Default parameter**: typically from `self._config[key]`

## Environment Variable Override

Any config key can be overridden with the `RAG_` prefix, or with `LASAGNA_`, which wins when both are set:

```bash
export RAG_VECTOR_BACKEND=weaviate
export LASAGNA_DEFAULT_TOP_K=10
export RAG_ENABLE_RERANKING=true
export RAG_EMBEDDING_DIM=384
```

Type coercion rules (`_parse_env_value`):
- If default is `bool` → `true`/`1`/`yes`/`on` → `True`, `false`/`0`/`no`/`off` → `False`; anything else is an error
- If default is `int` → `int(value)`
- If default is `float` → `float(value)`
- If default is a `list` or `dict` → parsed as JSON, and it must be a JSON list or object
- Otherwise → `str`

An empty value counts as unset. Numeric range validation is applied to known keys (e.g., `default_top_k` must be between 1 and 1000), and some string settings take one of a fixed set (`database_backend`, `cache_backend`, `tenant_isolation`, `log_level`, `routing_default_tier`, `routing_strategy`). An invalid value raises `ConfigError` when the config loads, naming the variable; a `RAG_` name that is no setting logs a warning with the closest one.

## Property Categories

### Vector Backend

| Property | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `vector_backend` | str | `"milvus"` | `RAG_VECTOR_BACKEND` | Vector store backend name |
| `collection_name` | str | `"rag_documents_v2"` | `RAG_COLLECTION_NAME` | Vector collection name; the default gives each embedding model its own (`rag_documents_v2__<model>__<dim>`) |
| `embedding_dim` | int | 1024 | `RAG_EMBEDDING_DIM` | Embedding vector dimension |

A Milvus server address is `"uri"` in `vector_backend_config`, else `MILVUS_URI` (with `MILVUS_TOKEN`); without one the backend opens Milvus Lite at `"db_path"` or `<db dir>/rag_milvus.db`. Setting both a `"db_path"` and a server address is an error. Other backend arguments go in `vector_backend_config`.

### Embedding

| Property | Type | Default | Description |
|---|---|---|---|
| `embedding_model` | str | `"BAAI/bge-m3"` | HuggingFace model name |
| `use_gpu` | bool | `True` | Use a GPU when one is available |
| `batch_size` | int | 32 | Batch size for encoding |
| `normalize_embeddings` | bool | `True` | L2-normalize embeddings |

### Search & Retrieval

| Property | Type | Default | Description |
|---|---|---|---|
| `default_top_k` | int | 20 | Number of candidates to retrieve |
| `similarity_threshold` | float | 0.3 | Minimum similarity threshold |
| `enable_hybrid_search` | bool | `True` | Enable hybrid (dense + sparse) |
| `hybrid_alpha` | float | 0.7 | Dense vs sparse weight |
| `enable_mmr` | bool | `True` | Maximal Marginal Relevance |
| `mmr_lambda` | float | 0.5 | MMR relevance vs diversity weight |

### Reranking

| Property | Type | Default | Description |
|---|---|---|---|
| `enable_reranking` | bool | `True` | Enable cross-encoder reranking |
| `reranker_model` | str | `"mixedbread-ai/mxbai-rerank-large-v1"` | Reranker model |
| `rerank_top_k` | int | 5 | Final results after reranking |

### Chunking

| Property | Type | Default | Description |
|---|---|---|---|
| `chunking_strategy` | str | `"semantic"` | Built-in strategy: `semantic`, `fixed`, `sentence_window` or `recursive` (applies when `enable_hierarchical`, `preserve_headers` and `preserve_tables` are off) |
| `chunk_size` | int | 800 | Target chunk size, in `chunk_unit` |
| `chunk_unit` | str | `"chars"` | `chars`, `words` or `tokens` (of `chunk_tokenizer`), for the `fixed` and `recursive` chunkers |
| `chunk_overlap` | int | 0 | Overlap between chunks |
| `enable_hierarchical` | bool | `True` | Enable parent-child chunking |

### Query Enhancement

| Property | Type | Default | Description |
|---|---|---|---|
| `enable_hyde` | bool | `False` | Hypothetical Document Embeddings |
| `enable_query_expansion` | bool | `False` | Query expansion |
| `enable_query_decomposition` | bool | `False` | Split complex questions into sub-queries |
| `enable_rag_fusion` | bool | `False` | RAG Fusion multi-query |

### Agentic RAG

| Property | Type | Default | Description |
|---|---|---|---|
| `use_agentic_rag` | bool | `True` | Enable agentic RAG |
| `max_agent_iterations` | int | 3 | Max agentic retry iterations |

### Database

| Property | Type | Default | Description |
|---|---|---|---|
| `database_backend` | str | `"sqlite"` | `"sqlite"` or `"postgresql"` |
| `database_dsn` | str | `""` | PostgreSQL connection string |
| `database_pool_min` | int | 2 | Min pool connections |
| `database_pool_max` | int | 10 | Max pool connections |
| `db_path` | str | (derived) | SQLite database file path, `~/.lasagna/dbs/rag.db` by default (`LASAGNA_DB_DIR` or `RAG_DB_DIR` moves the directory) |

### Storage & Upload

| Property | Type | Default | Description |
|---|---|---|---|
| `upload_dir` | str | `"~/.lasagna/uploads"` | Document upload directory (`LASAGNA_UPLOAD_DIR` or `RAG_UPLOAD_DIR`; under `LASAGNA_DATA_DIR` when that is set) |
| `max_file_size` | int | 52428800 | Max file upload size in bytes (50 MB) |
| `storage_backend` | str | `"local"` | File storage backend |

### Rate Limiting & Quotas

| Property | Type | Default | Description |
|---|---|---|---|
| `rate_limiting_enabled` | bool | `False` | Enable rate limiting |
| `rate_limiting_strategy` | str | `"adaptive"` | `"adaptive"` or `"fixed"` |
| `rate_limiting_rpm` | float | 60.0 | Requests per minute per tenant |
| `quota_enabled` | bool | `False` | Enable quota enforcement |

### Tenant Isolation

| Property | Type | Default | Description |
|---|---|---|---|
| `tenant_isolation` | str | `"none"` | `"none"` or `"partition"` |

### Cache

| Property | Type | Default | Description |
|---|---|---|---|
| `cache_backend` | str | `"memory"` | `"memory"` or `"redis"` |
| `redis_url` | str | `"redis://localhost:6379"` | Redis connection URL |

### Component Settings

Each built-in component reads its own settings, documented beside it: retrievers, rerankers and query enhancers in [retrieval/README.md](../../retrieval/README.md#built-in-components), chunkers and document processors in [processing/README.md](../../processing/README.md), embedders in [infrastructure/README.md](../../infrastructure/README.md#embedder-components), and the published agentic loops in [agentic/docs/config.md](../../agentic/docs/config.md#published-loop-settings). They resolve through the same chain as every property above.

## Session-Scoped Config

```python
# Create a session-scoped config instance
session_config = rag_config.with_session("sess_123")

# Properties now check session overrides first
print(session_config.default_top_k)  # may differ from rag_config.default_top_k
```

The `with_session()` method returns a shallow copy of `RAGConfig` bound to a
session ID. Session overrides are applied by the `KnowledgeConfigManager`
via `patch_config()`.

## TOML Configuration File

Default location: `src/lasagna/settings.lasagna.toml`

Override with: `LASAGNA_SETTINGS_FILE=/path/to/custom.toml` (or `RAG_SETTINGS_FILE`)

```toml
[rag]
vector_backend = "milvus"
embedding_model = "BAAI/bge-m3"
default_top_k = 5
enable_hybrid_search = false
enable_reranking = false
```
