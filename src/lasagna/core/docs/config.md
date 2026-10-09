# RAGConfig Property Reference

## Overview

`RAGConfig` in `config.py` exposes **over 180 `@property` decorators** providing
type-safe access to the merged configuration. Properties resolve through a
4-level priority chain:

```
Session override  →  Environment variable (RAG_*)  →  TOML setting  →  Default
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
1. **Session overrides** — from `session_provider.get_resolved_config(session_id)`
2. **Environment variables** — `self._env_overrides[leaf_key]`, type-coerced
3. **TOML settings** — `self._knowledge_settings` (Dynaconf)
4. **Default parameter** — typically from `self._config[key]`

## Environment Variable Override

Any config key can be overridden via `RAG_` prefix:

```bash
export RAG_VECTOR_BACKEND=weaviate
export RAG_DEFAULT_TOP_K=10
export RAG_ENABLE_RERANKING=true
export RAG_EMBEDDING_DIM=384
```

Type coercion rules:
- If default is `bool` → `"true"/"1"/"yes"` → `True`
- If default is `int` → `int(value)`
- If default is `float` → `float(value)`
- If default is `list` → `value.split(",")` (comma-separated)
- Otherwise → `str`

Numeric range validation is applied to known keys (e.g., `default_top_k` must be between 1 and 1000).

## Property Categories

### Vector Backend

| Property | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `vector_backend` | str | `"milvus"` | `RAG_VECTOR_BACKEND` | Vector store backend name |
| `collection_name` | str | `"rag_documents_v2"` | `RAG_COLLECTION_NAME` | Vector collection name |
| `embedding_dim` | int | 1024 | `RAG_EMBEDDING_DIM` | Embedding vector dimension |

A Milvus server address is read from `MILVUS_URI` (and `MILVUS_TOKEN`) by the backend registry; other backend arguments go in `vector_backend_config`.

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
| `chunking_strategy` | str | `"semantic"` | Chunking algorithm |
| `chunk_size` | int | 800 | Target chunk size (characters) |
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
| `db_path` | str | (derived) | SQLite database file path |

### Storage & Upload

| Property | Type | Default | Description |
|---|---|---|---|
| `upload_dir` | str | `"~/.rag_knowledge/uploads"` | Document upload directory |
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

Override with: `RAG_SETTINGS_FILE=/path/to/custom.toml`

```toml
[rag]
vector_backend = "milvus"
embedding_model = "BAAI/bge-m3"
default_top_k = 5
enable_hybrid_search = false
enable_reranking = false
```
