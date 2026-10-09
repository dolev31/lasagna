# RAGConfig Property Reference

## Overview

`RAGConfig` in `config.py` exposes **160 `@property` decorators** providing
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
    return self._get_knowledge_setting("vector_backend", default="milvus")
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
export RAG_TOP_K=10
export RAG_RERANKING_ENABLED=true
export RAG_EMBEDDING_DIM=384
```

Type coercion rules:
- If default is `bool` → `"true"/"1"/"yes"` → `True`
- If default is `int` → `int(value)`
- If default is `float` → `float(value)`
- If default is `list` → `value.split(",")` (comma-separated)
- Otherwise → `str`

Numeric range validation is applied to known keys (e.g., `top_k` must be > 0).

## Property Categories

### Vector Backend

| Property | Type | Default | Env Var | Description |
|---|---|---|---|---|
| `vector_backend` | str | `"milvus"` | `RAG_VECTOR_BACKEND` | Vector store backend name |
| `milvus_uri` | str | `"./milvus_lite.db"` | `RAG_MILVUS_URI` | Milvus connection URI |
| `milvus_collection_name` | str | `"rag_documents"` | `RAG_MILVUS_COLLECTION_NAME` | Milvus collection name |
| `embedding_dim` | int | 1024 | `RAG_EMBEDDING_DIM` | Embedding vector dimension |

### Embedding

| Property | Type | Default | Description |
|---|---|---|---|
| `embedding_model` | str | `"BAAI/bge-m3"` | HuggingFace model name |
| `embedding_device` | str | `"cpu"` | Compute device |
| `embedding_batch_size` | int | 32 | Batch size for encoding |
| `embedding_normalize` | bool | `True` | L2-normalize embeddings |

### Search & Retrieval

| Property | Type | Default | Description |
|---|---|---|---|
| `top_k` | int | 5 | Number of results to return |
| `min_score` | float | 0.0 | Minimum similarity threshold |
| `hybrid_search_enabled` | bool | `False` | Enable hybrid (dense + sparse) |
| `hybrid_alpha` | float | 0.7 | Dense vs sparse weight |
| `mmr_enabled` | bool | `False` | Maximal Marginal Relevance |
| `mmr_diversity_factor` | float | 0.3 | MMR lambda parameter |

### Reranking

| Property | Type | Default | Description |
|---|---|---|---|
| `reranking_enabled` | bool | `False` | Enable cross-encoder reranking |
| `reranker_model` | str | `"cross-encoder/ms-marco-MiniLM-L-12-v2"` | Reranker model |
| `reranker_top_k` | int | 5 | Final results after reranking |

### Chunking

| Property | Type | Default | Description |
|---|---|---|---|
| `chunk_strategy` | str | `"semantic"` | Chunking algorithm |
| `chunk_size` | int | 512 | Target chunk size (tokens) |
| `chunk_overlap` | int | 50 | Overlap between chunks |
| `hierarchical_chunking` | bool | `False` | Enable parent-child chunking |

### Query Enhancement

| Property | Type | Default | Description |
|---|---|---|---|
| `query_enhancement_enabled` | bool | `False` | Enable query enhancement |
| `hyde_enabled` | bool | `False` | Hypothetical Document Embeddings |
| `query_expansion_enabled` | bool | `False` | Query expansion |
| `rag_fusion_enabled` | bool | `False` | RAG Fusion multi-query |

### Agentic RAG

| Property | Type | Default | Description |
|---|---|---|---|
| `agentic_rag_enabled` | bool | `False` | Enable agentic RAG graph |
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
| `max_file_size_mb` | int | 100 | Max file upload size |
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
print(session_config.top_k)  # may differ from rag_config.top_k
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
top_k = 5
hybrid_search_enabled = false
reranking_enabled = false
```
