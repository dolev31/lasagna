# Embedding Manager: Deep Dive

## Overview

`EmbeddingManager` handles vector generation for the RAG pipeline. It supports two backends (BGE-M3 via FlagEmbedding, sentence-transformers), optional GPU acceleration, and an LRU embedding cache. It is the `default` embedder component. The other five (`sentence_transformers`, `bge_m3`, `openai`, `cohere`, `voyage`) live in `embedding/embedders.py` and are described in the [infrastructure README](../README.md#embedder-components).

---

## Backend Selection

```
embedding_backend = "auto" (default)
├── Model name contains "bge-m3"?
│   ├── Yes + FlagEmbedding installed → "flag-embedding"
│   └── Yes + FlagEmbedding NOT installed → "sentence-transformers" (fallback)
└── No → "sentence-transformers"
```

### BGE-M3 (FlagEmbedding)

- Dense + sparse (lexical) + ColBERT embeddings
- 8,192 token context window
- 100+ languages
- Native query/document asymmetric encoding
- `encode_sparse()` returns `{token: weight}` dicts for hybrid search

### Sentence-Transformers

- Universal fallback for any HuggingFace model
- Query and document prefixes follow the model's card, resolved from the model name, a specific family before a general one (see [Model-Card Prefixes](#model-card-prefixes))
- `encode_sparse()` returns `None` (not supported)

### Model-Card Prefixes

| Model family (matched in the name) | Query prefix | Document prefix |
| ---------------------------------- | ------------ | --------------- |
| Qwen3-Embedding, gte-Qwen, E5-Mistral, E5 instruct | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:` | none |
| E5 | `query: ` | `passage: ` |
| BGE-M3 | none (it is trained without a query instruction) | none |
| BGE (Chinese, `-zh`) | the Chinese BGE retrieval instruction | none |
| BGE v1.5, mxbai-embed, Arctic-Embed v1 | `Represent this sentence for searching relevant passages: ` | none |
| Arctic-Embed v2 | `query: ` | none |
| Nomic | `search_query: ` | `search_document: ` |
| anything else (MiniLM, GTE, ...) | none | none |

BGE-M3 gets no query instruction on either backend: the table maps it to none, and FlagEmbedding applies no prefix.

### Late Chunking

With `enable_late_chunking`, `EmbeddingManager` embeds the whole document once, and each chunk's vector is the mean of the token embeddings inside its span (Günther et al., arXiv:2409.04701). The chunk boundaries do not change, so it works with any chunker. A document longer than the model's window is processed as overlapping windows: consecutive windows share `late_chunking_overlap` tokens (512, capped at half a window), and a shared token keeps its embedding from the earlier window, where it had context on its left. It needs a sentence-transformers model with a fast tokenizer and no Dense projection. When it cannot run (the FlagEmbedding backend, another model, a chunk that is not a slice of the document), it falls back to standard encoding outside evaluation and raises under evaluation. It warns once when the model was not trained with mean pooling.

---

## GPU Detection Flow

```
use_gpu = true?
├── No → CPU
└── Yes
    ├── torch.cuda.is_available()? → CUDA (fp16 enabled for FlagEmbedding)
    ├── torch.backends.mps.is_available()? → MPS (fp16 disabled: MPS limitation)
    └── Neither → CPU (warning logged)
```

Device selection happens at model load time (`_load_flag_embedding()` / `_load_sentence_transformer()`). Changing `use_gpu` requires a config-drift reload.

---

## EmbeddingCache

Thread-safe LRU cache for computed embeddings.

### Key Computation

```python
key = SHA256(f"{prefix}:{text}")[:16]   # 16 hex chars = 64-bit key space
```

- **Query prefix**: the model's query prefix (for BGE v1.5, `"Represent this sentence for searching relevant passages: "`)
- **Document prefix**: the model's document prefix (empty for most models)

Queries and documents for the same text produce different cache keys when their prefixes differ.

### Collision Analysis

SHA256 truncated to 16 hex chars (64-bit):

| Cache Size | Collision Probability |
| ---------- | --------------------- |
| 10,000 | ~2.7 x 10^-12 (negligible) |
| 100,000 | ~2.7 x 10^-10 (negligible) |
| 1,000,000 | ~2.7 x 10^-8 (safe) |
| 10,000,000 | ~2.7 x 10^-6 (safe) |

At the default `max_size=10,000`, collisions are astronomically unlikely.

### Memory Footprint

Each entry: 1 `np.ndarray` of shape `(embedding_dim,)` with `float32` dtype.

| Embedding Dim | Max Size | Memory |
| ------------- | -------- | ------ |
| 1024 (BGE-M3, BGE-large) | 10,000 | ~40 MB |
| 384 (MiniLM) | 10,000 | ~15 MB |
| 768 (all-mpnet) | 10,000 | ~30 MB |

Plus `OrderedDict` overhead (~100 bytes/entry) = ~1 MB additional.

### Thread Safety

All operations use `threading.RLock` (re-entrant). `hit_rate` and `stats()` also acquire the lock for consistent reads.

---

## Batch Encoding Pipeline

```
encode_documents_batch(texts)
├── For each text:
│   ├── Cache hit? → results[i] = cached
│   └── Cache miss → add to texts_to_compute
├── Log cache hits (if any)
├── Batch encode uncached texts:
│   ├── FlagEmbedding: model.encode(texts, return_dense=True, batch_size=N)
│   │   └── Normalize if config.normalize_embeddings
│   └── sentence-transformers: model.encode(texts, batch_size=N, show_progress_bar=...)
├── Fill results array + update cache
└── Return List[np.ndarray]
```

**Concurrency note**: `encode_documents_batch()` is NOT internally locked. Concurrent calls with overlapping texts may compute the same embedding twice (cache miss for both), but the cache will eventually contain the correct value. This is safe but slightly wasteful.

---

## Adapter Pattern

`EmbeddingManagerAdapter` wraps `EmbeddingManager` to satisfy the `BaseEmbedder` ABC from `core/base_components.py`:

```python
class EmbeddingManagerAdapter(BaseEmbedder):
    def __init__(self, config=None, **kwargs):
        self._inner = get_embedding_manager(config)    # Uses singleton

    def encode(self, texts: List[str]) -> np.ndarray:  # ABC method
        results = self._inner.encode_documents_batch(texts)
        # A failed entry raises under evaluation (a zero vector is a document
        # no query can find); outside evaluation it becomes a zero vector.
        return np.array(results)

    def encode_query(self, query: str) -> np.ndarray:   # ABC method
        return self._inner.encode_query(query)

    @property
    def dimension(self) -> int:                          # ABC property
        return self._inner.dimension
```

Registered in `core/registry.py` as the default `"embedder"` component.

---

## Config-Drift Reload

The singleton detects config changes via a 10-field signature tuple (`_config_signature` in `embedding/manager.py`):

```python
signature = (
    config.embedding_model,
    config.embedding_model_fallback,
    config.embedding_dim,
    config.batch_size,
    config.normalize_embeddings,
    config.use_gpu,
    config.enable_embedding_cache,
    config.embedding_cache_size,
    config.embedding_backend,
    getattr(config, "embedding_truncate_dim", None),
)
```

On mismatch, a new `EmbeddingManager` is created under `_embedding_lock` (double-checked). The old manager is NOT explicitly destroyed to avoid breaking in-flight requests. Best-effort `torch.cuda.empty_cache()` is called after the swap.

**Thread safety**: `get_embedding_manager()` captures a local snapshot of the global reference before any checks. All return paths use either the local snapshot or a value read under `_embedding_lock`, never the bare global, so a concurrent `reset_embedding_manager()` cannot cause the function to return `None`.

**GPU memory implication**: Until all references to the old manager are dropped, both old and new models coexist in GPU memory. For BGE-M3 (~1.5 GB), this means a brief spike to ~3 GB during reload. In practice, config drift is rare (typically only during development or profile switching).
