# Embedding Manager -- Deep Dive

## Overview

`EmbeddingManager` handles vector generation for the RAG pipeline. It supports two backends (BGE-M3 via FlagEmbedding, sentence-transformers), optional GPU acceleration, and an LRU embedding cache.

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
- Explicit query prefix applied: `"Represent this sentence for searching relevant passages: "`
- `encode_sparse()` returns `None` (not supported)

---

## GPU Detection Flow

```
use_gpu = true?
├── No → CPU
└── Yes
    ├── torch.cuda.is_available()? → CUDA (fp16 enabled for FlagEmbedding)
    ├── torch.backends.mps.is_available()? → MPS (fp16 disabled — MPS limitation)
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

- **Query prefix**: `"Represent this sentence for searching relevant passages: "`
- **Document prefix**: `""` (empty)

Queries and documents for the same text produce different cache keys due to different prefixes.

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
        # Guard: replace None entries with zero vectors
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

The singleton detects config changes via a 9-field signature tuple:

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
)
```

On mismatch, a new `EmbeddingManager` is created under `_embedding_lock` (double-checked). The old manager is NOT explicitly destroyed to avoid breaking in-flight requests. Best-effort `torch.cuda.empty_cache()` is called after the swap.

**Thread safety**: `get_embedding_manager()` captures a local snapshot of the global reference before any checks. All return paths use either the local snapshot or a value read under `_embedding_lock` — never the bare global — so a concurrent `reset_embedding_manager()` cannot cause the function to return `None`.

**GPU memory implication**: Until all references to the old manager are dropped, both old and new models coexist in GPU memory. For BGE-M3 (~1.5 GB), this means a brief spike to ~3 GB during reload. In practice, config drift is rare (typically only during development or profile switching).
