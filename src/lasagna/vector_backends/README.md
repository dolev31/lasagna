# Vector Database Backend Abstraction Layer

A production-grade abstraction layer enabling RAG knowledge system to operate seamlessly across **any** vector database or search engine. Switch between Milvus, Qdrant, Pinecone, Weaviate, PgVector, and OpenSearch — per session, at runtime — without changing a single line of application code.

## Architecture

```
                    RAGBackend / MCP Server
                           |
                   _resolve_backend(session_id)
                           |
              +------------+------------+
              |                         |
         Fast Path                 Slow Path
    (no session / default)    (session has override)
              |                         |
     self._vector_backend       BackendRegistry
                               get_backend(name, **config)
                                        |
              +----------+---------+---------+---------+-----------+
              |          |         |         |         |           |
           Milvus    Qdrant   Pinecone  Weaviate   PgVector   OpenSearch
```

**Design Principles:**
1. **Minimal Interface** — Only methods the RAG system actually uses
2. **Backend-Agnostic Filters** — Write once, translates to all backends
3. **Capability Detection** — Runtime feature checks for graceful degradation
4. **Score Normalization** — Consistent `[0, 1]` similarity across all backends
5. **Zero Breaking Changes** — Existing code works without modification
6. **Thread-Safe** — Concurrent session resolution with bounded resource tracking

## Package Structure

| File | Purpose |
|------|---------|
| `base.py` | `VectorStoreAdapter` ABC, data classes, `Fields` constants |
| `filters.py` | `FilterExpression` AST with 6 backend translators |
| `registry.py` | `BackendRegistry` singleton factory with config-hash caching |
| `pool.py` | `PooledVectorBackend` — thread-safe connection pool wrapper |
| `milvus_backend.py` | Milvus Lite / Standalone / Distributed |
| `qdrant_backend.py` | Qdrant in-memory / local / cloud |
| `pinecone_backend.py` | Pinecone Serverless |
| `weaviate_backend.py` | Weaviate Embedded / Local / Cloud (v3 + v4) |
| `pgvector_backend.py` | PostgreSQL + pgvector with connection pooling and hybrid RRF |
| `opensearch_backend.py` | OpenSearch search engine with native Lucene BM25 + k-NN hybrid |

## Supported Backends

Milvus Lite ships with the core package. The other backends need their client library, installed through an extra:

| Backend | Install |
|---------|---------|
| Qdrant, Pinecone, Weaviate | `pip install "lasagna-rag[vector-backends]"` |
| PgVector | `pip install "lasagna-rag[pgvector]"` |
| OpenSearch | `pip install "lasagna-rag[opensearch]"` |

Without it, `get_backend()` raises `ValueError` and names the extra to install.

### Milvus (Default)

The default backend with the broadest feature set. Supports Lite mode (embedded, zero-config) and Standalone/Distributed (full hybrid search with BM25).

```python
from lasagna.vector_backends import get_backend

# Milvus Lite (embedded, local file)
backend = get_backend("milvus", db_path="/data/milvus.db")

# Milvus Standalone (remote server)
backend = get_backend("milvus", uri="http://localhost:19530")

# Milvus Cloud (with auth)
backend = get_backend("milvus", uri="https://...", token="your_token")
```

| Capability | Lite | Standalone |
|------------|------|------------|
| Hybrid Search (Dense + BM25) | - | Yes |
| Native Sparse Vectors | - | Yes |
| GPU Acceleration | - | Yes |
| JSON Filters | Yes | Yes |
| Dynamic Schema | Yes | Yes |
| Max Batch Size | 1000 | 1000 |
| Max Vector Dim | 32768 | 32768 |

### Qdrant

High-performance vector database with excellent filtering. Zero-config in-memory mode for development.

```python
# In-memory (zero config, ideal for tests)
backend = get_backend("qdrant")

# Local persistent storage
backend = get_backend("qdrant", path="/data/qdrant")

# Remote server
backend = get_backend("qdrant", url="http://localhost:6333")

# Qdrant Cloud
backend = get_backend("qdrant", url="https://...", api_key="your_key")
```

| Capability | Value |
|------------|-------|
| Hybrid Search | Not yet wired |
| Sparse Vectors | Yes |
| JSON Filters | Yes |
| Dynamic Schema | Yes |
| Max Batch Size | 1000 |
| Max Vector Dim | 65536 |

### Pinecone

Fully managed serverless vector database. No infrastructure to maintain.

```python
# Serverless (default: AWS us-east-1)
backend = get_backend("pinecone", api_key="your_key")

# Custom region
backend = get_backend("pinecone", api_key="your_key", cloud="gcp", region="us-central1")
```

| Capability | Value |
|------------|-------|
| Hybrid Search | Not yet wired |
| Sparse Vectors | Yes |
| JSON Filters | Yes |
| Dynamic Schema | Yes |
| Max Batch Size | 100 |
| Max Vector Dim | 20000 |
| Max Metadata | 40KB per vector |

### Weaviate

Native hybrid search (BM25 + vector) built in. Supports embedded mode for local development.

```python
# Embedded (zero config)
backend = get_backend("weaviate", embedded=True)

# Local server
backend = get_backend("weaviate", url="http://localhost:8080")

# Weaviate Cloud
backend = get_backend("weaviate", url="https://...", api_key="your_key")
```

| Capability | Value |
|------------|-------|
| Hybrid Search | Yes (native BM25 + vector) |
| Native BM25 | Yes |
| JSON Filters | Yes |
| Dynamic Schema | Yes |
| Max Batch Size | 1000 |
| Max Vector Dim | 65536 |

### PgVector (PostgreSQL)

Uses a single PostgreSQL instance with the pgvector extension for both metadata and vector storage. Native hybrid search via tsvector/tsquery + Reciprocal Rank Fusion (RRF) in a single SQL query.

```python
# Local PostgreSQL
backend = get_backend("pgvector", dsn="postgresql://rag:rag@localhost:5432/rag")

# With connection pool tuning
backend = get_backend("pgvector", dsn="...", pool_min=2, pool_max=10)
```

| Capability | Value |
|------------|-------|
| Hybrid Search | Yes (tsvector + RRF) |
| Native BM25 | No (uses tsvector/tsquery) |
| JSON Filters | Yes (JSONB operators) |
| Dynamic Schema | No |
| Max Batch Size | 5000 |
| Max Vector Dim | 16000 |
| Connection Pooling | Built-in (psycopg_pool) |
| HNSW + IVF Indexes | Yes |

### OpenSearch

Full search engine with native Lucene BM25 + k-NN vector hybrid search. Unlike vector DBs, OpenSearch provides production-grade text search (30+ language analyzers, fuzzy matching, phrase search) fused server-side with dense vector similarity in a single query.

```python
# Local dev
backend = get_backend("opensearch", hosts=["http://localhost:9200"])

# With authentication
backend = get_backend("opensearch", hosts=["https://search.example.com:9200"],
                      http_auth=("user", "pass"), use_ssl=True)

# With language-specific BM25 (German stemming, stop-words, etc.)
backend = get_backend("opensearch", language="german")
```

| Capability | Value |
|------------|-------|
| Hybrid Search | Yes (native Lucene BM25 + k-NN, server-side fusion) |
| Native BM25 | Yes (Lucene — 30+ language analyzers, fuzzy, phrase) |
| Fuzzy Matching | Yes (Levenshtein distance, catches typos) |
| Phrase Boosting | Yes (exact phrases rank higher) |
| Highlighting | Yes (returns matching text spans for citations) |
| Multi-Search | Yes (_msearch — batch queries in 1 round-trip) |
| JSON Filters | Yes |
| Dynamic Schema | Yes |
| Max Batch Size | 10000 |
| Max Vector Dim | 16000 |
| k-NN Engines | FAISS, Lucene, NMSLIB |

## Abstract Interface

All backends implement `VectorStoreAdapter` — 14 abstract methods covering the complete lifecycle:

### Connection

```python
backend.connect()           # Establish connection (thread-safe, double-checked locking)
backend.disconnect()        # Close connection
backend.is_connected()      # Check status
```

### Collection Management

```python
backend.has_collection("my_collection")     # Check existence
backend.create_collection(name, schema, index_config)
backend.drop_collection(name)
backend.describe_collection(name)           # -> CollectionInfo
backend.get_collection_stats(name)          # -> {"row_count": ...}
```

### Data Operations

```python
# Insert (single or batch)
ids = backend.insert("my_collection", documents)

# Delete by filter
count = backend.delete("my_collection", FilterExpression.eq("doc_id", "abc"))

# Query by filter (non-vector)
results = backend.query("my_collection", filter_expr, output_fields, limit)
```

### Vector Search

```python
# Dense search
results = backend.search(
    collection_name="my_collection",
    query_embedding=[0.1, 0.2, ...],
    filter_expr=FilterExpression.eq("session_id", "user123"),
    limit=10,
    output_fields=["content", "document_id", "metadata"],
)

# Hybrid search (dense + BM25, if supported)
if backend.capabilities.supports_hybrid_search:
    results = backend.hybrid_search(
        collection_name="my_collection",
        query_text="search terms",
        query_embedding=[0.1, 0.2, ...],
        alpha=0.7,  # 0.7 dense / 0.3 sparse
    )
```

### Search Results

Every backend returns normalized `VectorSearchResult` objects:

```python
for r in results:
    r.id            # Unique ID (int or str)
    r.score         # Normalized similarity [0, 1], higher = better
    r.entity        # Dict of all fields
    r.document_id   # Shortcut for entity["document_id"]
    r.content       # Shortcut for entity["content"]
    r.metadata      # Shortcut for entity["metadata"]
    r.distance      # 1 - score (lower = better)
```

## Filter Expression System

Write filters once — they translate to native syntax for each backend automatically.

### Creating Filters

```python
from lasagna.vector_backends import FilterExpression

# Comparison
FilterExpression.eq("field", "value")       # ==
FilterExpression.ne("field", "value")       # !=
FilterExpression.gt("field", 10)            # >
FilterExpression.gte("field", 10)           # >=
FilterExpression.lt("field", 100)           # <
FilterExpression.lte("field", 100)          # <=

# Collections
FilterExpression.in_list("field", [1, 2, 3])
FilterExpression.not_in_list("field", [4, 5])

# String matching
FilterExpression.like("field", "%pattern%")
FilterExpression.prefix("field", "pre")

# JSON nested fields
FilterExpression.json_eq("metadata", "source", "doc1")
FilterExpression.json_contains("metadata", "tags", "python")

# Composition (operator overloads)
expr = (
    FilterExpression.eq("session_id", "user123")
    & FilterExpression.gte("score", 0.5)
    | FilterExpression.eq("status", "active")
)
negated = ~FilterExpression.eq("archived", True)
```

### Translation Examples

```python
expr = FilterExpression.eq("session_id", "user123")
```

| Backend | Native Format |
|---------|---------------|
| Milvus | `session_id == "user123"` |
| Qdrant | `{"key": "session_id", "match": {"value": "user123"}}` |
| Pinecone | `{"session_id": {"$eq": "user123"}}` |
| Weaviate | `{"path": ["session_id"], "operator": "Equal", "valueText": "user123"}` |
| PgVector | `WHERE session_id = %s` (parameterized) |
| OpenSearch | `{"term": {"session_id": {"value": "user123"}}}` |

### Operator Support Matrix

| Operator | Milvus | Qdrant | Pinecone | Weaviate | PgVector | OpenSearch |
|----------|--------|--------|----------|----------|----------|------------|
| EQ / NE | Yes | Yes | Yes | Yes | Yes | Yes |
| GT / GTE / LT / LTE | Yes | Yes | Yes | Yes | Yes | Yes |
| IN / NOT_IN | Yes | Yes | Yes | Yes | Yes | Yes |
| LIKE | Yes | Yes | - | Yes | Yes | Yes |
| PREFIX | Yes | Yes | - | Yes | Yes | Yes |
| AND / OR / NOT | Yes | Yes | Partial | Yes | Yes | Yes |
| JSON_EQ | Yes | Yes | - | - | Yes | Yes |
| JSON_CONTAINS | Yes | Yes | - | - | Yes | Yes |

Unsupported operators raise `UnsupportedFilterError` with the operator and backend name — never silently dropped.

### Security

Filter expressions enforce input validation at construction time:
- String values capped at 65,535 characters
- JSON keys validated (alphanumeric + underscore/hyphen, max 256 chars)
- Control characters escaped (null bytes, tabs, newlines)
- NaN/Inf rejected for numeric values
- List values limited to 1,000 items
- PgVector uses parameterized queries (`%s`) — no SQL injection possible

## Backend Registry

The `BackendRegistry` is an **in-process Python singleton** — not a server, no port, no network. It's a dictionary that caches backend instances by `name + MD5(config)` so the same configuration always returns the same connected client:

```python
from lasagna.vector_backends import (
    get_backend,
    get_available_backends,
    register_backend,
)

# List available backends (only those whose client library is installed)
print(get_available_backends())  # ['milvus', 'qdrant', 'pinecone', 'weaviate', 'pgvector', 'opensearch']

# Get or create (cached by name + config hash)
# "qdrant" needs pip install "lasagna-rag[vector-backends]"
backend = get_backend("qdrant", url="http://localhost:6333")

# Same config returns same instance (singleton)
backend2 = get_backend("qdrant", url="http://localhost:6333")
assert backend is backend2

# Different config returns different instance
backend3 = get_backend("qdrant", url="http://other-host:6333")
assert backend is not backend3

# Register a custom backend
register_backend("my_custom_db", MyCustomBackend)
```

### Adding a New Backend

Implement `VectorStoreAdapter` and register it:

```python
from lasagna.vector_backends.base import (
    VectorStoreAdapter,
    BackendCapabilities,
    VectorSearchResult,
    CollectionSchema,
    CollectionInfo,
    IndexConfig,
)
from lasagna.vector_backends.filters import FilterExpression
from lasagna.vector_backends import get_backend, register_backend

class ChromaBackend(VectorStoreAdapter):

    @property
    def backend_name(self) -> str:
        return "chroma"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_hybrid_search=False,
            supports_json_filters=True,
            max_batch_size=5000,
        )

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def has_collection(self, name: str) -> bool: ...
    def create_collection(self, name, schema, index_config=None) -> None: ...
    def drop_collection(self, name: str) -> None: ...
    def describe_collection(self, name: str) -> CollectionInfo: ...
    def get_collection_stats(self, name: str) -> dict: ...
    def create_index(self, name, field, config) -> None: ...
    def list_indexes(self, name: str) -> list: ...
    def insert(self, name, data) -> list: ...
    def delete(self, name, filter_expr) -> int: ...
    def query(self, name, filter_expr=None, output_fields=None, limit=100, offset=0) -> list: ...
    def search(self, name, query_embedding, **kwargs) -> list: ...

# Register and use
register_backend("chroma", ChromaBackend)
backend = get_backend("chroma")
```

Add a filter translator in `filters.py` if the backend needs custom filter syntax:

```python
class _ChromaTranslator:
    @classmethod
    def translate(cls, expr: FilterExpression) -> dict:
        # Convert FilterExpression to Chroma's native format
        ...

# Then add to FilterExpression:
def to_chroma(self) -> dict:
    return _ChromaTranslator.translate(self)
```

## Per-Session Backend Switching

The system supports switching backends per user session at runtime — configured from the UI or SDK:

```
Session A (Milvus)  ──┐
                      │    RAGBackend._resolve_backend(session_id)
Session B (Qdrant)  ──┤              │
                      │    BackendRegistry (cached instances)
Session C (PgVector) ─┘              │
                              ┌──────┴──────┐
                          Fast Path     Slow Path
                       (same as default)  (different)
                              │              │
                     self._vector_backend  get_backend(name, **config)
                                             │
                                    connect + ensure collection
                                    (thread-safe, LRU bounded)
```

**Key behaviors:**
- **Zero overhead** for sessions using the default backend (fast path)
- **Lazy initialization** — collections created on first access, not on config change
- **Thread-safe** — `_collection_init_lock` prevents duplicate initialization
- **Bounded memory** — LRU eviction at 500 initialized collections (oldest 10% evicted)
- **Correct deletion** — SQLite tracks `vector_backend_name` per document, so deletions target the right backend even after switching

**Limitations:**

- Switching `vector_backend` at runtime does NOT migrate existing data. Documents indexed in the previous backend remain there.
- The registry singleton caches instances — a full restart is needed to reclaim connections from an abandoned backend.

## Capability Detection

Check what a backend supports before using advanced features:

```python
caps = backend.capabilities

# Feature flags
caps.supports_hybrid_search      # Dense + sparse search
caps.supports_native_bm25        # Built-in BM25
caps.supports_sparse_vectors     # Sparse vector storage
caps.supports_gpu                # GPU acceleration
caps.supports_json_filters       # JSON field filtering
caps.supports_dynamic_schema     # Schema-less fields

# Limits
caps.max_batch_size              # Max docs per insert
caps.max_varchar_length          # Max string field length
caps.max_vector_dimension        # Max embedding dimension

# Convenience
backend.supports_feature("hybrid_search")   # String-based check
backend.get_effective_batch_size(5000)       # Clamps to backend limit
```

## Score Normalization

All backends normalize scores to a consistent `[0, 1]` similarity range via `normalize_score()`:

| Metric | Raw Range | Normalized |
|--------|-----------|------------|
| Cosine similarity | [-1, 1] | [0, 1] |
| Cosine distance | [0, 2] | [0, 1] |
| L2 distance | [0, inf) | [0, 1] via `1/(1+d)` |
| Inner product | (-inf, inf) | [0, 1] clamped |

This ensures that `result.score = 0.85` means the same thing regardless of which backend produced it.

## Field Name Constants

Use the `Fields` class instead of magic strings:

```python
from lasagna.vector_backends import Fields

Fields.CHUNK_ID      # "chunk_id"
Fields.DOCUMENT_ID   # "document_id"
Fields.SESSION_ID    # "session_id"
Fields.CONTENT       # "content"
Fields.EMBEDDING     # "embedding"
Fields.METADATA      # "metadata"
```

## Configuration

Vector backend is configurable via multiple surfaces:

| Surface | How |
|---------|-----|
| Environment variable | `RAG_VECTOR_BACKEND=pgvector` |
| TOML config | `vector_backend = "pgvector"` in `settings.lasagna.toml` |
| SDK | `KnowledgeConfig(vector_backend="pgvector")` |
| REST API | `PATCH /api/v1/config {"vector_backend": "pgvector"}` |
| CLI | `lasagna init` interactive wizard |
| Session override | `session_config.set_vector_backend("qdrant")` |

Valid values: `milvus`, `qdrant`, `pinecone`, `weaviate`, `pgvector`, `opensearch`

## Testing

The test suite validates every backend against the same contract:

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_vector_backends.py` | 43 | Filter translation, capabilities, base class |
| `test_backend_resolution.py` | 23 | Resolution, concurrency, LRU eviction |
| `test_backend_contract.py` | 101 | Full CRUD + search across all backends |
| `test_backend_switching.py` | 9 | Data isolation, cross-backend consistency |
| `test_reranking_quality.py` | 16 | Reranking pipeline integration |
| **Total** | **192** | |

```bash
# Run all vector backend tests
pytest tests/unit/test_vector_backends.py tests/unit/test_backend_resolution.py -v

# Run integration tests (requires Qdrant + Milvus Lite)
pytest tests/integration/knowledge/ -v

# Run everything
pytest tests/unit/test_backend_resolution.py tests/unit/test_vector_backends.py \
       tests/unit/test_reranking_quality.py tests/integration/knowledge/ -q
```

## Design Overview

### Layered Architecture

The vector backends package has four layers:

1. **Contract Layer** (`base.py`): The `VectorStoreAdapter` ABC defines 14 abstract methods plus optional methods with defaults (hybrid search, async operations, partitions). `VectorSearchResult` enforces normalized `[0, 1]` similarity scores. `BackendCapabilities` enables runtime feature detection.

2. **Filter Layer** (`filters.py`): A portable AST (`FilterExpression`) with factory methods (`eq`, `in_list`, `json_eq`) and operator overloads (`&`, `|`, `~`). Six backend translators (Milvus, Qdrant, Pinecone, Weaviate, PgVector, OpenSearch) convert the AST to native syntax. Input validation prevents injection at construction time.

3. **Implementation Layer** (6 backends): Each backend implements the ABC, translates filters via the filter layer, normalizes scores via `normalize_score()`, and handles connection lifecycle with thread-safe double-checked locking.

4. **Infrastructure Layer** (`registry.py`, `pool.py`): The singleton registry manages backend class registration, instance caching by config hash, and plugin discovery. The connection pool wraps any backend for parallel search across multiple connections using a semaphore-bounded deque.

### Key Design Decisions

- **Immutable filter AST**: `FilterExpression` objects are composed at construction time; translation is deferred to the point of use, keeping filters backend-agnostic until the last moment.
- **Score normalization as a contract**: The base class provides `normalize_score()` and all backends must produce `[0, 1]` similarity — callers never need to know which metric type the backend uses internally.
- **Singleton registry with config-hash keying**: Same `(name, config)` tuple always returns the same instance, preventing connection proliferation while allowing different configs to coexist.
- **Capability-based degradation**: Callers check `caps.supports_hybrid_search` before calling `hybrid_search()`; the base class provides a dense-only fallback for backends that don't support it.
- **Shared UUID namespace**: Backends that require UUID-format IDs (Qdrant, Weaviate) use a shared `LASAGNA_UUID_NAMESPACE` for deterministic conversion, ensuring the same string ID maps to the same UUID across backends.
