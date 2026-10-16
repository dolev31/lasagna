# Vector backends

Lasagna stores and searches chunk embeddings through one interface,
`VectorStoreAdapter`, with six implementations: Milvus, Qdrant, Pinecone,
Weaviate, PgVector and OpenSearch. You pick one with the `vector_backend`
setting; the rest of the pipeline does not change.

## Choose a backend

Milvus (with Milvus Lite) ships with the core package. The others need their
client library, installed through an extra. Without it, `get_backend(name)`
raises `ValueError` and names the extra to install.

| Backend | Install | Read from the environment when no arguments are given | Compose fragment |
| --- | --- | --- | --- |
| `milvus` | core | `MILVUS_URI`, `MILVUS_TOKEN`; otherwise Milvus Lite at `<db dir>/rag_milvus.db` (see the note on Milvus servers below) | `milvus-lite.yml`, `milvus.yml` |
| `qdrant` | `pip install "lasagna-rag[vector-backends]"` | `QDRANT_URL`, `QDRANT_API_KEY`; otherwise local files in `<db dir>/qdrant_data` | `qdrant.yml` |
| `pinecone` | `pip install "lasagna-rag[vector-backends]"` | `PINECONE_API_KEY` | `pinecone.yml` |
| `weaviate` | `pip install "lasagna-rag[vector-backends]"` | `WEAVIATE_URL`, `WEAVIATE_API_KEY`; otherwise embedded Weaviate | `weaviate.yml` |
| `pgvector` | `pip install "lasagna-rag[pgvector]"` | `PGVECTOR_DSN`, then `RAG_DATABASE_DSN`; otherwise `postgresql://rag:rag@localhost:5432/rag` | `pgvector.yml` |
| `opensearch` | `pip install "lasagna-rag[opensearch]"` | `OPENSEARCH_HOSTS` (default `http://localhost:9200`), `OPENSEARCH_USER`, `OPENSEARCH_PASSWORD`, `OPENSEARCH_USE_SSL`, `OPENSEARCH_VERIFY_CERTS`, `OPENSEARCH_TIMEOUT` | `opensearch.yml` |

`<db dir>` is `~/.lasagna/dbs` unless `LASAGNA_DB_DIR` or `LASAGNA_DATA_DIR`
says otherwise (the `RAG_` names of both still work).

The compose fragments live in [`../deploy/backends/`](../deploy/backends/).
From `src/lasagna/deploy`, `make up VECTOR_BACKEND=qdrant` starts the stack with
that fragment; see [the deployment guide](../deploy/README.md#vector-backends).

## Select it

| Where | How |
| --- | --- |
| Environment | `LASAGNA_VECTOR_BACKEND=qdrant` (or `RAG_VECTOR_BACKEND`) |
| Settings file | `vector_backend = "qdrant"` in the `[rag]` section of `settings.lasagna.toml` |
| SDK | `KnowledgeConfig(vector_backend="qdrant", vector_backend_config={"url": "http://localhost:6333"})` |
| One session, over REST | `PATCH /api/v1/config` with body `{"vector_backend": "qdrant"}` and the session's `X-Session-ID` header |
| CLI | `lasagna init` asks for one and writes a local configuration file |

Valid names: `milvus`, `qdrant`, `pinecone`, `weaviate`, `pgvector`, `opensearch`.

`vector_backend_config` (`LASAGNA_VECTOR_BACKEND_CONFIG` as a JSON object) holds
the constructor arguments listed below. When it is empty, the backend reads the
environment variables in the table above. The process-wide backend is built
once, at startup: changing either setting afterwards needs a restart (the SDK
logs a warning if you try). A session switched over REST gets the named backend
with its environment configuration. Every setting is listed in
[the settings reference](../../../docs/reference/settings.md).

Switching backends moves no data: documents indexed in the previous backend
stay there. Each document row records `vector_backend_name` and
`collection_name`, and deleting the document deletes its vectors from that
backend (opened with its default configuration) and collection.

**Milvus servers.** The backend Lasagna builds at startup connects to a Milvus
server when one is configured: `"uri"` in `vector_backend_config` (what
`backends/milvus.yml` sets), else `MILVUS_URI`, with `"token"` or `MILVUS_TOKEN`.
Otherwise it opens Milvus Lite at the `"db_path"` in `vector_backend_config`, or
at `<db dir>/rag_milvus.db`. The startup log names the one it opened
(`Initializing vector backend: Milvus server at http://milvus:19530 (from
vector_backend_config)`). A `"db_path"` together with a server address stops
startup with a `ConfigError` that names both, and `MilvusBackend(db_path=...,
uri=...)` raises `ValueError`: before 0.1.0 the file won silently.

## Collections: one per embedding model

The default collection name is `rag_documents_v2__<model>__<dim>`, where
`<model>` is the model id lowercased with every run of other characters
replaced by `_` (an id longer than 32 characters is cut and given an 8-character
hash). The default model, `BAAI/bge-m3` at 1024 dimensions, writes to
`rag_documents_v2__baai_bge_m3__1024`. A `collection_name` other than
`rag_documents_v2` is used exactly as given. The functions are
`collection_name_for` and `resolve_collection_name` in `core/config.py`.

When the collection manager (`api/_collection_manager.py`) creates a collection,
it records the model and dimension in the `collection_meta` table of the
metadata database. Opening an existing collection on the default backend with
another model, or at another dimension, raises `CollectionIdentityError`, which
names the command that switches back. Nothing is dropped, except an empty
collection at the wrong dimension, which is rebuilt at the new one. See
[One collection per embedding model](../../../docs/configuration.md#one-collection-per-embedding-model)
for what this means when you change models.

## Constructor arguments

`get_backend(name, **kwargs)` passes `kwargs` to the backend's constructor.

```python
from lasagna.vector_backends import get_backend

get_backend("milvus", db_path="/data/milvus.db")                     # Milvus Lite
get_backend("milvus", uri="http://localhost:19530", token="...")     # Milvus server or Zilliz Cloud
get_backend("qdrant", path="/data/qdrant")                           # local files
get_backend("qdrant", url="https://...", api_key="...")              # server or Qdrant Cloud
get_backend("pinecone", api_key="...", cloud="aws", region="us-east-1")
get_backend("weaviate", embedded=True)
get_backend("weaviate", url="https://...", api_key="...")            # an api_key means Weaviate Cloud
get_backend("pgvector", dsn="postgresql://rag:rag@localhost:5432/rag", pool_min=2, pool_max=10)
get_backend("opensearch", hosts=["https://search.example.com:9200"],
            http_auth=("user", "pass"), use_ssl=True, language="german")
```

`QdrantBackend()` with no arguments is in-memory, but `get_backend("qdrant")`
with no arguments fills in the environment defaults above. OpenSearch's
`language` picks the BM25 analyzer for text fields (default `english`, also
`OPENSEARCH_LANGUAGE`); `engine` and `space_type` pick the k-NN method
(default `lucene` and `cosinesimil`). `OPENSEARCH_SHARDS` and
`OPENSEARCH_REPLICAS` (default 1 and 0) apply when an index is created.

## What each backend supports

From each backend's `capabilities` property:

| | Milvus Lite | Milvus server | Qdrant | Pinecone | Weaviate | PgVector | OpenSearch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Native hybrid search | no | yes | no | no | yes | yes (tsvector + RRF) | yes (BM25 + k-NN) |
| Native BM25 | no | yes | no | no | yes | no | yes |
| Partitions / namespaces | yes | yes | no | yes | no | no | no |
| Dynamic schema | yes | yes | yes | yes | yes | no | yes |
| Max batch size | 1000 | 1000 | 1000 | 100 | 1000 | 5000 | 10000 |
| Max vector dimension | 32768 | 32768 | 65536 | 20000 | 65536 | 16000 | 16000 |

Calling `hybrid_search()` on a backend without native hybrid search falls back
to dense search on Milvus Lite and Pinecone, and raises `NotImplementedError`
on Qdrant. The search pipeline does not call it there: with
`enable_hybrid_search` on, it fuses dense results with an in-process BM25 index
(`rank_bm25`) instead.

## The interface

`VectorStoreAdapter` (`base.py`) has two abstract properties (`backend_name`,
`capabilities`) and 14 abstract methods. A minimal round trip on Milvus Lite:

```python
from lasagna.core.schema import create_abstract_rag_schema
from lasagna.vector_backends import FilterExpression, get_backend
from lasagna.vector_backends.base import IndexConfig, IndexType, MetricType

backend = get_backend("milvus", db_path="/tmp/demo_milvus.db")
backend.connect()

backend.create_collection(
    "demo",
    create_abstract_rag_schema(4),  # chunk_id, document_id, session_id, content, embedding, metadata
    IndexConfig(index_type=IndexType.FLAT, metric_type=MetricType.COSINE),
)
backend.insert("demo", [
    {"chunk_id": "c1", "document_id": "d1", "session_id": "s1",
     "content": "zebrafish hearts regenerate", "embedding": [1, 0, 0, 0], "metadata": {}},
    {"chunk_id": "c2", "document_id": "d2", "session_id": "s2",
     "content": "mouse hearts scar", "embedding": [0.9, 0.1, 0, 0], "metadata": {}},
])
backend.flush("demo")

hits = backend.search(
    "demo",
    query_embedding=[1, 0, 0, 0],
    filter_expr=FilterExpression.eq("session_id", "s1"),
    limit=5,
    output_fields=["content", "document_id"],
)
for hit in hits:
    print(hit.id, hit.score, hit.document_id, hit.content)

backend.delete("demo", FilterExpression.eq("document_id", "d1"))
```

The same code runs on Qdrant with `get_backend("qdrant", path="/tmp/demo_qdrant")`.

Other methods: `disconnect`, `is_connected`, `has_collection`, `drop_collection`,
`describe_collection`, `get_collection_stats`, `create_index`, `list_indexes`,
`query` (filter only, with `limit` and `offset`). With defaults you can
override: `hybrid_search`, `iter_all` (every matching row, page by page),
`list_document_ids`, `session_filter`, `describe_fields`, `async_search`,
`async_insert`, the partition methods, `flush` and `compact`.

Each hit is a `VectorSearchResult` with `id`, `score` (higher is better),
`entity` (the returned fields), the shortcuts `document_id`, `content`,
`metadata`, `embedding`, and `distance` (`max(0, 1 - score)`).

**Scores.** Milvus, Qdrant, Pinecone, and the dense paths of Weaviate and
PgVector convert raw scores with `normalize_score()`: cosine similarity
`s` becomes `(s + 1) / 2`, cosine distance `d` becomes `1 - d/2`, L2 distance
becomes `1 / (1 + d)`, and inner product is kept as is (floored at 0, so it can
exceed 1). OpenSearch returns its own scores unchanged (BM25 scores are
unbounded). Native hybrid results carry the backend's fused score. Compare
scores within one backend, not across backends.

`capabilities` is a `BackendCapabilities` dataclass;
`backend.supports_feature("hybrid_search")` checks a flag by name and
`backend.get_effective_batch_size(n)` caps `n` at the backend's batch limit.
Field names are constants on `Fields` (`CHUNK_ID`, `DOCUMENT_ID`, `SESSION_ID`,
`CONTENT`, `EMBEDDING`, `METADATA`).

## Filters

`FilterExpression` (`filters.py`) is written once and translated per backend.

```python
from lasagna.vector_backends import FilterExpression

FilterExpression.eq("session_id", "s1")          # also ne, gt, gte, lt, lte
FilterExpression.in_list("document_id", ["d1", "d2"])   # also not_in_list
FilterExpression.like("content", "%heart%")
FilterExpression.prefix("document_id", "doc_")
FilterExpression.json_eq("metadata", "source", "pubmed")
FilterExpression.json_contains("metadata", "tags", "biology")

expr = (FilterExpression.eq("session_id", "s1") & FilterExpression.gte("page", 2)) | FilterExpression.eq("pinned", True)
negated = ~FilterExpression.eq("archived", True)
```

`FilterExpression.eq("session_id", "s1")` becomes:

| Backend | Native form |
| --- | --- |
| Milvus | `session_id == "s1"` |
| Qdrant | `{"key": "session_id", "match": {"value": "s1"}}` |
| Pinecone | `{"session_id": {"$eq": "s1"}}` |
| Weaviate | `Filter.by_property("session_id").equal("s1")` (weaviate-client v4) |
| PgVector | `session_id = %s` with parameters `["s1"]` |
| OpenSearch | `{"term": {"session_id": {"value": "s1"}}}` |

| Operator | Milvus | Qdrant | Pinecone | Weaviate | PgVector | OpenSearch |
| --- | --- | --- | --- | --- | --- | --- |
| EQ, NE, GT, GTE, LT, LTE | yes | yes | yes | yes | yes | yes |
| IN, NOT_IN | yes | yes | yes | yes | yes | yes |
| LIKE, PREFIX | yes | full-text match | no | yes | yes | yes |
| AND, OR, NOT | yes | yes | NOT only over EQ, IN, AND, OR | yes | yes | yes |
| JSON_EQ, JSON_CONTAINS | yes | yes | no | no | yes | yes |

An operator a backend cannot express raises `UnsupportedFilterError` naming the
operator and backend; it is never dropped. `in_list` and `not_in_list` accept
at most 1,000 values, and JSON keys must start with a letter or underscore and
contain only letters, digits, `_` and `-` (at most 256 characters); both are
checked when the expression is built. The Milvus translator rejects field names
that are not identifiers, strings over 65,535 characters, NaN and infinity, and
control characters other than newline, carriage return and tab, and escapes
quotes and backslashes. PgVector passes every value as a query parameter.
`tests/test_filter_injection.py` checks that a value cannot change the filter.

## Registry and pooling

`registry.py` holds one `BackendRegistry` per process. `get_backend(name, **kwargs)`
returns the same instance for the same name and arguments (keyed by a hash of
the arguments); `singleton=False` builds a new one. `get_available_backends()`
lists the backends whose client library is installed. With
`connection_pool_size` above 1 (or `get_backend(..., pool_size=n)`), the backend
is wrapped in `PooledVectorBackend` (`pool.py`), which spreads calls over `n`
connections. Instances are cached until `BackendRegistry().clear_instances()` or
a restart, so an abandoned backend keeps its connection.

## Add a backend

Subclass `VectorStoreAdapter`, implement the abstract members, and register it:

```python
from lasagna.vector_backends import BackendCapabilities, VectorStoreAdapter, get_backend, register_backend
from lasagna.vector_backends.base import CollectionInfo


class ChromaBackend(VectorStoreAdapter):
    @property
    def backend_name(self) -> str:
        return "chroma"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(supports_hybrid_search=False, max_batch_size=5000)

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def has_collection(self, collection_name) -> bool: ...
    def create_collection(self, collection_name, schema, index_config=None) -> None: ...
    def drop_collection(self, collection_name) -> None: ...
    def describe_collection(self, collection_name) -> CollectionInfo: ...
    def get_collection_stats(self, collection_name) -> dict: ...
    def create_index(self, collection_name, field_name, index_config) -> None: ...
    def list_indexes(self, collection_name) -> list: ...
    def insert(self, collection_name, data, partition_key=None) -> list: ...
    def delete(self, collection_name, filter_expr, partition_key=None) -> int: ...
    def query(self, collection_name, filter_expr=None, output_fields=None,
              limit=100, offset=0, partition_key=None) -> list: ...
    def search(self, collection_name, query_embedding, vector_field="embedding",
               filter_expr=None, limit=10, output_fields=None,
               search_params=None, partition_key=None) -> list: ...


register_backend("chroma", ChromaBackend)
backend = get_backend("chroma")
```

Translate `FilterExpression` trees in your backend (PgVector keeps its
translator in `pgvector_backend.py`; the others are in `filters.py`). Override
`list_document_ids` if your store can list distinct document ids natively: the
orphan detector calls it, and the default scans every row through `query()`.
Backends that need UUID ids derive them with `uuid5(LASAGNA_UUID_NAMESPACE, id)`
(Qdrant and Weaviate do). An installed package named `lasagna_vector_<name>`
with a `register_backend(registry)` function is registered when you call
`discover_backends()` from `lasagna.vector_backends.registry`.

## Tests

[`tests/test_vector_backend_conformance.py`](../../../tests/test_vector_backend_conformance.py)
asserts one contract (collections, insert and search, filters, delete, paging,
declared capabilities) against every backend it can reach, and skips the others
with the reason. It finds them through environment variables:
`LASAGNA_MILVUS_DB_PATH` or `LASAGNA_MILVUS_URI`, `QDRANT_PATH` or `QDRANT_URL`,
`PGVECTOR_DSN`, `WEAVIATE_URL`, `OPENSEARCH_HOSTS`. Pinecone is left out because
running it creates billable cloud resources.

```bash
LASAGNA_MILVUS_DB_PATH=/tmp/conf_milvus.db QDRANT_PATH=/tmp/conf_qdrant \
  pytest tests/test_vector_backend_conformance.py -q
```

The `Tests` workflow (`.github/workflows/tests.yml`) runs that command's
equivalent against Milvus Lite and Qdrant's local mode, both in-process, on
every push to `main` and every pull request. PgVector, Weaviate and OpenSearch
run in the suite only when you point it at a service. In the same workflow,
`tests/test_opensearch_backend.py` and `tests/test_pgvector_backend.py` test
those two backends against mocked clients.

## Files

| File | Contents |
| --- | --- |
| `base.py` | `VectorStoreAdapter`, `BackendCapabilities`, `VectorSearchResult`, schema and index dataclasses, `Fields` |
| `filters.py` | `FilterExpression` and the Milvus, Qdrant, Pinecone, Weaviate and OpenSearch translators |
| `registry.py` | `BackendRegistry`, `get_backend`, default configuration per backend, plugin discovery |
| `pool.py` | `PooledVectorBackend` |
| `_pymilvus.py` | Imports pymilvus without letting it read a `.env` file |
| `milvus_backend.py`, `qdrant_backend.py`, `pinecone_backend.py`, `weaviate_backend.py`, `pgvector_backend.py`, `opensearch_backend.py` | The six backends |
