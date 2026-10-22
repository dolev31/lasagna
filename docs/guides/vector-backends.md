# Choose a vector backend

Lasagna stores and searches chunk embeddings through one interface,
`VectorStoreAdapter`. Six backends implement it: Milvus, Qdrant, Pinecone,
Weaviate, PgVector and OpenSearch. You pick one with the `vector_backend`
setting. The rest of the pipeline does not change.

## Install the client

Milvus, with Milvus Lite, ships with the core package. The others need their
client library through an extra. Without it, `get_backend(name)` raises
`ValueError` and names the extra to install.

| Backend | Install | Read from the environment when no arguments are given |
|---|---|---|
| `milvus` | core | `MILVUS_URI`, `MILVUS_TOKEN`. Otherwise Milvus Lite at `<db dir>/rag_milvus.db`. |
| `qdrant` | `pip install "lasagna-rag[vector-backends]"` | `QDRANT_URL`, `QDRANT_API_KEY`. Otherwise local files in `<db dir>/qdrant_data`. |
| `pinecone` | `pip install "lasagna-rag[vector-backends]"` | `PINECONE_API_KEY` |
| `weaviate` | `pip install "lasagna-rag[vector-backends]"` | `WEAVIATE_URL`, `WEAVIATE_API_KEY`. Otherwise embedded Weaviate. |
| `pgvector` | `pip install "lasagna-rag[pgvector]"` | `PGVECTOR_DSN`, then `RAG_DATABASE_DSN`. Otherwise `postgresql://rag:rag@localhost:5432/rag`. |
| `opensearch` | `pip install "lasagna-rag[opensearch]"` | `OPENSEARCH_HOSTS` (default `http://localhost:9200`), `OPENSEARCH_USER`, `OPENSEARCH_PASSWORD`, `OPENSEARCH_USE_SSL`, `OPENSEARCH_VERIFY_CERTS`, `OPENSEARCH_TIMEOUT` |

`<db dir>` is `~/.lasagna/dbs` unless `LASAGNA_DB_DIR` or `LASAGNA_DATA_DIR` says
otherwise.

To run a backend in containers, see [Docker and Compose](../serving/docker.md#choose-a-vector-store).

## Select it

| Where | How |
|---|---|
| Environment | `LASAGNA_VECTOR_BACKEND=qdrant` (or `RAG_VECTOR_BACKEND`) |
| Settings file | `vector_backend = "qdrant"` in the `[rag]` section of `settings.lasagna.toml` |
| SDK | `KnowledgeConfig(vector_backend="qdrant", vector_backend_config={"url": "http://localhost:6333"})` |
| One session, over REST | `PATCH /api/v1/config` with `{"vector_backend": "qdrant"}` and the session's `X-Session-ID` header |
| Setup | `lasagna init` asks for one and writes a local configuration file |

`vector_backend_config` (or `LASAGNA_VECTOR_BACKEND_CONFIG` as a JSON object)
holds the constructor arguments. When it is empty, the backend reads the
environment variables in the table above. The process-wide backend is built once,
at startup, so changing either setting later needs a restart.

!!! note "Switching moves no data"
    Documents indexed in the previous backend stay there. Each document row
    records its `vector_backend_name` and `collection_name`, and deleting the
    document deletes its vectors from that backend and collection.

Every setting is in the [settings reference](../reference/settings.md).

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

A `db_path` together with a server address stops startup with a `ConfigError`
that names both.

## Collections: one per embedding model

The default collection name is `rag_documents_v2__<model>__<dim>`. The default
model, `BAAI/bge-m3` at 1024 dimensions, writes to
`rag_documents_v2__baai_bge_m3__1024`.

Opening an existing collection with another model, or at another dimension,
raises `CollectionIdentityError`, which names the command that switches back.
Nothing is dropped, except an empty collection at the wrong dimension, which is
rebuilt. [One collection per embedding model](../configuration.md#one-collection-per-embedding-model)
explains what this means when you change models.

## What each backend supports

| | Milvus Lite | Milvus server | Qdrant | Pinecone | Weaviate | PgVector | OpenSearch |
|---|---|---|---|---|---|---|---|
| Native hybrid search | no | yes | no | no | yes | yes | yes |
| Native BM25 | no | yes | no | no | yes | no | yes |
| Partitions or namespaces | yes | yes | no | yes | no | no | no |
| Dynamic schema | yes | yes | yes | yes | yes | no | yes |
| Max batch size | 1000 | 1000 | 1000 | 100 | 1000 | 5000 | 10000 |
| Max vector dimension | 32768 | 32768 | 65536 | 20000 | 65536 | 16000 | 16000 |

Without native hybrid search, the pipeline fuses dense results with an in-process
BM25 index when `enable_hybrid_search` is on.

Scores are normalized for most backends, and OpenSearch returns its own. Compare
scores within one backend, not across backends.

## Filters

`FilterExpression` is written once and translated for each backend.

```python
from lasagna.vector_backends import FilterExpression

FilterExpression.eq("session_id", "s1")          # also ne, gt, gte, lt, lte
FilterExpression.in_list("document_id", ["d1", "d2"])   # also not_in_list
FilterExpression.like("content", "%heart%")
FilterExpression.prefix("document_id", "doc_")
FilterExpression.json_eq("metadata", "source", "pubmed")

expr = (FilterExpression.eq("session_id", "s1") & FilterExpression.gte("page", 2)) | FilterExpression.eq("pinned", True)
negated = ~FilterExpression.eq("archived", True)
```

An operator a backend cannot express raises `UnsupportedFilterError` naming the
operator and the backend. It is never dropped.

## Add your own backend

Subclass `VectorStoreAdapter`, implement its abstract members, and register it:

```python
from lasagna.vector_backends import VectorStoreAdapter, get_backend, register_backend


class ChromaBackend(VectorStoreAdapter):
    ...  # backend_name, capabilities and the abstract methods


register_backend("chroma", ChromaBackend)
backend = get_backend("chroma")
```

The [vector backends README](https://github.com/dolev31/lasagna/blob/main/src/lasagna/vector_backends/README.md)
lists the members to implement, with a runnable round trip.

`tests/test_vector_backend_conformance.py` asserts one contract against every
backend it can reach: collections, insert and search, filters, delete, paging and
declared capabilities. Run it against yours to find out where it differs.

## API

`VectorStoreAdapter` and `register_backend` are documented in the
[Python API reference](../reference/python-api.md).
