# Backends

## Vector Backends (BYOVDB)

### Default Backend (Milvus)

```python
from lasagna.sdk.backends import get_backend, MilvusBackend

# Use factory
backend = get_backend("milvus")

# Or direct instantiation
backend = MilvusBackend(db_path="./vectors.db")
```

### Available Backends

```python
from lasagna.sdk.backends import list_backends

print(list_backends())
# ['milvus', ...]: the built-in backends whose client library is installed
```

Milvus Lite is always available. The others need an extra:

- Pinecone, Qdrant and Weaviate: `pip install "lasagna-rag[vector-backends]"`
- pgvector: `pip install "lasagna-rag[pgvector]"`
- OpenSearch: `pip install "lasagna-rag[opensearch]"`

### Using Pinecone

```python
from lasagna.sdk import KnowledgeClient, KnowledgeConfig, VectorBackend

config = KnowledgeConfig(
    vector_backend=VectorBackend.PINECONE,
    vector_backend_config={
        "api_key": "your-api-key",
        "environment": "us-east-1",
        "index_name": "knowledge-index",
    },
)
client = KnowledgeClient(config=config)
```

### Using Qdrant

```python
config = KnowledgeConfig(
    vector_backend=VectorBackend.QDRANT,
    vector_backend_config={
        # QdrantBackend arguments: url, api_key, host, port, path, ...
        "url": "http://localhost:6333",
    },
)
```

### Custom Backend

```python
from lasagna.sdk import KnowledgeConfig
from lasagna.sdk.backends import (
    BackendCapabilities,
    VectorBackendBase,
    register_backend,
)

class MyCustomBackend(VectorBackendBase):
    """Custom vector backend implementation.

    Every member below is abstract in the base class, so the class cannot be
    created until all of them exist. The methods are synchronous. Replace
    each ``...`` with calls to your database.
    """

    @property
    def backend_name(self) -> str:
        return "my_custom"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    # Connection
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...

    # Collections and indexes
    def has_collection(self, collection_name) -> bool: ...
    def create_collection(self, collection_name, schema, index_config=None) -> None: ...
    def drop_collection(self, collection_name) -> None: ...
    def describe_collection(self, collection_name): ...  # returns CollectionInfo
    def get_collection_stats(self, collection_name) -> dict: ...
    def create_index(self, collection_name, field_name, index_config) -> None: ...
    def list_indexes(self, collection_name) -> list: ...

    # Data
    def insert(self, collection_name, data, partition_key=None) -> list: ...
    def delete(self, collection_name, filter_expr, partition_key=None) -> int: ...
    def query(self, collection_name, filter_expr=None, output_fields=None,
              limit=100, offset=0, partition_key=None) -> list: ...
    def search(self, collection_name, query_embedding, vector_field="embedding",
               filter_expr=None, limit=10, output_fields=None,
               search_params=None, partition_key=None) -> list: ...  # VectorSearchResult items

# Register
register_backend("custom", MyCustomBackend)

# Use
config = KnowledgeConfig(vector_backend="custom")
```

---

## Storage Backends

Pluggable file storage for uploaded documents. Default is local filesystem (zero config).

### Listing Storage Backends

```python
from lasagna.sdk import get_available_storage_backends

print(get_available_storage_backends())
# ['local', 's3', 'fsspec']  (s3/fsspec only if boto3/fsspec installed)
```

### Configuration via SDK

```python
from lasagna.sdk import KnowledgeClient, KnowledgeConfig

# Local (default, zero config)
client = KnowledgeClient()

# S3-compatible (AWS S3, MinIO, R2, Backblaze, Wasabi, etc.)
config = KnowledgeConfig(
    storage_backend="s3",
    storage_backend_config={
        "bucket": "my-documents",
        "prefix": "rag",
        "endpoint_url": "http://minio:9000",  # omit for AWS S3
        "region_name": "us-east-1",
    },
)
client = KnowledgeClient(config=config)

# Google Cloud Storage via fsspec
config = KnowledgeConfig(
    storage_backend="fsspec",
    storage_backend_config={
        "protocol": "gcs",
        "base_path": "my-bucket/rag-documents",
        "storage_options": {"project": "my-gcp-project"},
    },
)
client = KnowledgeClient(config=config)
```

### Production Preset

```python
from lasagna.sdk import config_for_production

config = config_for_production(
    storage_backend="s3",
    storage_backend_config={
        "bucket": "prod-docs",
        "endpoint_url": "http://minio:9000",
    },
)
```

### Custom Storage Backend

```python
from lasagna.sdk import (
    StorageBackend,
    StorageCapabilities,
    StorageFileInfo,
    register_storage_backend,
)

class MyCloudStorage(StorageBackend):
    @property
    def backend_name(self) -> str:
        return "my_cloud"

    @property
    def capabilities(self) -> StorageCapabilities:
        return StorageCapabilities(supports_presigned_urls=True)

    def store(self, source, document_id, filename, **kw) -> StorageFileInfo:
        # ... upload to your service ...
        return StorageFileInfo(uri=f"my_cloud://{document_id}/{filename}", size=1024)

    def retrieve(self, uri):
        # ... download from your service ...
        ...

    def delete(self, uri) -> bool:
        ...

    def exists(self, uri) -> bool:
        ...

    def get_size(self, uri) -> int:
        ...

# Register and use
register_storage_backend("my_cloud", MyCloudStorage)
config = KnowledgeConfig(storage_backend="my_cloud")
```

### Storage Environment Variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `RAG_STORAGE_BACKEND` | `local` | `local`, `s3`, or `fsspec` |
| `RAG_STORAGE_S3_BUCKET` | *(empty)* | S3 bucket name |
| `RAG_STORAGE_S3_ENDPOINT_URL` | *(empty)* | S3-compatible endpoint |
| `RAG_STORAGE_S3_REGION` | `us-east-1` | AWS region |
| `RAG_STORAGE_FSSPEC_PROTOCOL` | *(empty)* | fsspec protocol (`gcs`, `az`, etc.) |
| `RAG_STORAGE_FSSPEC_BASE_PATH` | *(empty)* | Base path in the filesystem |

Install: `pip install lasagna-rag[storage-s3]` for S3, or `pip install lasagna-rag[storage-all]` for all providers.

---

## See Also

- [Configuration](configuration.md)
- [Deployment](deployment.md)
