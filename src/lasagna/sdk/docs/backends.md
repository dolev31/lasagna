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
# ['milvus', 'pinecone', 'qdrant', 'weaviate']
```

### Using Pinecone

```python
from lasagna.sdk import KnowledgeConfig, VectorBackend

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
        "url": "http://localhost:6333",
        "collection_name": "knowledge",
    },
)
```

### Custom Backend

```python
from lasagna.sdk.backends import (
    VectorBackendBase,
    register_backend,
)

class MyCustomBackend(VectorBackendBase):
    """Custom vector backend implementation."""

    async def insert(self, vectors, metadata):
        ...

    async def search(self, query_vector, top_k):
        ...

    async def delete(self, ids):
        ...

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
