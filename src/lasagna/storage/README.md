# Storage — Pluggable File Storage Backends

Abstraction layer that lets the RAG knowledge package store, retrieve, and delete document files through any storage service without changing core logic.

```python
from lasagna.storage import get_storage_backend

storage = get_storage_backend()                    # local (default)
storage = get_storage_backend("s3", bucket="docs") # or S3, MinIO, R2, ...

with open("report.pdf", "rb") as f:
    info = storage.store(f, "doc_abc123", "report.pdf")

local_path = storage.ensure_local(info.uri)        # cloud → temp file
storage.delete(info.uri)
```

## Backends

| Backend    | Install                       | URI Format                    | Presigned URLs | Multipart | Max Size | Use When                     |
|------------|-------------------------------|-------------------------------|----------------|-----------|----------|------------------------------|
| **Local**  | built-in                      | `/path/to/doc/file`           | No             | No        | 10 GB    | Development, single-node     |
| **S3**     | `pip install boto3>=1.28`     | `s3://bucket/prefix/doc/file` | Yes            | Yes*      | 5 TB     | AWS, MinIO, R2, B2, Wasabi   |
| **Fsspec** | `pip install fsspec` + driver | `proto://base/doc/file`       | No             | No        | 5 GB     | GCS, Azure, FTP, SFTP, HDFS  |

*Multipart is automatic via boto3 `upload_fileobj()` for files >8 MB.

---

## Architecture

```text
  REST API / SDK / MCP
         │
         ▼
  ┌─ DocumentPipeline ──────────────────────────────────────────────┐
  │                                                                  │
  │  upload ──► storage.store(stream, doc_id, filename)              │
  │                 └─► StorageFileInfo { uri, size, etag }          │
  │                 └─► DB saves uri + storage_backend_name          │
  │                                                                  │
  │  process ─► storage.ensure_local(uri) ──► Docling / OCR         │
  │                 ├─ Local: return path directly (zero-copy)       │
  │                 └─ Cloud: download → temp file → processor       │
  │                                       (caller cleans up)        │
  │                                                                  │
  │  delete ──► storage.delete(uri)                                  │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─ StorageRegistry (singleton, thread-safe) ──────────────────────┐
  │                                                                  │
  │  get_storage_backend(name?, **kwargs)                            │
  │    1. Resolve name:   RAG_STORAGE_BACKEND → rag_config → local  │
  │    2. Resolve config: kwargs → RAG_STORAGE_S3_* env vars        │
  │    3. Return cached singleton (MD5-keyed by config)              │
  │                                                                  │
  │  ┌────────┐  ┌────────┐  ┌──────────┐  ┌────────────────────┐   │
  │  │ local  │  │  s3    │  │  fsspec   │  │ custom plugins     │   │
  │  │(always)│  │(boto3) │  │(fsspec)   │  │(lasagna_storage_* )   │   │
  │  └────────┘  └────────┘  └──────────┘  └────────────────────┘   │
  └──────────────────────────────────────────────────────────────────┘
```

### Module Files

| File                | Role                                                                       |
|---------------------|----------------------------------------------------------------------------|
| `base.py`           | `StorageBackend` ABC, `StorageCapabilities`, `StorageFileInfo`, exceptions |
| `local_backend.py`  | Local filesystem -- zero dependencies, lazy base-dir from config           |
| `s3_backend.py`     | S3-compatible -- boto3, presigned URLs, paginated delete                   |
| `fsspec_backend.py` | Universal -- 30+ protocols via fsspec ecosystem                            |
| `registry.py`       | Singleton registry, factory, env-var defaults, plugin discovery            |
| `__init__.py`       | Public API, graceful optional imports                                      |

---

## Configuration

Storage is configured at deploy time (not per-session). The backend name and backend-specific settings are resolved once during initialization.

### Environment Variables (recommended for cloud backends)

```bash
# Backend selection
RAG_STORAGE_BACKEND=local              # "local" | "s3" | "fsspec"

# S3 (required when backend=s3)
RAG_STORAGE_S3_BUCKET=my-docs
RAG_STORAGE_S3_PREFIX=rag-documents    # default: rag-documents
RAG_STORAGE_S3_ENDPOINT_URL=http://minio:9000   # for S3-compatible services
RAG_STORAGE_S3_REGION=us-east-1

# Fsspec (required when backend=fsspec)
RAG_STORAGE_FSSPEC_PROTOCOL=gcs        # gcs | az | ftp | sftp | hdfs | ...
RAG_STORAGE_FSSPEC_BASE_PATH=bucket/rag
RAG_STORAGE_FSSPEC_OPTIONS='{"project":"my-proj"}'
```

### SDK

```python
from lasagna.sdk import KnowledgeConfig, config_for_production

# Backend name is propagated correctly
config = KnowledgeConfig(storage_backend="s3")
config = config_for_production(storage_backend="s3")

# Backend-specific config: use env vars (see Limitations below)
# os.environ["RAG_STORAGE_S3_BUCKET"] = "my-docs"
```

### TOML (`settings.lasagna.toml`)

Add to the `[rag]` section:

```toml
[rag]
storage_backend = "s3"   # or "local", "fsspec"
```

Backend-specific settings (bucket, endpoint, etc.) are configured via `RAG_STORAGE_S3_*` environment variables, not in TOML.

### Configurator UI

The **Advanced > File Storage** tab exposes backend selection and S3/fsspec parameters for schema validation. These are deploy-time settings — changes require a restart.

### What Works Where

| Setting                   | Env Var | SDK  | TOML | UI Schema |
|---------------------------|---------|------|------|-----------|
| Backend name              | Yes     | Yes  | Yes  | Yes       |
| S3 bucket, region, prefix | Yes     | No*  | No   | Yes       |
| Fsspec protocol, path     | Yes     | No*  | No   | Yes       |

*SDK and TOML `storage_backend_config` exists but is not yet wired to the backend constructor. See [Limitations](#limitations).

---

## Core Concepts

### `ensure_local(uri) → str`

Processors like Docling and OCR require local file paths. `ensure_local()` bridges cloud storage and these processors transparently:

| Backend     | What happens              | Memory cost        |
|-------------|---------------------------|--------------------|
| Local       | Returns the URI as-is     | Zero (no copy)     |
| S3 / Fsspec | Downloads to a temp file  | ~2x file size*     |

*Cloud backends currently buffer the full file in memory before writing to temp. See [Limitations](#limitations).

**Caller owns temp file cleanup:**

```python
local_path = storage.ensure_local(uri)
is_temp = storage.get_local_path(uri) is None  # True for cloud backends
try:
    process(local_path)
finally:
    if is_temp:
        os.unlink(local_path)
```

### `StorageCapabilities`

Backends advertise features at runtime — callers adapt without `isinstance` checks:

```python
if storage.capabilities.supports_presigned_urls:
    url = storage.get_presigned_url(uri, expires_in=3600)
```

Fields: `supports_presigned_urls`, `supports_streaming`, `supports_multipart_upload`, `max_file_size`.

### `StorageFileInfo`

Returned by `store()` — the `uri` is the canonical identifier for all subsequent operations:

```python
info = storage.store(f, "doc_abc123", "report.pdf")
info.uri           # "/uploads/doc_abc123/report.pdf" or "s3://bucket/..."
info.size          # file size in bytes
info.etag          # S3 ETag (None for local/fsspec)
info.content_type  # MIME type if provided
info.metadata      # custom key-value pairs
```

### Exception Hierarchy

```text
StorageError                  ← base (I/O, validation)
├── StorageNotFoundError      ← file/object does not exist
├── StoragePermissionError    ← auth/permission failure
└── StorageConnectionError    ← cannot reach backend
```

### Per-Document Backend Tracking

Each document's `storage_backend_name` is saved in metadata on upload. During reindex, the system resolves the correct backend per document — enabling mixed-backend scenarios and safe backend migrations.

---

## Extending

### 1. Implement the ABC

```python
from lasagna.storage import (
    StorageBackend, StorageCapabilities, StorageFileInfo,
)
from typing import BinaryIO, Dict, Optional

class MyBackend(StorageBackend):
    @property
    def backend_name(self) -> str:
        return "my_backend"

    @property
    def capabilities(self) -> StorageCapabilities:
        return StorageCapabilities()

    def store(
        self,
        source: BinaryIO,
        document_id: str,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageFileInfo:
        self._validate_store_inputs(document_id, filename)
        # ... write bytes ...
        return StorageFileInfo(uri="my://path", size=1024)

    def retrieve(self, uri: str) -> BinaryIO: ...
    def delete(self, uri: str) -> bool: ...
    def exists(self, uri: str) -> bool: ...
    def get_size(self, uri: str) -> int: ...
```

### 2. Register

```python
from lasagna.storage import register_storage_backend

register_storage_backend("my_backend", MyBackend)
storage = get_storage_backend("my_backend", **my_config)
```

### 3. Plugin Auto-Discovery (optional)

Create a package named `lasagna_storage_<name>` with a top-level function:

```python
# lasagna_storage_azure/__init__.py
def register_backend(registry):
    from .backend import AzureStorageBackend
    registry.register("azure", AzureStorageBackend)
```

Discovery uses `pkgutil.iter_modules()` prefix scanning (not setuptools entry points). Call `discover_storage_backends()` to trigger it.

---

## Security

### Path Traversal Prevention

`_validate_store_inputs()` runs before any I/O in `store()`, rejecting:

- Path separators (`/`, `\`)
- Traversal sequences (`..`)
- Null bytes (`\0`)
- IDs longer than 512 characters

The local backend adds `os.path.realpath()` containment checks in `store()`.

### Input Validation Coverage

| Operation                                            | Validated                                                     |
|------------------------------------------------------|---------------------------------------------------------------|
| `store()`                                            | `_validate_store_inputs()` + realpath containment (local)     |
| `retrieve()`, `delete()`, `exists()`, `get_size()`   | URI accepted as-is -- trust depends on database integrity     |

URIs are written by `store()` and persisted in the database. Read-path operations trust the stored URI. See [Limitations](#limitations) for the implications.

---

## Design Decisions

**ABC over Protocol** — Explicit contract with `@abstractmethod` enforcement. The registry uses `issubclass()` for type checking. Protocols would allow duck-typing but lose the forced implementation guarantee.

**Singleton Registry** — One backend instance per unique config (MD5 hash of kwargs). Avoids redundant S3 client creation. Thread-safe via double-checked locking. Cache clearable via `registry.clear_instances()`.

**`ensure_local()` Bridge** — Docling and OCR require `file_path: str` arguments. Rather than modifying every processor, the storage layer downloads cloud files to temp paths transparently.

**Optional Dependencies** — `boto3` and `fsspec` are imported with `try/except`. Backends self-register only when their library is importable. The local backend always works with zero dependencies.

**Prefix-Scan Plugin Discovery** — `pkgutil.iter_modules()` scans for `lasagna_storage_*` packages. Simpler than entry points, but has reduced compatibility with namespace packages and frozen applications.

---

## Limitations

1. **Config wiring gap** — `storage_backend_config` from SDK/TOML is stored in `rag_config` but `get_storage_backend()` reads `RAG_STORAGE_S3_*` env vars directly instead. The backend *name* propagates correctly; the backend *config dict* does not. **Workaround:** use individual env vars for cloud config. ([registry.py:308](registry.py#L308))

2. **URI trust boundary** — Local backend `retrieve()` and `delete()` open/unlink the URI without verifying it is within the base directory. `store()` is properly guarded with realpath checks. If the metadata database is compromised, arbitrary file read/delete is possible. ([local_backend.py:123](local_backend.py#L123))

3. **Cloud retrieve double-buffering** — S3 and fsspec `retrieve()` read the full file into a `BytesIO` buffer. When `ensure_local()` then copies this to a temp file, peak memory is ~2x file size. For files >100 MB this is significant. A streaming `ensure_local()` override for cloud backends would reduce this to ~8 MB constant. ([s3_backend.py:190](s3_backend.py#L190), [base.py:320](base.py#L320))

4. **S3 delete partial failure** — `delete_objects()` response `Errors` list is not inspected. Partial failures (e.g., Object Lock, permission on individual keys) are silently counted as successes. ([s3_backend.py:224](s3_backend.py#L224))
