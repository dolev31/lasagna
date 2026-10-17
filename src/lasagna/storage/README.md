# Storage backends

Where Lasagna keeps the files you upload. The document pipeline stores each
upload through a `StorageBackend`, records the returned URI and the backend's
name on the document row, hands processors (Docling, OCR, the text extractors)
a local path from `ensure_local()`, and deletes the file through the same
backend.

```python
from lasagna.storage import get_storage_backend

storage = get_storage_backend()                    # the configured backend, local by default

with open("report.pdf", "rb") as f:
    info = storage.store(f, "doc_abc123", "report.pdf")

print(info.uri)        # ~/.lasagna/uploads/doc_abc123/report.pdf, as an absolute path
local_path = storage.ensure_local(info.uri)        # a cloud backend downloads to a temp file
storage.delete(info.uri)                           # removes the document's directory
```

## Backends

| Name | Install | URI | Presigned URLs |
| --- | --- | --- | --- |
| `local` | core | `<upload dir>/<document_id>/<filename>` | no |
| `s3` | `pip install "lasagna-rag[storage-s3]"` | `s3://<bucket>/<prefix>/<document_id>/<filename>` | yes |
| `fsspec` | `pip install "lasagna-rag[storage-all]"` | `<protocol>://<base_path>/<document_id>/<filename>` | no |

`<upload dir>` is `~/.lasagna/uploads` unless `LASAGNA_UPLOAD_DIR` or
`LASAGNA_DATA_DIR` says otherwise (the `RAG_` names of both still work).

`s3` works with any service that speaks the S3 API (AWS, MinIO, Cloudflare R2
and others) through `endpoint_url`; the bucket must already exist. Uploads go
through boto3's `upload_fileobj`, which switches to multipart upload for large
files. `fsspec` wraps any fsspec filesystem: `storage-all` installs the S3
(`s3fs`), Google Cloud Storage (`gcsfs`) and Azure (`adlfs`) drivers, and
protocols such as SFTP or HDFS need their own driver package.

Each backend advertises a `max_file_size` in its capabilities (local 10 GB, S3
5 TB, fsspec 5 GB), but nothing enforces it. Uploads are limited by the
`max_file_size` setting, 50 MB by default.

Without its library, `get_storage_backend("s3")` raises `ValueError` and names
the extra to install. `get_available_storage_backends()` lists what is
installed.

## Configure

Storage is chosen once per process, when the backend starts; changing it needs
a restart. It is not a per-session setting.

| Setting | Environment | Settings file (`[rag]` in `settings.lasagna.toml`) | SDK |
| --- | --- | --- | --- |
| Backend | `LASAGNA_STORAGE_BACKEND` or `RAG_STORAGE_BACKEND` | `storage_backend = "s3"` | `KnowledgeConfig(storage_backend="s3")` |
| Its arguments | `LASAGNA_STORAGE_BACKEND_CONFIG` (JSON object) | `storage_backend_config = { bucket = "my-docs" }` | `KnowledgeConfig(storage_backend_config={"bucket": "my-docs"})` |

`config_for_production()` takes the same two arguments. `storage_backend_config`
holds the constructor arguments:

- `s3`: `bucket` (required), `prefix` (default `rag-documents`), `endpoint_url`,
  `region_name` (default `us-east-1`), `aws_access_key_id`,
  `aws_secret_access_key`. Without explicit keys, boto3 uses its usual
  credential chain (`AWS_ACCESS_KEY_ID` and so on).
- `fsspec`: `protocol` (required, for example `gcs`, `az`, `s3`, `memory`),
  `base_path`, `storage_options` (passed to `fsspec.filesystem()`).
- `local`: `base_dir` (default: the upload directory).

When `storage_backend_config` is empty or lacks the required key, these
variables fill in (they have no `LASAGNA_` form, and entries in
`storage_backend_config` win over them):

```bash
RAG_STORAGE_S3_BUCKET=my-docs
RAG_STORAGE_S3_PREFIX=rag-documents
RAG_STORAGE_S3_ENDPOINT_URL=http://minio:9000
RAG_STORAGE_S3_REGION=us-east-1

RAG_STORAGE_FSSPEC_PROTOCOL=gcs
RAG_STORAGE_FSSPEC_BASE_PATH=my-bucket/rag
RAG_STORAGE_FSSPEC_OPTIONS='{"project": "my-project"}'
```

The configurator's Advanced > File Storage field sets only the backend name;
its help text names the variables above.

Every setting is listed in [the settings reference](../../../docs/reference/settings.md).

## Using a backend directly

```python
from lasagna.storage import get_storage_backend

s3 = get_storage_backend("s3", bucket="my-docs", endpoint_url="http://localhost:9000")
gcs = get_storage_backend("fsspec", protocol="gcs", base_path="my-bucket/rag")
mem = get_storage_backend("fsspec", protocol="memory", base_path="/test-rag")
```

`get_storage_backend(name, **kwargs)` returns one cached instance per name and
arguments (`singleton=False` builds a new one; `get_storage_registry().clear_instances()`
empties the cache). With no arguments it uses the configuration above.

`store()` returns a `StorageFileInfo` with `uri` (the identifier every other call
takes), `size`, `content_type`, `etag` (S3 only) and `metadata`.

`ensure_local(uri)` returns the file itself for `local`. For `s3` and `fsspec` it
writes a temporary copy, and the caller deletes it; the document pipeline does:

```python
import os

local_path = storage.ensure_local(uri)
is_temp = local_path != storage.get_local_path(uri)   # get_local_path is None for cloud backends
try:
    ...  # read local_path
finally:
    if is_temp:
        os.unlink(local_path)
```

`capabilities` is a `StorageCapabilities` dataclass with `supports_presigned_urls`,
`supports_streaming`, `supports_multipart_upload` and `max_file_size`. On S3,
`get_presigned_url(uri, expires_in=3600)` returns a download link; elsewhere it
returns `None`.

Errors are `StorageError` and its subclasses `StorageNotFoundError`,
`StoragePermissionError` and `StorageConnectionError`.

Each document row records `storage_backend_name`. Reindexing a document
(`POST /api/v1/documents/{id}/reindex`) opens the backend named there, with
that backend's default configuration; when that is not the configured
backend, the reindexed copy is stored on the configured one. Processing an
upload and deleting a document use the configured backend.

## Add a backend

```python
from typing import BinaryIO, Dict, Optional

from lasagna.storage import (
    StorageBackend,
    StorageCapabilities,
    StorageFileInfo,
    get_storage_backend,
    register_storage_backend,
)


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
        ...  # write the bytes
        return StorageFileInfo(uri="my://path", size=1024)

    def retrieve(self, uri: str) -> BinaryIO: ...
    def delete(self, uri: str) -> bool: ...
    def exists(self, uri: str) -> bool: ...
    def get_size(self, uri: str) -> int: ...


register_storage_backend("my_backend", MyBackend)
storage = get_storage_backend("my_backend")
```

The default `ensure_local()` downloads through `retrieve()`; override
`get_local_path()` if your files are already on local disk. An installed package
named `lasagna_storage_<name>` with a `register_backend(registry)` function is
registered when you call `discover_storage_backends()`.

## Security

`_validate_store_inputs()` runs before any write and rejects an empty
`document_id` or `filename`, either one containing `/`, `\`, `..` or a null
byte, and a `document_id` longer than 512 characters. The local backend also
checks, with `os.path.realpath`, that the destination stays inside its base
directory. The document pipeline sanitizes the filename before calling
`store()`.

`retrieve()`, `delete()`, `exists()` and `get_size()` take the URI as stored in
the metadata database and do not check it. If that database can be written by
someone else, they can make the local backend read any file the process can
read, or delete one (`delete()` removes a whole document directory only when it
is inside the base directory, and otherwise just the file).

## Limitations

- `s3` and `fsspec` read the whole object into memory in `retrieve()`, so
  `ensure_local()` holds a full copy in memory before writing the temp file.
- S3 `delete()` does not inspect the `Errors` list of `delete_objects`, so an
  object it failed to delete (for example under Object Lock) is still counted
  as deleted.
- No backend can list its files, so the orphan detector does not find files
  without a document row.

`tests/test_storage_backends.py` covers the local backend, the registry,
custom backends and path traversal.

## Files

| File | Contents |
| --- | --- |
| `base.py` | `StorageBackend`, `StorageCapabilities`, `StorageFileInfo`, the exceptions |
| `local_backend.py` | Local filesystem |
| `s3_backend.py` | S3-compatible storage through boto3 |
| `fsspec_backend.py` | Any fsspec filesystem |
| `registry.py` | `StorageRegistry`, `get_storage_backend`, defaults from configuration and environment, plugin discovery |
