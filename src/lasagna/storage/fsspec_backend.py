"""Universal storage backend via fsspec.

The "never get married" escape hatch — supports 30+ storage backends
through Python's filesystem abstraction standard.

Supported services (via fsspec protocols):
    - ``s3``: AWS S3, MinIO, R2, etc.
    - ``gcs`` / ``gs``: Google Cloud Storage
    - ``az`` / ``abfs``: Azure Blob Storage
    - ``ftp`` / ``sftp``: FTP / SFTP servers
    - ``hdfs``: Hadoop HDFS
    - ``memory``: In-memory (for testing)
    - ``file``: Local filesystem
    - 20+ more via fsspec ecosystem

Requires: ``pip install fsspec`` (plus protocol-specific packages)
    - GCS: ``pip install gcsfs``
    - Azure: ``pip install adlfs``
    - S3: ``pip install s3fs``
    - HDFS: ``pip install pyarrow``

Usage::

    from lasagna.storage import get_storage_backend

    # Google Cloud Storage
    storage = get_storage_backend(
        "fsspec",
        protocol="gcs",
        base_path="my-bucket/rag-documents",
    )

    # Azure Blob Storage
    storage = get_storage_backend(
        "fsspec",
        protocol="az",
        base_path="my-container/rag-documents",
        storage_options={"account_name": "myaccount"},
    )

    # In-memory (for testing)
    storage = get_storage_backend(
        "fsspec",
        protocol="memory",
        base_path="/test-rag",
    )
"""

import io
from typing import Any, BinaryIO, Dict, Optional

from loguru import logger
from lasagna.core.provenance import redact_url

from .base import (
    StorageBackend,
    StorageCapabilities,
    StorageFileInfo,
    StorageError,
    StorageNotFoundError,
)

try:
    import fsspec

    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False


class FsspecStorageBackend(StorageBackend):
    """Universal file storage backend via fsspec.

    Wraps any fsspec-compatible filesystem. The ``protocol`` parameter
    selects which filesystem to use, and ``storage_options`` passes
    authentication / configuration to the underlying driver.

    URI format: ``<protocol>://<base_path>/<document_id>/<filename>``
    """

    def __init__(
        self,
        protocol: str,
        base_path: str = "",
        storage_options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize fsspec storage backend.

        Args:
            protocol: Filesystem protocol (e.g. "gcs", "az", "s3", "memory")
            base_path: Base path within the filesystem
            storage_options: Dict of options passed to ``fsspec.filesystem()``
                            (credentials, project, etc.)
        """
        if not FSSPEC_AVAILABLE:
            raise ImportError(
                "Fsspec storage backend requires fsspec. "
                "Install with: pip install fsspec (or pip install lasagna-rag[storage-all])"
            )

        self._protocol = protocol
        self._base_path = base_path.strip("/")
        self._storage_options = storage_options or {}

        try:
            self._fs = fsspec.filesystem(protocol, **self._storage_options)
        except Exception as e:
            raise StorageError(f"Failed to initialize fsspec filesystem '{protocol}': {e}") from e

        logger.info(f"Fsspec storage initialized: protocol={protocol}, base_path={base_path}")

    @property
    def backend_name(self) -> str:
        return "fsspec"

    @property
    def capabilities(self) -> StorageCapabilities:
        return StorageCapabilities(
            supports_presigned_urls=False,
            supports_streaming=True,
            supports_multipart_upload=False,
            max_file_size=5 * 1024**3,  # 5 GB default
        )

    def _full_path(self, document_id: str, filename: str) -> str:
        """Build the full path within the filesystem."""
        if self._base_path:
            return f"{self._base_path}/{document_id}/{filename}"
        return f"{document_id}/{filename}"

    def _path_from_uri(self, uri: str) -> str:
        """Extract filesystem path from URI."""
        prefix = f"{self._protocol}://"
        if uri.startswith(prefix):
            return uri[len(prefix) :]
        return uri

    def _to_uri(self, path: str) -> str:
        """Build URI from filesystem path."""
        return f"{self._protocol}://{path}"

    # =========================================================================
    # Core Operations
    # =========================================================================

    def store(
        self,
        source: BinaryIO,
        document_id: str,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageFileInfo:
        """Store file via fsspec filesystem."""
        self._validate_store_inputs(document_id, filename)
        path = self._full_path(document_id, filename)

        # Ensure parent directory exists (for local-like filesystems)
        parent = "/".join(path.split("/")[:-1])
        if parent:
            try:
                self._fs.mkdirs(parent, exist_ok=True)
            except (NotImplementedError, AttributeError):
                pass  # Some filesystems don't need/support mkdirs

        try:
            with self._fs.open(path, "wb") as f:
                # Read in chunks to handle large files
                while True:
                    chunk = source.read(8 * 1024 * 1024)  # 8MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as e:
            raise StorageError(f"Fsspec write failed ({self._protocol}): {e}") from e

        # Get size
        try:
            size = self._fs.size(path) or 0
        except Exception:
            size = 0

        uri = self._to_uri(path)
        logger.info(f"File stored via fsspec: {redact_url(uri)}")
        return StorageFileInfo(
            uri=uri,
            size=size,
            content_type=content_type,
            metadata=metadata or {},
        )

    def retrieve(self, uri: str) -> BinaryIO:
        """Read file via fsspec into an in-memory buffer.

        Uses streaming open() to avoid double memory allocation from cat_file().
        """
        path = self._path_from_uri(uri)
        try:
            buf = io.BytesIO()
            with self._fs.open(path, "rb") as f:
                while True:
                    chunk = f.read(8 * 1024 * 1024)  # 8MB chunks
                    if not chunk:
                        break
                    buf.write(chunk)
            buf.seek(0)
            return buf
        except FileNotFoundError as e:
            raise StorageNotFoundError(f"File not found ({self._protocol}): {uri}") from e
        except Exception as e:
            raise StorageError(f"Fsspec read failed ({self._protocol}): {e}") from e

    def delete(self, uri: str) -> bool:
        """Delete a file and its document directory via fsspec."""
        path = self._path_from_uri(uri)
        # Delete the document directory (parent of the file)
        doc_dir = "/".join(path.split("/")[:-1])

        try:
            if self._fs.exists(doc_dir):
                self._fs.rm(doc_dir, recursive=True)
                logger.info(f"Deleted via fsspec: {doc_dir}")
                return True
            elif self._fs.exists(path):
                self._fs.rm(path)
                logger.info(f"Deleted via fsspec: {path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Fsspec delete failed ({self._protocol}): {e}")
            return False

    def exists(self, uri: str) -> bool:
        """Check if file exists via fsspec."""
        if not uri:
            return False
        path = self._path_from_uri(uri)
        try:
            return self._fs.exists(path)
        except Exception:
            return False

    def get_size(self, uri: str) -> int:
        """Get file size via fsspec."""
        path = self._path_from_uri(uri)
        try:
            size = self._fs.size(path)
            if size is None:
                raise StorageNotFoundError(f"File not found ({self._protocol}): {uri}")
            return size
        except FileNotFoundError as e:
            raise StorageNotFoundError(f"File not found ({self._protocol}): {uri}") from e
