"""Base abstractions for File Storage backends.

This module defines the core interface that all storage backends must implement.
The design mirrors vector_backends/base.py — minimal interface, capability
detection, and graceful degradation.

Key design decisions:
1. ABC with @abstractmethod for required operations (store, retrieve, delete, exists)
2. Dataclasses for structured data (type safety, IDE support)
3. Optional methods with default implementations (presigned URLs, local path)
4. ensure_local() bridge for processors that need file paths (Docling, OCR)
5. Typed exceptions for granular error handling
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, Optional


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StorageCapabilities:
    """Capabilities advertised by a storage backend.

    Used for runtime feature detection and graceful degradation.
    """

    supports_presigned_urls: bool = False
    supports_streaming: bool = True
    supports_multipart_upload: bool = False
    max_file_size: int = 5 * 1024**3  # 5 GB default


@dataclass
class StorageFileInfo:
    """Metadata returned after storing a file.

    The ``uri`` is the canonical identifier used by retrieve/delete/exists.
    For local backends it's an absolute path; for cloud backends it's a
    scheme-prefixed URI (e.g. ``s3://bucket/prefix/doc_id/file.pdf``).
    """

    uri: str
    size: int
    content_type: Optional[str] = None
    etag: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Exceptions
# =============================================================================


class StorageError(Exception):
    """Base exception for storage operations."""


class StorageNotFoundError(StorageError):
    """File not found in storage backend."""


class StoragePermissionError(StorageError):
    """Insufficient permissions for storage operation."""


class StorageConnectionError(StorageError):
    """Cannot connect to storage backend."""


# =============================================================================
# Abstract Base Class
# =============================================================================


class StorageBackend(ABC):
    """Abstract base class for file storage backends.

    All storage backends (local, S3, GCS, Azure, fsspec, etc.) must implement
    this interface to be usable with Lasagna's document pipeline.

    Design Philosophy:
    - Minimal interface: only methods needed by the document pipeline
    - ``ensure_local()`` bridges cloud storage with file-path-only processors
    - Capability detection for runtime feature checks
    - Typed exceptions for granular error handling

    Example Implementation::

        class MyStorage(StorageBackend):
            @property
            def backend_name(self) -> str:
                return "my_storage"

            @property
            def capabilities(self) -> StorageCapabilities:
                return StorageCapabilities()

            def store(self, source, document_id, filename, **kw):
                # ... write bytes ...
                return StorageFileInfo(uri="my://path", size=1024)

            def retrieve(self, uri):
                return open(uri, "rb")

            def delete(self, uri):
                os.unlink(uri)
                return True

            def exists(self, uri):
                return os.path.exists(uri)

            def get_size(self, uri):
                return os.path.getsize(uri)
    """

    # =========================================================================
    # Identity & Capabilities
    # =========================================================================

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend (e.g., 'local', 's3', 'fsspec')."""

    @property
    @abstractmethod
    def capabilities(self) -> StorageCapabilities:
        """Return capabilities of this storage backend."""

    # =========================================================================
    # Core Operations (required)
    # =========================================================================

    @abstractmethod
    def store(
        self,
        source: BinaryIO,
        document_id: str,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> StorageFileInfo:
        """Store a file.

        Args:
            source: Readable binary stream
            document_id: Document identifier (used as directory/prefix)
            filename: Original filename (sanitized by caller)
            content_type: Optional MIME type
            metadata: Optional key-value metadata

        Returns:
            StorageFileInfo with the URI and size

        Raises:
            StorageError: On write failure
            StoragePermissionError: On permission failure
        """

    @abstractmethod
    def retrieve(self, uri: str) -> BinaryIO:
        """Open a file for reading.

        Args:
            uri: URI returned by store()

        Returns:
            Readable binary stream (caller must close)

        Raises:
            StorageNotFoundError: If file doesn't exist
        """

    @abstractmethod
    def delete(self, uri: str) -> bool:
        """Delete a file and its parent directory (if document-scoped).

        Args:
            uri: URI returned by store()

        Returns:
            True if deleted, False if not found
        """

    @abstractmethod
    def exists(self, uri: str) -> bool:
        """Check if a file exists.

        Args:
            uri: URI returned by store()

        Returns:
            True if file exists
        """

    @abstractmethod
    def get_size(self, uri: str) -> int:
        """Get file size in bytes.

        Args:
            uri: URI returned by store()

        Returns:
            File size in bytes

        Raises:
            StorageNotFoundError: If file doesn't exist
        """

    # =========================================================================
    # Input Validation
    # =========================================================================

    @staticmethod
    def _validate_store_inputs(document_id: str, filename: str) -> None:
        """Validate document_id and filename before any storage I/O.

        All backends construct paths as ``<base>/<document_id>/<filename>``.
        This rejects values that could escape that structure in any
        filesystem model (local, S3 keys, fsspec protocols).

        Raises:
            StorageError: If either value contains unsafe patterns.
        """
        # --- document_id ---
        if not document_id or not isinstance(document_id, str):
            raise StorageError("document_id must be a non-empty string")
        if len(document_id) > 512:
            raise StorageError(f"document_id too long: {len(document_id)} chars")
        if "/" in document_id or "\\" in document_id:
            raise StorageError(f"Path separator in document_id: {document_id!r}")
        if ".." in document_id:
            raise StorageError(f"Traversal sequence in document_id: {document_id!r}")
        if "\0" in document_id:
            raise StorageError("Null byte in document_id")
        # --- filename ---
        if not filename or not isinstance(filename, str):
            raise StorageError("filename must be a non-empty string")
        if "/" in filename or "\\" in filename:
            raise StorageError(f"Path separator in filename: {filename!r}")
        if ".." in filename:
            raise StorageError(f"Traversal sequence in filename: {filename!r}")
        if "\0" in filename:
            raise StorageError("Null byte in filename")

    # =========================================================================
    # Optional Methods (with default implementations)
    # =========================================================================

    def get_presigned_url(self, uri: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a pre-signed download URL.

        Only supported by cloud backends with ``supports_presigned_urls``.

        Args:
            uri: URI returned by store()
            expires_in: Expiry time in seconds (default 1 hour)

        Returns:
            Pre-signed URL string, or None if not supported
        """
        return None

    def get_local_path(self, uri: str) -> Optional[str]:
        """Return the local filesystem path for a file, if available.

        For local backends, returns the actual path (zero-copy).
        For cloud backends, returns None (use ``ensure_local()`` instead).

        Args:
            uri: URI returned by store()

        Returns:
            Local filesystem path, or None if file is remote
        """
        return None

    def ensure_local(self, uri: str) -> str:
        """Return a local file path, downloading to a temp file if needed.

        This is the bridge between cloud storage and processors that require
        local file paths (Docling's ``converter.convert(file_path)``,
        ``DocumentProcessor``, OCR).

        For local backends, this returns the actual path (zero-copy).
        For cloud backends, this downloads to a temp file.

        **The caller is responsible for cleaning up temp files** when the
        returned path differs from ``get_local_path(uri)``.

        Args:
            uri: URI returned by store()

        Returns:
            Local filesystem path (may be a temp file)

        Raises:
            StorageNotFoundError: If file doesn't exist
        """
        local = self.get_local_path(uri)
        if local:
            return local

        import shutil
        import tempfile

        # Derive suffix from filename for processors that check extensions
        basename = uri.rsplit("/", 1)[-1] if "/" in uri else uri
        suffix = ""
        if "." in basename:
            suffix = "." + basename.rsplit(".", 1)[-1]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            stream = self.retrieve(uri)
            try:
                shutil.copyfileobj(stream, tmp)
            finally:
                stream.close()
        except Exception:
            tmp.close()
            import os

            os.unlink(tmp.name)
            raise
        tmp.close()
        return tmp.name
