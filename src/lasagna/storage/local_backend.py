"""Local filesystem storage backend.

Zero-dependency default backend that stores files on the local filesystem.
Extracts and centralizes file I/O logic previously scattered across the
document pipeline.

URI format: absolute file paths (backward-compatible with existing DB values).

Usage::

    from lasagna.storage import get_storage_backend

    storage = get_storage_backend("local")
    # or simply: storage = get_storage_backend()  (local is default)

    with open("report.pdf", "rb") as f:
        info = storage.store(f, "doc_abc123", "report.pdf")
    print(info.uri)  # /path/to/rag_uploads/doc_abc123/report.pdf
"""

import os
import shutil
import threading
from typing import BinaryIO, Dict, Optional

from loguru import logger

from .base import (
    StorageBackend,
    StorageCapabilities,
    StorageFileInfo,
    StorageError,
    StorageNotFoundError,
)

LOCAL_AVAILABLE = True  # Always available (no external deps)


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend.

    Stores files under ``<base_dir>/<document_id>/<filename>``.
    When ``base_dir`` is not provided, it is lazily resolved from
    ``rag_config.upload_dir``.
    """

    _init_lock = threading.Lock()

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize local storage.

        Args:
            base_dir: Root directory for file storage.
                     If None, uses ``rag_config.upload_dir`` (lazy).
        """
        self._base_dir = base_dir

    @property
    def _resolved_base_dir(self) -> str:
        """Lazily resolve the base directory (thread-safe)."""
        if self._base_dir is None:
            with self._init_lock:
                if self._base_dir is None:  # Double-checked locking
                    from lasagna.core.config import rag_config

                    self._base_dir = rag_config.upload_dir
        os.makedirs(self._base_dir, exist_ok=True)
        return self._base_dir

    @property
    def backend_name(self) -> str:
        return "local"

    @property
    def capabilities(self) -> StorageCapabilities:
        return StorageCapabilities(
            supports_presigned_urls=False,
            supports_streaming=True,
            supports_multipart_upload=False,
            max_file_size=10 * 1024**3,  # 10 GB
        )

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
        """Store file on local filesystem.

        Security: path traversal is prevented by checking that the resolved
        destination stays within base_dir.
        """
        self._validate_store_inputs(document_id, filename)

        doc_dir = os.path.join(self._resolved_base_dir, document_id)
        dest_path = os.path.join(doc_dir, filename)

        # Path traversal prevention — check BEFORE creating any directories
        real_base = os.path.realpath(self._resolved_base_dir)
        real_dir = os.path.realpath(doc_dir)
        real_dest = os.path.realpath(dest_path)
        if not real_dir.startswith(real_base + os.sep):
            raise StorageError(f"Path traversal attempt detected: {document_id}")
        if not real_dest.startswith(real_base + os.sep):
            raise StorageError(f"Path traversal attempt detected: {filename}")

        os.makedirs(doc_dir, exist_ok=True)

        with open(dest_path, "wb") as f:
            shutil.copyfileobj(source, f)

        size = os.path.getsize(dest_path)
        logger.info(f"File stored: {dest_path}")
        return StorageFileInfo(uri=dest_path, size=size, content_type=content_type)

    def retrieve(self, uri: str) -> BinaryIO:
        """Open a local file for reading."""
        if not os.path.exists(uri):
            raise StorageNotFoundError(f"File not found: {uri}")
        return open(uri, "rb")

    def delete(self, uri: str) -> bool:
        """Delete a file and its parent document directory."""
        if not uri or not os.path.exists(uri):
            return False

        doc_dir = os.path.dirname(uri)
        try:
            # Remove entire document directory (same behavior as before)
            real_dir = os.path.realpath(doc_dir)
            real_base = os.path.realpath(self._resolved_base_dir)
            if real_dir.startswith(real_base + os.sep):
                shutil.rmtree(doc_dir)
            else:
                # Safety: only delete the specific file if dir is outside base
                os.unlink(uri)
            logger.info(f"Deleted document files: {doc_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document files: {e}")
            return False

    def exists(self, uri: str) -> bool:
        """Check if a local file exists."""
        return os.path.exists(uri) if uri else False

    def get_size(self, uri: str) -> int:
        """Get local file size."""
        if not os.path.exists(uri):
            raise StorageNotFoundError(f"File not found: {uri}")
        return os.path.getsize(uri)

    # =========================================================================
    # Local-Specific Methods
    # =========================================================================

    def get_local_path(self, uri: str) -> Optional[str]:
        """Return the URI itself (it's already a local path)."""
        if uri and os.path.exists(uri):
            return uri
        return None

    def ensure_local(self, uri: str) -> str:
        """For local storage, the URI IS the local path."""
        if not uri or not os.path.exists(uri):
            raise StorageNotFoundError(f"File not found: {uri}")
        return uri
