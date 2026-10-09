"""File Storage Backend Abstraction Layer.

This module provides a clean abstraction over file storage backends,
enabling Lasagna's RAG system to work with multiple storage services
(local filesystem, S3, GCS, Azure, etc.) without changing core logic.

Architecture:
    - StorageBackend: Abstract base class defining the interface
    - StorageRegistry: Factory for creating backend instances
    - LocalStorageBackend: Default implementation (zero deps)
    - S3StorageBackend: S3-compatible (boto3, 15+ services)
    - FsspecStorageBackend: Universal (fsspec, 30+ backends)

Usage::

    from lasagna.storage import get_storage_backend

    # Get the configured backend (default: local)
    storage = get_storage_backend()

    # Store a file
    with open("report.pdf", "rb") as f:
        info = storage.store(f, "doc_abc123", "report.pdf")

    # Get local path (for Docling / OCR)
    local_path = storage.ensure_local(info.uri)

    # Delete
    storage.delete(info.uri)

Design Principles:
    1. Minimal Interface: Only expose methods needed by document pipeline
    2. ensure_local() Bridge: Cloud files → local temp for Docling/OCR
    3. Capability Detection: Runtime detection of backend features
    4. Graceful Degradation: Fallback when features unavailable
    5. Zero Breaking Changes: Existing code works without modification
"""

from .base import (
    StorageBackend,
    StorageCapabilities,
    StorageFileInfo,
    StorageError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageConnectionError,
)
from .registry import (
    StorageRegistry,
    get_storage_backend,
    register_storage_backend,
    get_available_storage_backends,
    get_storage_registry,
    discover_storage_backends,
)
from .local_backend import LocalStorageBackend

# Optional imports for backends with external dependencies
try:
    from .s3_backend import S3StorageBackend
except ImportError:
    S3StorageBackend = None  # type: ignore

try:
    from .fsspec_backend import FsspecStorageBackend
except ImportError:
    FsspecStorageBackend = None  # type: ignore

__all__ = [
    # Base classes
    "StorageBackend",
    "StorageCapabilities",
    "StorageFileInfo",
    # Exceptions
    "StorageError",
    "StorageNotFoundError",
    "StoragePermissionError",
    "StorageConnectionError",
    # Registry
    "StorageRegistry",
    "get_storage_backend",
    "register_storage_backend",
    "get_available_storage_backends",
    "get_storage_registry",
    "discover_storage_backends",
    # Implementations
    "LocalStorageBackend",
    "S3StorageBackend",
    "FsspecStorageBackend",
]
