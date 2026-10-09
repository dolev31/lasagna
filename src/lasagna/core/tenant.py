"""Partition-based tenant isolation manager.

Maps ``session_id`` → ``partition_key`` and ensures the corresponding
partition/namespace exists on the vector backend before data operations.

Dual-mode:
    - **none** (default, single-user): No partitions.  Current metadata-filter
      behaviour is preserved — zero overhead.
    - **partition**: Uses native backend partitions where supported (Milvus
      ``partition_name``, Pinecone ``namespace``).  Falls back to metadata
      filtering for backends that don't support partitions (Qdrant, Weaviate).

Usage::

    from lasagna.core.tenant import get_tenant_manager

    tm = get_tenant_manager()
    pk = tm.resolve_partition_key("user_abc")   # → "t_user_abc" or None
    tm.ensure_partition(backend, collection, pk)
"""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING, Optional, Set

from loguru import logger

if TYPE_CHECKING:
    from lasagna.vector_backends.base import VectorStoreAdapter


# Safe characters for partition names (alphanumeric, underscore, hyphen)
_SAFE_RE = re.compile(r"[^a-zA-Z0-9_-]")
_MAX_PARTITION_NAME = 128


def _sanitize_partition_name(session_id: str) -> str:
    """Convert a session_id to a safe partition name.

    Prefixes with ``t_`` to avoid clashes with internal partition names
    and replaces unsafe characters with underscores.
    """
    safe = _SAFE_RE.sub("_", session_id)
    name = f"t_{safe}"
    if len(name) > _MAX_PARTITION_NAME:
        # Truncate and append a hash suffix for uniqueness
        import hashlib

        suffix = hashlib.sha256(session_id.encode()).hexdigest()[:8]
        name = name[: _MAX_PARTITION_NAME - 9] + "_" + suffix
    return name


class TenantManager:
    """Manages partition-based tenant isolation.

    Thread-safe.  The ``_created_partitions`` cache prevents redundant
    ``create_partition`` calls for partitions that are already known to exist.
    """

    def __init__(self, isolation_mode: str = "none") -> None:
        """
        Args:
            isolation_mode: ``"none"`` (metadata filtering only) or
                ``"partition"`` (native backend partitions + metadata filter
                as defense-in-depth).
        """
        if isolation_mode not in ("none", "partition"):
            raise ValueError(f"Unknown isolation_mode: {isolation_mode!r}")
        self._mode = isolation_mode
        # Cache of (backend_name, collection, partition) tuples known to exist
        self._created_partitions: Set[tuple] = set()
        self._lock = threading.Lock()

    @property
    def is_partition_mode(self) -> bool:
        return self._mode == "partition"

    def resolve_partition_key(self, session_id: Optional[str]) -> Optional[str]:
        """Map a session_id to a partition key.

        Returns ``None`` when isolation is disabled or session_id is empty,
        signalling callers to use the default (unpartitioned) behaviour.
        """
        if not self.is_partition_mode or not session_id:
            return None
        return _sanitize_partition_name(session_id)

    def ensure_partition(
        self,
        backend: VectorStoreAdapter,
        collection_name: str,
        partition_key: Optional[str],
    ) -> None:
        """Ensure the partition exists, creating it if necessary.

        No-op when ``partition_key`` is ``None`` or the backend doesn't
        support partitions.
        """
        if partition_key is None:
            return
        if not getattr(backend.capabilities, "supports_partitions", False):
            return

        cache_key = (backend.backend_name, collection_name, partition_key)
        if cache_key in self._created_partitions:
            return

        with self._lock:
            if cache_key in self._created_partitions:
                return
            try:
                if not backend.has_partition(collection_name, partition_key):
                    backend.create_partition(collection_name, partition_key)
                    logger.info(f"Created partition {partition_key!r} in {collection_name}")
                self._created_partitions.add(cache_key)
            except Exception as exc:
                logger.warning(
                    f"Failed to create partition {partition_key!r}: {exc}. "
                    "Falling back to metadata filtering."
                )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[TenantManager] = None
_instance_lock = threading.Lock()


def get_tenant_manager() -> TenantManager:
    """Return the global TenantManager singleton.

    Reads ``tenant_isolation`` from the RAG config on first call.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                from lasagna.core.config import rag_config

                mode = getattr(rag_config, "tenant_isolation", "none")
                _instance = TenantManager(isolation_mode=mode)
                logger.debug(f"TenantManager initialised: mode={mode}")
    return _instance


def configure_tenant_manager(isolation_mode: str) -> None:
    """Reconfigure the global TenantManager.

    Called during ``RAGBackend._do_init()`` after config is loaded.
    """
    global _instance
    with _instance_lock:
        _instance = TenantManager(isolation_mode=isolation_mode)
    logger.info(f"TenantManager configured: mode={isolation_mode}")
