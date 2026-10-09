"""Connection pool wrapper for vector store backends.

Provides a pooled adapter that distributes operations across multiple
backend connections for true parallel search capability.

In single-user mode (pool_size=1), this is a thin passthrough.
In production mode, multiple connections allow concurrent searches
without queueing at the connection level.

Usage::

    from lasagna.vector_backends.pool import PooledVectorBackend

    # Wrap any backend with a pool
    pooled = PooledVectorBackend(
        factory=lambda: MilvusBackend(uri="http://milvus:19530"),
        pool_size=10,
    )
    pooled.connect()
    results = pooled.search(...)  # Distributed across 10 connections
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from .base import (
    BackendCapabilities,
    CollectionInfo,
    CollectionSchema,
    IndexConfig,
    SearchParams,
    VectorSearchResult,
    VectorStoreAdapter,
)
from .filters import FilterExpression


class PooledVectorBackend(VectorStoreAdapter):
    """Thread-safe connection pool for vector store backends.

    Maintains a fixed-size pool of backend instances and distributes
    operations across them.  Each backend instance holds its own
    connection to the vector database.

    Thread-safety: Operations acquire a backend from the pool, use it,
    and release it back.  The pool is bounded — if all backends are in
    use, the caller blocks until one is released.
    """

    _MAX_POOL_SIZE = 100  # Prevent misconfigured massive pools

    def __init__(
        self,
        factory: Callable[[], VectorStoreAdapter],
        pool_size: int = 1,
    ) -> None:
        """Create a pooled backend.

        Args:
            factory: Callable that creates a new VectorStoreAdapter instance.
            pool_size: Number of backend connections to maintain (1–100).
        """
        self._factory = factory
        self._pool_size = max(1, min(pool_size, self._MAX_POOL_SIZE))
        if pool_size > self._MAX_POOL_SIZE:
            logger.warning(f"Requested pool_size={pool_size} capped to {self._MAX_POOL_SIZE}")
        self._pool: deque[VectorStoreAdapter] = deque()
        self._semaphore = threading.Semaphore(self._pool_size)
        self._lock = threading.Lock()
        self._connected = False
        self._backend_name: str = ""
        self._capabilities: Optional[BackendCapabilities] = None

    # =========================================================================
    # Capability Detection
    # =========================================================================

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def capabilities(self) -> BackendCapabilities:
        if self._capabilities is None:
            return BackendCapabilities()
        return self._capabilities

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> None:
        """Create and connect all pooled backend instances.

        If a factory or connect call fails partway through, all
        already-connected backends are cleaned up before re-raising.
        """
        if self._connected:
            return
        with self._lock:
            if self._connected:
                return
            try:
                for _ in range(self._pool_size):
                    backend = self._factory()
                    backend.connect()
                    self._pool.append(backend)
            except Exception:
                # Clean up partially-connected backends
                while self._pool:
                    try:
                        self._pool.popleft().disconnect()
                    except Exception as cleanup_err:
                        logger.debug(f"Error cleaning up partial pool: {cleanup_err}")
                raise
            # Cache name and capabilities from the first instance
            first = self._pool[0]
            self._backend_name = first.backend_name
            self._capabilities = first.capabilities
            self._connected = True
            logger.info(f"Connection pool created: {self._backend_name} x{self._pool_size}")

    def disconnect(self) -> None:
        """Disconnect all pooled backends.

        Drains all in-flight checkouts by acquiring every semaphore permit
        before proceeding, ensuring no backend is in use during cleanup.
        """
        # Acquire all permits — blocks until every checked-out backend returns
        for _ in range(self._pool_size):
            self._semaphore.acquire()
        try:
            with self._lock:
                while self._pool:
                    backend = self._pool.popleft()
                    try:
                        backend.disconnect()
                    except Exception as e:
                        logger.debug(f"Error disconnecting pooled backend: {e}")
                self._connected = False
        finally:
            # Release all permits so a subsequent connect() can work
            for _ in range(self._pool_size):
                self._semaphore.release()

    def is_connected(self) -> bool:
        return self._connected

    # =========================================================================
    # Internal pool operations
    # =========================================================================

    def _acquire(self) -> VectorStoreAdapter:
        """Acquire a backend from the pool (blocks if all in use)."""
        self._semaphore.acquire()
        try:
            with self._lock:
                if not self._connected or not self._pool:
                    raise RuntimeError("Pool is not connected")
                return self._pool.popleft()
        except BaseException:
            # Release the permit so we don't permanently leak it,
            # which would progressively reduce pool capacity until deadlock.
            self._semaphore.release()
            raise

    def _release(self, backend: VectorStoreAdapter) -> None:
        """Release a backend back to the pool."""
        with self._lock:
            self._pool.append(backend)
        self._semaphore.release()

    # =========================================================================
    # Collection Management — delegated
    # =========================================================================

    def has_collection(self, collection_name: str) -> bool:
        backend = self._acquire()
        try:
            return backend.has_collection(collection_name)
        finally:
            self._release(backend)

    def create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_config: Optional[IndexConfig] = None,
    ) -> None:
        backend = self._acquire()
        try:
            backend.create_collection(collection_name, schema, index_config)
        finally:
            self._release(backend)

    def drop_collection(self, collection_name: str) -> None:
        backend = self._acquire()
        try:
            backend.drop_collection(collection_name)
        finally:
            self._release(backend)

    def describe_collection(self, collection_name: str) -> CollectionInfo:
        backend = self._acquire()
        try:
            return backend.describe_collection(collection_name)
        finally:
            self._release(backend)

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        backend = self._acquire()
        try:
            return backend.get_collection_stats(collection_name)
        finally:
            self._release(backend)

    # =========================================================================
    # Index Management — delegated
    # =========================================================================

    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_config: IndexConfig,
    ) -> None:
        backend = self._acquire()
        try:
            backend.create_index(collection_name, field_name, index_config)
        finally:
            self._release(backend)

    def list_indexes(self, collection_name: str) -> List[str]:
        backend = self._acquire()
        try:
            return backend.list_indexes(collection_name)
        finally:
            self._release(backend)

    # =========================================================================
    # Data Operations — delegated
    # =========================================================================

    def insert(
        self,
        collection_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        partition_key: Optional[str] = None,
    ) -> List[Union[int, str]]:
        backend = self._acquire()
        try:
            return backend.insert(collection_name, data, partition_key=partition_key)
        finally:
            self._release(backend)

    def delete(
        self,
        collection_name: str,
        filter_expr: FilterExpression,
        partition_key: Optional[str] = None,
    ) -> int:
        backend = self._acquire()
        try:
            return backend.delete(collection_name, filter_expr, partition_key=partition_key)
        finally:
            self._release(backend)

    def query(
        self,
        collection_name: str,
        filter_expr: Optional[FilterExpression] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        partition_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        backend = self._acquire()
        try:
            return backend.query(
                collection_name,
                filter_expr=filter_expr,
                output_fields=output_fields,
                limit=limit,
                offset=offset,
                partition_key=partition_key,
            )
        finally:
            self._release(backend)

    def list_document_ids(
        self,
        collection_name: str,
        session_id: Optional[str] = None,
        *,
        limit: int = 10_000,
        offset: int = 0,
    ) -> List[str]:
        backend = self._acquire()
        try:
            return backend.list_document_ids(
                collection_name,
                session_id=session_id,
                limit=limit,
                offset=offset,
            )
        finally:
            self._release(backend)

    # =========================================================================
    # Vector Search — delegated
    # =========================================================================

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        vector_field: str = "embedding",
        filter_expr: Optional[FilterExpression] = None,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[SearchParams] = None,
        partition_key: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        backend = self._acquire()
        try:
            return backend.search(
                collection_name,
                query_embedding,
                vector_field=vector_field,
                filter_expr=filter_expr,
                limit=limit,
                output_fields=output_fields,
                search_params=search_params,
                partition_key=partition_key,
            )
        finally:
            self._release(backend)

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_embedding: List[float],
        vector_field: str = "embedding",
        sparse_field: str = "sparse_embedding",
        filter_expr: Optional[FilterExpression] = None,
        limit: int = 10,
        alpha: float = 0.7,
        output_fields: Optional[List[str]] = None,
        partition_key: Optional[str] = None,
        **kwargs,
    ) -> List[VectorSearchResult]:
        backend = self._acquire()
        try:
            return backend.hybrid_search(
                collection_name,
                query_text,
                query_embedding,
                vector_field=vector_field,
                sparse_field=sparse_field,
                filter_expr=filter_expr,
                limit=limit,
                alpha=alpha,
                output_fields=output_fields,
                partition_key=partition_key,
                **kwargs,
            )
        finally:
            self._release(backend)

    # =========================================================================
    # Partition Management — delegated
    # =========================================================================

    def has_partition(self, collection_name: str, partition_name: str) -> bool:
        backend = self._acquire()
        try:
            return backend.has_partition(collection_name, partition_name)
        finally:
            self._release(backend)

    def create_partition(self, collection_name: str, partition_name: str) -> None:
        backend = self._acquire()
        try:
            backend.create_partition(collection_name, partition_name)
        finally:
            self._release(backend)

    def drop_partition(self, collection_name: str, partition_name: str) -> None:
        backend = self._acquire()
        try:
            backend.drop_partition(collection_name, partition_name)
        finally:
            self._release(backend)

    def list_partitions(self, collection_name: str) -> List[str]:
        backend = self._acquire()
        try:
            return backend.list_partitions(collection_name)
        finally:
            self._release(backend)

    # =========================================================================
    # Lifecycle Operations — delegated
    # =========================================================================

    def flush(self, collection_name: str) -> None:
        backend = self._acquire()
        try:
            backend.flush(collection_name)
        finally:
            self._release(backend)

    def compact(self, collection_name: str) -> None:
        backend = self._acquire()
        try:
            backend.compact(collection_name)
        finally:
            self._release(backend)

    def prepare_index_params(self) -> Any:
        backend = self._acquire()
        try:
            return backend.prepare_index_params()
        finally:
            self._release(backend)

    # =========================================================================
    # Backend-Specific Extensions — delegated when available
    # =========================================================================

    def multi_search(self, *args, **kwargs):
        """Delegate to underlying backend's multi_search (e.g. OpenSearch _msearch).

        Raises AttributeError if the underlying backend does not support it.
        """
        backend = self._acquire()
        try:
            method = getattr(backend, "multi_search", None)
            if method is None:
                raise AttributeError(
                    f"{backend.backend_name} backend does not support multi_search"
                )
            return method(*args, **kwargs)
        finally:
            self._release(backend)

    def multi_hybrid_search(self, *args, **kwargs):
        """Delegate to underlying backend's multi_hybrid_search (e.g. OpenSearch _msearch).

        Raises AttributeError if the underlying backend does not support it.
        """
        backend = self._acquire()
        try:
            method = getattr(backend, "multi_hybrid_search", None)
            if method is None:
                raise AttributeError(
                    f"{backend.backend_name} backend does not support multi_hybrid_search"
                )
            return method(*args, **kwargs)
        finally:
            self._release(backend)
