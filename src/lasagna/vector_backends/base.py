"""Base abstractions for Vector Store backends.

This module defines the core interface that all vector database backends must implement.
The design is intentionally minimal - only methods actually used by Lasagna's RAG system
are included.

Key design decisions:
1. Use dataclasses for structured data (type safety, IDE support)
2. Abstract base class with @abstractmethod for required operations
3. Optional methods with default implementations for graceful degradation
4. Capability detection for runtime feature checks
5. Score normalization for consistent results across backends
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union
import math
import uuid

from .filters import FilterExpression


# =============================================================================
# Standard Field Names — use these instead of magic strings across backends
# =============================================================================


class Fields:
    """Canonical field names for the RAG document schema.

    All backends MUST use these names for interoperability. This eliminates
    magic strings and ensures consistent field access across Milvus, Qdrant,
    Pinecone, and Weaviate.
    """

    CHUNK_ID = "chunk_id"
    DOCUMENT_ID = "document_id"
    SESSION_ID = "session_id"
    CONTENT = "content"
    EMBEDDING = "embedding"
    METADATA = "metadata"
    VECTOR = "vector"  # Alias used by some backends


class MetricType(str, Enum):
    """Supported distance/similarity metrics."""

    COSINE = "COSINE"
    L2 = "L2"
    IP = "IP"  # Inner Product


class IndexType(str, Enum):
    """Common index types across backends."""

    FLAT = "FLAT"
    HNSW = "HNSW"
    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    AUTOINDEX = "AUTOINDEX"


@dataclass
class BackendCapabilities:
    """Capabilities advertised by a vector backend.

    Used for runtime feature detection and graceful degradation.
    """

    # Core capabilities
    supports_hybrid_search: bool = False
    supports_native_bm25: bool = False
    supports_sparse_vectors: bool = False

    # Index capabilities
    supports_hnsw: bool = True
    supports_ivf: bool = True
    supports_gpu: bool = False

    # Filter capabilities
    supports_json_filters: bool = True
    supports_metadata_filtering: bool = True
    max_filter_complexity: int = 100  # Max conditions in filter

    # Batch capabilities
    max_batch_size: int = 1000
    supports_async_insert: bool = False

    # Partition / namespace capabilities (for tenant isolation)
    supports_partitions: bool = False  # Milvus partitions, Pinecone namespaces

    # Schema capabilities
    supports_dynamic_schema: bool = False
    max_varchar_length: int = 65535
    max_vector_dimension: int = 32768


@dataclass
class IndexConfig:
    """Configuration for vector index creation."""

    index_type: IndexType = IndexType.HNSW
    metric_type: MetricType = MetricType.COSINE
    params: Dict[str, Any] = field(default_factory=lambda: {"M": 16, "efConstruction": 256})


@dataclass
class SearchParams:
    """Parameters for vector search operations."""

    metric_type: MetricType = MetricType.COSINE
    params: Dict[str, Any] = field(default_factory=lambda: {"ef": 128})


@dataclass
class FieldSchema:
    """Schema definition for a collection field."""

    name: str
    dtype: str  # "INT64", "VARCHAR", "FLOAT_VECTOR", "JSON", etc.
    is_primary: bool = False
    auto_id: bool = False
    max_length: Optional[int] = None  # For VARCHAR
    dimension: Optional[int] = None  # For vectors
    description: str = ""


@dataclass
class CollectionSchema:
    """Schema definition for a collection."""

    fields: List[FieldSchema]
    description: str = ""
    enable_dynamic_field: bool = False


@dataclass
class CollectionInfo:
    """Information about an existing collection."""

    name: str
    fields: List[Dict[str, Any]]
    row_count: int = 0
    description: str = ""
    index_info: Optional[Dict[str, Any]] = None


@dataclass
class VectorSearchResult:
    """A single result from vector search.

    This is a backend-agnostic representation that all backends
    must convert their native results to.

    The ``score`` field is a **similarity** value (higher = better match),
    typically in [0, 1] for COSINE and L2 metrics.  For IP (inner product)
    with non-normalized vectors the score may exceed 1.0 — this is
    intentional so ranking signal is preserved.  Downstream components
    (ranker, citation manager) re-normalize for display/filtering.
    """

    id: Union[int, str]
    score: float  # Similarity score, higher = better match
    entity: Dict[str, Any]  # All fields from the document

    # Optional fields populated by some backends
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    raw_score: Optional[float] = None  # Original backend score before normalization

    @property
    def document_id(self) -> str:
        """Extract document_id from entity."""
        return self.entity.get("document_id", "")

    @property
    def content(self) -> str:
        """Extract content from entity."""
        return self.entity.get("content", "")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Extract metadata from entity."""
        return self.entity.get("metadata", {})

    @property
    def embedding(self) -> Optional[List[float]]:
        """Extract embedding from entity if present."""
        return self.entity.get("embedding")

    @property
    def distance(self) -> float:
        """Get distance (inverse of similarity) for backward compatibility.

        Returns:
            Distance value where lower = better match (clamped to >= 0)
        """
        return max(0.0, 1.0 - self.score)


class VectorStoreAdapter(ABC):
    """Abstract base class for vector database backends.

    All vector databases (Milvus, Pinecone, Qdrant, etc.) must implement
    this interface to be usable with Lasagna's RAG system.

    Design Philosophy:
    - Methods are organized by operation type (CRUD, search, admin)
    - Required methods are @abstractmethod
    - Optional methods have sensible defaults
    - Backend-specific features accessed via capabilities

    Example Implementation:
        class MyBackend(VectorStoreAdapter):
            def __init__(self, connection_string: str):
                self.client = MyVectorDB(connection_string)

            def insert(self, collection_name, data):
                self.client.upsert(collection_name, data)

            # ... implement other required methods
    """

    # =========================================================================
    # Capability Detection
    # =========================================================================

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return capabilities of this backend.

        This enables runtime feature detection so the RAG system
        can adapt its behavior based on what the backend supports.
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend (e.g., 'milvus', 'pinecone')."""
        pass

    # =========================================================================
    # Connection Management
    # =========================================================================

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the vector database.

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the vector database."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if currently connected to the database."""
        pass

    # =========================================================================
    # Collection Management
    # =========================================================================

    @abstractmethod
    def has_collection(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists
        """
        pass

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_config: Optional[IndexConfig] = None,
    ) -> None:
        """Create a new collection.

        Args:
            collection_name: Name for the new collection
            schema: Schema defining the collection fields
            index_config: Optional index configuration

        Raises:
            ValueError: If collection already exists
        """
        pass

    @abstractmethod
    def drop_collection(self, collection_name: str) -> None:
        """Delete a collection and all its data.

        Args:
            collection_name: Name of collection to delete
        """
        pass

    @abstractmethod
    def describe_collection(self, collection_name: str) -> CollectionInfo:
        """Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            CollectionInfo with schema and stats
        """
        pass

    def describe_fields(self, collection_name: str) -> List[Dict[str, Any]]:
        """The collection's field definitions, without its statistics.

        Every search checks the embedding size of the collection; a backend for
        which ``describe_collection`` costs a second call to count rows
        overrides this to skip it.
        """
        return self.describe_collection(collection_name).fields

    @abstractmethod
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dict with 'row_count' and other stats
        """
        pass

    # =========================================================================
    # Index Management
    # =========================================================================

    @abstractmethod
    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_config: IndexConfig,
    ) -> None:
        """Create an index on a vector field.

        Args:
            collection_name: Name of the collection
            field_name: Name of the vector field to index
            index_config: Index configuration
        """
        pass

    @abstractmethod
    def list_indexes(self, collection_name: str) -> List[str]:
        """List indexed fields in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            List of field names that have indexes
        """
        pass

    def has_index(self, collection_name: str, field_name: str) -> bool:
        """Check if a field has an index.

        Default implementation uses list_indexes().
        Override for more efficient checks.
        """
        return field_name in self.list_indexes(collection_name)

    # =========================================================================
    # Data Operations (CRUD)
    # =========================================================================

    @abstractmethod
    def insert(
        self,
        collection_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        partition_key: Optional[str] = None,
    ) -> List[Union[int, str]]:
        """Insert one or more documents into a collection.

        Args:
            collection_name: Name of the collection
            data: Single document or list of documents
            partition_key: Optional partition/namespace for tenant isolation.
                Backends with ``supports_partitions`` use this natively.
                Others ignore it (tenant filtering via metadata).

        Returns:
            List of inserted IDs
        """
        pass

    @abstractmethod
    def delete(
        self,
        collection_name: str,
        filter_expr: FilterExpression,
        partition_key: Optional[str] = None,
    ) -> int:
        """Delete documents matching a filter.

        Args:
            collection_name: Name of the collection
            filter_expr: Backend-agnostic filter expression
            partition_key: Optional partition/namespace for tenant isolation.

        Returns:
            Number of deleted documents

        Raises:
            ValueError: If filter_expr is empty or would match no conditions.
        """
        pass

    @abstractmethod
    def query(
        self,
        collection_name: str,
        filter_expr: Optional[FilterExpression] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        partition_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query documents by filter (non-vector search).

        Args:
            collection_name: Name of the collection
            filter_expr: Optional filter expression
            output_fields: Fields to return (None = all)
            limit: Maximum results
            offset: Number of results to skip
            partition_key: Optional partition/namespace for tenant isolation.

        Returns:
            List of matching documents
        """
        pass

    def iter_all(
        self,
        collection_name: str,
        filter_expr: Optional[FilterExpression] = None,
        output_fields: Optional[List[str]] = None,
        *,
        partition_key: Optional[str] = None,
        page_size: int = 1000,
    ) -> Iterator[Dict[str, Any]]:
        """Stream every row matching a filter, however many there are.

        ``query()`` takes ``limit``/``offset``, which is the wrong tool for a
        full scan on backends that emulate offset by over-fetching: Milvus has
        no native offset, so it asks for ``limit + offset`` rows and slices
        locally. That is quadratic across pages, and past its 16384 cap the
        request is rejected outright — so a caller paginating a large corpus
        crashes partway through rather than finishing.

        This is deliberately **not** abstract. The default below works on any
        backend through the existing ``query()`` contract, so a third-party
        backend keeps working untouched; backends with a native cursor override
        it. That matters because backends are a published plugin point.

        Yields:
            Every matching row, in backend order.
        """
        offset = 0
        while True:
            rows = self.query(
                collection_name=collection_name,
                filter_expr=filter_expr,
                output_fields=output_fields,
                limit=page_size,
                offset=offset,
                partition_key=partition_key,
            )
            if not rows:
                return
            yield from rows
            if len(rows) < page_size:
                return
            offset += len(rows)

    def list_document_ids(
        self,
        collection_name: str,
        session_id: Optional[str] = None,
        *,
        limit: int = 10_000,
        offset: int = 0,
    ) -> List[str]:
        """Return distinct document_id values stored in this backend.

        Used by OrphanDetector for cross-store consistency checks.

        Args:
            collection_name: Name of the collection to scan
            session_id: Optional session filter to scope the scan
            limit: Maximum number of document IDs to return
            offset: Number of results to skip (for pagination)

        Returns:
            List of distinct document_id strings.

        Note:
            Default implementation uses ``query()`` with a metadata filter.
            Backends may override with a more efficient native distinct query.
            Pinecone does not support true distinct queries — this operation
            may be O(N) on the number of stored vectors.
        """
        filter_expr = self.session_filter(collection_name, session_id) if session_id else None
        # Scan chunk rows rather than paging: a collection with many chunks per
        # document would otherwise return only a fraction of documents in the
        # first page, and paging past a backend's limit+offset ceiling fails
        # outright. `iter_all` uses a native cursor where one exists.
        seen: set = set()
        result: List[str] = []
        skipped = 0
        for row in self.iter_all(
            collection_name=collection_name,
            filter_expr=filter_expr,
            output_fields=["document_id"],
        ):
            doc_id = row.get("document_id") or row.get("metadata", {}).get("document_id")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            if skipped < offset:
                skipped += 1
                continue
            result.append(doc_id)
            if len(result) >= limit:
                break
        return result

    def session_filter(self, collection_name: str, session_id: str) -> FilterExpression:
        """The filter that selects one session's rows in ``collection_name``.

        Every writer stores ``session_id`` as a top-level field, so the default
        matches on it. A backend whose collections may lack that field
        overrides this (Milvus).
        """
        return FilterExpression.eq(Fields.SESSION_ID, session_id)

    # =========================================================================
    # Vector Search
    # =========================================================================

    @abstractmethod
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
        """Perform vector similarity search.

        This is the core search operation used by the RAG system.

        Args:
            collection_name: Name of the collection
            query_embedding: Query vector
            vector_field: Name of the vector field to search
            filter_expr: Optional filter to narrow results
            limit: Maximum number of results
            output_fields: Fields to include in results
            search_params: Search parameters (metric, ef, etc.)
            partition_key: Optional partition/namespace for tenant isolation.
                When provided and the backend supports partitions, the
                search is scoped to only that partition's data.

        Returns:
            List of VectorSearchResult sorted by similarity
        """
        pass

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
        **kwargs: Any,
    ) -> List[VectorSearchResult]:
        """Perform hybrid dense + sparse search.

        Default implementation falls back to dense-only search.
        Override for backends with native hybrid support.

        Args:
            collection_name: Name of the collection
            query_text: Raw query text (for BM25/sparse)
            query_embedding: Dense query vector
            vector_field: Dense vector field name
            sparse_field: Sparse vector field name
            filter_expr: Optional filter
            limit: Maximum results
            alpha: Weight for dense (1.0 = pure dense)
            output_fields: Fields to include
            partition_key: Optional partition/namespace for tenant isolation.
            **kwargs: Backend-specific options (e.g., query_tokens, group_by_document).

        Returns:
            List of VectorSearchResult with combined scores
        """
        # Default: fall back to dense search
        return self.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            vector_field=vector_field,
            filter_expr=filter_expr,
            limit=limit,
            output_fields=output_fields,
            partition_key=partition_key,
        )

    # =========================================================================
    # Async Methods (default: wraps sync via run_in_executor)
    # =========================================================================

    async def async_search(
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
        """Async vector search.  Default wraps sync ``search()`` in executor."""
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(
                collection_name=collection_name,
                query_embedding=query_embedding,
                vector_field=vector_field,
                filter_expr=filter_expr,
                limit=limit,
                output_fields=output_fields,
                search_params=search_params,
                partition_key=partition_key,
            ),
        )

    async def async_insert(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
        partition_key: Optional[str] = None,
    ) -> None:
        """Async insert.  Default wraps sync ``insert()`` in executor."""
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.insert(
                collection_name=collection_name,
                data=data,
                partition_key=partition_key,
            ),
        )

    # =========================================================================
    # Partition Management (optional, for tenant isolation)
    # =========================================================================

    def has_partition(self, collection_name: str, partition_name: str) -> bool:
        """Check if a partition exists.  Returns False by default."""
        return False

    def create_partition(self, collection_name: str, partition_name: str) -> None:
        """Create a partition.  No-op by default."""
        pass

    def drop_partition(self, collection_name: str, partition_name: str) -> None:
        """Drop a partition.  No-op by default."""
        pass

    def list_partitions(self, collection_name: str) -> List[str]:
        """List partitions in a collection.  Returns empty list by default."""
        return []

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def prepare_index_params(self) -> Any:
        """Return backend-specific index parameter builder.

        For Milvus compatibility. Returns a builder object that backends
        can use to configure index parameters.

        Default implementation returns a simple dict-based builder.
        """
        return _IndexParamsBuilder()

    # =========================================================================
    # Score Normalization (for consistent results across backends)
    # =========================================================================

    def normalize_score(
        self,
        raw_score: float,
        metric_type: MetricType = MetricType.COSINE,
        is_distance: bool = False,
    ) -> float:
        """Normalize a raw score to a similarity value (higher = better).

        Converts backend-native scores to a consistent similarity semantic.
        Scores are typically in [0, 1] but may exceed 1.0 for IP metric with
        non-normalized vectors — this is intentional to preserve ranking signal.
        Downstream components (ranker, citation manager) apply their own
        normalization (sigmoid, minmax) for display and filtering.

        Guarantees:
        - Higher is always better
        - 0.0 = no similarity
        - Ordering is preserved (monotonic transformation)

        Args:
            raw_score: The raw score from the backend
            metric_type: The metric used for scoring
            is_distance: True if raw_score is distance (lower=better),
                        False if similarity (higher=better)

        Returns:
            Similarity score (typically [0, 1], may exceed 1.0 for IP)
        """
        if math.isnan(raw_score) or math.isinf(raw_score):
            return 0.0

        if is_distance:
            # Convert distance to similarity
            if metric_type == MetricType.L2:
                # L2 distance: 0 = identical, infinity = very different
                # Use inverse: 1 / (1 + distance)  →  always in (0, 1]
                return 1.0 / (1.0 + raw_score)
            elif metric_type == MetricType.COSINE:
                # Cosine distance: 0 = identical, 2 = opposite
                # Convert: 1 - (distance / 2)  →  [0, 1]
                return max(0.0, 1.0 - (raw_score / 2.0))
            else:
                # IP distance (rare): use inverse  →  always in (0, 1]
                return 1.0 / (1.0 + abs(raw_score))
        else:
            # Already similarity (higher = better)
            if metric_type == MetricType.COSINE:
                # Cosine similarity: [-1, 1] → [0, 1]
                # Apply uniform transformation to preserve monotonicity.
                # Backends returning [0,1] will map to [0.5,1], which is
                # fine — ranking order is preserved, and downstream
                # min-max normalization restores the full [0,1] range.
                return (raw_score + 1.0) / 2.0
            elif metric_type == MetricType.IP:
                # Inner product: unbounded for non-normalized vectors.
                # Preserve the raw value to maintain ranking signal;
                # downstream (ranker/citation) re-normalizes for display.
                return max(0.0, raw_score)
            else:
                # L2 similarity (rare): expected [0, 1]
                return max(0.0, raw_score)

    # =========================================================================
    # Capability-Aware Feature Methods
    # =========================================================================

    def supports_feature(self, feature: str) -> bool:
        """Check if this backend supports a specific feature.

        Args:
            feature: Feature name (e.g., 'hybrid_search', 'json_filters')

        Returns:
            True if feature is supported
        """
        caps = self.capabilities
        feature_map = {
            "hybrid_search": caps.supports_hybrid_search,
            "native_bm25": caps.supports_native_bm25,
            "sparse_vectors": caps.supports_sparse_vectors,
            "hnsw": caps.supports_hnsw,
            "ivf": caps.supports_ivf,
            "gpu": caps.supports_gpu,
            "json_filters": caps.supports_json_filters,
            "metadata_filtering": caps.supports_metadata_filtering,
            "async_insert": caps.supports_async_insert,
            "dynamic_schema": caps.supports_dynamic_schema,
        }
        return feature_map.get(feature, False)

    def get_effective_batch_size(self, requested: int) -> int:
        """Get the effective batch size respecting backend limits.

        Args:
            requested: Requested batch size

        Returns:
            Effective batch size (min of requested and max supported)
        """
        return min(requested, self.capabilities.max_batch_size)

    # =========================================================================
    # Lifecycle Operations (optional, default no-ops)
    # =========================================================================

    def flush(self, collection_name: str) -> None:
        """Flush pending writes to storage.

        Default is no-op. Override for backends that need explicit flush.
        """
        pass

    def compact(self, collection_name: str) -> None:
        """Compact collection storage.

        Default is no-op. Override for backends that support compaction.
        """
        pass


# =========================================================================
# Deterministic UUID Namespace — shared across backends that need UUID IDs
# =========================================================================

LASAGNA_UUID_NAMESPACE = uuid.UUID("7c5e3a4b-9f12-4d8e-a1c3-6b2e8f0d5a7e")


class _IndexParamsBuilder:
    """Generic index parameters builder for non-Milvus backends."""

    def __init__(self):
        self._params: List[Dict[str, Any]] = []

    def add_index(
        self,
        field_name: str,
        index_type: str = "HNSW",
        metric_type: str = "COSINE",
        params: Optional[Dict[str, Any]] = None,
    ) -> "_IndexParamsBuilder":
        """Add an index configuration."""
        self._params.append(
            {
                "field_name": field_name,
                "index_type": index_type,
                "metric_type": metric_type,
                "params": params or {},
            }
        )
        return self

    def to_list(self) -> List[Dict[str, Any]]:
        """Return list of index configurations."""
        return self._params

    def __iter__(self):
        return iter(self._params)
