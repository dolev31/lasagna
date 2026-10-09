"""Pinecone Vector Database Backend.

This module implements the VectorStoreAdapter interface for Pinecone,
enabling Lasagna to use Pinecone as a vector database backend.

Pinecone is a fully managed, cloud-native vector database service.

Requirements:
    pip install pinecone-client

Configuration:
    - api_key: Pinecone API key (required)
    - environment: Pinecone environment (optional, for pod-based)
    - index_name: Default index name (optional)
    - namespace: Default namespace for multi-tenancy (optional)
"""

import threading
import uuid
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from .base import (
    BackendCapabilities,
    CollectionInfo,
    CollectionSchema,
    IndexConfig,
    MetricType,
    SearchParams,
    VectorSearchResult,
    VectorStoreAdapter,
)
from .filters import FilterExpression

# Check if pinecone is available
try:
    from pinecone import Pinecone, ServerlessSpec

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    Pinecone = None
    ServerlessSpec = None


class PineconeBackend(VectorStoreAdapter):
    """Pinecone vector database backend.

    Supports Pinecone Serverless and Pod-based deployments.

    Example:
        # Initialize with API key
        backend = PineconeBackend(
            api_key="your-api-key",
            index_name="my-index"
        )

        # Connect and use
        backend.connect()
        results = backend.search(
            collection_name="my-index",
            query_embedding=[0.1, 0.2, ...],
            limit=10
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        namespace: str = "",
        cloud: str = "aws",
        region: str = "us-east-1",
        **kwargs,
    ):
        """Initialize Pinecone backend.

        Args:
            api_key: Pinecone API key (or set PINECONE_API_KEY env var)
            environment: Pinecone environment (for pod-based, deprecated)
            index_name: Default index name
            namespace: Default namespace for multi-tenancy
            cloud: Cloud provider for serverless (aws, gcp, azure)
            region: Region for serverless deployment
            **kwargs: Additional Pinecone client arguments
        """
        if not PINECONE_AVAILABLE:
            raise ImportError(
                "pinecone-client is required for PineconeBackend. "
                "Install it with: pip install pinecone-client"
            )

        self._api_key = api_key
        self._environment = environment
        self._default_index_name = index_name
        self._default_namespace = namespace
        self._cloud = cloud
        self._region = region
        self._kwargs = kwargs

        self._client: Optional[Pinecone] = None
        self._indexes: Dict[str, Any] = {}  # Cache of index objects
        self._capabilities: Optional[BackendCapabilities] = None
        self._connection_lock = threading.Lock()

    # =========================================================================
    # Capability Detection
    # =========================================================================

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return Pinecone capabilities."""
        if self._capabilities is None:
            self._capabilities = BackendCapabilities(
                supports_hybrid_search=False,  # Sparse encoder not yet wired up
                supports_native_bm25=False,  # Requires sparse encoding configuration
                supports_sparse_vectors=True,
                supports_hnsw=True,
                supports_ivf=False,
                supports_gpu=False,  # Managed service
                supports_json_filters=True,
                supports_metadata_filtering=True,
                max_filter_complexity=100,
                max_batch_size=100,  # Pinecone recommends smaller batches
                supports_async_insert=True,
                supports_partitions=True,  # Pinecone namespaces = partitions
                supports_dynamic_schema=True,
                max_varchar_length=40960,  # 40KB metadata per vector
                max_vector_dimension=20000,
            )
        return self._capabilities

    @property
    def backend_name(self) -> str:
        """Return backend name."""
        return "pinecone"

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> None:
        """Connect to Pinecone."""
        if self._client is not None:
            return

        with self._connection_lock:
            if self._client is not None:
                return

            try:
                import os

                api_key = self._api_key or os.environ.get("PINECONE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "Pinecone API key is required. "
                        "Pass api_key or set PINECONE_API_KEY environment variable."
                    )

                self._client = Pinecone(api_key=api_key, **self._kwargs)
                logger.info("Connected to Pinecone")

            except Exception as e:
                raise ConnectionError(f"Failed to connect to Pinecone: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from Pinecone."""
        self._indexes.clear()
        self._client = None
        logger.debug("Disconnected from Pinecone")

    def is_connected(self) -> bool:
        """Check if connected to Pinecone."""
        return self._client is not None

    def _ensure_connected(self) -> Pinecone:
        """Ensure connection and return client."""
        if self._client is None:
            self.connect()
        return self._client

    def _get_index(self, name: str):
        """Get or create index connection."""
        if name not in self._indexes:
            client = self._ensure_connected()
            self._indexes[name] = client.Index(name)
        return self._indexes[name]

    # =========================================================================
    # Collection Management
    # =========================================================================

    def has_collection(self, collection_name: str) -> bool:
        """Check if an index exists."""
        client = self._ensure_connected()
        existing = [idx.name for idx in client.list_indexes()]
        return collection_name in existing

    def create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_config: Optional[IndexConfig] = None,
    ) -> None:
        """Create a new Pinecone index."""
        client = self._ensure_connected()

        if self.has_collection(collection_name):
            raise ValueError(f"Index {collection_name} already exists")

        # Find vector dimension from schema
        dimension = None
        for field in schema.fields:
            if field.dtype == "FLOAT_VECTOR" and field.dimension:
                dimension = field.dimension
                break

        if dimension is None:
            raise ValueError("Schema must include a FLOAT_VECTOR field with dimension")

        # Map metric types
        metric_map = {
            MetricType.COSINE: "cosine",
            MetricType.L2: "euclidean",
            MetricType.IP: "dotproduct",
        }

        metric = "cosine"
        if index_config:
            metric = metric_map.get(index_config.metric_type, "cosine")

        # Create serverless index
        client.create_index(
            name=collection_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=self._cloud,
                region=self._region,
            ),
        )

        logger.info(f"Created Pinecone index: {collection_name}")

    def drop_collection(self, collection_name: str) -> None:
        """Delete a Pinecone index."""
        client = self._ensure_connected()

        if not self.has_collection(collection_name):
            logger.warning(f"Index {collection_name} does not exist")
            return

        client.delete_index(collection_name)
        if collection_name in self._indexes:
            del self._indexes[collection_name]

        logger.info(f"Deleted Pinecone index: {collection_name}")

    def describe_collection(self, collection_name: str) -> CollectionInfo:
        """Get information about a Pinecone index."""
        if not self.has_collection(collection_name):
            raise ValueError(f"Index {collection_name} not found")

        index = self._get_index(collection_name)
        stats = index.describe_index_stats()

        return CollectionInfo(
            name=collection_name,
            fields=[
                {"name": "id", "type": "VARCHAR"},
                {"name": "values", "type": "FLOAT_VECTOR", "dimension": stats.dimension},
                {"name": "metadata", "type": "JSON"},
            ],
            row_count=stats.total_vector_count,
            description=f"Pinecone index with {stats.total_vector_count} vectors",
        )

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a Pinecone index."""
        index = self._get_index(collection_name)
        stats = index.describe_index_stats()

        return {
            "row_count": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
        }

    # =========================================================================
    # Index Management
    # =========================================================================

    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_config: IndexConfig,
    ) -> None:
        """Pinecone indexes are created with the collection."""
        # Pinecone automatically indexes vectors
        logger.debug(f"Pinecone auto-indexes vectors for {collection_name}")

    def list_indexes(self, collection_name: str) -> List[str]:
        """List indexed fields (Pinecone always indexes vectors)."""
        return ["values"]  # Pinecone's vector field

    # =========================================================================
    # Data Operations
    # =========================================================================

    def insert(
        self,
        collection_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        partition_key: Optional[str] = None,
    ) -> List[Union[int, str]]:
        """Insert vectors into Pinecone."""
        index = self._get_index(collection_name)

        if isinstance(data, dict):
            data = [data]

        # Convert to Pinecone format
        records = []
        ids = []
        for item in data:
            # Generate or use provided ID - Pinecone requires string IDs
            record_id = item.get("id") or item.get("chunk_id") or item.get("document_id")
            if not record_id:
                record_id = str(uuid.uuid4())
            record_id = str(record_id)
            ids.append(record_id)

            # Extract vector (could be 'embedding' or 'values')
            vector = item.get("embedding") or item.get("values")
            if vector is None:
                raise ValueError("Each record must have 'embedding' or 'values'")

            # Build metadata (everything except id and vector)
            # Pinecone metadata must be flat, so we flatten nested dicts
            metadata = {}

            # First, copy top-level fields
            for key, value in item.items():
                if key in ("id", "chunk_id", "embedding", "values"):
                    continue
                if key == "metadata" and isinstance(value, dict):
                    # Flatten nested metadata dict - these are the RAG chunk metadata
                    for k, v in value.items():
                        if isinstance(v, (str, int, float, bool)):
                            metadata[k] = v
                        elif isinstance(v, list) and all(
                            isinstance(x, (str, int, float, bool)) for x in v
                        ):
                            metadata[k] = v
                elif isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                elif isinstance(value, list) and all(
                    isinstance(x, (str, int, float, bool)) for x in value
                ):
                    metadata[key] = value

            records.append(
                {
                    "id": record_id,
                    "values": vector,
                    "metadata": metadata,
                }
            )

        # Batch upsert (Pinecone recommends batches of 100)
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            index.upsert(vectors=batch, namespace=partition_key or self._default_namespace)

        logger.debug(f"Inserted {len(records)} vectors into Pinecone index {collection_name}")
        return ids

    def delete(
        self,
        collection_name: str,
        filter_expr: FilterExpression,
        partition_key: Optional[str] = None,
    ) -> int:
        """Delete vectors from Pinecone."""
        index = self._get_index(collection_name)

        # Convert filter to Pinecone format
        pc_filter = filter_expr.to_pinecone()

        if not pc_filter:
            raise ValueError(
                "Cannot delete with empty filter — this would be a no-op. "
                "Build a FilterExpression with at least one condition."
            )

        index.delete(filter=pc_filter, namespace=partition_key or self._default_namespace)

        # Pinecone doesn't return delete count for filter deletes
        return -1

    def query(
        self,
        collection_name: str,
        filter_expr: Optional[FilterExpression] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        partition_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query vectors by filter using zero-vector search.

        Pinecone doesn't support pure metadata queries without a vector.
        This implementation uses a zero vector with the metadata filter to
        achieve similar functionality. Results are ordered by metadata match
        rather than vector similarity.

        Note: For offset support, we fetch offset + limit and slice.
        """
        index = self._get_index(collection_name)

        # Get the dimension from index stats
        stats = index.describe_index_stats()
        dimension = stats.dimension

        if dimension is None or dimension == 0:
            logger.warning(f"Cannot determine dimension for {collection_name}")
            return []

        # Create zero vector for metadata-only query
        zero_vector = [0.0] * dimension

        # Build Pinecone filter
        pc_filter = None
        if filter_expr:
            pc_filter = filter_expr.to_pinecone()

        # Fetch extra for offset support
        fetch_limit = limit + offset

        results = index.query(
            vector=zero_vector,
            top_k=fetch_limit,
            filter=pc_filter,
            include_values=True,
            include_metadata=True,
            namespace=partition_key or self._default_namespace,
        )

        # Convert matches to dict format and apply offset
        entities = []
        for match in results.matches[offset:]:
            raw_metadata = dict(match.metadata) if match.metadata else {}

            # Extract flattened fields back to entity structure
            entity = {
                "id": match.id,
                "document_id": raw_metadata.pop("document_id", ""),
                "content": raw_metadata.pop("content", ""),
                "metadata": raw_metadata,
            }

            # Include vector if values is requested
            if output_fields and "values" in output_fields:
                entity["values"] = list(match.values) if match.values else []

            entities.append(entity)

        return entities

    # =========================================================================
    # Vector Search
    # =========================================================================

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        vector_field: str = "values",
        filter_expr: Optional[FilterExpression] = None,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[SearchParams] = None,
        partition_key: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Perform vector similarity search in Pinecone."""
        index = self._get_index(collection_name)

        # Determine metric from search_params (default COSINE)
        metric = MetricType.COSINE
        if search_params and search_params.metric_type:
            metric = search_params.metric_type

        # Build Pinecone filter
        pc_filter = None
        if filter_expr:
            pc_filter = filter_expr.to_pinecone()

        results = index.query(
            vector=query_embedding,
            top_k=limit,
            filter=pc_filter,
            include_values=True,
            include_metadata=True,
            namespace=partition_key or self._default_namespace,
        )

        search_results = []
        for match in results.matches:
            raw_metadata = dict(match.metadata) if match.metadata else {}

            # Extract top-level fields that were flattened during insert
            content = raw_metadata.pop("content", "")
            document_id = raw_metadata.pop("document_id", "")

            # Pinecone returns similarity scores (higher=better)
            search_results.append(
                VectorSearchResult(
                    id=match.id,
                    score=self.normalize_score(match.score, metric, is_distance=False),
                    entity={
                        "id": match.id,
                        "content": content,
                        "embedding": match.values,
                        "metadata": raw_metadata,  # session_id, chunk_index, etc.
                        "document_id": document_id,
                    },
                )
            )

        return search_results

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_embedding: List[float],
        vector_field: str = "values",
        sparse_field: str = "sparse_values",
        filter_expr: Optional[FilterExpression] = None,
        limit: int = 10,
        alpha: float = 0.7,
        output_fields: Optional[List[str]] = None,
        partition_key: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Perform hybrid search using Pinecone sparse-dense.

        Note: This requires the index to be configured for sparse-dense
        and a sparse encoder to be provided. For simplicity, this falls
        back to dense-only search unless sparse vectors are provided.
        """
        # TODO: Implement sparse encoding for true hybrid search
        # For now, fall back to dense search
        logger.warning(
            "Pinecone hybrid search requires sparse encoder configuration. Using dense-only search."
        )
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
    # Partition Management (Pinecone namespaces)
    # =========================================================================

    def has_partition(self, collection_name: str, partition_name: str) -> bool:
        """Check if a partition (namespace) exists.

        Pinecone namespaces are created implicitly on first write, so this
        always returns True.  Callers can safely proceed without a
        prior create_partition call.
        """
        return True

    def create_partition(self, collection_name: str, partition_name: str) -> None:
        """Create a partition (namespace).

        No-op for Pinecone -- namespaces are auto-created on first upsert.
        """
        logger.debug(
            f"Pinecone namespaces are auto-created; "
            f"skipping explicit creation of '{partition_name}' "
            f"in index '{collection_name}'"
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def flush(self, collection_name: str) -> None:
        """Pinecone handles persistence automatically."""
        pass

    def compact(self, collection_name: str) -> None:
        """Pinecone handles compaction automatically."""
        pass

    # =========================================================================
    # Legacy Compatibility (for RAG backend migration)
    # =========================================================================

    def search_raw(
        self,
        collection_name: str,
        data: List[List[float]],
        anns_field: str,
        filter: str,
        limit: int,
        output_fields: List[str],
        search_params: Dict[str, Any],
    ) -> List[List[Dict[str, Any]]]:
        """Raw search for legacy RAG backend compatibility.

        Converts Milvus-style search parameters to Pinecone format.
        Returns results in Milvus-style nested list format.
        """
        # Parse legacy Milvus filter string to FilterExpression
        filter_expr = None
        if filter:
            from .filters import parse_legacy_milvus_filter

            filter_expr = parse_legacy_milvus_filter(filter)

        # Perform search for each query vector
        all_results = []
        for query_vec in data:
            results = self.search(
                collection_name=collection_name,
                query_embedding=query_vec,
                filter_expr=filter_expr,
                limit=limit,
                output_fields=output_fields,
            )
            # Convert to Milvus-style result format
            milvus_results = []
            for r in results:
                milvus_results.append(
                    {
                        "id": r.id,
                        "distance": 1.0 - r.score,  # Pinecone uses similarity, Milvus uses distance
                        "entity": r.entity,
                    }
                )
            all_results.append(milvus_results)

        return all_results

    def insert_raw(
        self,
        collection_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> Any:
        """Raw insert for legacy RAG backend compatibility."""
        return self.insert(collection_name, data)

    def delete_raw(self, collection_name: str, filter: str) -> Any:
        """Raw delete for legacy RAG backend compatibility."""
        from .filters import parse_legacy_milvus_filter

        filter_expr = parse_legacy_milvus_filter(filter)
        if filter_expr:
            return self.delete(collection_name, filter_expr)
        return 0

    def query_raw(
        self,
        collection_name: str,
        filter: str,
        output_fields: List[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Raw query for legacy RAG backend compatibility."""
        from .filters import parse_legacy_milvus_filter

        filter_expr = parse_legacy_milvus_filter(filter)
        return self.query(collection_name, filter_expr, output_fields, limit)
