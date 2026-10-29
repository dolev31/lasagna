"""Qdrant Vector Database Backend.

This module implements the VectorStoreAdapter interface for Qdrant,
enabling Lasagna to use Qdrant as a vector database backend.

Qdrant is an open-source vector similarity search engine with filtering support.

Requirements:
    pip install qdrant-client

Configuration:
    - url: Qdrant server URL (default: http://localhost:6333)
    - api_key: API key for Qdrant Cloud (optional)
    - collection_name: Default collection name (optional)
    - prefer_grpc: Use gRPC instead of REST (optional)
"""

from __future__ import annotations

import threading
import uuid
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from lasagna.core.provenance import redact_url

from .base import (
    BackendCapabilities,
    LASAGNA_UUID_NAMESPACE,
    CollectionExistsError,
    CollectionInfo,
    CollectionSchema,
    IndexConfig,
    MetricType,
    SearchParams,
    VectorSearchResult,
    VectorStoreAdapter,
)
from .filters import FilterExpression

# Check if qdrant-client is available
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
        Filter,
        FieldCondition,
        MatchValue,
        MatchAny,
        Range,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None


class QdrantBackend(VectorStoreAdapter):
    """Qdrant vector database backend.

    Supports both local (in-memory/file) and cloud deployments.

    Example:
        # Initialize for local development
        backend = QdrantBackend(url="http://localhost:6333")

        # Initialize for Qdrant Cloud
        backend = QdrantBackend(
            url="https://your-cluster.cloud.qdrant.io",
            api_key="your-api-key"
        )

        # Connect and use
        backend.connect()
        results = backend.search(
            collection_name="my-collection",
            query_embedding=[0.1, 0.2, ...],
            limit=10
        )
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        path: Optional[str] = None,  # For local file-based storage
        **kwargs,
    ):
        """Initialize Qdrant backend.

        Args:
            url: Qdrant server URL (e.g., http://localhost:6333)
            api_key: API key for Qdrant Cloud
            host: Qdrant server host (alternative to url)
            port: REST API port (default: 6333)
            grpc_port: gRPC port (default: 6334)
            prefer_grpc: Use gRPC instead of REST API
            path: Path for local file-based storage
            **kwargs: Additional QdrantClient arguments
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required for QdrantBackend. "
                "Install it with: pip install qdrant-client"
            )

        self._url = url
        self._api_key = api_key
        self._host = host
        self._port = port
        self._grpc_port = grpc_port
        self._prefer_grpc = prefer_grpc
        self._path = path
        self._kwargs = kwargs

        self._client: Optional[QdrantClient] = None
        self._capabilities: Optional[BackendCapabilities] = None
        self._connection_lock = threading.Lock()

    # =========================================================================
    # Capability Detection
    # =========================================================================

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return Qdrant capabilities."""
        if self._capabilities is None:
            self._capabilities = BackendCapabilities(
                supports_hybrid_search=False,  # Sparse-dense hybrid not yet wired up
                supports_native_bm25=False,  # Requires sparse vector configuration
                supports_sparse_vectors=True,
                supports_hnsw=True,
                supports_ivf=False,
                supports_gpu=False,
                supports_json_filters=True,
                supports_metadata_filtering=True,
                max_filter_complexity=100,
                max_batch_size=1000,
                supports_async_insert=True,
                supports_dynamic_schema=True,
                max_varchar_length=65535,
                max_vector_dimension=65536,
            )
        return self._capabilities

    @property
    def backend_name(self) -> str:
        """Return backend name."""
        return "qdrant"

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> None:
        """Connect to Qdrant."""
        if self._client is not None:
            return

        with self._connection_lock:
            if self._client is not None:
                return
            self._connect_internal()

    def _connect_internal(self) -> None:
        """Internal connection logic (called under lock)."""
        try:
            if self._path:
                # Local file-based storage
                logger.info(f"Connecting to Qdrant (local): {self._path}")
                self._client = QdrantClient(path=self._path)
            elif self._url:
                # URL-based connection
                logger.info(f"Connecting to Qdrant: {redact_url(self._url)}")
                self._client = QdrantClient(
                    url=self._url,
                    api_key=self._api_key,
                    prefer_grpc=self._prefer_grpc,
                    **self._kwargs,
                )
            elif self._host:
                # Host/port connection
                logger.info(f"Connecting to Qdrant: {self._host}:{self._port}")
                self._client = QdrantClient(
                    host=self._host,
                    port=self._port,
                    grpc_port=self._grpc_port,
                    api_key=self._api_key,
                    prefer_grpc=self._prefer_grpc,
                    **self._kwargs,
                )
            else:
                # Default: in-memory
                logger.info("Connecting to Qdrant (in-memory)")
                self._client = QdrantClient(":memory:")

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from Qdrant."""
        if self._client:
            self._client.close()
            self._client = None
        logger.debug("Disconnected from Qdrant")

    def is_connected(self) -> bool:
        """Check if connected to Qdrant."""
        return self._client is not None

    def _ensure_connected(self) -> QdrantClient:
        """Ensure connection and return client."""
        if self._client is None:
            self.connect()
        return self._client

    # =========================================================================
    # Collection Management
    # =========================================================================

    def has_collection(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        client = self._ensure_connected()
        collections = client.get_collections().collections
        return any(c.name == collection_name for c in collections)

    def create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_config: Optional[IndexConfig] = None,
    ) -> None:
        """Create a new Qdrant collection."""
        client = self._ensure_connected()

        if self.has_collection(collection_name):
            raise CollectionExistsError(f"Collection {collection_name} already exists")

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
            MetricType.COSINE: Distance.COSINE,
            MetricType.L2: Distance.EUCLID,
            MetricType.IP: Distance.DOT,
        }

        distance = Distance.COSINE
        if index_config:
            distance = metric_map.get(index_config.metric_type, Distance.COSINE)

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=distance,
            ),
        )

        logger.info(f"Created Qdrant collection: {collection_name}")

    def drop_collection(self, collection_name: str) -> None:
        """Delete a Qdrant collection."""
        client = self._ensure_connected()

        if not self.has_collection(collection_name):
            logger.warning(f"Collection {collection_name} does not exist")
            return

        client.delete_collection(collection_name)
        logger.info(f"Deleted Qdrant collection: {collection_name}")

    def describe_collection(self, collection_name: str) -> CollectionInfo:
        """Get information about a Qdrant collection."""
        client = self._ensure_connected()

        if not self.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} not found")

        info = client.get_collection(collection_name)

        return CollectionInfo(
            name=collection_name,
            fields=[
                {"name": "id", "type": "VARCHAR"},
                {
                    "name": "vector",
                    "type": "FLOAT_VECTOR",
                    "dimension": info.config.params.vectors.size,
                },
                {"name": "payload", "type": "JSON"},
            ],
            row_count=info.points_count,
            description=f"Qdrant collection with {info.points_count} vectors",
        )

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a Qdrant collection."""
        client = self._ensure_connected()
        info = client.get_collection(collection_name)

        return {
            "row_count": info.points_count,
            "vectors_count": getattr(info, "vectors_count", info.points_count),
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
            "status": info.status.value if info.status else "unknown",
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
        """Create a payload index for filtering."""
        client = self._ensure_connected()

        # Qdrant auto-indexes vectors; this creates payload indexes
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema="keyword",  # or "integer", "float", "text"
        )
        logger.debug(f"Created payload index on {field_name} in {collection_name}")

    def list_indexes(self, collection_name: str) -> List[str]:
        """List indexed fields in a collection."""
        client = self._ensure_connected()
        info = client.get_collection(collection_name)

        # Return payload index fields
        if info.payload_schema:
            return list(info.payload_schema.keys())
        return []

    # =========================================================================
    # Data Operations
    # =========================================================================

    def insert(
        self,
        collection_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        partition_key: Optional[str] = None,
    ) -> List[Union[int, str]]:
        """Insert vectors into Qdrant.

        Respects max_batch_size from capabilities for optimal performance.
        """
        client = self._ensure_connected()

        if isinstance(data, dict):
            data = [data]

        points = []
        ids = []

        for item in data:
            # Generate or use provided ID
            original_id = item.get("id") or item.get("chunk_id") or str(uuid.uuid4())
            ids.append(original_id)

            # Qdrant requires UUID-format string IDs - convert non-UUID strings
            point_id = original_id
            if isinstance(point_id, str):
                try:
                    uuid.UUID(point_id)  # Already valid UUID
                except ValueError:
                    # Convert non-UUID string to deterministic UUID
                    point_id = str(uuid.uuid5(LASAGNA_UUID_NAMESPACE, point_id))

            # Extract vector
            vector = item.get("embedding") or item.get("vector")
            if vector is None:
                raise ValueError("Each record must have 'embedding' or 'vector'")

            # Build payload (everything except id and vector)
            # Keep dicts nested (e.g. metadata) for consistent filter access across backends
            # Include chunk_id in payload so it's available in search/query results
            payload = {}
            for key, value in item.items():
                if key not in ("id", "embedding", "vector"):
                    payload[key] = value
            # Ensure chunk_id is always in payload
            if "chunk_id" not in payload and original_id:
                payload["chunk_id"] = original_id

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        # Batch upsert respecting max_batch_size
        batch_size = self.get_effective_batch_size(len(points))
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch,
            )

        logger.debug(f"Inserted {len(points)} vectors into Qdrant collection {collection_name}")
        return ids

    def delete(
        self,
        collection_name: str,
        filter_expr: FilterExpression,
        partition_key: Optional[str] = None,
    ) -> int:
        """Delete vectors from Qdrant."""
        client = self._ensure_connected()

        # Convert filter to Qdrant format
        qdrant_filter = self._build_qdrant_filter(filter_expr)

        if not qdrant_filter:
            raise ValueError(
                "Cannot delete with empty filter: this would be a no-op. "
                "Build a FilterExpression with at least one condition."
            )

        client.delete(
            collection_name=collection_name,
            points_selector=qdrant_filter,
        )

        # Qdrant delete API doesn't return affected count — return -1 to signal success
        # without a count. Callers should treat -1 as "deletion submitted successfully"
        # and verify via subsequent query if exact count is needed.
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
        """Query vectors by filter (scroll through points)."""
        client = self._ensure_connected()

        qdrant_filter = None
        if filter_expr:
            qdrant_filter = self._build_qdrant_filter(filter_expr)

        # Qdrant's scroll() paginates by cursor: its `offset` argument is a
        # *point ID to resume from*, not a row count. Passing the caller's
        # numeric offset straight through meant it was silently meaningless —
        # every page returned the same first `limit` rows, and an offset past
        # the end still returned data. Callers that page until exhaustion
        # (base.list_document_ids advances by `current_offset += len(rows)`)
        # would loop on duplicates forever.
        #
        # The abstract interface promises numeric offsets, so emulate one:
        # over-fetch and drop the skipped prefix. This mirrors what the Milvus
        # backend does and carries the same O(offset) cost.
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=limit + offset,
            offset=None,
            with_payload=True,
            with_vectors=True,
        )
        if offset:
            points = points[offset:]

        results = []
        for point in points:
            payload = point.payload or {}
            result = {
                "id": payload.get("chunk_id", point.id),
                "embedding": point.vector,
                **payload,
            }
            results.append(result)

        return results

    # =========================================================================
    # Vector Search
    # =========================================================================

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        vector_field: str = "vector",
        filter_expr: Optional[FilterExpression] = None,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[SearchParams] = None,
        partition_key: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Perform vector similarity search in Qdrant."""
        client = self._ensure_connected()

        # Determine metric from search_params (default COSINE)
        metric = MetricType.COSINE
        if search_params and search_params.metric_type:
            metric = search_params.metric_type

        # Build Qdrant filter
        qdrant_filter = None
        if filter_expr:
            qdrant_filter = self._build_qdrant_filter(filter_expr)

        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        search_results = []
        for hit in response.points:
            payload = hit.payload or {}

            # Use chunk_id from payload as the canonical ID (falls back to point UUID)
            result_id = payload.get("chunk_id", str(hit.id))

            # Build entity from all payload fields
            entity = dict(payload)
            entity["id"] = result_id
            if hit.vector is not None:
                entity["embedding"] = hit.vector

            # Qdrant query_points score semantics by metric:
            #   COSINE → similarity [0, 1] (higher=better)
            #   L2     → negative distance ≤ 0 (higher=closer)
            #   IP     → raw dot product (higher=better)
            if metric == MetricType.L2:
                # Negate to get positive distance, then convert distance→similarity
                score = self.normalize_score(abs(hit.score), metric, is_distance=True)
            else:
                score = self.normalize_score(hit.score, metric, is_distance=False)

            search_results.append(
                VectorSearchResult(
                    id=result_id,
                    score=score,
                    entity=entity,
                )
            )

        return search_results

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_embedding: List[float],
        vector_field: str = "vector",
        sparse_field: str = "sparse_vector",
        filter_expr: Optional[FilterExpression] = None,
        limit: int = 10,
        alpha: float = 0.7,
        output_fields: Optional[List[str]] = None,
        partition_key: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Perform hybrid search using Qdrant sparse-dense.

        Qdrant supports sparse vectors for BM25-like retrieval.
        This requires the collection to have sparse vector configuration.
        """
        # Not implemented. This used to log a warning and return dense-only
        # results, which is the worst available option for an evaluation
        # framework: a hybrid-vs-dense comparison on Qdrant compared dense
        # against dense, produced a difference of ~0, and looked like a finding.
        # The caller asked a question this backend cannot answer, so say so.
        raise NotImplementedError(
            "Qdrant hybrid search is not implemented: it requires sparse "
            "vector configuration that this backend does not yet set up. "
            "Returning dense-only results would silently answer a different "
            "question than the one asked. Use hybrid on milvus, pgvector or "
            "opensearch, or set enable_hybrid_search=False for dense retrieval "
            "on Qdrant."
        )

    def _build_qdrant_filter(self, filter_expr: FilterExpression) -> Optional[Filter]:
        """Build Qdrant filter from FilterExpression."""
        if filter_expr.is_empty():
            return None

        # Use the filter's to_qdrant method which returns a dict
        filter_dict = filter_expr.to_qdrant()
        if not filter_dict:
            return None

        # Convert dict to Qdrant Filter object
        return self._dict_to_qdrant_filter(filter_dict)

    def _dict_to_qdrant_filter(self, filter_dict: Dict[str, Any]) -> Filter:
        """Convert filter dict to Qdrant Filter object."""
        must = []
        should = []
        must_not = []

        if "must" in filter_dict:
            for item in filter_dict["must"]:
                must.append(self._build_condition(item))

        if "should" in filter_dict:
            for item in filter_dict["should"]:
                should.append(self._build_condition(item))

        if "must_not" in filter_dict:
            for item in filter_dict["must_not"]:
                must_not.append(self._build_condition(item))

        # Single condition at top level
        if "key" in filter_dict:
            must.append(self._build_condition(filter_dict))

        return Filter(
            must=must if must else None,
            should=should if should else None,
            must_not=must_not if must_not else None,
        )

    def _build_condition(self, cond: Dict[str, Any]) -> FieldCondition:
        """Build a single Qdrant condition."""
        key = cond.get("key", "")

        if "match" in cond:
            match = cond["match"]
            if "value" in match:
                return FieldCondition(key=key, match=MatchValue(value=match["value"]))
            if "any" in match:
                return FieldCondition(key=key, match=MatchAny(any=match["any"]))

        if "range" in cond:
            r = cond["range"]
            return FieldCondition(
                key=key,
                range=Range(
                    gt=r.get("gt"),
                    gte=r.get("gte"),
                    lt=r.get("lt"),
                    lte=r.get("lte"),
                ),
            )

        # Handle nested compound filters (must/should/must_not in condition position)
        if "must" in cond or "should" in cond or "must_not" in cond:
            return self._dict_to_qdrant_filter(cond)

        raise ValueError(
            f"Unrecognized Qdrant filter condition structure: {cond!r}. "
            f"Expected 'match' or 'range' key."
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def flush(self, collection_name: str) -> None:
        """Qdrant handles persistence automatically."""
        pass

    def compact(self, collection_name: str) -> None:
        """Qdrant handles compaction automatically."""
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

        Converts Milvus-style search parameters to Qdrant format.
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
                        "distance": 1.0 - r.score,  # Convert similarity to distance
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

    # get_collection_stats is defined in the main Collection Management section above
