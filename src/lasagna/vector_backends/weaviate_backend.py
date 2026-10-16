"""Weaviate Vector Database Backend.

This module implements the VectorStoreAdapter interface for Weaviate,
enabling Lasagna to use Weaviate as a vector database backend.

Weaviate is an AI-native vector database with built-in vectorization modules.

Requirements:
    pip install weaviate-client

Configuration:
    - url: Weaviate server URL (default: http://localhost:8080)
    - api_key: API key for Weaviate Cloud (optional)
    - additional_headers: Additional headers for authentication (optional)
"""

import threading
import uuid
from typing import Any, Dict, List, Optional, Union

# UUID namespace imported from base for consistent ID generation across backends.

from loguru import logger
from lasagna.core.provenance import redact_url

from .base import (
    BackendCapabilities,
    LASAGNA_UUID_NAMESPACE,
    CollectionInfo,
    CollectionSchema,
    IndexConfig,
    MetricType,
    SearchParams,
    VectorSearchResult,
    VectorStoreAdapter,
)
from .filters import FilterExpression

# Check if weaviate-client is available
try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType, VectorDistances
    from weaviate.classes.query import MetadataQuery
    from weaviate.collections.classes.config_vectors import _VectorConfigCreate

    WEAVIATE_AVAILABLE = True
    WEAVIATE_V4 = True  # Using v4 client
except ImportError:
    try:
        # Try v3 client
        import weaviate

        WEAVIATE_AVAILABLE = True
        WEAVIATE_V4 = False
    except ImportError:
        WEAVIATE_AVAILABLE = False
        WEAVIATE_V4 = False
        weaviate = None


class WeaviateBackend(VectorStoreAdapter):
    """Weaviate vector database backend.

    Supports both local and Weaviate Cloud deployments.

    Example:
        # Initialize for local development
        backend = WeaviateBackend(url="http://localhost:8080")

        # Initialize for Weaviate Cloud
        backend = WeaviateBackend(
            url="https://your-cluster.weaviate.cloud",
            api_key="your-api-key"
        )

        # Connect and use
        backend.connect()
        results = backend.search(
            collection_name="Document",
            query_embedding=[0.1, 0.2, ...],
            limit=10
        )
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        embedded: bool = False,
        **kwargs,
    ):
        """Initialize Weaviate backend.

        Args:
            url: Weaviate server URL (e.g., http://localhost:8080)
            api_key: API key for Weaviate Cloud
            additional_headers: Additional headers (e.g., for OpenAI API key)
            embedded: Use embedded Weaviate (for development)
            **kwargs: Additional client arguments
        """
        if not WEAVIATE_AVAILABLE:
            raise ImportError(
                "weaviate-client is required for WeaviateBackend. "
                "Install it with: pip install weaviate-client"
            )

        self._url = url or "http://localhost:8080"
        self._api_key = api_key
        self._additional_headers = additional_headers or {}
        self._embedded = embedded
        self._kwargs = kwargs

        self._client = None
        self._capabilities: Optional[BackendCapabilities] = None
        self._connection_lock = threading.Lock()

    # =========================================================================
    # Capability Detection
    # =========================================================================

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return Weaviate capabilities."""
        if self._capabilities is None:
            self._capabilities = BackendCapabilities(
                supports_hybrid_search=True,
                supports_native_bm25=True,  # Weaviate has built-in BM25
                supports_sparse_vectors=False,  # Uses keyword search instead
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
        return "weaviate"

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> None:
        """Connect to Weaviate."""
        if self._client is not None:
            return

        with self._connection_lock:
            if self._client is not None:
                return

            try:
                if WEAVIATE_V4:
                    self._connect_v4()
                else:
                    self._connect_v3()

                logger.info(f"Connected to Weaviate: {redact_url(self._url)}")

            except Exception as e:
                raise ConnectionError(f"Failed to connect to Weaviate: {e}") from e

    def _connect_v4(self) -> None:
        """Connect using Weaviate v4 client."""
        if self._embedded:
            self._client = weaviate.connect_to_embedded()
        elif self._api_key:
            # Weaviate Cloud
            self._client = weaviate.connect_to_wcs(
                cluster_url=self._url,
                auth_credentials=weaviate.auth.AuthApiKey(self._api_key),
                headers=self._additional_headers,
            )
        else:
            # Local/custom deployment — use urllib for robust URL parsing
            from urllib.parse import urlparse

            parsed = urlparse(self._url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8080
            self._client = weaviate.connect_to_local(
                host=host,
                port=port,
                headers=self._additional_headers,
            )

    def _connect_v3(self) -> None:
        """Connect using Weaviate v3 client (legacy)."""
        auth_config = None
        if self._api_key:
            auth_config = weaviate.AuthApiKey(api_key=self._api_key)

        self._client = weaviate.Client(
            url=self._url,
            auth_client_secret=auth_config,
            additional_headers=self._additional_headers,
        )

    def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        if self._client:
            if WEAVIATE_V4 and hasattr(self._client, "close"):
                self._client.close()
            self._client = None
        logger.debug("Disconnected from Weaviate")

    def is_connected(self) -> bool:
        """Check if connected to Weaviate."""
        if self._client is None:
            return False
        try:
            return self._client.is_ready()
        except Exception:
            return False

    def _ensure_connected(self):
        """Ensure connection and return client."""
        if self._client is None:
            self.connect()
        return self._client

    # =========================================================================
    # Collection Management (Weaviate calls these "Classes")
    # =========================================================================

    def has_collection(self, collection_name: str) -> bool:
        """Check if a class exists."""
        client = self._ensure_connected()

        if WEAVIATE_V4:
            return client.collections.exists(collection_name)
        else:
            schema = client.schema.get()
            return any(c["class"] == collection_name for c in schema.get("classes", []))

    def create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_config: Optional[IndexConfig] = None,
    ) -> None:
        """Create a new Weaviate class."""
        client = self._ensure_connected()

        if self.has_collection(collection_name):
            raise ValueError(f"Class {collection_name} already exists")

        # Find vector dimension from schema
        dimension = None
        for field in schema.fields:
            if field.dtype == "FLOAT_VECTOR" and field.dimension:
                dimension = field.dimension
                break

        if WEAVIATE_V4:
            self._create_collection_v4(client, collection_name, schema, dimension, index_config)
        else:
            self._create_collection_v3(client, collection_name, schema, dimension, index_config)

        logger.info(f"Created Weaviate class: {collection_name}")

    def _create_collection_v4(
        self,
        client,
        collection_name: str,
        schema: CollectionSchema,
        dimension: Optional[int],
        index_config: Optional[IndexConfig],
    ) -> None:
        """Create collection using v4 client."""
        # Build properties from schema
        properties = []
        for field in schema.fields:
            if field.dtype == "FLOAT_VECTOR":
                continue  # Vectors are handled separately
            if field.dtype == "JSON":
                continue  # Skip JSON/OBJECT — Weaviate requires nested property
                # definitions for OBJECT type.  The insert method flattens
                # metadata dicts into top-level properties instead.
            if field.dtype == "INT64":
                properties.append(Property(name=field.name, data_type=DataType.INT))
            elif field.dtype == "VARCHAR":
                properties.append(Property(name=field.name, data_type=DataType.TEXT))
            elif field.dtype == "BOOL":
                properties.append(Property(name=field.name, data_type=DataType.BOOL))
            else:
                logger.warning(
                    f"Weaviate: skipping field '{field.name}' with unsupported "
                    f"dtype '{field.dtype}': not mapped to a Weaviate DataType"
                )

        # Add standard properties
        if not any(p.name == "content" for p in properties):
            properties.append(Property(name="content", data_type=DataType.TEXT))
        if not any(p.name == "document_id" for p in properties):
            properties.append(Property(name="document_id", data_type=DataType.TEXT))
        # Store original non-UUID IDs for lossless round-tripping
        if not any(p.name == "_original_id" for p in properties):
            properties.append(Property(name="_original_id", data_type=DataType.TEXT))

        # Map metric types to VectorDistances enum (weaviate-client v4)
        metric_map = {
            MetricType.COSINE: VectorDistances.COSINE,
            MetricType.L2: VectorDistances.L2_SQUARED,
            MetricType.IP: VectorDistances.DOT,
        }

        distance_metric = VectorDistances.COSINE
        if index_config:
            distance_metric = metric_map.get(index_config.metric_type, VectorDistances.COSINE)

        # Use vector_config (v4.20+ API) instead of deprecated
        # vectorizer_config + vector_index_config
        client.collections.create(
            name=collection_name,
            properties=properties,
            vector_config=_VectorConfigCreate(
                name="default",
                vectorizer=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=distance_metric,
                ),
            ),
        )

    def _create_collection_v3(
        self,
        client,
        collection_name: str,
        schema: CollectionSchema,
        dimension: Optional[int],
        index_config: Optional[IndexConfig],
    ) -> None:
        """Create collection using v3 client (legacy)."""
        # Build properties
        properties = [
            {"name": "content", "dataType": ["text"]},
            {"name": "document_id", "dataType": ["string"]},
            {"name": "chunk_id", "dataType": ["string"]},
            {"name": "_original_id", "dataType": ["string"]},
        ]

        class_obj = {
            "class": collection_name,
            "description": schema.description or "Lasagna document collection",
            "properties": properties,
            "vectorizer": "none",  # We provide our own vectors
        }

        client.schema.create_class(class_obj)

    def drop_collection(self, collection_name: str) -> None:
        """Delete a Weaviate class."""
        client = self._ensure_connected()

        if not self.has_collection(collection_name):
            logger.warning(f"Class {collection_name} does not exist")
            return

        if WEAVIATE_V4:
            client.collections.delete(collection_name)
        else:
            client.schema.delete_class(collection_name)

        logger.info(f"Deleted Weaviate class: {collection_name}")

    def describe_collection(self, collection_name: str) -> CollectionInfo:
        """Get information about a Weaviate class."""
        client = self._ensure_connected()

        if not self.has_collection(collection_name):
            raise ValueError(f"Class {collection_name} not found")

        if WEAVIATE_V4:
            collection = client.collections.get(collection_name)
            config = collection.config.get()
            count = collection.aggregate.over_all(total_count=True).total_count

            return CollectionInfo(
                name=collection_name,
                fields=[{"name": p.name, "type": str(p.data_type)} for p in config.properties],
                row_count=count or 0,
                description=f"Weaviate class with {count} objects",
            )
        else:
            schema = client.schema.get(collection_name)
            count = client.query.aggregate(collection_name).with_meta_count().do()
            total = (
                count.get("data", {})
                .get("Aggregate", {})
                .get(collection_name, [{}])[0]
                .get("meta", {})
                .get("count", 0)
            )

            return CollectionInfo(
                name=collection_name,
                fields=[
                    {"name": p["name"], "type": p["dataType"][0]}
                    for p in schema.get("properties", [])
                ],
                row_count=total,
                description=f"Weaviate class with {total} objects",
            )

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a Weaviate class."""
        info = self.describe_collection(collection_name)
        return {
            "row_count": info.row_count,
            "field_count": len(info.fields),
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
        """Weaviate auto-indexes properties."""
        logger.debug(f"Weaviate auto-indexes properties for {collection_name}")

    def list_indexes(self, collection_name: str) -> List[str]:
        """List indexed fields (Weaviate indexes all properties)."""
        info = self.describe_collection(collection_name)
        return [f["name"] for f in info.fields]

    # =========================================================================
    # Data Operations
    # =========================================================================

    def insert(
        self,
        collection_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        partition_key: Optional[str] = None,
    ) -> List[Union[int, str]]:
        """Insert objects into Weaviate."""
        client = self._ensure_connected()

        if isinstance(data, dict):
            data = [data]

        ids = []

        if WEAVIATE_V4:
            collection = client.collections.get(collection_name)
            objects = []

            for item in data:
                original_id = item.get("id") or item.get("chunk_id") or str(uuid.uuid4())
                ids.append(original_id)

                # Weaviate requires valid UUIDs — convert non-UUID IDs
                # deterministically using uuid5
                try:
                    weaviate_uuid = str(uuid.UUID(original_id))
                except (ValueError, AttributeError):
                    weaviate_uuid = str(uuid.uuid5(LASAGNA_UUID_NAMESPACE, original_id))

                vector = item.get("embedding") or item.get("vector")

                # Build properties — store original ID for retrieval
                properties = {"_original_id": original_id}
                for key, value in item.items():
                    if key not in ("id", "chunk_id", "embedding", "vector"):
                        if isinstance(value, dict):
                            # Flatten metadata
                            for k, v in value.items():
                                properties[k] = v
                        else:
                            properties[key] = value

                objects.append(
                    {
                        "uuid": weaviate_uuid,
                        "properties": properties,
                        "vector": vector,
                    }
                )

            # Batch insert — use named vector "default" (v4.20+ API)
            with collection.batch.dynamic() as batch:
                for obj in objects:
                    vec = obj["vector"]
                    batch.add_object(
                        uuid=obj["uuid"],
                        properties=obj["properties"],
                        vector={"default": vec} if vec else None,
                    )

            if batch.number_errors > 0:
                failed_details = []
                if hasattr(batch, "failed_objects"):
                    for failed in batch.failed_objects[:5]:
                        failed_details.append(str(failed))
                error_msg = (
                    f"Weaviate batch insert had {batch.number_errors} errors "
                    f"out of {len(objects)} objects in {collection_name}"
                )
                if failed_details:
                    error_msg += f". First errors: {'; '.join(failed_details)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        else:
            # v3 client
            with client.batch as batch:
                for item in data:
                    object_id = item.get("id") or item.get("chunk_id") or str(uuid.uuid4())
                    ids.append(object_id)

                    # Weaviate requires valid UUIDs — convert non-UUID IDs
                    # deterministically using uuid5 (same logic as v4 path)
                    try:
                        weaviate_uuid = str(uuid.UUID(object_id))
                    except (ValueError, AttributeError):
                        weaviate_uuid = str(uuid.uuid5(LASAGNA_UUID_NAMESPACE, object_id))

                    vector = item.get("embedding") or item.get("vector")

                    properties = {
                        k: v
                        for k, v in item.items()
                        if k not in ("id", "chunk_id", "embedding", "vector")
                    }
                    properties["_original_id"] = object_id

                    batch.add_data_object(
                        data_object=properties,
                        class_name=collection_name,
                        uuid=weaviate_uuid,
                        vector=vector,
                    )

        logger.debug(f"Inserted {len(ids)} objects into Weaviate class {collection_name}")
        return ids

    def delete(
        self,
        collection_name: str,
        filter_expr: FilterExpression,
        partition_key: Optional[str] = None,
    ) -> int:
        """Delete objects from Weaviate."""
        client = self._ensure_connected()

        # Build Weaviate filter
        where_filter = self._build_weaviate_filter(filter_expr)

        if where_filter is None:
            raise ValueError(
                "Cannot delete with empty filter: this could delete all objects. "
                "Build a FilterExpression with at least one condition."
            )

        if WEAVIATE_V4:
            collection = client.collections.get(collection_name)
            result = collection.data.delete_many(where=where_filter)
            return result.successful if hasattr(result, "successful") else -1
        else:
            client.batch.delete_objects(
                class_name=collection_name,
                where=where_filter,
            )
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
        """Query objects by filter."""
        client = self._ensure_connected()

        where_filter = None
        if filter_expr:
            where_filter = self._build_weaviate_filter(filter_expr)

        if WEAVIATE_V4:
            collection = client.collections.get(collection_name)
            response = collection.query.fetch_objects(
                filters=where_filter,
                limit=limit,
                offset=offset,
                include_vector=True,
            )
            results = []
            for obj in response.objects:
                vec = obj.vector
                if isinstance(vec, dict):
                    vec = vec.get("default")
                props = dict(obj.properties)
                obj_id = props.pop("_original_id", None) or str(obj.uuid)
                results.append({"id": obj_id, "embedding": vec, **props})
            return results
        else:
            v3_fields = output_fields or ["content", "document_id"]
            if "_original_id" not in v3_fields:
                v3_fields = list(v3_fields) + ["_original_id"]
            query = client.query.get(collection_name, v3_fields)
            query = query.with_additional(["id", "vector"])
            if where_filter:
                query = query.with_where(where_filter)
            query = query.with_limit(limit)

            result = query.do()
            objects = result.get("data", {}).get("Get", {}).get(collection_name, [])
            results = []
            for obj in objects:
                additional = obj.get("_additional", {})
                obj_id = obj.get("_original_id") or additional.get("id", "")
                vec = additional.get("vector")
                props = {k: v for k, v in obj.items() if k not in ("_additional", "_original_id")}
                results.append({"id": obj_id, "embedding": vec, **props})
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
        """Perform vector similarity search in Weaviate."""
        client = self._ensure_connected()

        # Determine metric from search_params (default COSINE)
        metric = MetricType.COSINE
        if search_params and search_params.metric_type:
            metric = search_params.metric_type

        where_filter = None
        if filter_expr:
            where_filter = self._build_weaviate_filter(filter_expr)

        if WEAVIATE_V4:
            collection = client.collections.get(collection_name)
            response = collection.query.near_vector(
                near_vector=query_embedding,
                target_vector="default",
                filters=where_filter,
                limit=limit,
                include_vector=True,
                return_metadata=MetadataQuery(distance=True),
            )

            results = []
            for obj in response.objects:
                props = dict(obj.properties)

                # Recover original ID if stored, else fall back to weaviate UUID
                obj_id = props.pop("_original_id", None) or str(obj.uuid)
                content = props.get("content", "")
                document_id = props.get("document_id", "")
                metadata = {k: v for k, v in props.items() if k not in ("content", "document_id")}

                # Extract vector — named vectors return a dict
                vec = obj.vector
                if isinstance(vec, dict):
                    vec = vec.get("default")

                # Weaviate distance semantics by metric:
                #   COSINE → cosine distance [0, 2], lower=identical
                #   L2     → L2 distance [0, ∞), lower=identical
                #   DOT    → negative inner product, lower=higher similarity
                raw_distance = obj.metadata.distance or 0.0
                if metric == MetricType.IP:
                    # DOT distance = negative IP; negate to get similarity
                    score = self.normalize_score(-raw_distance, metric, is_distance=False)
                else:
                    score = self.normalize_score(raw_distance, metric, is_distance=True)

                results.append(
                    VectorSearchResult(
                        id=obj_id,
                        score=score,
                        entity={
                            "id": obj_id,
                            "content": content,
                            "embedding": vec,
                            "metadata": metadata,
                            "document_id": document_id,
                        },
                    )
                )
            return results
        else:
            # v3 client
            v3_fields = output_fields or ["content", "document_id", "_original_id"]
            if "_original_id" not in v3_fields:
                v3_fields = list(v3_fields) + ["_original_id"]
            query = (
                client.query.get(collection_name, v3_fields)
                .with_near_vector({"vector": query_embedding})
                .with_limit(limit)
                .with_additional(["id", "distance", "vector"])
            )

            if where_filter:
                query = query.with_where(where_filter)

            result = query.do()
            objects = result.get("data", {}).get("Get", {}).get(collection_name, [])

            results = []
            for obj in objects:
                additional = obj.get("_additional", {})

                # Recover original ID if stored, else fall back to Weaviate UUID
                obj_id = obj.get("_original_id") or additional.get("id", "")

                content = obj.get("content", "")
                document_id = obj.get("document_id", "")
                metadata = {
                    k: v
                    for k, v in obj.items()
                    if k not in ("content", "document_id", "_additional", "_original_id")
                }

                # Same distance semantics as v4 path above
                raw_distance = float(additional.get("distance", 0.0))
                if metric == MetricType.IP:
                    score = self.normalize_score(-raw_distance, metric, is_distance=False)
                else:
                    score = self.normalize_score(raw_distance, metric, is_distance=True)

                results.append(
                    VectorSearchResult(
                        id=obj_id,
                        score=score,
                        entity={
                            "id": obj_id,
                            "content": content,
                            "embedding": additional.get("vector"),
                            "metadata": metadata,
                            "document_id": document_id,
                        },
                    )
                )
            return results

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_embedding: List[float],
        vector_field: str = "vector",
        sparse_field: str = "content",
        filter_expr: Optional[FilterExpression] = None,
        limit: int = 10,
        alpha: float = 0.7,
        output_fields: Optional[List[str]] = None,
        partition_key: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Perform hybrid search using Weaviate's built-in BM25 + vector search."""
        client = self._ensure_connected()

        where_filter = None
        if filter_expr:
            where_filter = self._build_weaviate_filter(filter_expr)

        if WEAVIATE_V4:
            collection = client.collections.get(collection_name)
            response = collection.query.hybrid(
                query=query_text,
                vector=query_embedding,
                target_vector="default",
                alpha=alpha,  # 1.0 = pure vector, 0.0 = pure BM25
                filters=where_filter,
                limit=limit,
                include_vector=True,
                return_metadata=MetadataQuery(score=True),
            )

            results = []
            for obj in response.objects:
                props = dict(obj.properties)

                # Recover original ID if stored, else fall back to weaviate UUID
                obj_id = props.pop("_original_id", None) or str(obj.uuid)
                content = props.get("content", "")
                document_id = props.get("document_id", "")
                metadata = {k: v for k, v in props.items() if k not in ("content", "document_id")}

                # Extract vector — named vectors return a dict
                vec = obj.vector
                if isinstance(vec, dict):
                    vec = vec.get("default")

                results.append(
                    VectorSearchResult(
                        id=obj_id,
                        score=obj.metadata.score or 0.0,
                        entity={
                            "id": obj_id,
                            "content": content,
                            "embedding": vec,
                            "metadata": metadata,
                            "document_id": document_id,
                        },
                    )
                )
            return results
        else:
            # v3 doesn't have direct hybrid, combine manually
            logger.warning("Weaviate v3 doesn't support direct hybrid search, using vector only")
            return self.search(
                collection_name=collection_name,
                query_embedding=query_embedding,
                filter_expr=filter_expr,
                limit=limit,
                partition_key=partition_key,
            )

    def _build_weaviate_filter(self, filter_expr: FilterExpression):
        """Build Weaviate v4 Filter object from FilterExpression.

        Returns a ``_Filters`` instance (or None) suitable for the v4
        client's ``filters=`` / ``where=`` parameters.
        """
        if filter_expr.is_empty():
            return None

        return filter_expr.to_weaviate()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def flush(self, collection_name: str) -> None:
        """Weaviate handles persistence automatically."""
        pass

    def compact(self, collection_name: str) -> None:
        """Weaviate handles compaction automatically."""
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

        Converts Milvus-style search parameters to Weaviate format.
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
