"""OpenSearch vector backend with native hybrid search (BM25 + k-NN).

OpenSearch provides Lucene-quality BM25 text search and server-side hybrid
fusion of dense vectors + BM25 scores in a single query. This backend wraps
the opensearch-py SDK — we call their API, not reimplement their search logic.

What OpenSearch handles (millions of lines of Lucene + OpenSearch):
    - BM25 scoring with 30+ language analyzers, fuzzy matching, phrase search
    - FAISS/Lucene/NMSLIB approximate nearest neighbor search
    - Server-side score normalization and combination
    - Bulk ingestion with batching, retries, error handling
    - Index management, Lucene segment merging

What this file does (~500 lines of translation):
    - Maps our VectorStoreAdapter interface to OpenSearch query DSL
    - Translates CollectionSchema to OpenSearch index mappings
    - Translates FilterExpression to OpenSearch bool queries
    - Passes through scores (hybrid pipeline normalizes; k-NN/BM25 raw)

Usage:
    from lasagna.vector_backends import get_backend

    backend = get_backend("opensearch", hosts=["http://localhost:9200"])
    backend.search(collection, embedding, filter_expr=..., limit=10)
    backend.hybrid_search(collection, "query text", embedding, alpha=0.7)

Environment variables:
    OPENSEARCH_HOSTS: Comma-separated list of hosts (default: http://localhost:9200)
    OPENSEARCH_USER: Basic auth username (default: unset — no auth is sent)
    OPENSEARCH_PASSWORD: Basic auth password (default: unset)
    OPENSEARCH_USE_SSL: Enable SSL (default: false)
    OPENSEARCH_VERIFY_CERTS: Verify SSL certificates (default: false)
    OPENSEARCH_TIMEOUT: Request timeout in seconds (default: 30)
    OPENSEARCH_LANGUAGE: BM25 language analyzer (default: english). Supports 30+
        languages: english, german, french, spanish, italian, cjk, arabic, etc.
    OPENSEARCH_SHARDS: Number of index shards (default: 1)
    OPENSEARCH_REPLICAS: Number of index replicas (default: 0, set >=1 for production)
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Union

from .base import (
    BackendCapabilities,
    CollectionInfo,
    CollectionSchema,
    Fields,
    IndexConfig,
    MetricType,
    SearchParams,
    VectorSearchResult,
    VectorStoreAdapter,
)
from .filters import FilterExpression

logger = logging.getLogger(__name__)

# Check if opensearch-py is installed
try:
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from opensearchpy.helpers import bulk as os_bulk

    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False

# Default k-NN engine and space type.
# lucene supports cosinesimil + filtered search. nmslib does not support filters.
# faiss only supports l2/innerproduct space types.
_DEFAULT_ENGINE = "lucene"
_DEFAULT_SPACE_TYPE = "cosinesimil"
_DEFAULT_HNSW_M = 16
_DEFAULT_HNSW_EF_CONSTRUCTION = 256
_DEFAULT_HNSW_EF_SEARCH = 128

# Neural sparse (3-way hybrid) constants
_NEURAL_SPARSE_FIELD = "sparse_tokens"


def _neural_sparse_enabled() -> bool:
    """Check if 3-way hybrid (dense + neural sparse + BM25) is enabled."""
    return os.environ.get("OPENSEARCH_ENABLE_NEURAL_SPARSE", "false").lower() == "true"


def _get_default_opensearch_config() -> Dict[str, Any]:
    """Resolve OpenSearch configuration from environment variables."""
    hosts_str = os.environ.get("OPENSEARCH_HOSTS", "http://localhost:9200")
    hosts = [h.strip() for h in hosts_str.split(",") if h.strip()]

    user = os.environ.get("OPENSEARCH_USER", "")
    password = os.environ.get("OPENSEARCH_PASSWORD", "")
    http_auth = (user, password) if user else None

    use_ssl = os.environ.get("OPENSEARCH_USE_SSL", "false").lower() == "true"
    verify_certs = os.environ.get("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"
    try:
        timeout = max(1, int(os.environ.get("OPENSEARCH_TIMEOUT", "30")))
    except ValueError:
        logger.warning("Invalid OPENSEARCH_TIMEOUT value, using default 30")
        timeout = 30

    return {
        "hosts": hosts,
        "http_auth": http_auth,
        "use_ssl": use_ssl,
        "verify_certs": verify_certs,
        "timeout": timeout,
    }


class OpenSearchBackend(VectorStoreAdapter):
    """OpenSearch vector backend with native Lucene BM25 + k-NN hybrid search.

    This backend adapts our VectorStoreAdapter interface to OpenSearch's query DSL.
    OpenSearch handles all search logic (BM25, k-NN, hybrid fusion, score normalization)
    — this class only translates between the two APIs.
    """

    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        http_auth: Optional[tuple] = None,
        use_ssl: Optional[bool] = None,
        verify_certs: Optional[bool] = None,
        timeout: Optional[int] = None,
        engine: str = _DEFAULT_ENGINE,
        space_type: str = _DEFAULT_SPACE_TYPE,
        **kwargs,
    ):
        defaults = _get_default_opensearch_config()
        self._hosts = hosts if hosts is not None else defaults["hosts"]
        self._http_auth = http_auth if http_auth is not None else defaults["http_auth"]
        self._use_ssl = use_ssl if use_ssl is not None else defaults["use_ssl"]
        self._verify_certs = verify_certs if verify_certs is not None else defaults["verify_certs"]
        self._timeout = timeout if timeout is not None else defaults["timeout"]
        self._engine = engine
        self._space_type = space_type
        self._kwargs = kwargs

        self._client: Optional[OpenSearch] = None
        self._connection_lock = threading.Lock()
        self._pipeline_lock = threading.Lock()
        # collection_name -> pipeline_name (tracks current hybrid pipeline per index)
        self._pipeline_cache: Dict[str, str] = {}

    # =========================================================================
    # Capability Detection
    # =========================================================================

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_hybrid_search=True,
            supports_native_bm25=True,
            supports_sparse_vectors=_neural_sparse_enabled(),
            supports_hnsw=True,
            supports_ivf=False,
            supports_gpu=False,
            supports_json_filters=True,
            supports_metadata_filtering=True,
            max_filter_complexity=1000,
            max_batch_size=10000,
            supports_async_insert=True,
            supports_partitions=False,
            supports_dynamic_schema=True,
            max_varchar_length=32766,
            max_vector_dimension=16000,
        )

    @property
    def backend_name(self) -> str:
        return "opensearch"

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> None:
        if self._client is not None:
            return

        with self._connection_lock:
            if self._client is not None:
                return

            if not OPENSEARCH_AVAILABLE:
                raise ImportError(
                    "opensearch-py is required for the OpenSearch backend. "
                    "Install it with: pip install opensearch-py"
                )

            self._client = OpenSearch(
                hosts=self._hosts,
                http_auth=self._http_auth,
                use_ssl=self._use_ssl,
                verify_certs=self._verify_certs,
                timeout=self._timeout,
                connection_class=RequestsHttpConnection,
            )
            logger.info(f"Connected to OpenSearch: {self._hosts}")

    def disconnect(self) -> None:
        # Lock ordering: _connection_lock -> _pipeline_lock (same as _ensure_hybrid_pipeline)
        with self._connection_lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
                self._client = None
                with self._pipeline_lock:
                    self._pipeline_cache.clear()
                logger.info("Disconnected from OpenSearch")

    def is_connected(self) -> bool:
        if self._client is None:
            return False
        try:
            return self._client.ping()
        except Exception:
            return False

    def _ensure_connected(self) -> OpenSearch:
        if self._client is None:
            self.connect()
        client = self._client
        if client is None:
            raise RuntimeError(
                "OpenSearch client unavailable: was disconnect() called concurrently?"
            )
        return client

    # =========================================================================
    # Collection (Index) Management
    # =========================================================================

    def has_collection(self, collection_name: str) -> bool:
        client = self._ensure_connected()
        return client.indices.exists(index=collection_name)

    def create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_config: Optional[IndexConfig] = None,
    ) -> None:
        client = self._ensure_connected()

        if client.indices.exists(index=collection_name):
            logger.debug(f"Index '{collection_name}' already exists")
            return

        mapping = self._schema_to_mapping(schema, index_config)
        settings = self._build_index_settings(index_config)

        try:
            client.indices.create(
                index=collection_name,
                body={"settings": settings, "mappings": mapping},
            )
            logger.info(f"Created OpenSearch index: {collection_name}")
        except Exception as e:
            # Handle concurrent create race (TOCTOU): another thread may have
            # created the index between our exists() check and create() call.
            if "resource_already_exists_exception" in str(e):
                logger.debug(f"Index '{collection_name}' created by concurrent thread")
                return
            raise

        # Pre-create the hybrid search pipeline for this index
        self._ensure_hybrid_pipeline(collection_name)

    def drop_collection(self, collection_name: str) -> None:
        client = self._ensure_connected()
        if client.indices.exists(index=collection_name):
            client.indices.delete(index=collection_name)
            # Clean up associated hybrid search pipeline (under lock to
            # prevent race with concurrent _ensure_hybrid_pipeline)
            with self._pipeline_lock:
                cached_pipeline = self._pipeline_cache.pop(collection_name, None)
            if cached_pipeline:
                try:
                    client.transport.perform_request(
                        "DELETE", f"/_search/pipeline/{cached_pipeline}"
                    )
                except Exception:
                    pass
            logger.info(f"Dropped OpenSearch index: {collection_name}")

    def describe_collection(self, collection_name: str) -> CollectionInfo:
        client = self._ensure_connected()
        mapping = client.indices.get_mapping(index=collection_name)

        properties = mapping.get(collection_name, {}).get("mappings", {}).get("properties", {})

        fields = []
        for name, config in properties.items():
            fields.append({"name": name, "type": config.get("type", "unknown")})

        stats = client.indices.stats(index=collection_name)
        doc_count = (
            stats.get("indices", {})
            .get(collection_name, {})
            .get("primaries", {})
            .get("docs", {})
            .get("count", 0)
        )

        return CollectionInfo(
            name=collection_name,
            fields=fields,
            row_count=doc_count,
        )

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        client = self._ensure_connected()
        stats = client.indices.stats(index=collection_name)
        idx_stats = stats.get("indices", {}).get(collection_name, {}).get("primaries", {})
        return {
            "row_count": idx_stats.get("docs", {}).get("count", 0),
            "index_size_bytes": idx_stats.get("store", {}).get("size_in_bytes", 0),
            "deleted_docs": idx_stats.get("docs", {}).get("deleted", 0),
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
        # OpenSearch creates k-NN index at mapping time, not separately.
        # This is a no-op — the index is configured via the mapping in create_collection.
        logger.debug(
            f"Index creation is handled at mapping time for OpenSearch "
            f"(field={field_name}, collection={collection_name})"
        )

    def list_indexes(self, collection_name: str) -> List[str]:
        client = self._ensure_connected()
        mapping = client.indices.get_mapping(index=collection_name)
        properties = mapping.get(collection_name, {}).get("mappings", {}).get("properties", {})
        # Return fields that have k-NN index (knn_vector type)
        return [name for name, config in properties.items() if config.get("type") == "knn_vector"]

    # =========================================================================
    # Data Operations
    # =========================================================================

    def insert(
        self,
        collection_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        partition_key: Optional[str] = None,
    ) -> List[Union[int, str]]:
        if isinstance(data, dict):
            data = [data]

        if not data:
            return []

        client = self._ensure_connected()

        actions = []
        ids = []
        null_embedding_count = 0
        for doc in data:
            doc_id = doc.get(Fields.CHUNK_ID)
            if doc_id is None:
                doc_id = doc.get("id")
            # Generate a UUID for docs without an ID so we always return
            # a valid string (never None) and the caller can reference
            # the document afterward.
            if doc_id is None or (isinstance(doc_id, str) and not doc_id.strip()):
                doc_id = str(uuid.uuid4())
            source = {
                Fields.CHUNK_ID: doc_id,
                # Use ``or`` instead of default param: dict.get(key, default)
                # returns None when key exists with value None, while ``or``
                # coerces None → safe default.  Prevents downstream crashes
                # (ranker, citation_manager do direct bracket access on these).
                Fields.DOCUMENT_ID: doc.get(Fields.DOCUMENT_ID) or "",
                Fields.SESSION_ID: doc.get(Fields.SESSION_ID) or "",
                Fields.CONTENT: doc.get(Fields.CONTENT) or "",
                Fields.EMBEDDING: doc.get(Fields.EMBEDDING),
                Fields.METADATA: doc.get(Fields.METADATA) or {},
            }
            if partition_key is not None:
                source["partition_key"] = partition_key

            # Include sparse tokens for 3-way hybrid when available
            if _neural_sparse_enabled() and _NEURAL_SPARSE_FIELD in doc:
                source[_NEURAL_SPARSE_FIELD] = doc[_NEURAL_SPARSE_FIELD]

            # Track null embeddings — docs without embeddings are invisible
            # to vector search (k-NN queries cannot match them).
            if source[Fields.EMBEDDING] is None:
                null_embedding_count += 1

            action: Dict[str, Any] = {
                "_index": collection_name,
                "_source": source,
                "_id": doc_id,
            }
            actions.append(action)
            ids.append(doc_id)

        if null_embedding_count > 0:
            logger.warning(
                f"Inserting {null_embedding_count}/{len(data)} documents without embeddings "
                f"into '{collection_name}'. These documents will be invisible to vector search "
                f"(k-NN queries cannot match them). They will still be findable via BM25 text "
                f"search and filter queries."
            )

        success, errors = os_bulk(client, actions, refresh="wait_for", raise_on_error=False)
        if errors:
            # Remove IDs of documents that failed to ingest so callers
            # don't reference chunks that don't exist.
            failed_ids: set = set()
            for err in errors:
                for action_type in ("index", "create", "update", "delete"):
                    if action_type in err:
                        failed_id = err[action_type].get("_id")
                        if failed_id:
                            failed_ids.add(failed_id)
                        break
            logger.warning(
                f"Bulk insert had {len(errors)} errors ({len(failed_ids)} failed IDs): {errors[:3]}"
            )
            if failed_ids:
                ids = [doc_id for doc_id in ids if doc_id not in failed_ids]

        return ids

    def delete(
        self,
        collection_name: str,
        filter_expr: FilterExpression,
        partition_key: Optional[str] = None,
    ) -> int:
        if not filter_expr or filter_expr.is_empty():
            raise ValueError(
                "Cannot delete with empty filter: this would delete all documents. "
                "Build a FilterExpression with at least one condition."
            )
        client = self._ensure_connected()

        query = filter_expr.to_opensearch()
        if partition_key is not None:
            query = {
                "bool": {"must": [query, {"term": {"partition_key": {"value": partition_key}}}]}
            }

        response = client.delete_by_query(
            index=collection_name,
            body={"query": query},
            refresh=True,
        )
        deleted = response.get("deleted", 0)
        logger.debug(f"Deleted {deleted} documents from {collection_name}")
        return deleted

    def query(
        self,
        collection_name: str,
        filter_expr: Optional[FilterExpression] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        partition_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        client = self._ensure_connected()

        query = filter_expr.to_opensearch() if filter_expr else {"match_all": {}}
        if partition_key is not None:
            query = {
                "bool": {"must": [query, {"term": {"partition_key": {"value": partition_key}}}]}
            }

        body: Dict[str, Any] = {"query": query, "from": offset, "size": limit}
        if output_fields:
            body["_source"] = output_fields

        response = client.search(index=collection_name, body=body)
        return [{"id": hit["_id"], **hit["_source"]} for hit in response["hits"]["hits"]]

    def list_document_ids(
        self,
        collection_name: str,
        session_id: Optional[str] = None,
        *,
        limit: int = 10_000,
        offset: int = 0,
    ) -> List[str]:
        """List distinct document_ids using an OpenSearch terms aggregation.

        Much more efficient than scanning all vectors via query().
        """
        client = self._ensure_connected()

        agg_query: Dict[str, Any] = {"match_all": {}}
        if session_id is not None:
            agg_query = {"term": {"session_id": {"value": session_id}}}

        body: Dict[str, Any] = {
            "size": 0,
            "query": agg_query,
            "aggs": {
                "unique_doc_ids": {
                    "terms": {
                        "field": "document_id",
                        "size": limit,
                    }
                }
            },
        }
        response = client.search(index=collection_name, body=body)
        buckets = response.get("aggregations", {}).get("unique_doc_ids", {}).get("buckets", [])
        return [b["key"] for b in buckets]

    # =========================================================================
    # Vector Search
    # =========================================================================

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        vector_field: str = Fields.EMBEDDING,
        filter_expr: Optional[FilterExpression] = None,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[SearchParams] = None,
        partition_key: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Dense vector search using OpenSearch k-NN plugin."""
        if limit <= 0:
            return []
        client = self._ensure_connected()

        knn_query: Dict[str, Any] = {
            "vector": query_embedding,
            "k": limit,
        }

        # Build filter for k-NN pre-filtering
        os_filter = self._build_search_filter(filter_expr, partition_key)
        if os_filter:
            knn_query["filter"] = os_filter

        body: Dict[str, Any] = {
            "size": limit,
            "query": {"knn": {vector_field: knn_query}},
        }
        if output_fields:
            body["_source"] = output_fields

        response = client.search(index=collection_name, body=body)
        return [self._hit_to_result(hit) for hit in response["hits"]["hits"]]

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_embedding: List[float],
        vector_field: str = Fields.EMBEDDING,
        sparse_field: str = Fields.CONTENT,
        filter_expr: Optional[FilterExpression] = None,
        limit: int = 10,
        alpha: float = 0.7,
        output_fields: Optional[List[str]] = None,
        partition_key: Optional[str] = None,
        **kwargs,
    ) -> List[VectorSearchResult]:
        """Native hybrid search: k-NN + BM25 fused server-side.

        OpenSearch runs both dense vector and BM25 text search in a single
        query on the same shard, then combines scores via the normalization
        processor pipeline. This is fundamentally better than the Python
        BM25 fallback used by other backends.

        The BM25 sub-query uses fuzzy matching (catches typos) and phrase
        boosting (exact phrases rank higher) — features only available
        in a real search engine, not a vector DB.

        Note: Highlighting in hybrid queries requires OpenSearch 3.0+.
        On OpenSearch 2.x, highlights will be silently empty (no error).
        Collapse (group_by_document) with hybrid queries requires OpenSearch 3.2+;
        on older versions it is silently ignored.

        Args:
            alpha: Weight for dense search (1.0 = pure vector, 0.0 = pure BM25).
                   Default 0.7 = 70% vector, 30% BM25.
            **kwargs: OpenSearch-specific options:
                group_by_document (bool): If True, collapse results by document_id
                    to ensure diversity (max 3 chunks per document). Default False.
        """
        if limit <= 0:
            return []

        group_by_document = kwargs.get("group_by_document", False)

        # Short-circuit: empty query_text degrades to pure vector search
        if not query_text or not query_text.strip():
            return self.search(
                collection_name,
                query_embedding,
                vector_field,
                filter_expr,
                limit,
                output_fields,
                partition_key=partition_key,
            )

        # Short-circuit: alpha=1.0 means pure vector, skip BM25 overhead
        if alpha >= 1.0:
            return self.search(
                collection_name,
                query_embedding,
                vector_field,
                filter_expr,
                limit,
                output_fields,
                partition_key=partition_key,
            )

        client = self._ensure_connected()

        # Build filter for pre-filtering (applied inside each sub-query,
        # NOT as post_filter which would filter after scoring and risk
        # returning fewer than `limit` results in multi-tenant mode)
        os_filter = self._build_search_filter(filter_expr, partition_key)

        # Short-circuit: alpha=0.0 means pure BM25, skip vector overhead
        if alpha <= 0.0:
            bm25_match = {"match": {sparse_field: {"query": query_text, "fuzziness": "AUTO"}}}
            bm25_phrase = {"match_phrase": {sparse_field: {"query": query_text, "boost": 1.5}}}
            if os_filter:
                query = {
                    "bool": {
                        "should": [bm25_match, bm25_phrase],
                        "filter": [os_filter],
                        "minimum_should_match": 1,
                    }
                }
            else:
                query = {"bool": {"should": [bm25_match, bm25_phrase]}}
            body: Dict[str, Any] = {
                "size": limit,
                "query": query,
                "highlight": {
                    "fields": {sparse_field: {"fragment_size": 200, "number_of_fragments": 3}},
                    "pre_tags": [""],
                    "post_tags": [""],
                },
            }
            if output_fields:
                body["_source"] = output_fields
            if group_by_document:
                body["collapse"] = {
                    "field": Fields.DOCUMENT_ID,
                    "inner_hits": {"name": "chunks", "size": 3},
                }
            response = client.search(index=collection_name, body=body)
            return [self._hit_to_result(hit) for hit in response["hits"]["hits"]]

        # Extract optional query_tokens for 3-way hybrid
        query_tokens = kwargs.get("query_tokens")
        n_queries = 3 if (_neural_sparse_enabled() and query_tokens) else 2

        # Ensure the hybrid search pipeline exists with correct weights
        pipeline_name = self._ensure_hybrid_pipeline(collection_name, alpha, n_queries)

        body = self._build_hybrid_body(
            query_text,
            query_embedding,
            vector_field,
            sparse_field,
            os_filter,
            limit,
            output_fields,
            group_by_document,
            query_tokens=query_tokens,
        )

        response = client.search(
            index=collection_name,
            body=body,
            params={"search_pipeline": pipeline_name},
        )

        return [self._hit_to_result(hit) for hit in response["hits"]["hits"]]

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _hit_to_result(self, hit: Dict[str, Any]) -> VectorSearchResult:
        """Convert an OpenSearch hit to VectorSearchResult."""
        source = hit.get("_source", {})
        raw_score = hit.get("_score", 0.0)

        # Do NOT clamp scores to [0, 1]. Different query types produce
        # different score ranges:
        #   - Hybrid pipeline (min_max normalization): already [0, 1]
        #   - Pure k-NN (FAISS cosinesimil): [0, 2] on OpenSearch < 2.19,
        #     [0, 1] on 2.19+ (cosine_similarity + 1 historically)
        #   - Pure BM25: unbounded (commonly 1-20+)
        # Clamping would make top results indistinguishable and destroy
        # ranking signal. Downstream consumers (ranker, reranker) compare
        # scores relatively, not against absolute thresholds.
        normalized = raw_score if raw_score is not None else 0.0

        hit_id = hit.get("_id", "")

        entity: Dict[str, Any] = {
            Fields.CHUNK_ID: hit_id,
            Fields.DOCUMENT_ID: source.get(Fields.DOCUMENT_ID) or "",
            Fields.SESSION_ID: source.get(Fields.SESSION_ID) or "",
            Fields.CONTENT: source.get(Fields.CONTENT) or "",
            Fields.EMBEDDING: source.get(Fields.EMBEDDING),
            Fields.METADATA: source.get(Fields.METADATA) or {},
        }

        # Extract highlight fragments if present (from BM25 matching).
        # These provide exact text spans for precise citations.
        highlights = hit.get("highlight", {})
        if highlights:
            # Flatten all highlight fragments into a single list
            entity["highlights"] = [
                frag for field_frags in highlights.values() for frag in field_frags
            ]

        return VectorSearchResult(
            id=hit_id,
            score=normalized,
            entity=entity,
            raw_score=raw_score,
        )

    def _build_hybrid_body(
        self,
        query_text: str,
        query_embedding: List[float],
        vector_field: str,
        sparse_field: str,
        os_filter: Optional[Dict[str, Any]],
        limit: int,
        output_fields: Optional[List[str]] = None,
        group_by_document: bool = False,
        query_tokens: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Build the hybrid search request body (k-NN + BM25, optionally 3-way).

        Extracted from hybrid_search() so multi_hybrid_search() can reuse
        the same query construction for _msearch batching.

        Args:
            query_tokens: Optional sparse token weights for neural sparse
                sub-query. When provided and neural sparse is enabled,
                builds a 3-way hybrid query (dense + neural_sparse + BM25).
        """
        knn_clause: Dict[str, Any] = {
            "vector": query_embedding,
            "k": limit,
        }
        if os_filter:
            knn_clause["filter"] = os_filter
        knn_subquery: Dict[str, Any] = {"knn": {vector_field: knn_clause}}

        bm25_match: Dict[str, Any] = {
            "match": {sparse_field: {"query": query_text, "fuzziness": "AUTO"}}
        }
        bm25_phrase: Dict[str, Any] = {
            "match_phrase": {sparse_field: {"query": query_text, "boost": 1.5}}
        }
        if os_filter:
            bm25_subquery: Dict[str, Any] = {
                "bool": {
                    "should": [bm25_match, bm25_phrase],
                    "filter": [os_filter],
                    "minimum_should_match": 1,
                }
            }
        else:
            bm25_subquery = {"bool": {"should": [bm25_match, bm25_phrase]}}

        # Build sub-queries list: 2-way or 3-way hybrid
        sub_queries = [knn_subquery, bm25_subquery]

        # 3-way hybrid: add neural sparse sub-query when enabled and tokens available
        if _neural_sparse_enabled() and query_tokens:
            sparse_query: Dict[str, Any] = {
                "neural_sparse": {_NEURAL_SPARSE_FIELD: {"query_tokens": query_tokens}}
            }
            if os_filter:
                sparse_query = {"bool": {"must": [sparse_query], "filter": [os_filter]}}
            # Insert between knn and bm25: [dense, neural_sparse, bm25]
            sub_queries = [knn_subquery, sparse_query, bm25_subquery]

        body: Dict[str, Any] = {
            "size": limit,
            "query": {
                "hybrid": {
                    "queries": sub_queries,
                }
            },
            "highlight": {
                "fields": {
                    sparse_field: {
                        "fragment_size": 200,
                        "number_of_fragments": 3,
                    }
                },
                "pre_tags": [""],
                "post_tags": [""],
            },
        }
        if output_fields:
            body["_source"] = output_fields

        if group_by_document:
            body["collapse"] = {
                "field": Fields.DOCUMENT_ID,
                "inner_hits": {"name": "chunks", "size": 3},
            }

        return body

    def _build_search_filter(
        self,
        filter_expr: Optional[FilterExpression],
        partition_key: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Build OpenSearch filter from FilterExpression and partition_key."""
        parts = []
        if filter_expr and not filter_expr.is_empty():
            parts.append(filter_expr.to_opensearch())
        if partition_key is not None:
            parts.append({"term": {"partition_key": {"value": partition_key}}})

        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return {"bool": {"must": parts}}

    def _ensure_hybrid_pipeline(
        self,
        collection_name: str,
        alpha: float = 0.7,
        n_queries: int = 2,
    ) -> str:
        """Create or update the hybrid search pipeline for a collection.

        OpenSearch requires a search pipeline with a normalization processor
        to combine scores from multiple sub-queries. The pipeline name
        encodes the alpha weight and mode so that changes trigger re-creation.

        Args:
            n_queries: Number of sub-queries (2 = standard, 3 = neural sparse).

        Thread-safe: uses double-checked locking to avoid concurrent
        pipeline creation/deletion races. Lock ordering is
        _connection_lock -> _pipeline_lock (same as disconnect).
        """
        # Round alpha to 2 decimal places so the pipeline name, cache key,
        # and weight array are all consistent. Without this, alpha=0.701
        # produces pipeline name "a0.70" but weights use the unrounded
        # 0.701, causing weights to not sum to 1.0 and allowing different
        # unrounded alphas to silently share the same pipeline.
        alpha = round(alpha, 2)

        # Include alpha and mode in the pipeline name so different configs
        # get different pipelines.
        alpha_key = f"{alpha:.2f}"
        mode_tag = "3w" if n_queries == 3 else "2w"
        pipeline_name = f"rag-hybrid-{collection_name}-{mode_tag}-a{alpha_key}"

        # Fast path (no lock) — pipeline already cached
        if self._pipeline_cache.get(collection_name) == pipeline_name:
            return pipeline_name

        # Connect BEFORE acquiring _pipeline_lock to maintain lock ordering:
        # _connection_lock -> _pipeline_lock (prevents deadlock with disconnect)
        client = self._ensure_connected()

        with self._pipeline_lock:
            # Re-check under lock
            cached = self._pipeline_cache.get(collection_name)
            if cached == pipeline_name:
                return pipeline_name

            # Config changed or first call — delete old pipeline if exists
            if cached and cached != pipeline_name:
                try:
                    client.transport.perform_request("DELETE", f"/_search/pipeline/{cached}")
                except Exception:
                    pass

            # Compute weights based on mode.
            # Alpha controls dense weight in both 2-way and 3-way modes.
            if n_queries == 3:
                # 3-way: [dense, neural_sparse, bm25]
                # Dense gets `alpha`, remaining (1-alpha) split equally
                # between neural sparse and BM25.
                remainder = round(1.0 - alpha, 2)
                sparse_w = round(remainder / 2, 2)
                bm25_w = round(remainder - sparse_w, 2)  # avoid float drift
                weights = [alpha, sparse_w, bm25_w]
            else:
                # 2-way: [dense, bm25]
                weights = [alpha, round(1.0 - alpha, 2)]

            pipeline_body = {
                "description": (
                    f"RAG hybrid search pipeline for {collection_name} "
                    f"({mode_tag}, alpha={alpha_key})"
                ),
                "phase_results_processors": [
                    {
                        "normalization-processor": {
                            "normalization": {"technique": "min_max"},
                            "combination": {
                                "technique": "arithmetic_mean",
                                "parameters": {"weights": weights},
                            },
                        }
                    }
                ],
            }
            try:
                client.transport.perform_request(
                    "PUT",
                    f"/_search/pipeline/{pipeline_name}",
                    body=pipeline_body,
                )
                self._pipeline_cache[collection_name] = pipeline_name
                logger.debug(f"Created hybrid search pipeline: {pipeline_name}")
            except Exception as e:
                logger.error(
                    f"Failed to create hybrid search pipeline '{pipeline_name}': {e}. "
                    f"Hybrid search will fail until the pipeline is available."
                )
                raise RuntimeError(f"Cannot create OpenSearch hybrid search pipeline: {e}") from e

        return pipeline_name

    def _schema_to_mapping(
        self,
        schema: CollectionSchema,
        index_config: Optional[IndexConfig] = None,
    ) -> Dict[str, Any]:
        """Convert CollectionSchema to OpenSearch index mapping.

        Text fields use a language-specific analyzer (default: "english") for
        proper stemming, stop-word removal, and linguistic normalization.
        This means "acquired" will match "acquisition", "running" matches "run",
        etc. — a major BM25 quality advantage over the standard analyzer.
        """
        properties: Dict[str, Any] = {}
        metric = index_config.metric_type if index_config else MetricType.COSINE

        # Language analyzer for BM25 text fields.
        # OpenSearch ships 30+ analyzers (english, german, french, cjk, etc.)
        # Configurable via OPENSEARCH_LANGUAGE env var or create_collection kwargs.
        language = self._kwargs.get(
            "language",
            os.environ.get("OPENSEARCH_LANGUAGE", "english"),
        )

        space_map = {
            MetricType.COSINE: "cosinesimil",
            MetricType.L2: "l2",
            MetricType.IP: "innerproduct",
        }

        for f in schema.fields:
            if f.dtype == "VARCHAR":
                if f.name in (Fields.CHUNK_ID, Fields.DOCUMENT_ID, Fields.SESSION_ID):
                    # ID and filter fields use keyword (exact match, no analysis)
                    properties[f.name] = {"type": "keyword"}
                else:
                    # Content fields use language-aware BM25 analysis.
                    # Lucene handles tokenization, stemming, stop-words automatically.
                    properties[f.name] = {"type": "text", "analyzer": language}

            elif f.dtype == "FLOAT_VECTOR":
                dim = f.dimension or 1024
                properties[f.name] = {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": space_map.get(metric, self._space_type),
                        "engine": self._engine,
                        "parameters": {
                            "ef_construction": _DEFAULT_HNSW_EF_CONSTRUCTION,
                            "m": _DEFAULT_HNSW_M,
                        },
                    },
                }

            elif f.dtype == "JSON":
                properties[f.name] = {"type": "object", "enabled": True}

            elif f.dtype == "INT64":
                properties[f.name] = {"type": "long"}

            else:
                # Fallback for unknown types
                properties[f.name] = {"type": "text"}

        # Always add partition_key field
        if "partition_key" not in properties:
            properties["partition_key"] = {"type": "keyword"}

        # Neural sparse: add rank_features field for sparse token weights
        if _neural_sparse_enabled():
            properties[_NEURAL_SPARSE_FIELD] = {"type": "rank_features"}

        return {"properties": properties}

    def _build_index_settings(self, index_config: Optional[IndexConfig] = None) -> Dict[str, Any]:
        """Build OpenSearch index settings.

        Shards and replicas are configurable via env vars for production safety.
        Default: 1 shard, 0 replicas (dev mode). Production should use >=1 replica.
        """
        try:
            shards = max(1, int(os.environ.get("OPENSEARCH_SHARDS", "1")))
        except ValueError:
            logger.warning("Invalid OPENSEARCH_SHARDS value, using default 1")
            shards = 1
        try:
            replicas = max(0, int(os.environ.get("OPENSEARCH_REPLICAS", "0")))
        except ValueError:
            logger.warning("Invalid OPENSEARCH_REPLICAS value, using default 0")
            replicas = 0

        return {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": _DEFAULT_HNSW_EF_SEARCH,
                "number_of_shards": shards,
                "number_of_replicas": replicas,
            }
        }

    # =========================================================================
    # Partition Support (not natively supported — use filtering)
    # =========================================================================

    def has_partition(self, collection_name: str, partition_name: str) -> bool:
        return False

    def create_partition(self, collection_name: str, partition_name: str) -> None:
        pass  # Use partition_key field filtering instead

    def drop_partition(self, collection_name: str, partition_name: str) -> None:
        pass

    def list_partitions(self, collection_name: str) -> List[str]:
        return []

    # =========================================================================
    # OpenSearch-Specific: Multi-Search for RAG Fusion
    # =========================================================================

    def multi_search(
        self,
        collection_name: str,
        searches: List[Dict[str, Any]],
    ) -> List[List[VectorSearchResult]]:
        """Execute multiple search queries in a single HTTP round-trip.

        This is an OpenSearch-specific optimization for RAG Fusion, which
        issues N reformulated queries. Using _msearch batches them into one
        request, cutting network latency by ~Nx.

        Args:
            collection_name: Index to search.
            searches: List of search body dicts (each a complete search body).

        Returns:
            List of result lists, one per input search.
        """
        client = self._ensure_connected()

        body_parts: List[Dict[str, Any]] = []
        for search_body in searches:
            body_parts.append({"index": collection_name})
            body_parts.append(search_body)

        response = client.msearch(body=body_parts)

        all_results = []
        for i, resp in enumerate(response.get("responses", [])):
            if "error" in resp:
                logger.warning(f"multi_search sub-query {i} failed: {resp['error']}")
                all_results.append([])
            else:
                hits = resp.get("hits", {}).get("hits", [])
                all_results.append([self._hit_to_result(hit) for hit in hits])

        # Pad with empty results if OpenSearch returned fewer responses
        # than expected (e.g., partial transport failure). Maintains
        # positional correspondence with input searches.
        while len(all_results) < len(searches):
            all_results.append([])

        return all_results

    def multi_hybrid_search(
        self,
        collection_name: str,
        query_texts: List[str],
        query_embeddings: List[List[float]],
        vector_field: str = Fields.EMBEDDING,
        sparse_field: str = Fields.CONTENT,
        filter_expr: Optional[FilterExpression] = None,
        limit: int = 10,
        alpha: float = 0.7,
        output_fields: Optional[List[str]] = None,
        partition_key: Optional[str] = None,
        query_tokens_list: Optional[List[Optional[Dict[str, float]]]] = None,
    ) -> List[List[VectorSearchResult]]:
        """Batch multiple hybrid searches into a single _msearch round-trip.

        Optimized for RAG Fusion, which issues N reformulated queries.
        Batching into one HTTP request cuts network latency by ~Nx.

        Args:
            query_texts: List of query strings.
            query_embeddings: List of embeddings, one per query.
            alpha: Weight for dense vs. BM25 (same for all queries).
            query_tokens_list: Optional sparse token weights per query
                for 3-way hybrid. Must match length of query_texts.
            Other args: Same as hybrid_search().

        Returns:
            List of result lists, one per input query.
        """
        if not query_texts or not query_embeddings:
            return []
        if limit <= 0:
            return [[] for _ in query_texts]
        if len(query_texts) != len(query_embeddings):
            raise ValueError(
                f"query_texts ({len(query_texts)}) and query_embeddings "
                f"({len(query_embeddings)}) must have the same length"
            )

        # Short-circuit: alpha=1.0 means pure vector, skip hybrid overhead.
        # Matches hybrid_search() behavior exactly.
        if alpha >= 1.0:
            return [
                self.search(
                    collection_name,
                    query_embeddings[i],
                    vector_field,
                    filter_expr,
                    limit,
                    output_fields,
                    partition_key=partition_key,
                )
                for i in range(len(query_texts))
            ]

        # Short-circuit: alpha=0.0 means pure BM25, skip vector overhead.
        # Matches hybrid_search() behavior exactly.
        if alpha <= 0.0:
            return [
                self.hybrid_search(
                    collection_name,
                    query_texts[i],
                    query_embeddings[i],
                    vector_field,
                    sparse_field,
                    filter_expr,
                    limit,
                    alpha,
                    output_fields,
                    partition_key=partition_key,
                )
                for i in range(len(query_texts))
            ]

        # Separate empty-text queries from non-empty ones.
        # hybrid_search() degrades empty query_text to pure vector search
        # (line ~558). We must do the same here to maintain behavioral
        # consistency — otherwise BM25 match on "" returns zero docs,
        # dragging down the hybrid score for that sub-query.
        non_empty_indices: List[int] = []
        empty_indices: List[int] = []
        for i, q_text in enumerate(query_texts):
            if not q_text or not q_text.strip():
                empty_indices.append(i)
            else:
                non_empty_indices.append(i)

        # Pre-allocate results array
        all_results: List[List[VectorSearchResult]] = [[] for _ in query_texts]

        # Handle empty-text queries with pure vector search
        for i in empty_indices:
            all_results[i] = self.search(
                collection_name,
                query_embeddings[i],
                vector_field=vector_field,
                filter_expr=filter_expr,
                limit=limit,
                output_fields=output_fields,
                partition_key=partition_key,
            )

        # If no non-empty queries remain, return early
        if not non_empty_indices:
            return all_results

        client = self._ensure_connected()
        os_filter = self._build_search_filter(filter_expr, partition_key)

        # Determine if 3-way hybrid is active. Only use 3-way when ALL
        # non-empty queries have tokens — otherwise the pipeline weight
        # count (3) would not match the sub-query count (2) for tokenless
        # queries, causing OpenSearch to reject those sub-searches.
        has_sparse = (
            _neural_sparse_enabled()
            and query_tokens_list
            and len(query_tokens_list) >= len(query_texts)
            and all(query_tokens_list[i] for i in non_empty_indices)
        )
        n_queries = 3 if has_sparse else 2
        pipeline_name = self._ensure_hybrid_pipeline(collection_name, alpha, n_queries)

        # Build hybrid bodies only for non-empty queries
        body_parts: List[Dict[str, Any]] = []
        for i in non_empty_indices:
            q_tokens = (
                query_tokens_list[i]
                if has_sparse and query_tokens_list and i < len(query_tokens_list)
                else None
            )
            body_parts.append(
                {
                    "index": collection_name,
                }
            )
            body_parts.append(
                self._build_hybrid_body(
                    query_texts[i],
                    query_embeddings[i],
                    vector_field,
                    sparse_field,
                    os_filter,
                    limit,
                    output_fields,
                    query_tokens=q_tokens,
                )
            )

        try:
            # Pass the pipeline as a URL query param rather than per-sub-query
            # header, matching the single-query path in hybrid_search().  One
            # pipeline applies to the whole batch (pipeline_name is computed
            # once above), so the per-header form would buy nothing while
            # requiring OpenSearch >= 2.19 and otherwise dropping every batch
            # into the sequential fallback below — defeating the round-trip
            # saving this method exists to provide.
            response = client.msearch(
                body=body_parts,
                params={"search_pipeline": pipeline_name},
            )
        except Exception as msearch_err:
            # Older clusters that reject the search_pipeline parameter still
            # fall back to sequential hybrid_search calls.
            if "search_pipeline" in str(msearch_err):
                logger.debug(
                    f"_msearch with search_pipeline not supported, "
                    f"falling back to sequential hybrid_search "
                    f"({len(non_empty_indices)} queries)"
                )
                for i in non_empty_indices:
                    q_tokens = (
                        query_tokens_list[i]
                        if has_sparse and query_tokens_list and i < len(query_tokens_list)
                        else None
                    )
                    all_results[i] = self.hybrid_search(
                        collection_name,
                        query_texts[i],
                        query_embeddings[i],
                        vector_field,
                        sparse_field,
                        filter_expr,
                        limit,
                        alpha,
                        output_fields,
                        partition_key=partition_key,
                        query_tokens=q_tokens,
                    )
                return all_results
            raise

        # Map msearch responses back to the original indices
        responses = response.get("responses", [])
        if len(responses) < len(non_empty_indices):
            logger.warning(
                f"multi_hybrid_search: expected {len(non_empty_indices)} responses "
                f"but got {len(responses)}: some queries will have empty results"
            )
        for resp_idx, resp in enumerate(responses):
            if resp_idx >= len(non_empty_indices):
                break
            orig_idx = non_empty_indices[resp_idx]
            if "error" in resp:
                logger.warning(f"multi_hybrid_search sub-query {orig_idx} failed: {resp['error']}")
            else:
                hits = resp.get("hits", {}).get("hits", [])
                all_results[orig_idx] = [self._hit_to_result(hit) for hit in hits]

        return all_results

    def flush(self, collection_name: str) -> None:
        client = self._ensure_connected()
        client.indices.refresh(index=collection_name)

    def compact(self, collection_name: str) -> None:
        client = self._ensure_connected()
        client.indices.forcemerge(index=collection_name, max_num_segments=1)
