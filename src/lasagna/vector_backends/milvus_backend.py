"""Milvus Vector Database Backend Implementation.

This module implements the VectorStoreAdapter interface for Milvus,
wrapping the existing Milvus client with the portable abstraction layer.

Supported Milvus Versions:
- Milvus Lite (embedded)
- Milvus Standalone
- Milvus Distributed

Features:
- Full CRUD operations
- HNSW and IVF index support
- JSON metadata filtering
- Optional hybrid search (with native BM25 on Standalone/Distributed)
"""

import logging
import sys
import threading
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from .base import (
    VectorStoreAdapter,
    VectorSearchResult,
    CollectionInfo,
    IndexConfig,
    SearchParams,
    BackendCapabilities,
    CollectionSchema,
    MetricType,
    IndexType,
)
from ..core.provenance import redact_url
from .filters import FilterExpression
from ._pymilvus import from_pymilvus

# Milvus imports with fallback. Through from_pymilvus: a bare import reads the
# nearest .env file into os.environ (see _pymilvus.py).
try:
    MilvusClient, DataType, MilvusCollectionSchema, MilvusFieldSchema = from_pymilvus(
        "MilvusClient", "DataType", "CollectionSchema", "FieldSchema"
    )

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    MilvusClient = None
    DataType = None
    MilvusCollectionSchema = None
    MilvusFieldSchema = None
    logger.warning("pymilvus not installed. MilvusBackend will not be available.")


class DataDirLockedError(ConnectionError):
    """Another process holds the Milvus Lite store this one tried to open.

    A ConnectionError, so code written for any failure to connect still
    catches it. The CLI recognizes it by name and prints only the fix.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        super().__init__(
            f"The Milvus Lite store at {db_path} is in use by another "
            f"process (a running `lasagna serve`, MCP server or benchmark). "
            f"Wait for it to finish or stop it, or give this run its own "
            f"store with RAG_DB_DIR=<another directory>."
        )


class _HideLockedStoreTraceback(logging.Filter):
    """Drop milvus_lite's traceback for a store another process holds.

    milvus_lite logs the lock failure with a 25-line traceback and returns
    nothing, so pymilvus raises a generic error. MilvusBackend.connect raises
    DataDirLockedError with the fix instead, which makes the traceback noise
    above the one line that matters.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        while exc is not None:
            if type(exc).__name__ == "DataDirLockedError":
                return False
            exc = exc.__cause__ or exc.__context__
        return True


logging.getLogger("milvus_lite.server_manager").addFilter(_HideLockedStoreTraceback())


def _lite_store_locked(db_path: str) -> bool:
    """Whether another process holds the Milvus Lite store's lock (see core.local_server)."""
    from lasagna.core.local_server import lite_store_locked

    return lite_store_locked(db_path)


def _served_by_this_process(db_path: str) -> bool:
    """Whether this process's own Milvus Lite server holds the store.

    milvus_lite runs its server as a child of the process that opened the store,
    and the child holds the lock. A failure to connect while our own server is
    up is some other failure (a gRPC error, a crashed child), not another
    program holding the store, and must not be reported as one.
    """
    import os

    # Read, never imported: a process that has not loaded milvus_lite has
    # started no server (and importing it here would bypass the .env guard).
    manager = sys.modules.get("milvus_lite.server_manager")
    servers = getattr(getattr(manager, "server_manager_instance", None), "_servers", None)
    return isinstance(servers, dict) and os.path.abspath(db_path) in servers


#: gRPC keepalive for a local Milvus Lite store. pymilvus pings every 10 s, idle
#: or not; the Lite server enforces gRPC's default (at most one ping per five
#: minutes without data) and closes an idle connection with GOAWAY
#: "too_many_pings" after about 20 s. The client then printed an error line on
#: every idle period, and a request that arrived as the connection closed
#: failed. A server deployment (Milvus standalone) allows the pings, so these
#: apply to Lite only. pymilvus before 3.0 ignores the argument.
_LITE_GRPC_OPTIONS = {
    "grpc.keepalive_permit_without_calls": False,
    "grpc.keepalive_time_ms": 300_000,
    "grpc.keepalive_timeout_ms": 20_000,
}


def _hit_primary_key(hit: Dict[str, Any]) -> str:
    """The primary key of a search hit, whatever pymilvus calls it.

    pymilvus 2.x returns it as ``id``; 3.x returns it under the primary field's
    own name (``chunk_id`` here) and has no ``id`` key. Reading only ``id``
    gave every chunk the id "0", which collapsed citations and any map keyed by
    chunk id (agentic context assembly, multi-query fusion) onto one chunk.
    """
    for key in ("id", "chunk_id"):
        value = hit.get(key)
        if value not in (None, ""):
            return str(value)
    return str((hit.get("entity") or {}).get("chunk_id", ""))


def _is_timeout(exc: BaseException) -> bool:
    """Whether a pymilvus or gRPC error is a deadline that ran out."""
    if isinstance(exc, TimeoutError):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return "deadline" in text or "timeout" in text or "timed out" in text


class MilvusBackend(VectorStoreAdapter):
    """Milvus implementation of VectorStoreAdapter.

    This is the default backend for Lasagna's RAG system, wrapping
    the existing MilvusClient with the portable abstraction.

    Example:
        backend = MilvusBackend(db_path="/path/to/milvus.db")
        backend.connect()

        # Create collection
        schema = CollectionSchema(fields=[...])
        backend.create_collection("my_collection", schema)

        # Insert documents
        backend.insert("my_collection", {"content": "text", "embedding": [...]})

        # Search
        results = backend.search(
            "my_collection",
            query_embedding=[...],
            filter_expr=FilterExpression.eq("session_id", "user123"),
            limit=10
        )
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs,
    ):
        """Initialize Milvus backend.

        Args:
            db_path: Path to Milvus Lite database file (local mode)
            uri: Milvus server URI (e.g., "http://localhost:19530")
            token: Authentication token for Milvus Cloud
            timeout: Connection timeout in seconds (default: 30)
            **kwargs: Additional MilvusClient arguments
        """
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is required for MilvusBackend. Install it with: pip install pymilvus"
            )
        if db_path and uri:
            # connect() used to take db_path and ignore uri, so a backend
            # configured for a server quietly wrote to a local file.
            raise ValueError(
                f"MilvusBackend got both db_path={db_path!r} (Milvus Lite) and "
                f"uri={redact_url(uri)!r} (a Milvus server). Pass one."
            )

        self._db_path = db_path
        self._uri = uri
        self._token = token
        self._timeout = timeout
        # Loading a large persisted collection takes longer than connecting,
        # but must still end: an unbounded wait hung every later run.
        self._load_timeout = float(kwargs.pop("load_timeout", 600.0))
        self._kwargs = kwargs
        self._client: Optional[MilvusClient] = None
        self._capabilities: Optional[BackendCapabilities] = None
        self._capabilities_lock = threading.Lock()
        self._connection_lock = threading.Lock()
        # Collections verified loaded on the current connection (see _ensure_loaded).
        self._loaded_collections: set = set()
        # Whether each collection has a scalar session_id field (see session_filter).
        self._session_column: Dict[str, bool] = {}
        # Whether each collection has the sparse field native hybrid search reads.
        self._native_hybrid: Dict[str, bool] = {}

    # =========================================================================
    # Capability Detection
    # =========================================================================

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return Milvus capabilities.

        Thread-safe lazy initialization of capabilities based on deployment type.
        """
        if self._capabilities is not None:
            return self._capabilities

        # Thread-safe double-checked locking
        with self._capabilities_lock:
            if self._capabilities is not None:
                return self._capabilities

            # Detect capabilities based on deployment type
            is_lite = self._db_path is not None

            self._capabilities = BackendCapabilities(
                # Milvus Lite doesn't support native BM25/sparse vectors
                supports_hybrid_search=not is_lite,
                supports_native_bm25=not is_lite,
                supports_sparse_vectors=not is_lite,
                # Index support
                supports_hnsw=True,
                supports_ivf=True,
                supports_gpu=not is_lite,
                # Filter capabilities
                supports_json_filters=True,
                supports_metadata_filtering=True,
                max_filter_complexity=100,
                # Batch capabilities
                max_batch_size=1000,
                supports_async_insert=True,
                # Partition / namespace capabilities
                supports_partitions=True,
                # Schema capabilities
                supports_dynamic_schema=True,
                max_varchar_length=65535,
                max_vector_dimension=32768,
            )

        return self._capabilities

    def collection_supports_native_hybrid(self, collection_name: str) -> bool:
        """Whether native hybrid search can run on this collection.

        A Milvus server can run it, but only on a collection with the
        ``sparse_embedding`` field ``hybrid_search`` reads, and the schema
        Lasagna creates has none: every hybrid search tried it, failed and was
        reported degraded. Read once per collection; a failed read is not kept.
        """
        if not self.capabilities.supports_hybrid_search:
            return False
        found = self._native_hybrid.get(collection_name)
        if found is None:
            try:
                fields = self.describe_fields(collection_name)
            except Exception as e:
                logger.debug(f"Could not read the fields of {collection_name}: {e}")
                return False
            found = any(f.get("name") == "sparse_embedding" for f in fields)
            self._native_hybrid[collection_name] = found
        return found

    @property
    def backend_name(self) -> str:
        """Return backend name."""
        return "milvus"

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> None:
        """Connect to Milvus.

        Thread-safe: uses ``_connection_lock`` to prevent duplicate connections
        (especially important for Milvus Lite where the DB file is exclusive).

        Raises:
            ConnectionError: If connection fails or times out
            ValueError: If neither db_path nor uri is provided
        """
        if self._client is not None:
            return

        with self._connection_lock:
            # Double-check after acquiring lock
            if self._client is not None:
                return

            try:
                if self._db_path:
                    # Milvus Lite (local file)
                    logger.info(f"Connecting to Milvus Lite: {self._db_path}")
                    self._client = MilvusClient(
                        self._db_path, timeout=self._timeout, grpc_options=dict(_LITE_GRPC_OPTIONS)
                    )
                elif self._uri:
                    # Milvus Standalone/Distributed
                    logger.info(f"Connecting to Milvus server: {redact_url(self._uri)}")
                    connect_kwargs = {"uri": self._uri, "timeout": self._timeout, **self._kwargs}
                    if self._token:
                        connect_kwargs["token"] = self._token
                    self._client = MilvusClient(**connect_kwargs)
                else:
                    raise ValueError(
                        "Either db_path (for Lite) or uri (for Standalone) is required"
                    )
            except Exception as e:
                if (
                    self._db_path
                    and _lite_store_locked(self._db_path)
                    and not _served_by_this_process(self._db_path)
                ):
                    # pymilvus reports only "Open local milvus failed"; the
                    # cause was in a traceback logged above it.
                    raise DataDirLockedError(self._db_path) from e
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    raise ConnectionError(f"Failed to connect to Milvus: {error_msg}") from e
                raise

    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.debug(f"Error closing Milvus connection: {e}")
            self._client = None
        self._loaded_collections.clear()
        self._session_column.clear()

    def is_connected(self) -> bool:
        """Check if connected to Milvus."""
        return self._client is not None

    @property
    def is_lite(self) -> bool:
        """Whether this is Milvus Lite (a local database file), not a server."""
        return bool(self._db_path)

    def _ensure_connected(self) -> MilvusClient:
        """Ensure we're connected and return the client."""
        if not self._client:
            self.connect()
        return self._client

    def _ensure_loaded(self, client: MilvusClient, collection_name: str) -> None:
        """Load a collection that this process found persisted but released.

        Milvus Lite 3.x reopens a database file with its collections in the
        ``released`` state, and search or query then fail with "call load()
        before search/get/query". Checked once per collection per connection.
        """
        if collection_name in self._loaded_collections:
            return
        try:
            state = str(
                client.get_load_state(collection_name, timeout=self._timeout).get("state", "")
            )
            if "NotLoad" in state:
                client.load_collection(collection_name, timeout=self._load_timeout)
        except Exception as e:
            if _is_timeout(e):
                where = self._db_path or self._uri
                raise TimeoutError(
                    f"Milvus did not load collection {collection_name!r} at {where} within "
                    f"{self._load_timeout:.0f}s. If an earlier run was interrupted, the store "
                    f"may be inconsistent: run `lasagna doctor`, or move the store aside and "
                    f"re-ingest."
                ) from e
            # Best effort otherwise: the read that follows reports the real error.
            logger.debug(f"Could not verify load state of {collection_name}: {e}")
        self._loaded_collections.add(collection_name)

    # =========================================================================
    # Collection Management
    # =========================================================================

    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists."""
        client = self._ensure_connected()
        return client.has_collection(collection_name)

    def create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_config: Optional[IndexConfig] = None,
    ) -> None:
        """Create a new collection."""
        client = self._ensure_connected()
        self._session_column.pop(collection_name, None)
        self._native_hybrid.pop(collection_name, None)

        # Convert our schema to Milvus schema
        milvus_schema = self._to_milvus_schema(schema)

        # Find embedding dimension from schema
        embedding_dim = None
        for field in schema.fields:
            if field.dtype == "FLOAT_VECTOR" and field.dimension:
                embedding_dim = field.dimension
                break

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            schema=milvus_schema,
            dimension=embedding_dim,
            auto_id=any(f.auto_id for f in schema.fields if f.is_primary),
        )

        # Create index if provided
        if index_config:
            # Find vector field
            vector_field = None
            for field in schema.fields:
                if field.dtype == "FLOAT_VECTOR":
                    vector_field = field.name
                    break

            if vector_field:
                self.create_index(collection_name, vector_field, index_config)

        logger.info(f"Created Milvus collection: {collection_name}")

    def drop_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        client = self._ensure_connected()
        client.drop_collection(collection_name)
        self._loaded_collections.discard(collection_name)
        self._session_column.pop(collection_name, None)
        self._native_hybrid.pop(collection_name, None)
        logger.info(f"Dropped Milvus collection: {collection_name}")

    def _describe(self, collection_name: str):
        """The server's description of the collection, and its fields as dicts."""
        client = self._ensure_connected()
        info = client.describe_collection(collection_name)

        # Extract fields
        fields = []
        if isinstance(info, dict):
            raw_fields = info.get("fields", [])
        else:
            raw_fields = getattr(info, "fields", [])

        for f in raw_fields:
            if isinstance(f, dict):
                fields.append(f)
            else:
                fields.append(
                    {
                        "name": getattr(f, "name", ""),
                        "dtype": str(getattr(f, "dtype", "")),
                        "params": getattr(f, "params", {}),
                    }
                )
        return info, fields

    def describe_collection(self, collection_name: str) -> CollectionInfo:
        """Get collection information."""
        info, fields = self._describe(collection_name)

        # Get row count
        stats = self.get_collection_stats(collection_name)
        row_count = stats.get("row_count", 0)

        return CollectionInfo(
            name=collection_name,
            fields=fields,
            row_count=row_count,
            description=info.get("description", "") if isinstance(info, dict) else "",
        )

    def describe_fields(self, collection_name: str) -> List[Dict[str, Any]]:
        """The fields alone: counting rows is a second call, as slow as the first."""
        return self._describe(collection_name)[1]

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics. Raises when the server cannot say.

        An error used to read as ``row_count: 0``, and the callers that drop
        an empty collection (`collection_is_empty`) took that for "empty":
        a timeout on a populated collection rebuilt it, deleting every
        session's documents.
        """
        client = self._ensure_connected()
        stats = client.get_collection_stats(collection_name)
        rows = stats["row_count"] if isinstance(stats, dict) else stats.row_count
        return {"row_count": int(rows)}

    def session_filter(self, collection_name: str, session_id: str) -> FilterExpression:
        """Select one session's rows: by the scalar column when the collection has one.

        Both writers (upload and benchmark ingest) store the same value in the
        ``session_id`` column and in ``metadata["session_id"]``, so the two
        filters select the same rows. The JSON path is parsed row by row: on
        Milvus Lite it made a filtered search about 20 times slower than the
        column. A collection without the column keeps the JSON path.
        """
        if self._has_session_column(collection_name):
            return FilterExpression.eq("session_id", session_id)
        return FilterExpression.json_eq("metadata", "session_id", session_id)

    def _has_session_column(self, collection_name: str) -> bool:
        known = self._session_column.get(collection_name)
        if known is not None:
            return known
        try:
            fields = self.describe_fields(collection_name)
        except Exception as e:
            # Not cached: the read that follows reports the real error.
            logger.debug(f"Could not describe {collection_name}: {e}")
            return False
        varchar = (DataType.VARCHAR, str(DataType.VARCHAR), "VARCHAR")
        known = any(
            f.get("name") == "session_id" and f.get("type", f.get("dtype")) in varchar
            for f in fields
        )
        self._session_column[collection_name] = known
        return known

    # =========================================================================
    # Index Management
    # =========================================================================

    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_config: Any,  # Can be IndexConfig or index_params builder
    ) -> None:
        """Create index on a vector field.

        Args:
            collection_name: Name of the collection
            field_name: Name of the field to index
            index_config: Either an IndexConfig object or a Milvus index_params builder
        """
        client = self._ensure_connected()

        # create_collection() already builds the index it is given. Asking
        # again makes pymilvus log a rejected RPC at ERROR with a full
        # traceback, which is the first thing a fresh install showed a user.
        try:
            if client.list_indexes(collection_name, field_name=field_name):
                logger.debug(f"Index already exists on {collection_name}.{field_name}")
                return
        except Exception as e:  # fall through: create_index reports the real error
            logger.debug(f"Could not list indexes on {collection_name}: {e}")

        # Map our IndexType to Milvus
        index_type_map = {
            IndexType.FLAT: "FLAT",
            IndexType.HNSW: "HNSW",
            IndexType.IVF_FLAT: "IVF_FLAT",
            IndexType.IVF_SQ8: "IVF_SQ8",
            IndexType.AUTOINDEX: "AUTOINDEX",
        }

        metric_type_map = {
            MetricType.COSINE: "COSINE",
            MetricType.L2: "L2",
            MetricType.IP: "IP",
        }

        try:
            # Check if index_config is already a Milvus index_params builder
            if hasattr(index_config, "_params") or hasattr(index_config, "add_index"):
                # It's a builder object from prepare_index_params()
                client.create_index(collection_name, index_config)
                logger.debug(f"Created index on {collection_name}.{field_name}")
                return

            # Otherwise, treat it as an IndexConfig object
            if self.is_lite and getattr(index_config, "index_type", None) != IndexType.FLAT:
                index_config = IndexConfig(
                    index_type=IndexType.FLAT,
                    metric_type=getattr(index_config, "metric_type", MetricType.COSINE),
                    params={},
                )
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name=field_name,
                index_type=index_type_map.get(index_config.index_type, "AUTOINDEX"),
                metric_type=metric_type_map.get(index_config.metric_type, "COSINE"),
                params=index_config.params,
            )
            client.create_index(collection_name, index_params)
            logger.debug(f"Created index on {collection_name}.{field_name}")
        except Exception as e:
            error_msg = str(e).lower()
            # Try AUTOINDEX as fallback (catch various Milvus error message patterns)
            if any(
                p in error_msg
                for p in [
                    "invalid index type",
                    "not support",
                    "unknown index type",
                    "unkown index type",
                    "invalid parameter",
                ]
            ):
                logger.warning("Index type not supported, trying AUTOINDEX...")
                try:
                    index_params = client.prepare_index_params()
                    index_params.add_index(
                        field_name=field_name,
                        index_type="AUTOINDEX",
                        metric_type=metric_type_map.get(
                            getattr(index_config, "metric_type", MetricType.COSINE), "COSINE"
                        ),
                        params={},
                    )
                    client.create_index(collection_name, index_params)
                except Exception as e2:
                    logger.warning(f"AUTOINDEX fallback also failed for {field_name}: {e2}")
            elif "already" in error_msg or "exist" in error_msg:
                logger.debug(f"Index already exists on {field_name}")
            else:
                raise

    def list_indexes(self, collection_name: str) -> List[str]:
        """List indexed fields."""
        client = self._ensure_connected()
        try:
            indexes = client.list_indexes(collection_name)
            return indexes if indexes else []
        except Exception as e:
            logger.debug(f"Could not list indexes: {e}")
            return []

    def prepare_index_params(self) -> Any:
        """Return Milvus index parameter builder for compatibility."""
        client = self._ensure_connected()
        return client.prepare_index_params()

    # =========================================================================
    # Data Operations
    # =========================================================================

    def insert(
        self,
        collection_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        partition_key: Optional[str] = None,
    ) -> List[Union[int, str]]:
        """Insert documents."""
        client = self._ensure_connected()

        # Normalize to list
        if isinstance(data, dict):
            data = [data]

        insert_kwargs: Dict[str, Any] = {
            "collection_name": collection_name,
            "data": data,
        }
        if partition_key is not None:
            insert_kwargs["partition_name"] = partition_key

        result = client.insert(**insert_kwargs)

        # Extract IDs from result
        if isinstance(result, dict):
            return result.get("ids", [])
        if hasattr(result, "primary_keys"):
            return list(result.primary_keys)
        return []

    def delete(
        self,
        collection_name: str,
        filter_expr: FilterExpression,
        partition_key: Optional[str] = None,
    ) -> int:
        """Delete documents matching filter."""
        client = self._ensure_connected()

        # Convert filter to Milvus syntax
        milvus_filter = filter_expr.to_milvus() if filter_expr else ""

        if not milvus_filter:
            raise ValueError(
                "Cannot delete with empty filter. "
                "Build a FilterExpression with at least one condition."
            )

        delete_kwargs: Dict[str, Any] = {
            "collection_name": collection_name,
            "filter": milvus_filter,
        }
        if partition_key is not None:
            delete_kwargs["partition_name"] = partition_key

        result = client.delete(**delete_kwargs)

        # Extract delete count
        if isinstance(result, dict):
            return result.get("delete_count", 0)
        if hasattr(result, "delete_count"):
            return result.delete_count
        return 0

    def query(
        self,
        collection_name: str,
        filter_expr: Optional[FilterExpression] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        partition_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query documents by filter."""
        client = self._ensure_connected()
        self._ensure_loaded(client, collection_name)

        milvus_filter = filter_expr.to_milvus() if filter_expr else ""

        query_kwargs: Dict[str, Any] = {
            "collection_name": collection_name,
            "filter": milvus_filter,
            "output_fields": output_fields,
            "limit": limit + offset,  # Milvus query doesn't have offset
        }
        if partition_key is not None:
            query_kwargs["partition_names"] = [partition_key]

        results = client.query(**query_kwargs)

        # Apply offset manually
        if offset > 0:
            results = results[offset:]

        return results[:limit]

    #: Milvus rejects a query whose ``limit + offset`` exceeds this. Because
    #: ``query()`` above emulates offset by over-fetching, paginating past it is
    #: not a slow scan — it is a hard failure partway through.
    _QUERY_LIMIT_OFFSET_CAP = 16_384

    def iter_all(
        self,
        collection_name: str,
        filter_expr=None,
        output_fields: Optional[List[str]] = None,
        *,
        partition_key: Optional[str] = None,
        page_size: int = 1000,
    ):
        """Stream every matching row using Milvus's native cursor.

        ``query_iterator`` keeps server-side position, so there is no
        ``limit + offset`` arithmetic and no 16384 ceiling. The base class
        implementation would crash on any corpus larger than that — which is
        not exotic: a few thousand documents chunked normally exceeds it.
        """
        client = self._ensure_connected()
        self._ensure_loaded(client, collection_name)
        kwargs: Dict[str, Any] = {
            "collection_name": collection_name,
            "filter": filter_expr.to_milvus() if filter_expr else "",
            "output_fields": output_fields,
            "batch_size": min(page_size, self._QUERY_LIMIT_OFFSET_CAP),
        }
        if partition_key is not None:
            kwargs["partition_names"] = [partition_key]

        try:
            iterator = client.query_iterator(**kwargs)
        except Exception:
            # Older pymilvus, or a deployment without iterator support. Fall
            # back to the paginated default rather than failing the caller --
            # but log it, because the caller silently gets the capped path.
            logger.warning(
                "Milvus query_iterator unavailable; falling back to paginated "
                "scan, which cannot read past %d rows.",
                self._QUERY_LIMIT_OFFSET_CAP,
            )
            yield from super().iter_all(
                collection_name,
                filter_expr,
                output_fields,
                partition_key=partition_key,
                page_size=page_size,
            )
            return

        try:
            while True:
                batch = iterator.next()
                if not batch:
                    return
                yield from batch
        finally:
            iterator.close()

    # =========================================================================
    # Vector Search
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
        """Perform vector similarity search."""
        client = self._ensure_connected()
        self._ensure_loaded(client, collection_name)

        # Convert filter to Milvus syntax
        milvus_filter = filter_expr.to_milvus() if filter_expr else ""

        # Build search params
        params = {}
        metric_type = "COSINE"
        actual_metric = MetricType.COSINE
        if search_params:
            metric_type = search_params.metric_type.value
            actual_metric = search_params.metric_type
            params = search_params.params
        # Milvus "distance" field semantics: COSINE/IP return similarity
        # (higher=better), L2 returns distance (lower=better)
        is_distance = actual_metric == MetricType.L2

        # Default to returning all scalar fields if none specified
        if output_fields is None:
            output_fields = ["*"]

        # Execute search
        search_kwargs: Dict[str, Any] = {
            "collection_name": collection_name,
            "data": [query_embedding],
            "anns_field": vector_field,
            "filter": milvus_filter,
            "limit": limit,
            "output_fields": output_fields,
            "search_params": {"metric_type": metric_type, **params},
        }
        if partition_key is not None:
            search_kwargs["partition_names"] = [partition_key]

        results = client.search(**search_kwargs)

        # Convert to VectorSearchResult
        search_results = []
        if results and results[0]:
            for hit in results[0]:
                # Handle both dict and Hit object formats
                if isinstance(hit, dict):
                    entity = hit.get("entity", {})
                    hit_id = _hit_primary_key(hit)
                    hit_score = hit.get("distance", 0.0)
                else:
                    # pymilvus Hit object
                    hit_id = getattr(hit, "id", 0)
                    hit_score = getattr(hit, "distance", 0.0)
                    if hasattr(hit, "entity"):
                        raw = dict(hit.entity)
                        # pymilvus may wrap fields in a nested 'entity' key
                        if "entity" in raw and isinstance(raw["entity"], dict):
                            entity = raw["entity"]
                        else:
                            entity = raw
                    else:
                        entity = {}

                search_results.append(
                    VectorSearchResult(
                        id=str(hit_id),
                        score=self.normalize_score(
                            hit_score, actual_metric, is_distance=is_distance
                        ),
                        entity=entity,
                    )
                )

        return search_results

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
    ) -> List[VectorSearchResult]:
        """Perform hybrid dense + sparse search.

        Only works on Milvus Standalone/Distributed with sparse vectors.
        Falls back to dense-only search on Milvus Lite.
        """
        if not self.capabilities.supports_hybrid_search:
            # Fallback to dense search
            return self.search(
                collection_name=collection_name,
                query_embedding=query_embedding,
                vector_field=vector_field,
                filter_expr=filter_expr,
                limit=limit,
                output_fields=output_fields,
                partition_key=partition_key,
            )

        # Native hybrid search (requires pymilvus hybrid components)
        try:
            AnnSearchRequest, RRFRanker = from_pymilvus("AnnSearchRequest", "RRFRanker")

            client = self._ensure_connected()
            self._ensure_loaded(client, collection_name)
            milvus_filter = filter_expr.to_milvus() if filter_expr else ""

            # Dense search request
            dense_request = AnnSearchRequest(
                data=[query_embedding],
                anns_field=vector_field,
                param={"metric_type": "COSINE"},
                limit=limit * 2,
            )

            # Sparse BM25 request
            sparse_request = AnnSearchRequest(
                data=[query_text],
                anns_field=sparse_field,
                param={"metric_type": "BM25"},
                limit=limit * 2,
            )

            ranker = RRFRanker()

            hybrid_kwargs: Dict[str, Any] = {
                "collection_name": collection_name,
                "reqs": [dense_request, sparse_request],
                "ranker": ranker,
                "filter": milvus_filter,
                "limit": limit,
                "output_fields": output_fields,
            }
            if partition_key is not None:
                hybrid_kwargs["partition_names"] = [partition_key]

            results = client.hybrid_search(**hybrid_kwargs)

            # Convert results (same entity extraction logic as search())
            search_results = []
            if results and results[0]:
                for hit in results[0]:
                    if isinstance(hit, dict):
                        entity = hit.get("entity", {})
                        hit_id = _hit_primary_key(hit)
                        hit_score = hit.get("distance", 0.0)
                    else:
                        # pymilvus Hit object
                        hit_id = getattr(hit, "id", 0)
                        hit_score = getattr(hit, "distance", 0.0)
                        if hasattr(hit, "entity"):
                            raw = dict(hit.entity)
                            # pymilvus may wrap fields in a nested 'entity' key
                            if "entity" in raw and isinstance(raw["entity"], dict):
                                entity = raw["entity"]
                            else:
                                entity = raw
                        else:
                            entity = {}
                    search_results.append(
                        VectorSearchResult(
                            id=str(hit_id),
                            score=self.normalize_score(
                                hit_score, MetricType.COSINE, is_distance=False
                            ),
                            entity=entity,
                        )
                    )

            return search_results

        except ImportError:
            logger.warning("Milvus hybrid search components not available")
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
    # Partition Management (tenant isolation)
    # =========================================================================

    def has_partition(self, collection_name: str, partition_name: str) -> bool:
        """Check if a partition exists in the collection."""
        client = self._ensure_connected()
        return client.has_partition(collection_name, partition_name)

    def create_partition(self, collection_name: str, partition_name: str) -> None:
        """Create a partition in the collection."""
        client = self._ensure_connected()
        client.create_partition(collection_name, partition_name)
        logger.info(f"Created partition '{partition_name}' in collection '{collection_name}'")

    def list_partitions(self, collection_name: str) -> List[str]:
        """List all partitions in the collection."""
        client = self._ensure_connected()
        return client.list_partitions(collection_name)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def flush(self, collection_name: str) -> None:
        """Flush pending writes."""
        # MilvusClient doesn't expose flush directly
        pass

    def _to_milvus_schema(self, schema: CollectionSchema) -> MilvusCollectionSchema:
        """Convert our schema to Milvus schema."""
        dtype_map = {
            "INT64": DataType.INT64,
            "VARCHAR": DataType.VARCHAR,
            "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
            "JSON": DataType.JSON,
            "BOOL": DataType.BOOL,
            "FLOAT": DataType.FLOAT,
            "DOUBLE": DataType.DOUBLE,
            "SPARSE_FLOAT_VECTOR": DataType.SPARSE_FLOAT_VECTOR,
        }

        milvus_fields = []
        for field in schema.fields:
            kwargs = {
                "name": field.name,
                "dtype": dtype_map.get(field.dtype, DataType.VARCHAR),
                "is_primary": field.is_primary,
                "description": field.description,
            }

            if field.auto_id:
                kwargs["auto_id"] = True

            if field.max_length and field.dtype == "VARCHAR":
                kwargs["max_length"] = field.max_length

            if field.dimension and field.dtype == "FLOAT_VECTOR":
                kwargs["dim"] = field.dimension

            milvus_fields.append(MilvusFieldSchema(**kwargs))

        return MilvusCollectionSchema(
            fields=milvus_fields,
            description=schema.description,
            enable_dynamic_field=schema.enable_dynamic_field,
        )

    # =========================================================================
    # Legacy Compatibility
    # =========================================================================

    @property
    def client(self) -> MilvusClient:
        """Direct access to MilvusClient for legacy code.

        DEPRECATED: Use the abstraction methods instead.
        This is provided for gradual migration only.
        """
        return self._ensure_connected()

    def search_raw(
        self,
        collection_name: str,
        data: List[List[float]],
        anns_field: str,
        filter: str,
        limit: int,
        output_fields: List[str],
        search_params: Dict[str, Any],
    ) -> Any:
        """Raw search for legacy compatibility.

        DEPRECATED: Use search() with FilterExpression instead.
        This is provided for gradual migration only.
        """
        client = self._ensure_connected()
        self._ensure_loaded(client, collection_name)
        return client.search(
            collection_name=collection_name,
            data=data,
            anns_field=anns_field,
            filter=filter,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )

    def insert_raw(
        self,
        collection_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> Any:
        """Raw insert for legacy compatibility.

        DEPRECATED: Use insert() instead.
        """
        client = self._ensure_connected()
        return client.insert(collection_name=collection_name, data=data)

    def delete_raw(self, collection_name: str, filter: str) -> Any:
        """Raw delete for legacy compatibility.

        DEPRECATED: Use delete() with FilterExpression instead.
        """
        client = self._ensure_connected()
        return client.delete(collection_name=collection_name, filter=filter)

    def query_raw(
        self,
        collection_name: str,
        filter: str,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
    ) -> Any:
        """Raw query for legacy compatibility.

        DEPRECATED: Use query() with FilterExpression instead.
        """
        client = self._ensure_connected()
        self._ensure_loaded(client, collection_name)
        return client.query(
            collection_name=collection_name,
            filter=filter,
            output_fields=output_fields,
            limit=limit,
        )
