"""PgVector (PostgreSQL) Vector Database Backend.

This module implements the VectorStoreAdapter interface for PostgreSQL
with the pgvector extension, enabling Lasagna to use a single PostgreSQL
instance for both metadata storage and vector search.

Hybrid search is implemented using pgvector for dense similarity search
and PostgreSQL's built-in tsvector/tsquery for full-text (BM25-like) ranking,
combined via Reciprocal Rank Fusion (RRF) in a single SQL query.

Requirements:
    pip install pgvector psycopg[binary] psycopg_pool

Configuration:
    - dsn: PostgreSQL connection string (e.g., postgresql://user:pass@host:5432/db)
    - pool_min: Minimum pool connections (default: 2)
    - pool_max: Maximum pool connections (default: 10)
"""

import json
import re
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from loguru import logger
from lasagna.core.provenance import redact_url

from .base import (
    BackendCapabilities,
    CollectionExistsError,
    CollectionInfo,
    CollectionSchema,
    Fields,
    IndexConfig,
    IndexType,
    MetricType,
    SearchParams,
    VectorSearchResult,
    VectorStoreAdapter,
)
from .filters import FilterExpression, FilterOperator, UnsupportedFilterError

# Check if pgvector + psycopg3 are available
try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.sql import SQL, Identifier, Literal, Placeholder
    from psycopg.types.json import Jsonb
    from psycopg_pool import ConnectionPool
    from pgvector.psycopg import register_vector

    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    psycopg = None  # type: ignore
    ConnectionPool = None  # type: ignore
    register_vector = None  # type: ignore


# =============================================================================
# Filter Translation
# =============================================================================


class _PgVectorTranslator:
    """Translate FilterExpression to parameterized PostgreSQL WHERE clauses.

    All user values flow through %s placeholders for SQL injection safety.
    Field names are validated against a strict regex.
    """

    OPS = {
        FilterOperator.EQ: "=",
        FilterOperator.NE: "!=",
        FilterOperator.GT: ">",
        FilterOperator.GTE: ">=",
        FilterOperator.LT: "<",
        FilterOperator.LTE: "<=",
    }

    @classmethod
    def translate(cls, expr: FilterExpression) -> Tuple[str, List[Any]]:
        """Convert expression to (sql_fragment, params) tuple.

        Returns:
            Tuple of (sql_string_with_placeholders, param_values_list).
            Empty string and empty list if the expression is empty.
        """
        if expr.is_empty():
            return ("", [])
        return cls._translate_node(expr)

    @classmethod
    def _translate_node(cls, expr: FilterExpression) -> Tuple[str, List[Any]]:
        op = expr.operator

        # --- Logical operators ---
        if op == FilterOperator.AND:
            parts, params = [], []
            for child in expr.children or []:
                child_sql, child_params = cls._translate_node(child)
                if child_sql:
                    parts.append(child_sql)
                    params.extend(child_params)
            if not parts:
                return ("", [])
            return ("(" + " AND ".join(parts) + ")", params)

        if op == FilterOperator.OR:
            parts, params = [], []
            for child in expr.children or []:
                child_sql, child_params = cls._translate_node(child)
                if child_sql:
                    parts.append(child_sql)
                    params.extend(child_params)
            if not parts:
                return ("", [])
            return ("(" + " OR ".join(parts) + ")", params)

        if op == FilterOperator.NOT:
            if not expr.children:
                return ("", [])
            child_sql, child_params = cls._translate_node(expr.children[0])
            if not child_sql:
                return ("", [])
            return (f"NOT ({child_sql})", child_params)

        # --- Leaf operators ---
        field = cls._validate_field(expr.field)

        if op in cls.OPS:
            return (f"{field} {cls.OPS[op]} %s", [expr.value])

        if op == FilterOperator.IN:
            if not expr.value:
                return ("FALSE", [])
            placeholders = ", ".join(["%s"] * len(expr.value))
            return (f"{field} IN ({placeholders})", list(expr.value))

        if op == FilterOperator.NOT_IN:
            if not expr.value:
                return ("TRUE", [])
            placeholders = ", ".join(["%s"] * len(expr.value))
            return (f"{field} NOT IN ({placeholders})", list(expr.value))

        if op == FilterOperator.LIKE:
            return (f"{field} LIKE %s", [expr.value])

        if op == FilterOperator.PREFIX:
            return (f"{field} LIKE %s", [f"{expr.value}%"])

        if op == FilterOperator.JSON_EQ:
            json_key = cls._validate_json_key(expr.json_key)
            return (f"{field}->>'{json_key}' = %s", [str(expr.value)])

        if op == FilterOperator.JSON_CONTAINS:
            json_key = cls._validate_json_key(expr.json_key)
            return (f"{field}->'{json_key}' @> %s::jsonb", [json.dumps(expr.value)])

        raise UnsupportedFilterError(op, "pgvector")

    @staticmethod
    def _validate_field(field: Optional[str]) -> str:
        """Validate field name to prevent SQL injection."""
        if not field or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", field):
            raise ValueError(f"Invalid field name: {field!r}")
        return field

    @staticmethod
    def _validate_json_key(key: Optional[str]) -> str:
        """Validate JSON key name."""
        if not key or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_\-]*$", key):
            raise ValueError(f"Invalid json_key: {key!r}")
        return key


# =============================================================================
# PgVector Backend
# =============================================================================


class PgVectorBackend(VectorStoreAdapter):
    """PostgreSQL + pgvector vector database backend.

    Uses a single PostgreSQL instance for vector storage and search.
    Each logical "collection" maps to a SQL table prefixed with ``vec_``
    to avoid collision with the metadata tables (rag_documents, etc.).

    Hybrid search combines pgvector ANN with PostgreSQL tsvector/tsquery
    full-text search via Reciprocal Rank Fusion (RRF) in a single query.

    Example:
        backend = PgVectorBackend(dsn="postgresql://rag:rag@localhost:5432/rag")
        backend.connect()
        results = backend.search(
            collection_name="my_collection",
            query_embedding=[0.1, 0.2, ...],
            limit=10,
        )
    """

    # Distance operators by metric type
    _DISTANCE_OPS = {
        MetricType.COSINE: "<=>",  # cosine distance
        MetricType.L2: "<->",  # L2 (Euclidean) distance
        MetricType.IP: "<#>",  # negative inner product
    }

    # Index operator classes by metric type
    _INDEX_OPS = {
        MetricType.COSINE: "vector_cosine_ops",
        MetricType.L2: "vector_l2_ops",
        MetricType.IP: "vector_ip_ops",
    }

    def __init__(
        self,
        dsn: Optional[str] = None,
        pool_min: int = 2,
        pool_max: int = 10,
        schema: str = "public",
        **kwargs,
    ):
        """Initialize PgVector backend.

        Args:
            dsn: PostgreSQL connection string. Falls back to PGVECTOR_DSN
                 then RAG_DATABASE_DSN env vars, then default local PG.
            pool_min: Minimum pool connections.
            pool_max: Maximum pool connections.
            schema: PostgreSQL schema (default: public).
        """
        if not PGVECTOR_AVAILABLE:
            raise ImportError(
                "pgvector backend requires pgvector, psycopg, and psycopg_pool. "
                "Install with: pip install pgvector 'psycopg[binary]' psycopg_pool"
            )

        import os

        self._dsn = (
            dsn
            or os.environ.get("PGVECTOR_DSN")
            or os.environ.get("RAG_DATABASE_DSN")
            or "postgresql://rag:rag@localhost:5432/rag"
        )
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._schema = schema
        self._pool: Optional[ConnectionPool] = None
        self._capabilities_cache: Optional[BackendCapabilities] = None
        self._connection_lock = threading.Lock()

    # =========================================================================
    # Capability Detection
    # =========================================================================

    @property
    def capabilities(self) -> BackendCapabilities:
        if self._capabilities_cache is None:
            self._capabilities_cache = BackendCapabilities(
                supports_hybrid_search=True,
                supports_native_bm25=False,
                supports_sparse_vectors=False,
                supports_hnsw=True,
                supports_ivf=True,
                supports_gpu=False,
                supports_json_filters=True,
                supports_metadata_filtering=True,
                max_filter_complexity=100,
                max_batch_size=5000,
                supports_async_insert=False,
                supports_partitions=False,
                supports_dynamic_schema=False,
                max_varchar_length=1048576,
                max_vector_dimension=16000,
            )
        return self._capabilities_cache

    @property
    def backend_name(self) -> str:
        return "pgvector"

    # =========================================================================
    # Connection Management
    # =========================================================================

    def connect(self) -> None:
        if self._pool is not None:
            return

        with self._connection_lock:
            if self._pool is not None:
                return

            # Mask password in logs
            logger.info(f"Connecting to PgVector: {redact_url(self._dsn)}")

            try:
                # Ensure pgvector extension exists BEFORE creating the pool.
                # register_vector() (called in pool's configure callback) needs
                # the vector type to be present in the database.
                with psycopg.connect(self._dsn) as conn:
                    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    conn.commit()

                self._pool = ConnectionPool(
                    self._dsn,
                    min_size=self._pool_min,
                    max_size=self._pool_max,
                    configure=self._configure_connection,
                )
                logger.info("PgVector backend connected and extension verified")
            except (psycopg.Error, OSError) as e:
                self._pool = None
                raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

    @staticmethod
    def _configure_connection(conn):
        """Configure each pooled connection with pgvector type adapter."""
        register_vector(conn)

    def disconnect(self) -> None:
        if self._pool:
            self._pool.close()
            self._pool = None
        logger.debug("Disconnected from PgVector")

    def is_connected(self) -> bool:
        return self._pool is not None

    @contextmanager
    def _conn(self):
        """Yield a connection from the pool."""
        if self._pool is None:
            self.connect()
        with self._pool.connection() as conn:
            yield conn

    def _ensure_connected(self):
        """Ensure pool is initialized."""
        if self._pool is None:
            self.connect()

    # =========================================================================
    # Table Name Sanitization
    # =========================================================================

    @staticmethod
    def _table_name(collection_name: str) -> str:
        """Convert collection name to safe SQL table identifier.

        Prefixed with ``vec_`` to avoid collision with metadata tables.
        Only alphanumeric and underscores allowed.
        """
        safe = re.sub(r"[^a-zA-Z0-9_]", "", collection_name)
        if not safe or not safe[0].isalpha():
            safe = "c_" + safe
        return f"vec_{safe}"

    # =========================================================================
    # Collection Management
    # =========================================================================

    def has_collection(self, collection_name: str) -> bool:
        table = self._table_name(collection_name)
        self._ensure_connected()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT EXISTS ("
                "  SELECT 1 FROM information_schema.tables "
                "  WHERE table_schema = %s AND table_name = %s"
                ")",
                [self._schema, table],
            ).fetchone()
            return row[0] if row else False

    def create_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        index_config: Optional[IndexConfig] = None,
    ) -> None:
        table = self._table_name(collection_name)

        if self.has_collection(collection_name):
            raise CollectionExistsError(f"Collection {collection_name} already exists")

        # Find vector dimension from schema fields
        dimension = None
        for f in schema.fields:
            if f.dtype == "FLOAT_VECTOR" and f.dimension:
                dimension = f.dimension
                break

        if dimension is None:
            raise ValueError("Schema must include a FLOAT_VECTOR field with dimension")

        self._ensure_connected()
        with self._conn() as conn:
            # Create table
            conn.execute(
                SQL("""
                CREATE TABLE IF NOT EXISTS {table} (
                    chunk_id     TEXT PRIMARY KEY,
                    document_id  TEXT NOT NULL,
                    session_id   TEXT NOT NULL,
                    content      TEXT NOT NULL DEFAULT '',
                    embedding    vector({dim}),
                    metadata     JSONB DEFAULT '{{}}'::jsonb,
                    content_tsv  tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', content)
                    ) STORED,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                )
            """).format(
                    table=Identifier(table),
                    dim=Literal(int(dimension)),
                )
            )

            # B-tree indexes for filtered queries
            conn.execute(
                SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} (session_id)").format(
                    idx=Identifier(f"idx_{table}_session_id"), table=Identifier(table)
                )
            )

            conn.execute(
                SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} (document_id)").format(
                    idx=Identifier(f"idx_{table}_document_id"), table=Identifier(table)
                )
            )

            # GIN index for tsvector full-text search
            conn.execute(
                SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} USING gin (content_tsv)").format(
                    idx=Identifier(f"idx_{table}_content_gin"), table=Identifier(table)
                )
            )

            # GIN index for JSONB metadata queries
            conn.execute(
                SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON {table} USING gin (metadata jsonb_path_ops)"
                ).format(idx=Identifier(f"idx_{table}_metadata_gin"), table=Identifier(table))
            )

            conn.commit()

        # Create HNSW vector index
        if index_config is None:
            index_config = IndexConfig()
        self.create_index(collection_name, Fields.EMBEDDING, index_config)

        logger.info(
            f"Created PgVector collection: {collection_name} (table={table}, dim={dimension})"
        )

    def drop_collection(self, collection_name: str) -> None:
        table = self._table_name(collection_name)
        self._ensure_connected()
        with self._conn() as conn:
            conn.execute(
                SQL("DROP TABLE IF EXISTS {table} CASCADE").format(
                    table=Identifier(table),
                )
            )
            conn.commit()
        logger.info(f"Dropped PgVector collection: {collection_name}")

    def describe_collection(self, collection_name: str) -> CollectionInfo:
        table = self._table_name(collection_name)
        self._ensure_connected()
        with self._conn() as conn:
            # Get column info
            rows = conn.execute(
                "SELECT column_name, data_type, character_maximum_length "
                "FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = %s "
                "ORDER BY ordinal_position",
                [self._schema, table],
            ).fetchall()

            if not rows:
                raise ValueError(f"Collection {collection_name} not found")

            fields = []
            for r in rows:
                fields.append(
                    {
                        "name": r[0],
                        "type": r[1],
                        "max_length": r[2],
                    }
                )

            # Get approximate row count (fast)
            count_row = conn.execute(
                SQL("SELECT reltuples::bigint FROM pg_class WHERE relname = %s"), [table]
            ).fetchone()
            row_count = int(count_row[0]) if count_row and count_row[0] >= 0 else 0

        return CollectionInfo(
            name=collection_name,
            fields=fields,
            row_count=row_count,
            description=f"PgVector collection (table={table})",
        )

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        table = self._table_name(collection_name)
        self._ensure_connected()
        with self._conn() as conn:
            row = conn.execute(
                SQL("SELECT COUNT(*) FROM {table}").format(table=Identifier(table))
            ).fetchone()
            row_count = row[0] if row else 0

        return {"row_count": row_count}

    # =========================================================================
    # Index Management
    # =========================================================================

    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_config: IndexConfig,
    ) -> None:
        table = self._table_name(collection_name)
        self._ensure_connected()

        if field_name in (Fields.EMBEDDING, "embedding", "vector"):
            # Vector index
            metric = index_config.metric_type if index_config else MetricType.COSINE
            ops_class = self._INDEX_OPS.get(metric, "vector_cosine_ops")

            if index_config and index_config.index_type == IndexType.IVF_FLAT:
                # IVFFlat index
                nlist = index_config.params.get("nlist", 100)
                idx_name = f"idx_{table}_embedding_ivfflat"
                sql = SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON {table} "
                    "USING ivfflat (embedding {ops}) WITH (lists = {nlist})"
                ).format(
                    idx=Identifier(idx_name),
                    table=Identifier(table),
                    ops=SQL(ops_class),
                    nlist=Literal(int(nlist)),
                )
            else:
                # HNSW index (default)
                m = index_config.params.get("M", 16) if index_config else 16
                ef_construction = (
                    index_config.params.get("efConstruction", 256) if index_config else 256
                )
                idx_name = f"idx_{table}_embedding_hnsw"
                sql = SQL(
                    "CREATE INDEX IF NOT EXISTS {idx} ON {table} "
                    "USING hnsw (embedding {ops}) "
                    "WITH (m = {m}, ef_construction = {efc})"
                ).format(
                    idx=Identifier(idx_name),
                    table=Identifier(table),
                    ops=SQL(ops_class),
                    m=Literal(int(m)),
                    efc=Literal(int(ef_construction)),
                )

            with self._conn() as conn:
                conn.execute(sql)
                conn.commit()

            logger.debug(f"Created vector index {idx_name} on {table}")
        else:
            # Payload / scalar index
            idx_name = f"idx_{table}_{field_name}"
            with self._conn() as conn:
                conn.execute(
                    SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} ({field})").format(
                        idx=Identifier(idx_name),
                        table=Identifier(table),
                        field=Identifier(field_name),
                    )
                )
                conn.commit()
            logger.debug(f"Created scalar index {idx_name} on {table}")

    def list_indexes(self, collection_name: str) -> List[str]:
        table = self._table_name(collection_name)
        self._ensure_connected()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = %s AND schemaname = %s",
                [table, self._schema],
            ).fetchall()
        return [r[0] for r in rows]

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

        table = self._table_name(collection_name)
        ids: List[Union[int, str]] = []
        batch_size = self.get_effective_batch_size(len(data))

        self._ensure_connected()
        with self._conn() as conn:
            for batch_start in range(0, len(data), batch_size):
                batch = data[batch_start : batch_start + batch_size]
                params: List[Any] = []

                for item in batch:
                    chunk_id = item.get("chunk_id") or item.get("id") or str(uuid.uuid4())
                    ids.append(chunk_id)

                    embedding = item.get("embedding") or item.get("vector")
                    if embedding is None:
                        raise ValueError("Each record must have 'embedding' or 'vector'")

                    content = item.get("content", "")
                    session_id = item.get("session_id", "")
                    document_id = item.get("document_id", "")
                    metadata = item.get("metadata", {})

                    params.extend(
                        [
                            chunk_id,
                            document_id,
                            session_id,
                            content,
                            np.array(embedding, dtype=np.float32),
                            Jsonb(metadata) if isinstance(metadata, dict) else metadata,
                        ]
                    )

                # Build multi-row VALUES clause using psycopg.sql composition
                row_sql = SQL("({})").format(SQL(", ").join([Placeholder()] * 6))
                values_sql = SQL(", ").join([row_sql] * len(batch))

                sql = SQL(
                    "INSERT INTO {table} "
                    "(chunk_id, document_id, session_id, content, embedding, metadata) "
                    "VALUES {values} "
                    "ON CONFLICT (chunk_id) DO UPDATE SET "
                    "embedding = EXCLUDED.embedding, "
                    "content = EXCLUDED.content, "
                    "metadata = EXCLUDED.metadata, "
                    "document_id = EXCLUDED.document_id, "
                    "session_id = EXCLUDED.session_id"
                ).format(table=Identifier(table), values=values_sql)

                conn.execute(sql, params)

            conn.commit()

        logger.debug(f"Inserted {len(ids)} vectors into PgVector collection {collection_name}")
        return ids

    def delete(
        self,
        collection_name: str,
        filter_expr: FilterExpression,
        partition_key: Optional[str] = None,
    ) -> int:
        where_sql, where_params = self._build_where(filter_expr)
        if not where_sql:
            raise ValueError(
                "Cannot delete with empty filter: this would delete all rows. "
                "Build a FilterExpression with at least one condition."
            )

        table = self._table_name(collection_name)
        self._ensure_connected()
        with self._conn() as conn:
            cur = conn.execute(
                SQL("DELETE FROM {table} {where}").format(
                    table=Identifier(table),
                    where=SQL(where_sql),
                ),
                where_params,
            )
            conn.commit()
            return cur.rowcount

    def query(
        self,
        collection_name: str,
        filter_expr: Optional[FilterExpression] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        partition_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        table = self._table_name(collection_name)
        where_sql, where_params = self._build_where(filter_expr)

        # Determine columns to select
        if output_fields and output_fields != ["*"]:
            # Validate each field name
            for f in output_fields:
                _PgVectorTranslator._validate_field(f)
            cols = SQL(", ").join(Identifier(f) for f in output_fields)
        else:
            cols = SQL("chunk_id, document_id, session_id, content, metadata")

        self._ensure_connected()
        with self._conn() as conn:
            cur = conn.cursor(row_factory=dict_row)
            rows = cur.execute(
                SQL("SELECT {cols} FROM {table} {where} LIMIT %s OFFSET %s").format(
                    cols=cols,
                    table=Identifier(table),
                    where=SQL(where_sql),
                ),
                where_params + [limit, offset],
            ).fetchall()

        return [dict(r) for r in rows]

    def list_document_ids(
        self,
        collection_name: str,
        session_id: Optional[str] = None,
        *,
        limit: int = 10_000,
        offset: int = 0,
    ) -> List[str]:
        """List distinct document_ids using a native SQL DISTINCT query."""
        table = self._table_name(collection_name)
        self._ensure_connected()
        with self._conn() as conn:
            cur = conn.cursor()
            if session_id:
                rows = cur.execute(
                    SQL(
                        "SELECT DISTINCT document_id FROM {table} WHERE session_id = %s LIMIT %s OFFSET %s"
                    ).format(table=Identifier(table)),
                    [session_id, limit, offset],
                ).fetchall()
            else:
                rows = cur.execute(
                    SQL("SELECT DISTINCT document_id FROM {table} LIMIT %s OFFSET %s").format(
                        table=Identifier(table)
                    ),
                    [limit, offset],
                ).fetchall()
        return [row[0] for row in rows]

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
        table = self._table_name(collection_name)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Determine metric and distance operator
        metric = MetricType.COSINE
        if search_params and search_params.metric_type:
            metric = search_params.metric_type
        dist_op = self._DISTANCE_OPS.get(metric, "<=>")

        # HNSW ef_search parameter
        ef_search = 128
        if search_params and search_params.params:
            ef_search = search_params.params.get("ef", 128)

        where_sql, where_params = self._build_where(filter_expr)

        self._ensure_connected()
        with self._conn() as conn:
            # SET LOCAL is transaction-scoped, safe for concurrent pool use
            conn.execute(SQL("SET LOCAL hnsw.ef_search = {}").format(Literal(int(ef_search))))

            query_sql = SQL(
                "SELECT chunk_id, document_id, session_id, content, metadata, "
                "embedding {dist_op} %s AS distance "
                "FROM {table} {where} "
                "ORDER BY embedding {dist_op} %s "
                "LIMIT %s"
            ).format(
                dist_op=SQL(dist_op),
                table=Identifier(table),
                where=SQL(where_sql),
            )

            # Parameter order must match left-to-right %s in the final SQL:
            # 1. embedding <=> %s (SELECT distance) — before WHERE
            # 2. %s... from where_sql (WHERE clause)
            # 3. embedding <=> %s (ORDER BY)
            # 4. %s (LIMIT)
            cur = conn.cursor(row_factory=dict_row)
            rows = cur.execute(
                query_sql,
                [query_embedding] + list(where_params) + [query_embedding, limit],
            ).fetchall()

            conn.commit()  # Release SET LOCAL scope

        results = []
        for row in rows:
            raw_distance = float(row["distance"])
            # pgvector distance operator semantics:
            #   COSINE (<=>): cosine distance [0, 2], lower=identical
            #   L2     (<->): L2 distance [0, ∞), lower=identical
            #   IP     (<#>): negative inner product, lower=higher similarity
            if metric == MetricType.IP:
                # <#> returns negative IP; negate to get similarity
                score = self.normalize_score(-raw_distance, metric, is_distance=False)
            else:
                score = self.normalize_score(raw_distance, metric, is_distance=True)

            entity = {
                Fields.CHUNK_ID: row["chunk_id"],
                Fields.DOCUMENT_ID: row["document_id"],
                Fields.SESSION_ID: row["session_id"],
                Fields.CONTENT: row["content"],
                Fields.METADATA: row["metadata"] or {},
            }

            results.append(
                VectorSearchResult(
                    id=row["chunk_id"],
                    score=score,
                    entity=entity,
                    raw_score=raw_distance,
                )
            )

        return results

    # =========================================================================
    # Hybrid Search (dense + full-text via RRF)
    # =========================================================================

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_embedding: List[float],
        vector_field: str = "embedding",
        sparse_field: str = "content_tsv",
        filter_expr: Optional[FilterExpression] = None,
        limit: int = 10,
        alpha: float = 0.7,
        output_fields: Optional[List[str]] = None,
        partition_key: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Hybrid search combining pgvector ANN + tsvector full-text via RRF.

        Uses two CTEs — one for dense vector search, one for tsvector
        full-text search — then fuses via Reciprocal Rank Fusion (RRF)
        weighted by ``alpha`` (1.0 = pure dense, 0.0 = pure sparse).
        """
        table = self._table_name(collection_name)
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Quick check: if no alphanumeric content, skip hybrid and go dense-only
        if not re.search(r"\w", query_text):
            return self.search(
                collection_name=collection_name,
                query_embedding=query_embedding,
                vector_field=vector_field,
                filter_expr=filter_expr,
                limit=limit,
                output_fields=output_fields,
                partition_key=partition_key,
            )

        where_sql, where_params = self._build_where(filter_expr)

        # RRF constant (standard from literature)
        k = 60
        # Retrieve more candidates than final limit for fusion
        candidate_limit = limit * 3

        self._ensure_connected()
        with self._conn() as conn:
            conn.execute(SQL("SET LOCAL hnsw.ef_search = {}").format(Literal(128)))

            # Hybrid RRF query following the official pgvector-python pattern:
            #   - RANK() for correct tie handling (vs ROW_NUMBER)
            #   - plainto_tsquery() via cross-join: evaluated once, referenced by alias
            #   - Where clause duplicated in both CTEs for correct filtering
            #
            # The sparse CTE cross-joins with plainto_tsquery() so the tsquery is
            # computed once and referenced as "query" — no repeated %s params.
            hybrid_sql = SQL("""
                WITH dense AS (
                    SELECT chunk_id,
                           RANK() OVER (ORDER BY embedding <=> %s) AS rank_d
                    FROM {table}
                    {where}
                    ORDER BY embedding <=> %s
                    LIMIT %s
                ),
                sparse AS (
                    SELECT chunk_id,
                           RANK() OVER (
                               ORDER BY ts_rank_cd(content_tsv, query) DESC
                           ) AS rank_s
                    FROM {table}, plainto_tsquery('english', %s) query
                    {where_with_tsv}
                    ORDER BY ts_rank_cd(content_tsv, query) DESC
                    LIMIT %s
                ),
                fused AS (
                    SELECT COALESCE(d.chunk_id, s.chunk_id) AS chunk_id,
                           COALESCE(%s * (1.0 / (%s + d.rank_d)), 0) +
                           COALESCE((1.0 - %s) * (1.0 / (%s + s.rank_s)), 0) AS rrf_score
                    FROM dense d
                    FULL OUTER JOIN sparse s ON d.chunk_id = s.chunk_id
                )
                SELECT f.rrf_score, t.chunk_id, t.document_id, t.session_id,
                       t.content, t.metadata
                FROM fused f
                JOIN {table} t ON f.chunk_id = t.chunk_id
                ORDER BY f.rrf_score DESC
                LIMIT %s
            """).format(
                table=Identifier(table),
                where=SQL(where_sql),
                where_with_tsv=SQL(
                    (where_sql + " AND " if where_sql else "WHERE ") + "content_tsv @@ query"
                ),
            )

            # Parameter order matches left-to-right %s in the final SQL.
            params: List[Any] = []
            # dense CTE
            params.append(query_embedding)  # RANK ORDER BY embedding <=> %s
            params.extend(where_params)  # WHERE clause %s placeholders
            params.append(query_embedding)  # ORDER BY embedding <=> %s
            params.append(candidate_limit)  # LIMIT %s
            # sparse CTE (plainto_tsquery cross-join — only one %s for query_text)
            params.append(query_text)  # plainto_tsquery('english', %s)
            params.extend(where_params)  # WHERE clause %s placeholders
            params.append(candidate_limit)  # LIMIT %s
            # fused CTE
            params.extend([float(alpha), k, float(alpha), k])
            # final
            params.append(limit)

            cur = conn.cursor(row_factory=dict_row)
            rows = cur.execute(hybrid_sql, params).fetchall()
            conn.commit()

        results = []
        for row in rows:
            rrf_score = float(row["rrf_score"])
            entity = {
                Fields.CHUNK_ID: row["chunk_id"],
                Fields.DOCUMENT_ID: row["document_id"],
                Fields.SESSION_ID: row["session_id"],
                Fields.CONTENT: row["content"],
                Fields.METADATA: row["metadata"] or {},
            }
            results.append(
                VectorSearchResult(
                    id=row["chunk_id"],
                    score=min(1.0, rrf_score),
                    entity=entity,
                    raw_score=rrf_score,
                )
            )

        return results

    # =========================================================================
    # Helper Methods
    # =========================================================================

    @staticmethod
    def _build_where(
        filter_expr: Optional[FilterExpression],
    ) -> Tuple[str, List[Any]]:
        """Build WHERE clause from FilterExpression.

        Returns:
            Tuple of ("WHERE ...", [params]) or ("", []).
        """
        if filter_expr is None or filter_expr.is_empty():
            return ("", [])
        sql, params = _PgVectorTranslator.translate(filter_expr)
        if sql:
            return (f"WHERE {sql}", params)
        return ("", [])
