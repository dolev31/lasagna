"""Tests for the PgVector (PostgreSQL + pgvector) vector database backend.

Unit tests run without PostgreSQL — they test the filter translator,
table name sanitization, and score normalization.

Integration tests (marked @pytest.mark.integration) require a running
PostgreSQL instance with the pgvector extension. They test the full
lifecycle: connect → create_collection → insert → search → hybrid_search
→ delete → drop → disconnect.

Run unit tests only:
    pytest tests/test_pgvector_backend.py -v -k "not integration"

Run all (requires PostgreSQL with pgvector on localhost:5432):
    PGVECTOR_DSN=postgresql://rag:rag@localhost:5432/rag \
    pytest tests/test_pgvector_backend.py -v
"""

import os
import uuid

import pytest

from lasagna.vector_backends.filters import (
    FilterExpression,
    FilterOperator,
)

# ---------------------------------------------------------------------------
# Conditional imports — skip gracefully if pgvector is not installed
# ---------------------------------------------------------------------------

try:
    from lasagna.vector_backends.pgvector_backend import (
        PGVECTOR_AVAILABLE,
        PgVectorBackend,
        _PgVectorTranslator,
    )
except ImportError:
    PGVECTOR_AVAILABLE = False
    PgVectorBackend = None  # type: ignore
    _PgVectorTranslator = None  # type: ignore

pytestmark = pytest.mark.skipif(
    not PGVECTOR_AVAILABLE,
    reason="pgvector / psycopg not installed",
)


# ============================================================================
# Unit Tests — Filter Translator
# ============================================================================


class TestPgVectorTranslator:
    """Test _PgVectorTranslator.translate() produces correct parameterized SQL."""

    def test_empty_expression(self):
        expr = FilterExpression(operator=FilterOperator.AND, children=[])
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == ""
        assert params == []

    def test_eq(self):
        expr = FilterExpression.eq("session_id", "user123")
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == "session_id = %s"
        assert params == ["user123"]

    def test_ne(self):
        expr = FilterExpression.ne("status", "deleted")
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == "status != %s"
        assert params == ["deleted"]

    def test_gt_gte_lt_lte(self):
        for op_name, op_enum, symbol in [
            ("gt", FilterOperator.GT, ">"),
            ("gte", FilterOperator.GTE, ">="),
            ("lt", FilterOperator.LT, "<"),
            ("lte", FilterOperator.LTE, "<="),
        ]:
            expr = FilterExpression(field="score", operator=op_enum, value=0.5)
            sql, params = _PgVectorTranslator.translate(expr)
            assert sql == f"score {symbol} %s", f"Failed for {op_name}"
            assert params == [0.5]

    def test_in(self):
        expr = FilterExpression.in_list("document_id", ["d1", "d2", "d3"])
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == "document_id IN (%s, %s, %s)"
        assert params == ["d1", "d2", "d3"]

    def test_in_empty_list(self):
        expr = FilterExpression(field="document_id", operator=FilterOperator.IN, value=[])
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == "FALSE"
        assert params == []

    def test_not_in(self):
        expr = FilterExpression(field="status", operator=FilterOperator.NOT_IN, value=["a", "b"])
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == "status NOT IN (%s, %s)"
        assert params == ["a", "b"]

    def test_like(self):
        expr = FilterExpression(field="content", operator=FilterOperator.LIKE, value="%quantum%")
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == "content LIKE %s"
        assert params == ["%quantum%"]

    def test_prefix(self):
        expr = FilterExpression(field="filename", operator=FilterOperator.PREFIX, value="report_")
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == "filename LIKE %s"
        assert params == ["report_%"]

    def test_and(self):
        expr = FilterExpression.eq("session_id", "s1") & FilterExpression.eq("document_id", "d1")
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == "(session_id = %s AND document_id = %s)"
        assert params == ["s1", "d1"]

    def test_or(self):
        expr = FilterExpression.eq("status", "ready") | FilterExpression.eq("status", "processing")
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == "(status = %s OR status = %s)"
        assert params == ["ready", "processing"]

    def test_not(self):
        expr = ~FilterExpression.eq("status", "deleted")
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == "NOT (status = %s)"
        assert params == ["deleted"]

    def test_nested_and_or(self):
        expr = FilterExpression.eq("session_id", "s1") & (
            FilterExpression.eq("status", "ready") | FilterExpression.eq("status", "processing")
        )
        sql, params = _PgVectorTranslator.translate(expr)
        assert "session_id = %s" in sql
        assert "(status = %s OR status = %s)" in sql
        assert params == ["s1", "ready", "processing"]

    def test_json_eq(self):
        expr = FilterExpression.json_eq("metadata", "author", "alice")
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == "metadata->>'author' = %s"
        assert params == ["alice"]

    def test_invalid_field_name_raises(self):
        with pytest.raises(ValueError, match="Invalid field name"):
            _PgVectorTranslator._validate_field("'; DROP TABLE--")

    def test_invalid_json_key_raises(self):
        with pytest.raises(ValueError, match="Invalid json_key"):
            _PgVectorTranslator._validate_json_key("'; DROP TABLE--")

    def test_filter_expression_via_translator(self):
        """Test translating a FilterExpression via _PgVectorTranslator directly."""
        expr = FilterExpression.eq("session_id", "abc")
        sql, params = _PgVectorTranslator.translate(expr)
        assert sql == "session_id = %s"
        assert params == ["abc"]


# ============================================================================
# Unit Tests — Table Name Sanitization
# ============================================================================


class TestTableNameSanitization:
    def test_normal_name(self):
        assert PgVectorBackend._table_name("my_collection") == "vec_my_collection"

    def test_strips_special_chars(self):
        assert PgVectorBackend._table_name("my-collection!@#") == "vec_mycollection"

    def test_numeric_prefix_gets_c_prefix(self):
        assert PgVectorBackend._table_name("123abc") == "vec_c_123abc"

    def test_empty_name_gets_c_prefix(self):
        assert PgVectorBackend._table_name("---") == "vec_c_"


# ============================================================================
# Unit Tests — Score Normalization
# ============================================================================


class TestScoreNormalization:
    def setup_method(self):
        # Create instance without connecting
        self.backend = PgVectorBackend.__new__(PgVectorBackend)

    def test_cosine_distance_zero_is_perfect(self):
        score = self.backend.normalize_score(0.0, is_distance=True)
        assert score == 1.0

    def test_cosine_distance_two_is_zero(self):
        score = self.backend.normalize_score(2.0, is_distance=True)
        assert score == 0.0

    def test_cosine_distance_mid(self):
        score = self.backend.normalize_score(1.0, is_distance=True)
        assert score == pytest.approx(0.5)


# ============================================================================
# Unit Tests — Library Adapter Usage (Jsonb, SQL composition)
# ============================================================================


class TestLibraryAdapterUsage:
    """Verify we use pgvector/psycopg library adapters instead of raw casts."""

    def test_no_vector_casts_in_source(self):
        """The source should not contain '::vector' casts — register_vector()
        makes psycopg3 auto-convert Python lists to pg vector type."""
        import inspect

        source = inspect.getsource(PgVectorBackend)
        assert "::vector" not in source, (
            "Found ::vector cast — register_vector() type adapter handles "
            "list-to-vector conversion automatically"
        )

    def test_insert_uses_jsonb_adapter(self):
        """The insert method should use psycopg.types.json.Jsonb for metadata
        instead of json.dumps() + '::jsonb' cast."""
        import inspect

        source = inspect.getsource(PgVectorBackend.insert)
        assert "Jsonb(" in source, "insert() should use Jsonb() adapter"
        assert "::jsonb" not in source, "insert() should not use ::jsonb cast"

    def test_sql_composition_no_string_concat(self):
        """INSERT SQL should use SQL().format() composition, not string
        concatenation with SQL() objects."""
        import inspect

        source = inspect.getsource(PgVectorBackend.insert)
        # Should use Placeholder() for VALUES
        assert "Placeholder()" in source, "insert() should use Placeholder()"
        # Should NOT concatenate Python strings into SQL()
        assert "+ values_sql +" not in source, (
            "insert() should not concatenate strings with SQL objects"
        )

    def test_hybrid_search_uses_official_patterns(self):
        """hybrid_search should follow official pgvector-python RRF patterns:
        - RANK() (not ROW_NUMBER) for correct tie handling
        - plainto_tsquery() (not to_tsquery with manual parsing)
        - Cross-join pattern for single tsquery evaluation"""
        import inspect

        source = inspect.getsource(PgVectorBackend.hybrid_search)
        assert "RANK()" in source, "Should use RANK() for correct RRF tie handling"
        assert "ROW_NUMBER()" not in source, "Should not use ROW_NUMBER() for RRF"
        assert "plainto_tsquery" in source, "Should use plainto_tsquery (official pattern)"
        # The SQL should not contain manual to_tsquery calls with %s params
        # (plainto_tsquery cross-join pattern evaluates tsquery once by alias)
        sql_lines = [
            line
            for line in source.split("\n")
            if "to_tsquery" in line and "%s" in line and "plainto" not in line
        ]
        assert len(sql_lines) == 0, (
            f"Found manual to_tsquery with %s param — use plainto_tsquery: {sql_lines}"
        )


# ============================================================================
# Integration Tests — Full Lifecycle (requires PostgreSQL with pgvector)
# ============================================================================

_INTEGRATION_DSN = os.environ.get("PGVECTOR_DSN", "postgresql://rag:rag@localhost:5432/rag")


def _pg_available() -> bool:
    """Check if PostgreSQL with pgvector is reachable."""
    if not PGVECTOR_AVAILABLE:
        return False
    try:
        backend = PgVectorBackend(dsn=_INTEGRATION_DSN, pool_min=1, pool_max=2)
        backend.connect()
        backend.disconnect()
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _pg_available(), reason="PostgreSQL with pgvector not available")
class TestPgVectorIntegration:
    """Integration tests that run against a real PostgreSQL instance."""

    COLLECTION = f"test_{uuid.uuid4().hex[:8]}"
    DIM = 4  # Small dimension for testing speed

    @pytest.fixture(autouse=True)
    def backend(self):
        """Provide a connected backend, drop test collection on teardown."""
        self._backend = PgVectorBackend(dsn=_INTEGRATION_DSN, pool_min=1, pool_max=2)
        self._backend.connect()
        yield self._backend
        # Cleanup
        try:
            self._backend.drop_collection(self.COLLECTION)
        except Exception:
            pass
        self._backend.disconnect()

    def _create_test_collection(self):
        from lasagna.vector_backends.base import (
            CollectionSchema,
            FieldSchema,
        )

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True, max_length=128),
                FieldSchema(name="embedding", dtype="FLOAT_VECTOR", dimension=self.DIM),
                FieldSchema(name="content", dtype="VARCHAR", max_length=4000),
                FieldSchema(name="metadata", dtype="JSON"),
            ]
        )
        self._backend.create_collection(self.COLLECTION, schema)

    # --- Collection Management ---

    def test_create_and_has_collection(self):
        assert not self._backend.has_collection(self.COLLECTION)
        self._create_test_collection()
        assert self._backend.has_collection(self.COLLECTION)

    def test_describe_collection(self):
        self._create_test_collection()
        info = self._backend.describe_collection(self.COLLECTION)
        assert info.name == self.COLLECTION
        field_names = [f["name"] for f in info.fields]
        assert "chunk_id" in field_names
        assert "embedding" in field_names

    def test_drop_collection(self):
        self._create_test_collection()
        self._backend.drop_collection(self.COLLECTION)
        assert not self._backend.has_collection(self.COLLECTION)

    # --- Insert and Search ---

    def test_insert_and_search(self):
        self._create_test_collection()

        self._backend.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "hello world of quantum computing",
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                    "metadata": {"page": 1},
                },
                {
                    "chunk_id": "c2",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "goodbye cruel world of classical computing",
                    "embedding": [0.0, 1.0, 0.0, 0.0],
                    "metadata": {"page": 2},
                },
            ],
        )

        results = self._backend.search(
            collection_name=self.COLLECTION,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            limit=2,
        )

        assert len(results) == 2
        assert results[0].id == "c1"  # Closest match
        assert results[0].score > results[1].score

    def test_search_with_filter(self):
        self._create_test_collection()

        self._backend.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "document one",
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                    "metadata": {},
                },
                {
                    "chunk_id": "c2",
                    "document_id": "d2",
                    "session_id": "s1",
                    "content": "document two",
                    "embedding": [0.9, 0.1, 0.0, 0.0],
                    "metadata": {},
                },
            ],
        )

        results = self._backend.search(
            collection_name=self.COLLECTION,
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            filter_expr=FilterExpression.eq("document_id", "d2"),
            limit=10,
        )

        assert len(results) == 1
        assert results[0].id == "c2"

    def test_upsert_overwrites(self):
        self._create_test_collection()

        self._backend.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "original",
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                    "metadata": {},
                }
            ],
        )

        # Upsert same chunk_id with different content
        self._backend.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "updated",
                    "embedding": [0.0, 1.0, 0.0, 0.0],
                    "metadata": {},
                }
            ],
        )

        stats = self._backend.get_collection_stats(self.COLLECTION)
        assert stats["row_count"] == 1  # Not duplicated

    # --- Delete ---

    def test_delete(self):
        self._create_test_collection()

        self._backend.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "keep",
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                    "metadata": {},
                },
                {
                    "chunk_id": "c2",
                    "document_id": "d2",
                    "session_id": "s1",
                    "content": "remove",
                    "embedding": [0.0, 1.0, 0.0, 0.0],
                    "metadata": {},
                },
            ],
        )

        deleted = self._backend.delete(
            self.COLLECTION,
            FilterExpression.eq("document_id", "d2"),
        )
        assert deleted == 1

        stats = self._backend.get_collection_stats(self.COLLECTION)
        assert stats["row_count"] == 1

    # --- Hybrid Search ---

    def test_hybrid_search(self):
        self._create_test_collection()

        self._backend.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "quantum entanglement is a fascinating phenomenon",
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                    "metadata": {},
                },
                {
                    "chunk_id": "c2",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "classical mechanics describes macroscopic objects",
                    "embedding": [0.0, 1.0, 0.0, 0.0],
                    "metadata": {},
                },
            ],
        )

        results = self._backend.hybrid_search(
            collection_name=self.COLLECTION,
            query_text="quantum entanglement",
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            limit=2,
        )

        assert len(results) >= 1
        # First result should be c1 (matches both dense and sparse)
        assert results[0].id == "c1"

    # --- Query (non-vector) ---

    def test_query_by_filter(self):
        self._create_test_collection()

        self._backend.insert(
            self.COLLECTION,
            [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "session_id": "s1",
                    "content": "hello",
                    "embedding": [1.0, 0.0, 0.0, 0.0],
                    "metadata": {"author": "alice"},
                }
            ],
        )

        rows = self._backend.query(
            self.COLLECTION,
            filter_expr=FilterExpression.eq("session_id", "s1"),
        )

        assert len(rows) == 1
        assert rows[0]["chunk_id"] == "c1"

    # --- Capabilities ---

    def test_capabilities(self):
        caps = self._backend.capabilities
        assert caps.supports_hybrid_search is True
        assert caps.supports_hnsw is True
        assert caps.supports_native_bm25 is False
        assert caps.supports_partitions is False

    def test_backend_name(self):
        assert self._backend.backend_name == "pgvector"
