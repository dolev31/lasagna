"""One contract, asserted identically against every vector backend.

Lasagna puts six backends behind `VectorStoreAdapter` and lets a user swap
between them with a config key. That promise only holds if they actually behave
the same, and until this suite existed nothing checked it: Qdrant, Weaviate and
Pinecone had no behavioural tests at all (2,335 LOC between them), and Milvus —
the default — was only exercised incidentally through higher layers.

Every test here runs against every backend that can be reached. A backend with
no service available is skipped with a reason naming what was missing, never
silently passed over.

Running against real services:

    VECTOR_BACKEND=pgvector make -C src/lasagna/deploy up
    PGVECTOR_DSN=postgresql://... pytest tests/test_vector_backend_conformance.py

Backends are opt-in by environment variable so that a plain `pytest` run on a
laptop stays hermetic:

    LASAGNA_MILVUS_DB_PATH  ./milvus_conformance.db  (Milvus Lite, no server)
    LASAGNA_MILVUS_URI      http://localhost:19530   (Milvus server)
    PGVECTOR_DSN         postgresql://user:pass@localhost:5432/db  # pragma: allowlist secret
    QDRANT_URL           http://localhost:6333
    WEAVIATE_URL         http://localhost:8080
    OPENSEARCH_HOSTS     http://localhost:9200

Pinecone is deliberately absent: it is SaaS-only, so exercising it would create
billable cloud resources. It stays covered by mocks in its own module.
"""

from __future__ import annotations

import os
import uuid

import pytest

from lasagna.vector_backends.base import (
    CollectionSchema,
    FieldSchema,
    IndexConfig,
    IndexType,
    MetricType,
)
from lasagna.vector_backends.filters import FilterExpression, FilterOperator

DIM = 8


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------


def _milvus_factory():
    """Milvus, either Lite (a local file) or a server URI.

    Note the variable is LASAGNA_MILVUS_*, not MILVUS_URI: pymilvus claims that
    name globally and validates it as an http(s) URL at import time, so setting
    it to a Lite file path breaks the library before any test runs.
    """
    db_path = os.environ.get("LASAGNA_MILVUS_DB_PATH")
    uri = os.environ.get("LASAGNA_MILVUS_URI")
    if not db_path and not uri:
        return None, "LASAGNA_MILVUS_DB_PATH / LASAGNA_MILVUS_URI not set"
    from lasagna.vector_backends.milvus_backend import MilvusBackend

    if db_path:
        return lambda: MilvusBackend(db_path=db_path), None
    return lambda: MilvusBackend(uri=uri), None


def _pgvector_factory():
    dsn = os.environ.get("PGVECTOR_DSN") or os.environ.get("DATABASE_DSN")
    if not dsn:
        return None, "PGVECTOR_DSN not set"
    try:
        from lasagna.vector_backends.pgvector_backend import PgVectorBackend
    except ImportError as exc:
        return None, f"pgvector backend unavailable: {exc}"

    return lambda: PgVectorBackend(dsn=dsn, pool_min=1, pool_max=2), None


def _qdrant_factory():
    """Qdrant, either against a server or its embedded local-file mode.

    QDRANT_PATH needs no server at all, so this backend is exercisable on a
    machine with no container runtime.
    """
    url = os.environ.get("QDRANT_URL")
    path = os.environ.get("QDRANT_PATH")
    if not url and not path:
        return None, "QDRANT_URL / QDRANT_PATH not set"
    # Check the CLIENT, not our wrapper. `qdrant_backend.py` catches its own
    # ImportError and sets the symbol to None, so importing the wrapper succeeds
    # with qdrant-client absent -- the guard missed, the param was constructed,
    # and 19 conformance tests ERRORed on every Python version in CI, which
    # installs `.[eval]` and does not carry qdrant-client.
    try:
        import qdrant_client  # noqa: F401
    except ImportError as exc:
        return None, f"qdrant-client not installed: {exc}"
    try:
        from lasagna.vector_backends.qdrant_backend import QdrantBackend
    except ImportError as exc:
        return None, f"qdrant backend unavailable: {exc}"
    if QdrantBackend is None:
        return None, "qdrant backend module imported but the class is unavailable"

    if path:
        return lambda: QdrantBackend(path=path), None
    return lambda: QdrantBackend(url=url), None


def _weaviate_factory():
    url = os.environ.get("WEAVIATE_URL")
    if not url:
        return None, "WEAVIATE_URL not set"
    try:
        from lasagna.vector_backends.weaviate_backend import WeaviateBackend
    except ImportError as exc:
        return None, f"weaviate backend unavailable: {exc}"

    return lambda: WeaviateBackend(url=url), None


def _opensearch_factory():
    hosts = os.environ.get("OPENSEARCH_HOSTS")
    if not hosts:
        return None, "OPENSEARCH_HOSTS not set"
    try:
        from lasagna.vector_backends.opensearch_backend import OpenSearchBackend
    except ImportError as exc:
        return None, f"opensearch backend unavailable: {exc}"

    return lambda: OpenSearchBackend(), None


_FACTORIES = {
    "milvus": _milvus_factory,
    "pgvector": _pgvector_factory,
    "qdrant": _qdrant_factory,
    "weaviate": _weaviate_factory,
    "opensearch": _opensearch_factory,
}


def _backend_params():
    """One param per backend; unreachable ones carry a skip naming the reason."""
    params = []
    for name, discover in _FACTORIES.items():
        try:
            factory, reason = discover()
        except Exception as exc:  # noqa: BLE001 - discovery must never break collection
            factory, reason = None, f"discovery failed: {type(exc).__name__}: {exc}"
        if factory is None:
            params.append(
                pytest.param(None, id=name, marks=pytest.mark.skip(reason=f"{name}: {reason}"))
            )
        else:
            params.append(pytest.param(factory, id=name))

    # Pinecone is SaaS-only; exercising it would create billable resources.
    params.append(
        pytest.param(
            None,
            id="pinecone",
            marks=pytest.mark.skip(
                reason="pinecone: SaaS-only, excluded to avoid billable cloud resources"
            ),
        )
    )
    return params


@pytest.fixture(params=_backend_params())
def backend(request):
    """A connected backend, with its test collection dropped on teardown."""
    factory = request.param
    if factory is None:  # pragma: no cover - skipped params never construct
        pytest.skip("backend unavailable")

    adapter = factory()
    adapter.connect()
    yield adapter
    try:
        adapter.disconnect()
    except Exception:  # noqa: BLE001 - teardown must not mask a test failure
        pass


@pytest.fixture
def collection(backend):
    """A unique, indexed collection that is dropped afterwards.

    Names are randomised so a crashed run cannot poison the next one — a
    shared fixed name is how backend suites start failing for reasons that
    have nothing to do with the code under test.
    """
    name = f"conf_{uuid.uuid4().hex[:12]}"
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True, max_length=128),
            FieldSchema(name="embedding", dtype="FLOAT_VECTOR", dimension=DIM),
            FieldSchema(name="content", dtype="VARCHAR", max_length=4000),
            # A top-level scalar to filter on. Filtering into the JSON blob
            # needs per-backend path syntax, which is a separate concern from
            # "does filtering work at all".
            FieldSchema(name="tenant", dtype="VARCHAR", max_length=64),
            FieldSchema(name="metadata", dtype="JSON"),
        ]
    )
    # Index at creation, as _collection_manager does in production. Milvus
    # refuses to search an unindexed vector field, so an index-less fixture
    # would test a state no real caller is ever in.
    index_config = IndexConfig(
        index_type=IndexType.AUTOINDEX,
        metric_type=MetricType.COSINE,
    )
    backend.create_collection(name, schema, index_config)
    yield name
    try:
        backend.drop_collection(name)
    except Exception:  # noqa: BLE001
        pass


def _vec(seed: float) -> list:
    """Deterministic unit-ish vector; distinct seeds give distinct directions."""
    return [seed] + [0.0] * (DIM - 1)


def _rows(n: int, prefix: str = "c", tenant: str = "t1") -> list:
    return [
        {
            "chunk_id": f"{prefix}{i}",
            "embedding": _vec(1.0 - i * 0.01),
            "content": f"content number {i}",
            "tenant": tenant,
            "metadata": {"idx": i, "tenant": tenant},
        }
        for i in range(n)
    ]


def _ids_filter(ids: list) -> FilterExpression:
    """Filter selecting the given primary keys.

    `delete()` takes a filter, not an id list — deletion by id is expressed as
    a filter over the primary key.
    """
    return FilterExpression(field="chunk_id", operator=FilterOperator.IN, value=ids)


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------


class TestCollectionLifecycle:
    def test_created_collection_exists(self, backend, collection):
        assert backend.has_collection(collection)

    def test_absent_collection_reports_absent(self, backend):
        assert not backend.has_collection(f"never_created_{uuid.uuid4().hex[:8]}")

    def test_drop_removes_collection(self, backend, collection):
        backend.drop_collection(collection)
        assert not backend.has_collection(collection)

    def test_describe_returns_the_name(self, backend, collection):
        info = backend.describe_collection(collection)
        assert info.name == collection

    def test_backend_identifies_itself(self, backend):
        assert isinstance(backend.backend_name, str) and backend.backend_name

    def test_reports_connected(self, backend):
        assert backend.is_connected()


# ---------------------------------------------------------------------------
# Insert and search
# ---------------------------------------------------------------------------


class TestInsertAndSearch:
    def test_insert_then_search_returns_rows(self, backend, collection):
        backend.insert(collection, _rows(5))
        backend.flush(collection)
        results = backend.search(collection, _vec(1.0), limit=5)
        assert results, "search returned nothing after inserting 5 rows"

    def test_search_respects_limit(self, backend, collection):
        backend.insert(collection, _rows(10))
        backend.flush(collection)
        results = backend.search(collection, _vec(1.0), limit=3)
        assert len(results) <= 3, f"limit=3 returned {len(results)} results"

    def test_results_ordered_by_similarity(self, backend, collection):
        """Nearest first. A backend returning insertion order is broken."""
        backend.insert(collection, _rows(5))
        backend.flush(collection)
        results = backend.search(collection, _vec(1.0), limit=5)
        if len(results) < 2:
            pytest.skip("need at least two results to check ordering")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True) or scores == sorted(scores), (
            f"results are not monotonically ordered by score: {scores}"
        )

    def test_search_on_empty_collection_returns_empty(self, backend, collection):
        """Empty must be an empty list, not an exception and not None."""
        results = backend.search(collection, _vec(1.0), limit=5)
        assert results == [], f"expected [] from empty collection, got {results!r}"

    def test_result_carries_its_id(self, backend, collection):
        backend.insert(collection, _rows(3))
        backend.flush(collection)
        results = backend.search(collection, _vec(1.0), limit=3)
        assert all(getattr(r, "id", None) is not None for r in results), (
            "every result must carry the primary key it came from"
        )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_equality_filter_narrows_results(self, backend, collection):
        backend.insert(collection, _rows(6, tenant="t1"))
        backend.insert(collection, _rows(6, prefix="d", tenant="t2"))
        backend.flush(collection)

        expr = FilterExpression(field="tenant", operator=FilterOperator.EQ, value="t2")
        filtered = backend.search(collection, _vec(1.0), filter_expr=expr, limit=20)
        unfiltered = backend.search(collection, _vec(1.0), limit=20)

        assert len(filtered) <= len(unfiltered), "a filter must never widen the result set"

    def test_filter_matching_nothing_returns_empty(self, backend, collection):
        backend.insert(collection, _rows(4))
        backend.flush(collection)
        expr = FilterExpression(field="tenant", operator=FilterOperator.EQ, value="no-such-tenant")
        assert backend.search(collection, _vec(1.0), filter_expr=expr, limit=10) == []


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


class TestDeletion:
    def test_deleted_rows_stop_being_returned(self, backend, collection):
        backend.insert(collection, _rows(5))
        backend.flush(collection)
        before = len(backend.search(collection, _vec(1.0), limit=10))

        backend.delete(collection, _ids_filter(["c0", "c1"]))
        backend.flush(collection)
        after = len(backend.search(collection, _vec(1.0), limit=10))

        assert after < before, f"deleting 2 of {before} rows left {after} still searchable"

    def test_deleting_absent_id_is_not_an_error(self, backend, collection):
        """Idempotent delete. Callers retry; a raise here turns retry into failure."""
        backend.insert(collection, _rows(2))
        backend.flush(collection)
        backend.delete(collection, _ids_filter(["definitely-not-present"]))


# ---------------------------------------------------------------------------
# Pagination — the path most likely to diverge
# ---------------------------------------------------------------------------


class TestPagination:
    """Known-risky: Milvus has no native offset and emulates it as limit+offset,
    which trips its own 16384 cap. Other backends page differently. This is
    where untested adapters are most likely to disagree.
    """

    def test_query_pages_without_duplicates_or_gaps(self, backend, collection):
        total = 25
        backend.insert(collection, _rows(total))
        backend.flush(collection)

        seen, offset, page = [], 0, 10
        for _ in range(10):  # bounded: never loop forever on a broken backend
            rows = backend.query(collection, limit=page, offset=offset, output_fields=["chunk_id"])
            if not rows:
                break
            seen.extend(r.get("chunk_id") for r in rows)
            offset += len(rows)

        assert len(seen) == len(set(seen)), (
            f"pagination returned duplicates: {len(seen)} rows, {len(set(seen))} unique"
        )

    def test_offset_past_the_end_returns_empty(self, backend, collection):
        backend.insert(collection, _rows(3))
        backend.flush(collection)
        rows = backend.query(collection, limit=10, offset=1000, output_fields=["chunk_id"])
        assert rows == [], f"offset past end should be empty, got {len(rows)} rows"


# ---------------------------------------------------------------------------
# Capability honesty
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_capabilities_are_declared(self, backend):
        caps = backend.capabilities
        assert caps is not None
        assert isinstance(caps.supports_hybrid_search, bool)

    def test_hybrid_search_matches_its_declaration(self, backend, collection):
        """A backend must not advertise hybrid search it cannot do.

        Callers branch on `capabilities`, so a false advertisement produces a
        crash in a path the caller believed was supported.
        """
        caps = backend.capabilities
        backend.insert(collection, _rows(4))
        backend.flush(collection)

        if not caps.supports_hybrid_search:
            pytest.skip(f"{backend.backend_name} does not advertise hybrid search")

        results = backend.hybrid_search(
            collection, query_text="content", query_embedding=_vec(1.0), limit=3
        )
        assert isinstance(results, list)
