"""On Milvus a search is confined to its session by the ``session_id`` column.

It was confined by ``metadata["session_id"]``, a path Milvus parses row by row.
Measured on Milvus Lite with 18,442 chunks in three sessions, a filtered search
took about 270 ms against about 12 ms on the column, and that filter was most of
the time of every query a benchmark or a server ran. Every collection this
package creates has the column, and both writers store in it the session they
also put in the metadata, so the two filters select the same rows. A collection
made without the column keeps the JSON path.

Each search also counted the collection's rows twice and used neither count.

The Milvus Lite tests are opt-in like the rest of the suite's:

    LASAGNA_MILVUS_DB_PATH=./milvus.db pytest tests/test_milvus_filters_sessions_by_column.py
"""

from __future__ import annotations

import collections
import os
from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.api._collection_manager import CollectionManager
from lasagna.core.config import rag_config
from lasagna.knowledge_evaluation_sdk._fast_ingest import _flush_insert
from lasagna.vector_backends.base import CollectionSchema, FieldSchema
from lasagna.vector_backends.filters import FilterExpression
from lasagna.vector_backends.milvus_backend import MilvusBackend

#: What Milvus Lite's describe_collection returns for a collection this package created.
COLUMNS = [
    {"name": "chunk_id", "type": "VARCHAR", "params": {"max_length": 128}, "is_primary": True},
    {"name": "document_id", "type": "VARCHAR", "params": {"max_length": 128}},
    {"name": "session_id", "type": "VARCHAR", "params": {"max_length": 128}},
    {"name": "content", "type": "VARCHAR", "params": {"max_length": 65535}},
    {"name": "embedding", "type": "FLOAT_VECTOR", "params": {"dim": 384}},
    {"name": "metadata", "type": "JSON", "params": {}},
]
WITHOUT_COLUMN = [f for f in COLUMNS if f["name"] != "session_id"]


class _Client:
    """Stands in for MilvusClient and records what the backend asked of it."""

    def __init__(self, fields):
        self.fields = fields
        self.calls = collections.Counter()
        self.filters = []

    def has_collection(self, name, **kw):
        return True

    def describe_collection(self, name, **kw):
        self.calls["describe_collection"] += 1
        return {"collection_name": name, "description": "", "fields": self.fields}

    def get_collection_stats(self, name, **kw):
        self.calls["get_collection_stats"] += 1
        return {"row_count": 7}

    def get_load_state(self, name, **kw):
        return {"state": "Loaded"}

    def search(self, **kw):
        self.filters.append(kw["filter"])
        return [[]]

    def drop_collection(self, name, **kw):
        pass


def _milvus(fields):
    backend = MilvusBackend(db_path="never-opened.db")
    backend._client = _Client(fields)
    return backend, backend._client


class TestTheFilter:
    def test_a_collection_with_the_column_is_filtered_on_it(self):
        backend, _ = _milvus(COLUMNS)
        assert backend.session_filter("c", "s1").to_milvus() == 'session_id == "s1"'

    def test_a_collection_without_it_keeps_the_json_path(self):
        backend, _ = _milvus(WITHOUT_COLUMN)
        assert backend.session_filter("c", "s1").to_milvus() == 'metadata["session_id"] == "s1"'

    def test_the_schema_is_read_once_and_again_after_the_collection_is_dropped(self):
        backend, client = _milvus(WITHOUT_COLUMN)
        for _ in range(3):
            backend.session_filter("c", "s1")
        assert client.calls["describe_collection"] == 1
        backend.drop_collection("c")
        client.fields = COLUMNS
        assert backend.session_filter("c", "s1").to_milvus() == 'session_id == "s1"'

    def test_a_schema_that_cannot_be_read_falls_back_and_is_asked_again(self):
        backend, client = _milvus(COLUMNS)

        def unreadable(name, **kw):
            raise RuntimeError("collection not loaded")

        client.describe_collection = unreadable
        assert backend.session_filter("c", "s1").to_milvus() == 'metadata["session_id"] == "s1"'
        del client.describe_collection
        assert backend.session_filter("c", "s1").to_milvus() == 'session_id == "s1"'

    def test_other_backends_filter_on_their_top_level_field(self):
        from lasagna.vector_backends.base import VectorStoreAdapter

        assert VectorStoreAdapter.session_filter(None, "c", "s1") == FilterExpression.eq(
            "session_id", "s1"
        )


class TestASearch:
    @pytest.fixture
    def orchestrator(self, monkeypatch):
        monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
        backend, client = _milvus(COLUMNS)
        o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
        o.embedding_manager = SimpleNamespace(
            get_cache_stats=lambda: {}, encode_query=lambda q: np.ones(3)
        )
        o.hybrid_searcher = SimpleNamespace()
        o.reranker = o.query_enhancer = o._auto_merger = None
        o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: backend)
        o._default_backend = backend
        o._get_session_config = lambda sid: rag_config
        o._session_embedder = lambda config: o.embedding_manager
        return o, client

    ARM = dict(
        enable_hybrid_search=False,
        enable_reranking=False,
        enable_query_routing=False,
        enable_query_classification=False,
        enable_search_cache=False,
        enable_hyde=False,
    )

    def test_it_filters_on_the_column_and_does_not_count_rows(self, orchestrator):
        o, client = orchestrator
        o.search("q", "sess1", top_k=5, query_config_overrides=self.ARM)
        assert client.filters == ['session_id == "sess1"']
        assert client.calls["get_collection_stats"] == 0

    def test_its_embedding_size_check_reads_the_schema_without_counting_rows(self):
        backend, client = _milvus(COLUMNS)
        manager = SimpleNamespace(_default_backend=backend)
        assert CollectionManager.get_collection_embedding_dim(manager, "c") == 384
        assert client.calls["get_collection_stats"] == 0


class _Recorder:
    backend_name = "milvus"

    def __init__(self):
        self.rows = []

    def insert(self, collection_name, data, partition_key=None):
        self.rows.extend(data)


def test_benchmark_ingest_writes_one_session_to_the_column_and_the_metadata():
    """A chunker's own `session_id` key must not move a chunk to another session."""
    store = _Recorder()
    chunks = [
        {"chunk_id": "d_chunk_0", "document_id": "d", "content": "x", "embedding": [1.0]},
        {
            "chunk_id": "d_chunk_1",
            "document_id": "d",
            "content": "y",
            "embedding": [1.0],
            "metadata": {"session_id": "someone-else", "source": "chunker"},
        },
    ]
    _flush_insert(chunks, store, "c", None, "arm", False, None)
    assert [(r["session_id"], r["metadata"]["session_id"]) for r in store.rows] == [
        ("arm", "arm"),
        ("arm", "arm"),
    ]
    assert store.rows[1]["metadata"]["source"] == "chunker"


# ---------------------------------------------------------------------------
# Against Milvus Lite
# ---------------------------------------------------------------------------

milvus_lite = pytest.mark.skipif(
    not os.environ.get("LASAGNA_MILVUS_DB_PATH"),
    reason="Milvus Lite is opt-in: set LASAGNA_MILVUS_DB_PATH to run",
)
DIM = 16
SESSIONS = ("alpha", "beta")


def _vectors(n, seed):
    v = np.random.default_rng(seed).normal(size=(n, DIM))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).tolist()


@pytest.fixture(scope="module")
def lite(tmp_path_factory):
    backend = MilvusBackend(db_path=str(tmp_path_factory.mktemp("milvus") / "sessions.db"))
    backend.connect()
    yield backend
    backend.disconnect()


def _fill(backend, name):
    """Both sessions hold the same vectors, so a leak would rank first."""
    vectors = _vectors(30, seed=7)
    for session in SESSIONS:
        chunks = [
            {
                "chunk_id": f"{session}_d{i // 3}_chunk_{i % 3}",
                "document_id": f"{session}_d{i // 3}",
                "content": f"chunk {i} of {session}",
                "embedding": vec,
                "metadata": {"chunk_index": i % 3},
            }
            for i, vec in enumerate(vectors)
        ]
        _flush_insert(chunks, backend, name, None, session, False, None)
    return vectors


def _hits(backend, name, vector, flt):
    return [
        (h.id, h.score, h.entity["metadata"]["session_id"])
        for h in backend.search(
            name, vector, filter_expr=flt, limit=100, output_fields=["metadata"]
        )
    ]


@milvus_lite
def test_a_session_never_sees_another_and_both_filters_agree(lite):
    name = "isolation"
    CollectionManager.create_collection_on_backend(None, lite, name, DIM, rag_config)
    try:
        vectors = _fill(lite, name)
        for session in SESSIONS:
            column = lite.session_filter(name, session)
            assert column.to_milvus() == f'session_id == "{session}"'
            json_path = FilterExpression.json_eq("metadata", "session_id", session)
            for vector in vectors[:10]:
                hits = _hits(lite, name, vector, column)
                assert len(hits) == 30 and {s for _, _, s in hits} == {session}
                assert hits == _hits(lite, name, vector, json_path)
            rows = lite.query(name, column, output_fields=["chunk_id"], limit=1000)
            same = lite.query(name, json_path, output_fields=["chunk_id"], limit=1000)
            assert {r["chunk_id"] for r in rows} == {r["chunk_id"] for r in same}
            assert sorted(lite.list_document_ids(name, session)) == [
                f"{session}_d{i}" for i in range(10)
            ]
    finally:
        lite.drop_collection(name)


@milvus_lite
def test_a_collection_without_the_column_is_still_isolated(lite):
    name = "no_column"
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True, max_length=128),
            FieldSchema(name="document_id", dtype="VARCHAR", max_length=128),
            FieldSchema(name="content", dtype="VARCHAR", max_length=4000),
            FieldSchema(name="embedding", dtype="FLOAT_VECTOR", dimension=DIM),
            FieldSchema(name="metadata", dtype="JSON"),
        ]
    )
    from lasagna.vector_backends.base import IndexConfig, IndexType, MetricType

    lite.create_collection(name, schema, IndexConfig(IndexType.FLAT, MetricType.COSINE))
    try:
        vectors = _vectors(4, seed=3)
        lite.insert(
            name,
            [
                {
                    "chunk_id": f"{s}_{i}",
                    "document_id": f"{s}_doc",
                    "content": "x",
                    "embedding": v,
                    "metadata": {"session_id": s},
                }
                for s in SESSIONS
                for i, v in enumerate(vectors)
            ],
        )
        flt = lite.session_filter(name, "alpha")
        assert flt.to_milvus() == 'metadata["session_id"] == "alpha"'
        hits = _hits(lite, name, vectors[0], flt)
        assert len(hits) == 4 and {s for _, _, s in hits} == {"alpha"}
    finally:
        lite.drop_collection(name)
