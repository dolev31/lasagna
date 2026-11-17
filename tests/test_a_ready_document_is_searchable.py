"""A document is marked ready only once a search can find it.

A Milvus server serves searches at the collection's level, Bounded by default,
so a search sent right after an insert missed the new rows. In the default
Compose stack, a search sent as a document turned ``ready`` missed it in 27
of 30 tries, and every one found it a second later; the live Milvus job of
CI failed the same way once (``test_a_switch_to_a_model_the_image_lacks_
leaves_the_document_searchable``). The pipeline now asks the store to wait
until a search sees the document's chunks before it writes ``ready``: one
Strong ``count(*)`` on a Milvus server, nothing on a store whose writes are
searchable when they return. Searches keep the fast Bounded level.
"""

from __future__ import annotations

import pytest
from loguru import logger

from lasagna.vector_backends.base import VectorStoreAdapter
from lasagna.vector_backends.filters import FilterExpression
from lasagna.vector_backends.milvus_backend import MilvusBackend
from tests._document_harness import MODEL_B, SESSION, Harness, MemoryVectors
from tests.test_a_milvus_server_reads_its_own_writes import _BoundedServer

TEXT = "the reranker lowered ndcg in the ablation and the dense retriever kept recall " * 3


class _WaitingVectors(MemoryVectors):
    """Records what the pipeline had written when it asked the store to catch up."""

    def __init__(self, harness_row, **kw):
        super().__init__(**kw)
        self._row = harness_row
        self.waits = []
        self.fail_wait = None

    def wait_until_searchable(self, collection_name, filter_expr, partition_key=None):
        document_id = filter_expr.value
        row = self._row(document_id) or {}
        stored = {
            name: [r for r in rows if r.get("document_id") == document_id]
            for name, rows in self.collections.items()
        }
        self.waits.append(
            {
                "collection": collection_name,
                "filter": (filter_expr.field, filter_expr.value),
                "partition_key": partition_key,
                "status": row.get("status"),
                "row_collection": row.get("collection_name"),
                "stored": {name: len(rows) for name, rows in stored.items() if rows},
            }
        )
        if self.fail_wait is not None:
            raise self.fail_wait


def _harness(tmp_path, monkeypatch):
    holder = {}
    vectors = _WaitingVectors(lambda doc_id: holder["h"].row(doc_id))
    h = Harness(tmp_path, monkeypatch, vectors=vectors)
    holder["h"] = h
    return h, vectors


def test_a_new_document_waits_for_search_before_it_is_ready(tmp_path, monkeypatch):
    h, vectors = _harness(tmp_path, monkeypatch)
    doc = h.upload(h.write(tmp_path, "paper.txt", TEXT))
    assert h.searchable(doc.document_id)
    (wait,) = vectors.waits
    assert wait["filter"] == ("document_id", doc.document_id)
    assert wait["collection"] == h.row(doc.document_id)["collection_name"]
    # Its chunks were stored, and the row did not say ready yet.
    assert wait["stored"] == {wait["collection"]: h.row(doc.document_id)["chunks_count"]}
    assert wait["status"] != "ready"


def test_a_reindex_waits_after_the_old_copy_is_retired_and_before_the_row_moves(
    tmp_path, monkeypatch
):
    h, vectors = _harness(tmp_path, monkeypatch)
    doc = h.upload(h.write(tmp_path, "paper.txt", TEXT))
    old_collection = h.row(doc.document_id)["collection_name"]
    h.settings = dict(MODEL_B)
    h.pipeline.reindex_document(doc.document_id, SESSION)
    new_collection = h.row(doc.document_id)["collection_name"]
    assert new_collection != old_collection and h.searchable(doc.document_id)

    wait = vectors.waits[-1]
    assert wait["collection"] == new_collection
    # The new copy is stored and the old one gone, so a search after the wait
    # sees neither a missing document nor two copies of it.
    assert wait["stored"] == {new_collection: h.row(doc.document_id)["chunks_count"]}
    # The row still named the old copy: it moves only after the wait.
    assert wait["row_collection"] == old_collection


def test_a_failed_wait_is_logged_and_the_document_is_still_ready(tmp_path, monkeypatch):
    h, vectors = _harness(tmp_path, monkeypatch)
    vectors.fail_wait = TimeoutError("count(*) timed out")
    seen = []
    sink = logger.add(seen.append, level="WARNING", format="{message}")
    try:
        doc = h.upload(h.write(tmp_path, "paper.txt", TEXT))
    finally:
        logger.remove(sink)
    assert h.searchable(doc.document_id)
    assert any(
        doc.document_id in str(m) and "TimeoutError" in str(m) and "marked ready" in str(m)
        for m in seen
    )


def test_a_store_without_the_method_is_not_warned_about(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch, vectors=MemoryVectors())
    assert not hasattr(h.vectors, "wait_until_searchable")
    seen = []
    sink = logger.add(seen.append, level="WARNING", format="{message}")
    try:
        doc = h.upload(h.write(tmp_path, "paper.txt", TEXT))
    finally:
        logger.remove(sink)
    assert h.searchable(doc.document_id)
    assert not any("Could not confirm that a search sees" in str(m) for m in seen)


# -- the Milvus backend -------------------------------------------------------


class _SearchingServer(_BoundedServer):
    """The bounded server, answering searches at the default level too."""

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return [
            [{"id": r["chunk_id"], "distance": 0.9, "entity": r} for r in self._visible(kwargs)]
        ]


def _server_backend(**kw) -> MilvusBackend:
    backend = MilvusBackend(**kw)
    backend._client = _SearchingServer([{"document_id": "doc_old", "chunk_id": "c0"}])
    backend._ensure_loaded = lambda client, name: None
    return backend


def test_on_a_server_a_default_read_misses_a_new_row_until_the_wait():
    backend = _server_backend(uri="http://milvus:19530")
    backend._client.rows.append({"document_id": "doc_new", "chunk_id": "c1"})
    new = FilterExpression.eq("document_id", "doc_new")
    assert all(r["document_id"] != "doc_new" for r in backend.query("rag_documents", limit=10))

    backend.wait_until_searchable("rag_documents", new, partition_key="t_s1")
    name, probe = backend._client.calls[-1]
    assert name == "query"
    assert probe["consistency_level"] == "Strong"
    assert probe["output_fields"] == ["count(*)"]
    assert probe["filter"] == new.to_milvus()
    assert probe["partition_names"] == ["t_s1"]
    # The reads after it, at the fast default, see the new row: a query and a
    # search, which stays on the fast default.
    assert any(r["document_id"] == "doc_new" for r in backend.query("rag_documents", limit=10))
    assert "consistency_level" not in backend._client.calls[-1][1]
    hits = backend.search("rag_documents", [0.1, 0.2], limit=10)
    name, sent = backend._client.calls[-1]
    assert name == "search" and "consistency_level" not in sent
    assert "c1" in {str(h.id) for h in hits}


def test_without_a_partition_the_probe_names_none():
    backend = _server_backend(uri="http://milvus:19530")
    backend.wait_until_searchable("rag_documents", FilterExpression.eq("document_id", "d"))
    assert "partition_names" not in backend._client.calls[-1][1]


def test_milvus_lite_waits_for_nothing(tmp_path):
    backend = _server_backend(db_path=str(tmp_path / "lite.db"))
    backend.wait_until_searchable("rag_documents", FilterExpression.eq("document_id", "d"))
    assert backend._client.calls == []


def test_a_failed_probe_raises_for_the_pipeline_to_log():
    backend = _server_backend(uri="http://milvus:19530")

    def refuse(**kwargs):
        raise TimeoutError("server busy")

    backend._client.query = refuse
    with pytest.raises(TimeoutError):
        backend.wait_until_searchable("rag_documents", FilterExpression.eq("document_id", "d"))


def test_the_base_store_waits_for_nothing():
    assert (
        VectorStoreAdapter.wait_until_searchable(
            object(), "c", FilterExpression.eq("document_id", "d")
        )
        is None
    )


def test_a_pool_waits_on_one_instance_and_frees_it():
    from lasagna.vector_backends.pool import PooledVectorBackend

    def factory():
        backend = _server_backend(uri="http://milvus:19530")
        backend.connect = lambda: None
        return backend

    pool = PooledVectorBackend(factory=factory, pool_size=1)
    pool.connect()
    pool.wait_until_searchable("rag_documents", FilterExpression.eq("document_id", "d"))
    (member,) = pool._members
    assert member._client.calls[-1][1]["consistency_level"] == "Strong"
    # The only instance is free again.
    assert pool.query("rag_documents", limit=1) is not None


# -- the benchmark's fast ingest --------------------------------------------


@pytest.mark.asyncio
async def test_the_fast_ingest_waits_once_for_the_whole_corpus(monkeypatch):
    from lasagna.knowledge_evaluation_sdk._fast_ingest import fast_ingest_corpus
    from tests.test_fast_ingest import (
        MockBackend,
        MockCollectionManager,
        MockConfig,
        MockVectorBackend,
        _make_corpus,
        _make_corpus_to_valid,
        _PartitionedTenants,
    )

    class _Waiting(MockVectorBackend):
        def __init__(self):
            super().__init__()
            self.waits = []

        def wait_until_searchable(self, collection_name, filter_expr, partition_key=None):
            self.waits.append(
                (
                    collection_name,
                    filter_expr.field,
                    filter_expr.value,
                    partition_key,
                    len(self.inserted),
                )
            )

    monkeypatch.setattr("lasagna.core.tenant.get_tenant_manager", lambda: _PartitionedTenants())
    backend = MockBackend()
    store = _Waiting()
    backend._vector_backend = store
    backend._collection_manager = MockCollectionManager(store, backend._embedding_manager)
    corpus = _make_corpus(3)
    total, ingested, errors = await fast_ingest_corpus(
        corpus=corpus,
        corpus_to_valid=_make_corpus_to_valid(corpus),
        ready_ids=set(),
        session_id="s1",
        config=MockConfig(),
        backend=backend,
    )
    assert total > 0 and not errors
    # Once, for the session, after every chunk was inserted.
    assert store.waits == [("test_collection", "session_id", "s1", "tenant_s1", total)]
