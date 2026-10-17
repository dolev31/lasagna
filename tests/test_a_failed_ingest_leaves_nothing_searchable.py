"""A document that fails to index leaves no chunk that search can serve.

``process_document`` did no cleanup after the vector insert, and the upload
only marked the row failed, so the chunks of a failed document stayed in the
store; search hid only archived documents and served them. A Celery retry, or
the broker delivering the message again after its worker died, inserted the
same chunk ids again, and Milvus ``insert`` kept both. Garbage collection then
told the user to upload again, and the passage came back twice. In a
replacement, a failure after the old chunks were retired deleted the new file
while the row still named the old file and collection: the new chunks had no
row pointing at them.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.core.config import rag_config
from tests._document_harness import MODEL_B, SESSION, Harness, MemoryVectors

TEXT = "the reranker lowered ndcg in the ablation and the dense retriever kept recall " * 3


@pytest.fixture(params=["insert", "upsert"])
def harness(request, tmp_path, monkeypatch):
    vectors = None if request.param == "insert" else MemoryVectors(upsert=True)
    h = Harness(tmp_path, monkeypatch, vectors=vectors)
    h.monkeypatch = monkeypatch
    return h


def _keyword_index_fails(h):
    h.settings = {**h.settings, "enable_hybrid_search": True}
    h.pipeline.hybrid_searcher.index_chunks = _raise(RuntimeError("the keyword index is full"))


def _parents_fail(h):
    h.settings = {**h.settings, "enable_hierarchical": True}
    chunk = h._chunk

    def with_parents(**k):
        k.pop("return_parents", None)
        return chunk(**k), [{"stable_id": "p", "document_id": k["document_id"]}]

    import lasagna.api._document_pipeline as pipeline_mod

    h.monkeypatch.setattr(pipeline_mod, "chunk_document", with_parents)
    h.monkeypatch.setattr(h.db, "save_parent_chunks", _raise(OSError("disk full")))


def _ready_cannot_be_written(h):
    update = h.db.update_document_status

    def refuse_ready(document_id, status, **kw):
        if status == "ready":
            raise OSError("database is locked")
        return update(document_id, status, **kw)

    h.monkeypatch.setattr(h.db, "update_document_status", refuse_ready)


def _raise(exc):
    def fail(*a, **k):
        raise exc

    return fail


AFTER_INSERT = {
    "keyword_index": _keyword_index_fails,
    "parent_chunks": _parents_fail,
    "ready_status": _ready_cannot_be_written,
}


@pytest.mark.parametrize("failure", AFTER_INSERT)
def test_a_new_document_that_fails_after_insert_leaves_no_chunks(harness, tmp_path, failure):
    h = harness
    AFTER_INSERT[failure](h)
    stored_ids = []
    insert = h.vectors.insert

    def recording_insert(collection_name, data, partition_key=None):
        stored_ids.extend(d["document_id"] for d in data)
        return insert(collection_name, data, partition_key)

    h.monkeypatch.setattr(h.vectors, "insert", recording_insert)

    with pytest.raises(Exception):
        h.upload(h.write(tmp_path, "paper.txt", TEXT))

    (document_id,) = set(stored_ids)
    assert h.chunks(document_id) == []
    assert h.row(document_id)["status"] == "error"


def _leftover(h, document_id, n=3):
    """What a worker killed after its insert leaves: the document's chunks, row processing."""
    config = h.config()
    h.vectors.insert(
        config.collection_name,
        [
            {
                "chunk_id": f"{document_id}_chunk_{i}",
                "document_id": document_id,
                "session_id": SESSION,
                "content": f"left over {i}",
                "embedding": [0.0] * 8,
                "metadata": {},
            }
            for i in range(n)
        ],
    )


def test_a_message_delivered_again_indexes_the_document_once(tmp_path, monkeypatch):
    """Milvus ``insert`` keeps both rows of an id: the redelivered task doubled the chunks."""
    from lasagna.infrastructure.tasks import document_tasks

    h = Harness(tmp_path, monkeypatch)
    monkeypatch.setattr("lasagna.api.rag_backend.get_rag_backend", lambda: h.backend)
    monkeypatch.setattr(document_tasks, "dispatch_processing", lambda *a, **k: True)
    import lasagna.api._document_pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "dispatch_processing", lambda *a, **k: True)
    doc = h.upload(h.write(tmp_path, "paper.txt", TEXT))  # queued, not processed
    _leftover(h, doc.document_id)

    row = h.row(doc.document_id)
    document_tasks._process_document_impl(doc.document_id, SESSION, row["file_path"], queued=True)

    assert h.searchable(doc.document_id)
    assert all(not c["content"].startswith("left over") for c in h.chunks(doc.document_id))


def test_a_thread_pool_retry_indexes_the_document_once(tmp_path, monkeypatch):
    from lasagna.infrastructure.tasks import document_tasks

    h = Harness(tmp_path, monkeypatch)
    monkeypatch.setattr("lasagna.api.rag_backend.get_rag_backend", lambda: h.backend)
    import lasagna.api._document_pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "dispatch_processing", lambda *a, **k: True)
    doc = h.upload(h.write(tmp_path, "paper.txt", TEXT))
    _leftover(h, doc.document_id)

    document_tasks._process_document_impl(
        doc.document_id,
        SESSION,
        h.row(doc.document_id)["file_path"],
        retry_ctx=document_tasks.TaskRetryContext(retries=1),
    )
    assert h.searchable(doc.document_id)


# -- search serves ready documents only ------------------------------------


class _Backend:
    backend_name = "fake"


def test_search_returns_chunks_of_ready_documents_only(tmp_path, monkeypatch):
    from lasagna.api import _search_orchestrator as orch_mod
    from lasagna.core.base_components import BaseRetriever, ScoredDocument
    from lasagna.vector_backends.base import VectorStoreAdapter

    h = Harness(tmp_path, monkeypatch)
    for document_id, status in (
        ("d_ready", "ready"),
        ("d_failed", "error"),
        ("d_busy", "processing"),
    ):
        h.db.save_document(
            {
                "document_id": document_id,
                "session_id": SESSION,
                "filename": f"{document_id}.txt",
                "file_size": 1,
                "file_type": ".txt",
                "status": status,
                "upload_date": "2026-10-17T00:00:00+00:00",
            },
            file_path="",
        )

    class Retriever(BaseRetriever):
        _forwards_session_config = True

        def retrieve(self, query, collection, top_k=5, **kwargs):
            return [
                ScoredDocument(content=d, score=s, document_id=d, chunk_id=f"{d}_chunk_0")
                for d, s in (("d_failed", 0.9), ("d_busy", 0.8), ("d_ready", 0.7))
            ]

    _Backend.session_filter = VectorStoreAdapter.session_filter
    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    backend = _Backend()
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.embedding_manager = SimpleNamespace(
        get_cache_stats=lambda: {}, encode_query=lambda q: np.ones(3)
    )
    o.hybrid_searcher = None
    o.reranker = o.query_enhancer = o._auto_merger = None
    o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: backend)
    o._document_pipeline = h.pipeline
    o._default_backend = backend
    o._get_session_config = lambda sid: rag_config
    o._session_embedder = lambda config: o.embedding_manager
    o.backend = backend
    plain = dict(
        enable_hybrid_search=False,
        enable_reranking=False,
        enable_query_routing=False,
        enable_query_classification=False,
        enable_search_cache=False,
        enable_hyde=False,
        enable_mmr=False,
        enable_context_reordering=False,
        enable_hierarchical=False,
        enable_rag_fusion=False,
        enable_query_expansion=False,
    )
    out = o.search(
        "q",
        SESSION,
        top_k=5,
        min_score=None,
        _retriever_override=Retriever(),
        query_config_overrides=plain,
    )
    assert [r.chunk.document_id for r in out] == ["d_ready"]


# -- a replacement stays consistent at every failure point -------------------


@pytest.fixture
def uploaded(harness, tmp_path):
    h = harness
    doc = h.upload(h.write(tmp_path, "paper.txt", TEXT))
    h.document_id = doc.document_id
    h.old_file = h.row(doc.document_id)["file_path"]
    h.settings = dict(MODEL_B)  # the reindex moves it to another collection
    h.chunk_words = 2
    return h


def _files(h):
    return sorted(str(p) for p in h.uploads.rglob("*") if p.is_file())


@pytest.mark.parametrize("failure", ["keyword_index", "parent_chunks"])
def test_a_failure_after_the_new_copy_is_recorded_keeps_the_new_copy(uploaded, tmp_path, failure):
    """The old chunks are gone by then: the row, file and chunks are the new copy's."""
    h = uploaded
    AFTER_INSERT[failure](h)
    replacement = h.write(tmp_path, "paper-v2.txt", TEXT + " a second version of the paper")

    h.upload(replacement, document_id=h.document_id)

    row = h.row(h.document_id)
    assert h.searchable(h.document_id), row
    assert row["collection_name"] == h.config().collection_name
    assert h.chunks(h.document_id) == h.chunks(h.document_id, row["collection_name"])
    assert _files(h) == [row["file_path"]] and row["file_path"] != h.old_file


def test_a_replacement_that_cannot_be_recorded_leaves_nothing_served(uploaded):
    """The old chunks are retired, the row cannot be pointed at the new ones."""
    h = uploaded
    h.monkeypatch.setattr(h.db, "update_document_fields", _raise(OSError("database is locked")))

    with pytest.raises(RuntimeError, match="Reindex it again"):
        h.pipeline.reindex_document(h.document_id, SESSION)

    row = h.row(h.document_id)
    assert row["status"] == "error" and "Reindex it again" in row["error_message"]
    assert h.chunks(h.document_id) == []
    assert _files(h) == [h.old_file]
