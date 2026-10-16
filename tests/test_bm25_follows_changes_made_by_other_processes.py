"""Each process's BM25 index follows uploads and deletes made by other processes.

The in-memory BM25 index was filled from the store once per session per
process, and only the process that uploaded or deleted a document changed its
own index. With several API workers and a Celery worker, a document processed
elsewhere never reached a worker's keyword search, and a deleted document's
chunks kept coming back as keyword-only hits from every other worker.

The metadata database now keeps a content generation per session, bumped on
every upload and delete; a process whose index was built at another
generation rebuilds it from the store on its next search. Keyword-only hits
whose chunk is gone from the store are dropped. The upload path also indexed
chunks under their position (0, 1, ...) instead of their stored id, so the
uploading process held every chunk twice once it read the store, and returned
the same chunk twice.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import lasagna.api._document_pipeline as dp
from lasagna.api._document_pipeline import DocumentPipeline
from lasagna.core import database
from lasagna.core.config import get_rag_config
from lasagna.core.database import RAGDatabase
from lasagna.core.schema import Document, DocumentStatus
from lasagna.retrieval.search.hybrid import HybridSearcher
from lasagna.vector_backends.filters import FilterExpression

SESSION = "shared-session"
COLLECTION = "rag_documents_v2__test__4"


class _Store:
    """One vector store both processes use, holding chunks by id."""

    backend_name = "fake"
    capabilities = SimpleNamespace(supports_hybrid_search=False, supports_native_bm25=False)

    def __init__(self):
        self.chunks = {}
        self.scans = 0

    def is_connected(self):
        return True

    def session_filter(self, collection_name, session_id):
        return FilterExpression.eq("session_id", session_id)

    def insert(self, collection_name, data, partition_key=None):
        for row in data:
            self.chunks[row["chunk_id"]] = row

    def delete(self, collection_name, filter_expr, partition_key=None):
        for cid in [c for c, r in self.chunks.items() if r["document_id"] == filter_expr.value]:
            del self.chunks[cid]

    def iter_all(self, collection_name, filter_expr=None, output_fields=None, partition_key=None):
        self.scans += 1
        return [dict(r) for r in self.chunks.values() if r["session_id"] == SESSION]

    def query(self, collection_name, filter_expr=None, output_fields=None, limit=100, **kw):
        wanted = set(filter_expr.value)
        return [{"chunk_id": c} for c in self.chunks if c in wanted][:limit]


@pytest.fixture
def world(tmp_path, monkeypatch):
    db = RAGDatabase(str(tmp_path / "metadata.db"))
    monkeypatch.setattr(database, "_rag_database", db)
    store = _Store()
    monkeypatch.setattr(dp, "get_backend", lambda name: store)
    monkeypatch.setattr(dp, "get_quota_manager", lambda: MagicMock())
    config = get_rag_config().with_query_overrides(
        {
            "collection_name": COLLECTION,
            "enable_hybrid_search": True,
            "hybrid_alpha": 0.5,
            "enable_contextual_retrieval": False,
            "enable_graph_retrieval": False,
            "enable_late_chunking": False,
            "enable_raptor": False,
            "enable_hierarchical": False,
        }
    )
    monkeypatch.setattr(
        dp,
        "chunk_document",
        lambda text, document_id, session_id, **kw: [
            (text, {"document_id": document_id, "session_id": session_id})
        ],
    )
    return SimpleNamespace(db=db, store=store, config=config)


def _searcher(store):
    searcher = HybridSearcher()
    searcher.set_vector_backend(store, COLLECTION)
    return searcher


def _process_a(world):
    """The process that takes uploads and deletes: a pipeline with its own index."""
    pipeline = DocumentPipeline.__new__(DocumentPipeline)
    embedder = SimpleNamespace(
        model_name="fake",
        dimension=4,
        encode_documents_batch=lambda texts, show_progress=False: [np.ones(4) for _ in texts],
    )
    pipeline._collection_manager = SimpleNamespace(
        embedder_for=lambda config: embedder,
        resolve_backend=lambda session_id: world.store,
        collection_is_empty=lambda name, backend=None: False,
        _default_backend_name="fake",
    )
    pipeline._storage = SimpleNamespace(
        ensure_local=lambda p: p, get_local_path=lambda p: p, delete=lambda p: None
    )
    pipeline._rebuild_lock = threading.Lock()
    pipeline._rebuilding = False
    pipeline.hybrid_searcher = _searcher(world.store)
    pipeline._should_use_docling = lambda path, config: False
    pipeline._extract_text_with_ocr = lambda path, file_type, config: world.texts[path]
    pipeline._update_document_status = lambda *a, **k: None
    world.texts = {}
    return pipeline


def _upload(world, pipeline, doc_id, text):
    world.texts[f"/up/{doc_id}"] = text
    world.db.save_document(
        {
            "document_id": doc_id,
            "session_id": SESSION,
            "filename": f"{doc_id}.txt",
            "file_size": len(text),
            "file_type": "txt",
            "status": "ready",
        },
        f"/up/{doc_id}",
        COLLECTION,
    )
    document = Document(
        document_id=doc_id,
        session_id=SESSION,
        filename=f"{doc_id}.txt",
        file_size=len(text),
        file_type="txt",
        status=DocumentStatus.PROCESSING,
    )
    pipeline.process_document(document, f"/up/{doc_id}", config=world.config)


def _search(world, searcher, query):
    """What a search on the BM25 fallback path does: fill the index, then fuse."""
    searcher.hydrate_session_index(
        SESSION, world.store, COLLECTION, world.store.session_filter(COLLECTION, SESSION)
    )
    hits = searcher.search_hybrid(query, [], SESSION, top_k=10, config=world.config)
    return [h.document_id for h in hits]


def _corpus(world, pipeline):
    for doc_id, text in [
        ("d_tide", "tide tables for the harbor mouth"),
        ("d_bridge", "the bridge opens at noon"),
        ("d_light", "lighthouse keeper logbook"),
        # Four documents, so three remain after a delete: BM25 gives a term
        # found in one of two documents a weight of exactly zero.
        ("d_ferry", "ferry timetable for weekdays"),
    ]:
        _upload(world, pipeline, doc_id, text)


def test_a_document_uploaded_by_another_process_is_found_by_keyword(world):
    a = _process_a(world)
    b = _searcher(world.store)
    _corpus(world, a)
    assert _search(world, b, "lighthouse") == ["d_light"]

    _upload(world, a, "d_moon", "moon phases calendar")

    assert _search(world, b, "moon") == ["d_moon"]


def test_a_document_deleted_by_another_process_stops_matching(world):
    a = _process_a(world)
    b = _searcher(world.store)
    _corpus(world, a)
    assert _search(world, b, "tide") == ["d_tide"]

    assert a.delete_document("d_tide", SESSION)

    assert _search(world, b, "tide") == []
    assert _search(world, b, "bridge") == ["d_bridge"]


def test_a_keyword_hit_whose_chunk_left_the_store_is_dropped(world):
    """Even before the generation says so (the deleting process died before it)."""
    a = _process_a(world)
    b = _searcher(world.store)
    _corpus(world, a)
    assert _search(world, b, "bridge") == ["d_bridge"]

    world.store.delete(COLLECTION, FilterExpression.eq("document_id", "d_bridge"))

    assert _search(world, b, "bridge") == []


def test_an_unchanged_session_is_not_read_again(world):
    a = _process_a(world)
    b = _searcher(world.store)
    _corpus(world, a)
    _search(world, b, "tide")
    scans = world.store.scans

    for query in ("bridge", "lighthouse", "tide"):
        _search(world, b, query)

    assert world.store.scans == scans


def test_the_uploading_process_holds_each_chunk_once(world):
    a = _process_a(world)
    _corpus(world, a)

    assert _search(world, a.hybrid_searcher, "lighthouse") == ["d_light"]
    indexed = [d["chunk_id"] for d in a.hybrid_searcher.get_session_index(SESSION).documents]
    assert sorted(indexed) == sorted(world.store.chunks)
