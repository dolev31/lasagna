"""Chunks of documents search will not return do not take places in the top-k cut.

Search attached each chunk to its document's row after the reranker and the cut
to ``top_k``, and dropped the chunks of archived documents (and, since then, of
failed or still-processing ones) there: a query whose best hits were in an
archived document came back with fewer than ``top_k`` results while servable
chunks ranked just below the cut. They are now removed before it, as repeated
chunk ids are. The evaluation harness registers every corpus document as
ready, so its candidate lists are unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.core.base_components import BaseRetriever, ScoredDocument
from lasagna.core.config import rag_config
from lasagna.vector_backends.base import VectorStoreAdapter
from tests._document_harness import SESSION, Harness

PLAIN = dict(
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


class _Backend:
    backend_name = "fake"
    session_filter = VectorStoreAdapter.session_filter


class _Retriever(BaseRetriever):
    _forwards_session_config = True

    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, collection, top_k=5, **kwargs):
        return [
            ScoredDocument(content=c, score=s, document_id=d, chunk_id=c) for d, c, s in self.hits
        ]


def _orchestrator(pipeline, monkeypatch):
    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    backend = _Backend()
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.embedding_manager = SimpleNamespace(
        get_cache_stats=lambda: {}, encode_query=lambda q: np.ones(3)
    )
    o.hybrid_searcher = None
    o.reranker = o.query_enhancer = o._auto_merger = None
    o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: backend)
    o._document_pipeline = pipeline
    o._default_backend = backend
    o._get_session_config = lambda sid: rag_config
    o._session_embedder = lambda config: o.embedding_manager
    o.backend = backend
    return o


def _register(h, document_id, status):
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


def test_an_archived_documents_chunks_leave_the_cut_full(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    _register(h, "d_archived", "soft_deleted")
    _register(h, "d_ready", "ready")
    hits = [
        ("d_archived", "a0", 0.95),
        ("d_archived", "a1", 0.9),
        ("d_ready", "r0", 0.8),
        ("d_ready", "r1", 0.7),
        ("d_ready", "r2", 0.6),
    ]
    out = _orchestrator(h.pipeline, monkeypatch).search(
        "q",
        SESSION,
        top_k=3,
        min_score=None,
        _retriever_override=_Retriever(hits),
        query_config_overrides=PLAIN,
    )
    assert [r.chunk.chunk_id for r in out] == ["r0", "r1", "r2"]


def test_with_every_document_ready_the_list_is_the_retrievers(tmp_path, monkeypatch):
    """What the harness sees: nothing is removed, so its numbers cannot move."""
    h = Harness(tmp_path, monkeypatch)
    for d in ("d1", "d2"):
        _register(h, d, "ready")
    hits = [("d1", "c0", 0.9), ("d2", "c1", 0.8), ("d1", "c2", 0.7)]
    out = _orchestrator(h.pipeline, monkeypatch).search(
        "q",
        SESSION,
        top_k=3,
        min_score=None,
        _retriever_override=_Retriever(hits),
        query_config_overrides=PLAIN,
    )
    assert [r.chunk.chunk_id for r in out] == ["c0", "c1", "c2"]
