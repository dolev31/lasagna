"""A chunk retrieved twice does not cost the caller a result.

Search returned each chunk once, but removed the repeats after reranking and
after the cut to ``top_k``: a list of ``top_k`` hits holding one chunk twice
came back one result short. Repeats are now removed from the candidates,
before reranking and the cut.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.core.base_components import BaseReranker, BaseRetriever, ScoredDocument
from lasagna.core.config import rag_config


class _Retriever(BaseRetriever):
    _forwards_session_config = True

    def retrieve(self, query, collection, top_k=5, **kwargs):
        return [
            ScoredDocument(content=c, score=s, document_id="d", chunk_id=c)
            for c, s in (("c0", 0.9), ("c0", 0.85), ("c1", 0.8), ("c2", 0.7))
        ]


class _KeepOrder(BaseReranker):
    def rerank(self, query, documents, top_k=5, **kwargs):
        return list(documents)[:top_k]


class _Backend:
    backend_name = "fake"


@pytest.fixture
def orchestrator(monkeypatch):
    from lasagna.vector_backends.base import VectorStoreAdapter

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
    o._document_pipeline = SimpleNamespace(
        get_documents_metadata_batch=lambda ids: {i: SimpleNamespace(document_id=i) for i in ids}
    )
    o._default_backend = backend
    o._get_session_config = lambda sid: rag_config
    o._session_embedder = lambda config: o.embedding_manager
    o.backend = backend
    return o


SETTINGS = dict(
    enable_hybrid_search=False,
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


@pytest.mark.parametrize("reranking", [False, True], ids=["no_reranker", "reranker"])
def test_top_k_distinct_chunks_come_back(orchestrator, reranking):
    out = orchestrator.search(
        "q",
        "s",
        top_k=2,
        min_score=None,
        _retriever_override=_Retriever(),
        _reranker_override=_KeepOrder() if reranking else None,
        query_config_overrides=dict(SETTINGS, enable_reranking=reranking),
    )
    assert [r.chunk.chunk_id for r in out] == ["c0", "c1"]
