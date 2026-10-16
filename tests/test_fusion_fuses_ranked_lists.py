"""Query fusion fuses each variation's ranked list, then reorders the fused list once.

With context reordering on, every variation's search came back reordered
(even ranks first, odd ranks reversed at the end), and reciprocal rank fusion
read those positions as ranks: a variation's second-best passage counted as its
last. The fused list was then never reordered, so the setting did not do what
it says either. Separately, the variations ignored the arm's retriever and
searched the store's dense index, so a retriever arm with query expansion
measured dense search under its own name.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.core.base_components import BaseQueryEnhancer, BaseRetriever, ScoredDocument
from lasagna.core.config import rag_config
from lasagna.core.strict import lost_in_the_middle, present_context_order

DOCS = [f"d{i}" for i in range(6)]
#: Each query's ranking: the variation ranks the documents in reverse.
RANKING = {"q": DOCS, "variant": DOCS[::-1]}
RRF_K = 60

CONFIG = dict(
    enable_hybrid_search=False,
    enable_reranking=False,
    enable_query_routing=False,
    enable_query_classification=False,
    enable_search_cache=False,
    enable_hyde=False,
    enable_mmr=False,
    enable_hierarchical=False,
    enable_query_expansion=False,
    enable_rag_fusion=True,
    rag_fusion_n_queries=1,
    rrf_k=RRF_K,
    similarity_threshold=0.0,
    enable_context_reordering=True,
)


class _Variant(BaseQueryEnhancer):
    def enhance(self, query, **kwargs):
        return ["variant"]


class _Retriever(BaseRetriever):
    def __init__(self):
        self.queries = []

    def retrieve(self, query, collection, top_k=5, **kwargs):
        self.queries.append(query)
        return [
            ScoredDocument(content=d, score=1.0 - i / 10, document_id=d, chunk_id=d)
            for i, d in enumerate(RANKING[query])
        ]


class _Store:
    """The store's own vector search: the ranking follows the query embedding."""

    backend_name = "fake"

    def search(self, query_embedding, **kw):
        query = "q" if query_embedding[0] else "variant"
        return [
            SimpleNamespace(
                id=d,
                score=1.0 - i / 10,
                distance=i / 10,
                entity={"document_id": d, "content": d, "metadata": {}},
            )
            for i, d in enumerate(RANKING[query])
        ]


@pytest.fixture
def orchestrator(monkeypatch):
    from lasagna.vector_backends.base import VectorStoreAdapter

    _Store.session_filter = VectorStoreAdapter.session_filter
    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    store = _Store()
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.embedding_manager = SimpleNamespace(
        get_cache_stats=lambda: {},
        encode_query=lambda q: np.array([1.0, 0.0] if q == "q" else [0.0, 1.0]),
    )
    o.hybrid_searcher = None
    o.reranker = o._auto_merger = None
    o.query_enhancer = _Variant()
    o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: store)
    o._document_pipeline = SimpleNamespace(
        get_documents_metadata_batch=lambda ids: {i: SimpleNamespace(document_id=i) for i in ids}
    )
    o._default_backend = store
    o._get_session_config = lambda sid: rag_config
    o._session_embedder = lambda config: o.embedding_manager
    o.backend = store
    return o


def _fused():
    """Weighted RRF over the true ranks: the original query counts twice."""
    score = {d: 0.0 for d in DOCS}
    for query, weight in (("q", 2.0), ("variant", 1.0)):
        for rank, d in enumerate(RANKING[query]):
            score[d] += weight / (RRF_K + rank)
    return sorted(DOCS, key=lambda d: -score[d])


def _ids(results):
    return [r.chunk.chunk_id for r in results]


def test_fusion_reads_ranks_and_reorders_the_fused_list(orchestrator):
    out = orchestrator.search("q", "s", top_k=6, query_config_overrides=CONFIG)
    assert _ids(out) == lost_in_the_middle(_fused())


def test_without_reordering_the_fused_list_is_ranked(orchestrator):
    config = {**CONFIG, "enable_context_reordering": False}
    out = orchestrator.search("q", "s", top_k=6, query_config_overrides=config)
    assert _ids(out) == _fused()


def test_a_caller_that_presents_the_passages_gets_the_ranked_fused_list(orchestrator):
    with present_context_order() as presented:
        out = orchestrator.search("q", "s", top_k=6, query_config_overrides=CONFIG)
    assert _ids(out) == _fused()
    assert presented.reorder is True


def test_each_variation_uses_the_arms_retriever(orchestrator):
    def _never(**kw):
        raise AssertionError("a variation searched the store instead of the arm's retriever")

    orchestrator._default_backend.search = _never
    retriever = _Retriever()
    config = {**CONFIG, "enable_context_reordering": False}
    out = orchestrator.search(
        "q", "s", top_k=6, query_config_overrides=config, _retriever_override=retriever
    )
    assert sorted(retriever.queries) == ["q", "variant"]
    assert _ids(out) == _fused()
