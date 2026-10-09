"""An agentic pipeline's own searches use the arm's components.

The pipeline issues one search per hop and passes only settings, so every hop
searched with the profile's reranker, enhancer and retriever: a reranker x
agentic comparison measured the agentic loop alone. And under evaluation a
failed agentic search fell back to standard results scored as the agentic
arm's.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lasagna.api import _search_orchestrator as orch
from lasagna.core.strict import strict


class _Sentinel(Exception):
    pass


def _orchestrator(monkeypatch):
    o = orch.SearchOrchestrator.__new__(orch.SearchOrchestrator)
    o._get_session_config = lambda sid: SimpleNamespace(
        with_query_overrides=lambda q: SimpleNamespace(max_agent_iterations=1, rerank_top_k=5),
        max_agent_iterations=1,
        rerank_top_k=5,
    )
    monkeypatch.setattr(
        orch, "get_quota_manager", lambda: SimpleNamespace(check_search=lambda s: None)
    )
    import lasagna.providers as providers

    monkeypatch.setattr(
        providers, "get_llm_provider", lambda: SimpleNamespace(get_llm=lambda: None)
    )
    return o


def test_the_arms_components_are_visible_to_each_hop(monkeypatch):
    o = _orchestrator(monkeypatch)
    seen = {}

    class _Agent:
        def __init__(self, **kw):
            pass

        async def retrieve(self, **kw):
            seen.update(orch._INNER_SEARCH_COMPONENTS.get())
            raise _Sentinel

    import lasagna.agentic.agentic_rag as agentic_rag

    monkeypatch.setattr(agentic_rag, "AgenticRAG", _Agent)
    reranker, retriever = object(), object()
    with strict(), pytest.raises(RuntimeError, match="Agentic search failed"):
        o._search_comprehensive(
            "q",
            "s",
            5,
            -100.0,
            None,
            None,
            None,
            _reranker_override=reranker,
            _retriever_override=retriever,
        )
    assert seen == {"_reranker_override": reranker, "_retriever_override": retriever}
    assert orch._INNER_SEARCH_COMPONENTS.get() == {}, "the arm's components outlived its search"


def test_a_hop_searches_with_them(monkeypatch):
    o = orch.SearchOrchestrator.__new__(orch.SearchOrchestrator)
    o.reranker = "profile reranker"
    used = {}

    def _adapt(r):
        used["reranker"] = r
        raise _Sentinel

    import lasagna.retrieval.search.ranker as ranker

    monkeypatch.setattr(ranker, "adapt_reranker_for_pipeline", _adapt)
    token = orch._INNER_SEARCH_COMPONENTS.set({"_reranker_override": "arm reranker"})
    try:
        with pytest.raises(_Sentinel):
            o.search("q", "s", _from_routing=True)
        assert used["reranker"] == "arm reranker"
        with pytest.raises(_Sentinel):
            o.search("q", "s")  # a top-level search is not a hop
        assert used["reranker"] == "profile reranker"
    finally:
        orch._INNER_SEARCH_COMPONENTS.reset(token)
