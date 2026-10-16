"""A search that falls back from routing says so.

The routing ``try`` in ``search()`` also wrapped the agentic call. Anything
raised there (a metrics error after the agent had run, a strict-evaluation
failure, a configuration error) was logged as "Query routing failed" and the
standard pipeline ran: no ``degraded_stages`` entry, no ``standard_fallback``
mark, and under evaluation the standard results were scored as the agentic
arm's.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lasagna.api import _search_orchestrator as orch
from lasagna.core.config import RAGConfig
from lasagna.core.strict import strict
from lasagna.utils import degraded


class _StandardPipelineRan(Exception):
    pass


def _orchestrator(monkeypatch, *, router_error=None, agent_error=None):
    o = orch.SearchOrchestrator.__new__(orch.SearchOrchestrator)
    o.reranker = None
    o.query_enhancer = None
    o._get_session_config = lambda sid: RAGConfig()

    def standard_pipeline_ran(enhancer):
        raise _StandardPipelineRan

    # The first step of the standard pipeline after routing.
    monkeypatch.setattr(orch, "_is_public_enhancer", standard_pipeline_ran)
    monkeypatch.setattr(
        orch,
        "get_quota_manager",
        lambda: SimpleNamespace(check_search=lambda s: None, record_search=lambda s: None),
    )
    if router_error is not None:

        def route(**kw):
            raise router_error

        monkeypatch.setattr(orch, "get_query_router", lambda: SimpleNamespace(route=route))

    import lasagna.agentic.agentic_rag as agentic_rag
    import lasagna.providers as providers

    monkeypatch.setattr(
        providers, "get_llm_provider", lambda: SimpleNamespace(get_llm=lambda: None)
    )

    class _Agent:
        def __init__(self, **kw):
            pass

        async def retrieve(self, **kw):
            raise agent_error

    monkeypatch.setattr(agentic_rag, "AgenticRAG", _Agent)
    return o


def test_a_failed_router_is_reported_as_a_degraded_stage(monkeypatch):
    o = _orchestrator(monkeypatch, router_error=RuntimeError("router broke"))
    with degraded.collecting() as stages, pytest.raises(_StandardPipelineRan):
        o.search("why?", "s1", routing_tier="comprehensive")
    assert stages == ["routing"]


def test_under_evaluation_a_failed_router_raises(monkeypatch):
    o = _orchestrator(monkeypatch, router_error=RuntimeError("router broke"))
    with strict(), pytest.raises(RuntimeError, match="Query routing failed: router broke"):
        o.search("why?", "s1", routing_tier="comprehensive")


def test_under_evaluation_a_failed_agent_is_not_served_as_standard_results(monkeypatch):
    o = _orchestrator(monkeypatch, agent_error=ConnectionError("LLM endpoint down"))
    with strict(), pytest.raises(RuntimeError, match="Agentic search failed: LLM endpoint down"):
        o.search("why?", "s1", routing_tier="comprehensive")


def test_a_configuration_error_in_the_agentic_path_is_raised(monkeypatch):
    o = _orchestrator(monkeypatch, agent_error=KeyError("unknown agentic pipeline 'self_ak'"))
    with pytest.raises(KeyError, match="self_ak"):
        o.search("why?", "s1", routing_tier="comprehensive")
