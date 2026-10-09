"""An agentic request served by standard retrieval must be visible as such.

When the agentic pipeline fails, the orchestrator falls back to standard search
and returns its (non-empty) results. The harness only flagged a fallback when
zero results came back, so those samples were scored as agentic using the
standard answer, biasing agentic-vs-standard comparisons toward no difference.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lasagna.api._search_orchestrator import SearchOrchestrator
from lasagna.knowledge_evaluation_sdk.experiment import _agentic_fell_back


def _result(meta=None):
    return SimpleNamespace(chunk=SimpleNamespace(metadata=dict(meta or {}), content="x"))


def test_a_failed_agentic_run_is_stamped_as_a_fallback():
    orchestrator = MagicMock(spec=SearchOrchestrator)
    orchestrator._get_session_config.return_value = SimpleNamespace(
        max_agent_iterations=1, rerank_top_k=5
    )
    orchestrator.search.return_value = [_result()]

    with (
        patch("lasagna.api._search_orchestrator.get_quota_manager"),
        patch("lasagna.providers.get_llm_provider", side_effect=RuntimeError("LLM endpoint down")),
    ):
        results = SearchOrchestrator._search_comprehensive(
            orchestrator,
            query="q",
            session_id="s1",
            top_k=5,
            min_score=0.0,
            document_ids=None,
            trace_id=None,
            routing_decision=None,
        )

    assert results[0].chunk.metadata["_routing_tier_used"] == "standard_fallback"
    assert _agentic_fell_back(True, results) is True


def test_the_harness_reads_the_stamp():
    assert _agentic_fell_back(True, []) is True
    assert _agentic_fell_back(True, [_result({"_routing_tier_used": "comprehensive"})]) is False
    assert _agentic_fell_back(True, [_result({"_routing_tier_used": "standard_fallback"})]) is True
    assert (
        _agentic_fell_back(False, [_result({"_routing_tier_used": "standard_fallback"})]) is False
    )
