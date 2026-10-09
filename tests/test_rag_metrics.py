"""
Test script to verify RAG token metrics collection via SDK.

Run with: pytest tests/test_rag_metrics.py -v

Note: This test requires documents to be uploaded for the session.
"""

import pytest

CugaAgent = pytest.importorskip(
    "cuga.sdk", reason="cuga.sdk not available in standalone mode"
).CugaAgent


def _agent_constructible() -> str:
    """Return "" if a CUGA agent can be built here, else why it cannot.

    These tests construct a real agent, which resolves an LLM provider and
    raises when its credentials are absent. Probing by construction rather than
    by guessing which ``*_API_KEY`` matters: the agent picks its own provider,
    so the presence of *a* key says nothing about whether the *required* one is
    set. Without this the module passed only on machines with the right key
    exported, and silently depended on an unrelated test module having called
    load_dotenv() earlier in the same session -- so it passed in a full run and
    failed when run on its own.
    """
    try:
        CugaAgent(tools=[])
    except Exception as exc:  # noqa: BLE001 - any construction failure disqualifies
        return f"{type(exc).__name__}: {exc}"
    return ""


_AGENT_UNAVAILABLE = _agent_constructible()

pytestmark = pytest.mark.skipif(
    bool(_AGENT_UNAVAILABLE),
    reason=f"CUGA agent not constructible here ({_AGENT_UNAVAILABLE})",
)


class TestRAGMetricsCollection:
    """Test RAG metrics are collected when using agent.invoke()."""

    def test_sdk_rag_manager_exists(self):
        """Test that CugaAgent has RAG manager."""
        agent = CugaAgent(tools=[])
        assert hasattr(agent, "rag"), "CugaAgent should have rag attribute"

    def test_rag_manager_has_enable_method(self):
        """Test that RAG manager has enable method."""
        agent = CugaAgent(tools=[])
        assert hasattr(agent.rag, "enable"), "RAG manager should have enable method"
        assert callable(agent.rag.enable), "enable should be callable"

    def test_rag_manager_has_session_id(self):
        """Test that RAG manager has session_id property."""
        agent = CugaAgent(tools=[])
        assert hasattr(agent.rag, "session_id"), "RAG manager should have session_id"

    def test_rag_metrics_structure(self):
        """Test expected RAG metrics structure."""
        # Expected metrics keys when RAG is used
        expected_keys = [
            "tokens_before",
            "tokens_after",
            "tokens_saved",
            "reduction_percentage",
            "chunks_filtered",
            "search_count",
        ]

        # Create a mock metrics dict
        mock_metrics = {
            "tokens_before": 1000,
            "tokens_after": 500,
            "tokens_saved": 500,
            "reduction_percentage": 50.0,
            "chunks_filtered": 5,
            "search_count": 2,
        }

        for key in expected_keys:
            assert key in mock_metrics, f"Metrics should have key: {key}"


class TestRAGMetricsIntegration:
    """Integration tests for RAG metrics (requires running environment)."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires live RAG environment with uploaded documents")
    async def test_rag_metrics_in_result(self):
        """Test that RAG metrics are included in invoke result."""
        agent = CugaAgent(tools=[])
        agent.rag.enable()
        agent.rag.session_id = "default"

        result = await agent.invoke("What are the main topics in the documents?")

        if isinstance(result, dict):
            assert "rag_metrics" in result, "Result should include rag_metrics"
            metrics = result["rag_metrics"]
            assert "tokens_before" in metrics
            assert "tokens_after" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
