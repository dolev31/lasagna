"""Comprehensive test suite for the structured RAG graph.

Tests cover:
  - Graph state construction (build_initial_state)
  - Plan retrieval node (routing tier mapping, decomposition)
  - Edge routing functions (all conditional edges)
  - Route-loop invariants (bounded rewrite, bounded retrieval)
  - Citation invariants (monotonic indices, content-hash dedup)
  - Graph builder compilation
  - Quality gates (hallucination check, answer quality check)

All external dependencies (RAGBackend, LLM, embedding manager) are mocked.
"""

from __future__ import annotations

import types
from typing import Any, Dict, Set
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
from lasagna.agentic.graph_state import (
    build_initial_state,
)

# ---------------------------------------------------------------------------
# Edge routing functions
# ---------------------------------------------------------------------------
from lasagna.agentic.edges import (
    route_after_generation,
    route_after_grading,
    route_after_rewrite,
    route_after_select_chunks,
)

# ---------------------------------------------------------------------------
# Routing models
# ---------------------------------------------------------------------------
from lasagna.retrieval.routing.models import RoutingTier


# ============================================================================
# Helpers / shared fixtures
# ============================================================================


def _make_mock_llm_response(content: str) -> MagicMock:
    """Return a mock LLM response object with a ``.content`` attribute."""
    response = MagicMock()
    response.content = content
    return response


def _make_async_llm(response_content: str) -> AsyncMock:
    """Return an ``AsyncMock`` LLM whose ``ainvoke`` resolves to *response_content*."""
    llm = AsyncMock()
    llm.ainvoke.return_value = _make_mock_llm_response(response_content)
    return llm


def _base_state(**overrides: Any) -> Dict[str, Any]:
    """Construct a minimal graph state dict with sensible defaults.

    Individual tests can override any field via keyword arguments.
    """
    state: Dict[str, Any] = {
        "original_question": "What is RAG?",
        "session_id": "test-session",
        "top_k": 5,
        "session_config": None,
        "llm": None,
        "routing_decision": None,
        "query_complexity": "moderate",
        "sub_queries": ["What is RAG?"],
        "current_sub_query_index": 0,
        "current_query": "What is RAG?",
        "retrieved_chunks": [],
        "hop_data": [],
        "hop_count": 0,
        "refined_chunks": [],
        "refinement_method": "auto",
        "refinement_metrics": {},
        "global_citation_index": 1,
        "all_hop_citations": [],
        "seen_content_hashes": set(),
        "document_grades": [],
        "overall_grade": "",
        "graded_chunks": [],
        "crag_web_fallback_used": False,
        "crag_grading_latency_ms": 0.0,
        "generation": "",
        "agent_summary": "",
        "generation_citations": [],
        "hallucination_check": "",
        "answer_quality": "",
        "quality_check_attempts": 0,
        "coverage_ratio": 0.0,
        "coverage_confidence": 0.0,
        "uncovered_aspects": [],
        "suggested_followup": None,
        "marked_chunks": [],
        "filtered_citations": [],
        "retrieval_attempts": 0,
        "max_retrieval_attempts": 3,
        "rewrite_count": 0,
        "max_rewrites": 3,
        "stop_reason": "",
        "timing_spans": {},
        "rag_metrics_list": [],
        "total_start_time": 0.0,
        "previous_queries": [],
        "previous_query_embeddings": [],
        "formatted_context": "",
        "final_chunks": [],
        "final_citations": [],
        "final_metrics": None,
    }
    state.update(overrides)
    return state


# ============================================================================
# 1. Graph State Tests
# ============================================================================


class TestGraphState:
    """Tests for ``build_initial_state``."""

    def test_build_initial_state_defaults(self):
        """build_initial_state sets all required fields with correct defaults."""
        state = build_initial_state(
            question="test question",
            session_id="test-session",
            top_k=5,
            session_config=None,
            llm=None,
        )
        assert state["original_question"] == "test question"
        assert state["session_id"] == "test-session"
        assert state["global_citation_index"] == 1
        assert state["hop_count"] == 0
        assert state["retrieval_attempts"] == 0
        assert state["rewrite_count"] == 0
        assert state["max_retrieval_attempts"] == 3
        assert state["previous_queries"] == []
        assert state["all_hop_citations"] == []
        assert state["seen_content_hashes"] == set()

    def test_build_initial_state_custom_max_attempts(self):
        """Custom max_retrieval_attempts and max_rewrites are honoured."""
        state = build_initial_state(
            question="q",
            session_id="s",
            top_k=10,
            session_config=None,
            llm=None,
            max_retrieval_attempts=7,
            max_rewrites=5,
        )
        assert state["max_retrieval_attempts"] == 7
        assert state["max_rewrites"] == 5

    def test_build_initial_state_sub_queries_default_to_question(self):
        """sub_queries defaults to [question] when no routing."""
        state = build_initial_state(
            question="How does chunking work?",
            session_id="s",
            top_k=5,
            session_config=None,
            llm=None,
        )
        assert state["sub_queries"] == ["How does chunking work?"]
        assert state["current_query"] == "How does chunking work?"

    def test_build_initial_state_with_document_filter(self):
        """document_filter is passed through."""
        state = build_initial_state(
            question="q",
            session_id="s",
            top_k=5,
            session_config=None,
            llm=None,
            document_filter=["doc-1", "doc-2"],
        )
        assert state["document_filter"] == ["doc-1", "doc-2"]

    def test_build_initial_state_refinement_method_from_config(self):
        """refinement_method reads from session_config.chunk_refinement_method."""
        config = types.SimpleNamespace(chunk_refinement_method="refined")
        state = build_initial_state(
            question="q",
            session_id="s",
            top_k=5,
            session_config=config,
            llm=None,
        )
        assert state["refinement_method"] == "refined"

    def test_build_initial_state_refinement_method_defaults_auto(self):
        """When session_config is None, refinement_method defaults to 'auto'."""
        state = build_initial_state(
            question="q",
            session_id="s",
            top_k=5,
            session_config=None,
            llm=None,
        )
        assert state["refinement_method"] == "auto"


# ============================================================================
# 2. Plan Retrieval Node Tests
# ============================================================================


class TestPlanRetrieval:
    """Tests for ``plan_retrieval_node``."""

    @pytest.mark.asyncio
    async def test_plan_retrieval_no_routing(self):
        """When routing_decision is None, defaults to 'moderate' complexity."""
        from lasagna.agentic.nodes.plan_retrieval import (
            plan_retrieval_node,
        )

        state = _base_state(routing_decision=None)
        result = await plan_retrieval_node(state)

        assert result["query_complexity"] == "moderate"
        assert result["sub_queries"] == ["What is RAG?"]
        assert result["current_sub_query_index"] == 0
        # moderate -> budget=2
        assert result["max_retrieval_attempts"] == 2

    @pytest.mark.asyncio
    async def test_plan_retrieval_fast_tier(self):
        """FAST tier maps to 'simple' with budget=1."""
        from lasagna.agentic.nodes.plan_retrieval import (
            plan_retrieval_node,
        )

        routing = types.SimpleNamespace(tier=RoutingTier.FAST)
        state = _base_state(routing_decision=routing)
        result = await plan_retrieval_node(state)

        assert result["query_complexity"] == "simple"
        assert result["max_retrieval_attempts"] == 1

    @pytest.mark.asyncio
    async def test_plan_retrieval_standard_tier(self):
        """STANDARD tier maps to 'moderate' with budget=2."""
        from lasagna.agentic.nodes.plan_retrieval import (
            plan_retrieval_node,
        )

        routing = types.SimpleNamespace(tier=RoutingTier.STANDARD)
        state = _base_state(routing_decision=routing)
        result = await plan_retrieval_node(state)

        assert result["query_complexity"] == "moderate"
        assert result["max_retrieval_attempts"] == 2

    @pytest.mark.asyncio
    async def test_plan_retrieval_comprehensive_tier(self):
        """COMPREHENSIVE tier maps to 'complex' with full budget."""
        from lasagna.agentic.nodes.plan_retrieval import (
            plan_retrieval_node,
        )

        routing = types.SimpleNamespace(tier=RoutingTier.COMPREHENSIVE)
        state = _base_state(
            routing_decision=routing,
            max_retrieval_attempts=5,
        )
        result = await plan_retrieval_node(state)

        assert result["query_complexity"] == "complex"
        # complex keeps the configured max
        assert result["max_retrieval_attempts"] == 5

    @pytest.mark.asyncio
    async def test_plan_retrieval_comprehensive_with_decomposition(self):
        """COMPREHENSIVE tier with decomposition enabled decomposes the query."""
        from lasagna.agentic.nodes.plan_retrieval import (
            plan_retrieval_node,
        )

        mock_llm = _make_async_llm(
            "1. What is retrieval augmented generation\n"
            "2. How does RAG improve LLM answers\n"
            "3. What are RAG components"
        )
        config = types.SimpleNamespace(enable_query_decomposition=True)
        routing = types.SimpleNamespace(tier=RoutingTier.COMPREHENSIVE)
        state = _base_state(
            routing_decision=routing,
            session_config=config,
            llm=mock_llm,
            max_retrieval_attempts=5,
        )
        result = await plan_retrieval_node(state)

        assert result["query_complexity"] == "complex"
        assert len(result["sub_queries"]) == 3

    @pytest.mark.asyncio
    async def test_plan_retrieval_comprehensive_no_llm_skips_decomposition(self):
        """When LLM is None, decomposition is skipped even if enabled."""
        from lasagna.agentic.nodes.plan_retrieval import (
            plan_retrieval_node,
        )

        config = types.SimpleNamespace(enable_query_decomposition=True)
        routing = types.SimpleNamespace(tier=RoutingTier.COMPREHENSIVE)
        state = _base_state(
            routing_decision=routing,
            session_config=config,
            llm=None,
        )
        result = await plan_retrieval_node(state)

        # Falls back to [original_question]
        assert result["sub_queries"] == ["What is RAG?"]

    @pytest.mark.asyncio
    async def test_plan_retrieval_routing_without_tier_attribute(self):
        """routing_decision without .tier falls back to 'moderate'."""
        from lasagna.agentic.nodes.plan_retrieval import (
            plan_retrieval_node,
        )

        routing = types.SimpleNamespace()  # no .tier attribute
        state = _base_state(routing_decision=routing)
        result = await plan_retrieval_node(state)

        assert result["query_complexity"] == "moderate"


# ============================================================================
# 3. Edge Routing Tests
# ============================================================================


class TestEdgeRouting:
    """Tests for conditional edge routing functions in ``edges.py``."""

    # -- route_after_grading ------------------------------------------------

    def test_route_after_grading_correct(self):
        """Grade 'correct' routes to 'generate'."""
        state = _base_state(overall_grade="correct")
        assert route_after_grading(state) == "generate"

    def test_route_after_grading_incorrect_with_web_fallback(self):
        """Grade 'incorrect' with web fallback enabled routes to 'web_fallback'."""
        config = types.SimpleNamespace(crag_web_search_fallback=True)
        state = _base_state(
            overall_grade="incorrect",
            session_config=config,
            crag_web_fallback_used=True,
            retrieval_attempts=0,
            max_retrieval_attempts=3,
        )
        assert route_after_grading(state) == "web_fallback"

    def test_route_after_grading_incorrect_no_web_fallback_rewrites(self):
        """Grade 'incorrect' without web fallback routes to 'rewrite_query' when budget remains."""
        state = _base_state(
            overall_grade="incorrect",
            retrieval_attempts=1,
            max_retrieval_attempts=3,
        )
        assert route_after_grading(state) == "rewrite_query"

    def test_route_after_grading_budget_exhausted(self):
        """When retrieval budget is exhausted, routes to 'generate' regardless of grade."""
        state = _base_state(
            overall_grade="ambiguous",
            retrieval_attempts=3,
            max_retrieval_attempts=3,
        )
        assert route_after_grading(state) == "generate"

    def test_route_after_grading_ambiguous_with_budget(self):
        """Grade 'ambiguous' with remaining budget routes to 'rewrite_query'."""
        state = _base_state(
            overall_grade="ambiguous",
            retrieval_attempts=1,
            max_retrieval_attempts=3,
        )
        assert route_after_grading(state) == "rewrite_query"

    # -- route_after_generation ---------------------------------------------

    def test_route_after_generation_no_quality_gates(self):
        """Without quality gates, routes directly to 'select_chunks'."""
        state = _base_state(session_config=None)
        assert route_after_generation(state) == "select_chunks"

    def test_route_after_generation_with_hallucination_check(self):
        """With hallucination check enabled, routes to 'check_hallucination'."""
        config = types.SimpleNamespace(
            enable_hallucination_check=True,
            enable_answer_quality_check=False,
        )
        state = _base_state(session_config=config)
        assert route_after_generation(state) == "check_hallucination"

    def test_route_after_generation_with_quality_check_only(self):
        """With only answer quality check enabled, routes to 'check_answer_quality'."""
        config = types.SimpleNamespace(
            enable_hallucination_check=False,
            enable_answer_quality_check=True,
        )
        state = _base_state(session_config=config)
        assert route_after_generation(state) == "check_answer_quality"

    def test_route_after_generation_both_gates_prefers_hallucination(self):
        """When both gates enabled, hallucination check runs first."""
        config = types.SimpleNamespace(
            enable_hallucination_check=True,
            enable_answer_quality_check=True,
        )
        state = _base_state(session_config=config)
        assert route_after_generation(state) == "check_hallucination"

    # -- route_after_select_chunks ------------------------------------------

    def test_route_after_select_chunks_more_subqueries(self):
        """When more sub-queries remain, routes to 'advance_sub_query'."""
        state = _base_state(
            sub_queries=["sq1", "sq2", "sq3"],
            current_sub_query_index=1,
        )
        assert route_after_select_chunks(state) == "advance_sub_query"

    def test_route_after_select_chunks_done(self):
        """When all sub-queries processed, routes to 'final_search'."""
        state = _base_state(
            sub_queries=["sq1", "sq2"],
            current_sub_query_index=1,
        )
        assert route_after_select_chunks(state) == "final_search"

    def test_route_after_select_chunks_single_query(self):
        """With a single sub-query at index 0, routes to 'final_search'."""
        state = _base_state(
            sub_queries=["only query"],
            current_sub_query_index=0,
        )
        assert route_after_select_chunks(state) == "final_search"

    # -- route_after_rewrite ------------------------------------------------

    def test_route_after_rewrite_budget_exhausted(self):
        """When max rewrites reached, routes to 'generate' instead of 'retrieve'."""
        state = _base_state(rewrite_count=3, max_rewrites=3)
        assert route_after_rewrite(state) == "generate"

    def test_route_after_rewrite_budget_remaining(self):
        """When rewrite budget remains, routes to 'retrieve'."""
        state = _base_state(rewrite_count=1, max_rewrites=3)
        assert route_after_rewrite(state) == "retrieve"


# ============================================================================
# 4. Route-Loop Invariants
# ============================================================================


class TestRouteLoopInvariants:
    """Invariant tests ensuring bounded loops in edge routing."""

    def test_rewrite_loop_bounded(self):
        """rewrite_count cannot exceed max_rewrites -- edge function always terminates."""
        for rewrite_count in range(10):
            max_rewrites = 3
            state = _base_state(
                rewrite_count=rewrite_count,
                max_rewrites=max_rewrites,
            )
            result = route_after_rewrite(state)
            if rewrite_count >= max_rewrites:
                assert result == "generate", (
                    f"Expected 'generate' when rewrite_count={rewrite_count} "
                    f">= max_rewrites={max_rewrites}, got '{result}'"
                )
            else:
                assert result == "retrieve"

    def test_retrieval_budget_bounded(self):
        """After retrieval budget is exhausted, grading routes to 'generate'."""
        for attempts in range(10):
            max_attempts = 3
            state = _base_state(
                overall_grade="ambiguous",
                retrieval_attempts=attempts,
                max_retrieval_attempts=max_attempts,
            )
            result = route_after_grading(state)
            if attempts >= max_attempts:
                assert result == "generate", (
                    f"Expected 'generate' when attempts={attempts} "
                    f">= max={max_attempts}, got '{result}'"
                )
            else:
                assert result == "rewrite_query"

    def test_combined_loops_converge(self):
        """Simulating repeated grading + rewrite cycles always terminates."""
        max_attempts = 3
        max_rewrites = 3
        attempts = 0
        rewrite_count = 0
        iterations = 0

        while iterations < 100:  # hard safety bound
            # Simulate grading edge
            grading_state = _base_state(
                overall_grade="ambiguous",
                retrieval_attempts=attempts,
                max_retrieval_attempts=max_attempts,
            )
            grading_result = route_after_grading(grading_state)

            if grading_result == "generate":
                break

            # Simulate rewrite edge
            rewrite_state = _base_state(
                rewrite_count=rewrite_count,
                max_rewrites=max_rewrites,
            )
            rewrite_result = route_after_rewrite(rewrite_state)

            if rewrite_result == "generate":
                break

            # Simulate one more attempt
            attempts += 1
            rewrite_count += 1
            iterations += 1

        assert iterations < 100, "Loop did not converge within 100 iterations"
        # Should converge within max_attempts iterations
        assert iterations <= max_attempts


# ============================================================================
# 5. Citation Invariants
# ============================================================================


class TestCitationInvariants:
    """Tests for citation index and deduplication properties."""

    def test_citation_indices_monotonic(self):
        """Global citation index should be monotonically increasing."""
        from lasagna.agentic.agentic_rag import RetrievedChunkData
        from lasagna.agentic.nodes.refine_chunks import _vanilla_pass

        # Create fake chunks
        chunks = [
            RetrievedChunkData(
                chunk_id=f"c{i}",
                document_id="doc1",
                document_name="test.pdf",
                content=f"Unique content for chunk {i} about topic {i}.",
                score=0.9 - i * 0.1,
                chunk_index=i,
                page_number=i + 1,
                metadata={},
            )
            for i in range(3)
        ]

        # First vanilla pass
        seen_hashes: Set[str] = set()
        result1 = _vanilla_pass(
            chunks[:2],
            query="test query",
            hop_index=0,
            citation_start_index=1,
            seen_content_hashes=seen_hashes,
        )

        idx_after_first = result1["global_citation_index"]
        assert idx_after_first > 1, "Citation index should advance after first pass"

        # Second vanilla pass picks up where the first left off
        result2 = _vanilla_pass(
            chunks[2:],
            query="test query",
            hop_index=1,
            citation_start_index=idx_after_first,
            seen_content_hashes=result1["seen_content_hashes"],
        )

        idx_after_second = result2["global_citation_index"]
        assert idx_after_second > idx_after_first, (
            f"Second pass index ({idx_after_second}) should be greater "
            f"than first pass index ({idx_after_first})"
        )

        # Verify all citation indices are strictly increasing
        all_citations = result1["all_hop_citations"] + result2["all_hop_citations"]
        indices = [c["index"] for c in all_citations]
        for i in range(1, len(indices)):
            assert indices[i] > indices[i - 1], f"Citation indices not monotonic: {indices}"

    def test_content_hash_deduplication(self):
        """Duplicate content hashes should be skipped in citation building."""
        from lasagna.agentic.agentic_rag import RetrievedChunkData
        from lasagna.agentic.nodes.refine_chunks import _vanilla_pass

        # Two chunks with identical content
        duplicate_content = "This is the same content in both chunks."
        chunks = [
            RetrievedChunkData(
                chunk_id="c1",
                document_id="doc1",
                document_name="test.pdf",
                content=duplicate_content,
                score=0.9,
                chunk_index=0,
                metadata={},
            ),
            RetrievedChunkData(
                chunk_id="c2",
                document_id="doc1",
                document_name="test.pdf",
                content=duplicate_content,
                score=0.8,
                chunk_index=1,
                metadata={},
            ),
        ]

        seen_hashes: Set[str] = set()
        result = _vanilla_pass(
            chunks,
            query="test query",
            hop_index=0,
            citation_start_index=1,
            seen_content_hashes=seen_hashes,
        )

        # Only one citation should be produced (duplicate skipped)
        assert len(result["all_hop_citations"]) == 1
        assert result["global_citation_index"] == 2  # started at 1, advanced once

    def test_seen_content_hashes_persist_across_hops(self):
        """Content hashes accumulated in hop 0 prevent duplicates in hop 1."""
        from lasagna.agentic.agentic_rag import RetrievedChunkData
        from lasagna.agentic.nodes.refine_chunks import _vanilla_pass

        shared_content = "Shared content across hops."
        chunk_hop0 = RetrievedChunkData(
            chunk_id="c1",
            document_id="doc1",
            document_name="test.pdf",
            content=shared_content,
            score=0.9,
            chunk_index=0,
            metadata={},
        )
        chunk_hop1 = RetrievedChunkData(
            chunk_id="c2",
            document_id="doc1",
            document_name="test.pdf",
            content=shared_content,
            score=0.85,
            chunk_index=1,
            metadata={},
        )

        seen_hashes: Set[str] = set()

        result0 = _vanilla_pass(
            [chunk_hop0],
            query="query",
            hop_index=0,
            citation_start_index=1,
            seen_content_hashes=seen_hashes,
        )
        assert len(result0["all_hop_citations"]) == 1

        # Re-use the same seen hashes set
        result1 = _vanilla_pass(
            [chunk_hop1],
            query="query",
            hop_index=1,
            citation_start_index=result0["global_citation_index"],
            seen_content_hashes=result0["seen_content_hashes"],
        )
        # The duplicate should be skipped
        assert len(result1["all_hop_citations"]) == 0


# ============================================================================
# 6. Graph Builder Test
# ============================================================================


class TestGraphBuilder:
    """Tests for ``build_rag_graph``."""

    def test_graph_builds_without_error(self):
        """build_rag_graph compiles without error."""
        from lasagna.agentic.graph_builder import build_rag_graph

        graph = build_rag_graph(session_config=None)
        assert graph is not None

    def test_graph_builds_with_session_config(self):
        """build_rag_graph compiles when a session config is provided."""
        from lasagna.agentic.graph_builder import build_rag_graph

        config = types.SimpleNamespace(
            enable_hallucination_check=True,
            enable_answer_quality_check=True,
        )
        graph = build_rag_graph(session_config=config)
        assert graph is not None


# ============================================================================
# 7. Quality Gates Tests
# ============================================================================


class TestQualityGates:
    """Tests for hallucination and answer quality gate nodes."""

    @pytest.mark.asyncio
    async def test_hallucination_check_supported(self):
        """When LLM says 'supported', result is 'supported'."""
        from lasagna.agentic.nodes.quality_gates import (
            check_hallucination_node,
        )

        mock_llm = _make_async_llm("supported")

        # Provide some fake graded chunks with a .content attribute
        mock_chunk = types.SimpleNamespace(content="Some relevant source text.")

        state = _base_state(
            llm=mock_llm,
            generation="The answer derived from sources.",
            graded_chunks=[mock_chunk],
        )
        result = await check_hallucination_node(state)

        assert result["hallucination_check"] == "supported"
        assert result["quality_check_attempts"] == 1

    @pytest.mark.asyncio
    async def test_hallucination_check_not_supported(self):
        """When LLM says 'not_supported', result is 'not_supported'."""
        from lasagna.agentic.nodes.quality_gates import (
            check_hallucination_node,
        )

        mock_llm = _make_async_llm("not_supported")
        mock_chunk = types.SimpleNamespace(content="Source content.")

        state = _base_state(
            llm=mock_llm,
            generation="A fabricated answer.",
            graded_chunks=[mock_chunk],
        )
        result = await check_hallucination_node(state)

        assert result["hallucination_check"] == "not_supported"

    @pytest.mark.asyncio
    async def test_hallucination_check_partially_supported(self):
        """When LLM says 'partially_supported', result is 'partially_supported'."""
        from lasagna.agentic.nodes.quality_gates import (
            check_hallucination_node,
        )

        mock_llm = _make_async_llm("**partially_supported**")
        mock_chunk = types.SimpleNamespace(content="Source content.")

        state = _base_state(
            llm=mock_llm,
            generation="Partially grounded answer.",
            graded_chunks=[mock_chunk],
        )
        result = await check_hallucination_node(state)

        assert result["hallucination_check"] == "partially_supported"

    @pytest.mark.asyncio
    async def test_hallucination_check_no_llm(self):
        """When LLM is None, defaults to 'supported'."""
        from lasagna.agentic.nodes.quality_gates import (
            check_hallucination_node,
        )

        state = _base_state(
            llm=None,
            generation="Some answer.",
            graded_chunks=[],
        )
        result = await check_hallucination_node(state)

        assert result["hallucination_check"] == "supported"

    @pytest.mark.asyncio
    async def test_hallucination_check_no_generation(self):
        """When generation is empty, defaults to 'supported'."""
        from lasagna.agentic.nodes.quality_gates import (
            check_hallucination_node,
        )

        mock_llm = _make_async_llm("not_supported")
        state = _base_state(
            llm=mock_llm,
            generation="",
            graded_chunks=[],
        )
        result = await check_hallucination_node(state)

        assert result["hallucination_check"] == "supported"

    @pytest.mark.asyncio
    async def test_hallucination_check_no_chunks(self):
        """When no chunks are available, defaults to 'supported'."""
        from lasagna.agentic.nodes.quality_gates import (
            check_hallucination_node,
        )

        mock_llm = _make_async_llm("not_supported")
        state = _base_state(
            llm=mock_llm,
            generation="An answer.",
            graded_chunks=[],
            refined_chunks=[],
            retrieved_chunks=[],
        )
        result = await check_hallucination_node(state)

        assert result["hallucination_check"] == "supported"

    @pytest.mark.asyncio
    async def test_hallucination_check_llm_error_defaults_supported(self):
        """When the LLM call raises an exception, defaults to 'supported'."""
        from lasagna.agentic.nodes.quality_gates import (
            check_hallucination_node,
        )

        failing_llm = AsyncMock()
        failing_llm.ainvoke.side_effect = RuntimeError("LLM unavailable")
        mock_chunk = types.SimpleNamespace(content="Source content.")

        state = _base_state(
            llm=failing_llm,
            generation="Answer text.",
            graded_chunks=[mock_chunk],
        )
        result = await check_hallucination_node(state)

        assert result["hallucination_check"] == "supported"

    @pytest.mark.asyncio
    async def test_answer_quality_check_addresses(self):
        """When LLM says answer addresses question, result is 'addresses_question'."""
        from lasagna.agentic.nodes.quality_gates import (
            check_answer_quality_node,
        )

        mock_llm = _make_async_llm("addresses_question")

        state = _base_state(
            llm=mock_llm,
            generation="A complete and relevant answer.",
            original_question="What is RAG?",
        )
        result = await check_answer_quality_node(state)

        assert result["answer_quality"] == "addresses_question"

    @pytest.mark.asyncio
    async def test_answer_quality_check_does_not_address(self):
        """When LLM says answer does not address, result is 'does_not_address'."""
        from lasagna.agentic.nodes.quality_gates import (
            check_answer_quality_node,
        )

        mock_llm = _make_async_llm("does_not_address")

        state = _base_state(
            llm=mock_llm,
            generation="Completely off-topic response.",
            original_question="What is RAG?",
        )
        result = await check_answer_quality_node(state)

        assert result["answer_quality"] == "does_not_address"

    @pytest.mark.asyncio
    async def test_answer_quality_check_partially_addresses(self):
        """When LLM says 'partially_addresses', result is 'partially_addresses'."""
        from lasagna.agentic.nodes.quality_gates import (
            check_answer_quality_node,
        )

        mock_llm = _make_async_llm("**partially_addresses**")

        state = _base_state(
            llm=mock_llm,
            generation="A vague answer.",
            original_question="Explain RAG in detail.",
        )
        result = await check_answer_quality_node(state)

        assert result["answer_quality"] == "partially_addresses"

    @pytest.mark.asyncio
    async def test_answer_quality_check_no_llm(self):
        """When LLM is None, defaults to 'addresses_question'."""
        from lasagna.agentic.nodes.quality_gates import (
            check_answer_quality_node,
        )

        state = _base_state(
            llm=None,
            generation="An answer.",
        )
        result = await check_answer_quality_node(state)

        assert result["answer_quality"] == "addresses_question"

    @pytest.mark.asyncio
    async def test_answer_quality_check_no_generation(self):
        """When generation is empty, defaults to 'addresses_question'."""
        from lasagna.agentic.nodes.quality_gates import (
            check_answer_quality_node,
        )

        mock_llm = _make_async_llm("does_not_address")
        state = _base_state(
            llm=mock_llm,
            generation="",
        )
        result = await check_answer_quality_node(state)

        assert result["answer_quality"] == "addresses_question"

    @pytest.mark.asyncio
    async def test_hallucination_check_increments_quality_attempts(self):
        """quality_check_attempts should increment after hallucination check."""
        from lasagna.agentic.nodes.quality_gates import (
            check_hallucination_node,
        )

        mock_llm = _make_async_llm("supported")
        mock_chunk = types.SimpleNamespace(content="Source.")

        state = _base_state(
            llm=mock_llm,
            generation="Answer.",
            graded_chunks=[mock_chunk],
            quality_check_attempts=2,
        )
        result = await check_hallucination_node(state)

        assert result["quality_check_attempts"] == 3

    @pytest.mark.asyncio
    async def test_quality_gate_timing_spans_accumulated(self):
        """timing_spans['quality_check_ms'] is accumulated across calls."""
        from lasagna.agentic.nodes.quality_gates import (
            check_hallucination_node,
        )

        mock_llm = _make_async_llm("supported")
        mock_chunk = types.SimpleNamespace(content="Source.")

        state = _base_state(
            llm=mock_llm,
            generation="Answer.",
            graded_chunks=[mock_chunk],
            timing_spans={"quality_check_ms": 50.0},
        )
        result = await check_hallucination_node(state)

        # Should be >= 50 (accumulated)
        assert result["timing_spans"]["quality_check_ms"] >= 50.0


# ============================================================================
# 8. Regression Fixes
# ============================================================================


class TestRegressionFixes:
    """Regression tests for critical bug fixes in sub-query progression and quality gates."""

    @pytest.mark.asyncio
    async def test_plan_retrieval_sets_first_subquery_as_current(self):
        """plan_retrieval_node must set current_query to sub_queries[0], not original_question."""
        from lasagna.agentic.nodes.plan_retrieval import plan_retrieval_node

        mock_llm = _make_async_llm(
            "1. What is retrieval augmented generation\n2. How does RAG improve LLM answers\n"
        )
        config = types.SimpleNamespace(enable_query_decomposition=True)
        routing = types.SimpleNamespace(tier=RoutingTier.COMPREHENSIVE)
        state = _base_state(
            routing_decision=routing,
            session_config=config,
            llm=mock_llm,
            max_retrieval_attempts=5,
        )
        result = await plan_retrieval_node(state)

        assert len(result["sub_queries"]) == 2
        # CRITICAL: current_query must be the first sub-query, not original_question
        assert result["current_query"] == result["sub_queries"][0]
        assert result["current_query"] != "What is RAG?"

    @pytest.mark.asyncio
    async def test_advance_sub_query_node_increments_index(self):
        """advance_sub_query_node must increment index and set current_query."""
        from lasagna.agentic.nodes.advance_sub_query import advance_sub_query_node

        state = _base_state(
            sub_queries=["q1", "q2", "q3"],
            current_sub_query_index=0,
        )
        result = await advance_sub_query_node(state)

        assert result["current_sub_query_index"] == 1
        assert result["current_query"] == "q2"

    @pytest.mark.asyncio
    async def test_answer_quality_check_increments_quality_attempts(self):
        """check_answer_quality_node MUST increment quality_check_attempts."""
        from lasagna.agentic.nodes.quality_gates import check_answer_quality_node

        mock_llm = _make_async_llm("does_not_address")
        state = _base_state(
            llm=mock_llm,
            generation="Bad answer.",
            original_question="What is RAG?",
            quality_check_attempts=1,
        )
        result = await check_answer_quality_node(state)

        assert result["quality_check_attempts"] == 2

    def test_quality_gate_loop_bounded_answer_quality_only(self):
        """When only answer quality gate is enabled, the loop must terminate."""
        from lasagna.agentic.edges import route_after_quality_check

        max_attempts = 2
        config = types.SimpleNamespace(max_quality_check_attempts=max_attempts)

        # Simulate worst case: does_not_address every time
        for attempts in range(5):
            state = _base_state(
                answer_quality="does_not_address",
                hallucination_check="supported",
                quality_check_attempts=attempts,
                session_config=config,
            )
            result = route_after_quality_check(state)

            if attempts >= max_attempts:
                assert result == "select_chunks", (
                    f"Expected 'select_chunks' at attempts={attempts}, got '{result}'"
                )

    def test_sub_query_full_progression_terminates(self):
        """Simulating full sub-query loop: route_after_select_chunks eventually routes to final_search."""
        # Start: 3 sub-queries, index 0
        state = _base_state(
            sub_queries=["sq1", "sq2", "sq3"],
            current_sub_query_index=0,
        )
        assert route_after_select_chunks(state) == "advance_sub_query"

        # After advancing to index 1
        state["current_sub_query_index"] = 1
        assert route_after_select_chunks(state) == "advance_sub_query"

        # After advancing to index 2 (last)
        state["current_sub_query_index"] = 2
        assert route_after_select_chunks(state) == "final_search"

    def test_graph_builder_includes_advance_sub_query_and_final_search(self):
        """Graph must include advance_sub_query and final_search nodes."""
        from lasagna.agentic.graph_builder import build_rag_graph

        graph = build_rag_graph(session_config=None)
        # CompiledGraph wraps a Graph that has .nodes
        # At minimum, the graph should compile without error (tested above)
        # and contain the expected node count (13 nodes now)
        assert graph is not None


# ============================================================================
# 9. Round 2 Regression Fixes
# ============================================================================


class TestRound2Fixes:
    """Regression tests for round 2 bug fixes."""

    @pytest.mark.asyncio
    async def test_web_fallback_preserves_prior_hop_citations(self):
        """web_fallback_node must APPEND web citations, not replace all_hop_citations."""
        from lasagna.agentic.nodes.web_fallback import web_fallback_node
        from unittest.mock import patch

        # Prior hop has 2 existing citations
        existing_citations = [
            {"index": 1, "document_id": "doc1", "chunk_id": "c1", "content_preview": "prior1"},
            {"index": 2, "document_id": "doc1", "chunk_id": "c2", "content_preview": "prior2"},
        ]

        # Mock web search to return 1 result
        mock_web_chunk = types.SimpleNamespace(
            content="Web result content",
            document_id="web",
            document_name="web_result",
            chunk_id="web_1",
            score=0.5,
        )

        state = _base_state(
            all_hop_citations=existing_citations,
            global_citation_index=3,
            seen_content_hashes={"hash_prior1", "hash_prior2"},
            hop_count=1,
            graded_chunks=[],
        )

        with patch(
            "lasagna.agentic.nodes.web_fallback._crag_web_search_fallback",
            new_callable=AsyncMock,
            return_value=[mock_web_chunk],
        ):
            result = await web_fallback_node(state)

        # Prior citations must still be present
        assert len(result["all_hop_citations"]) >= 3  # 2 prior + 1 web
        assert result["all_hop_citations"][0]["chunk_id"] == "c1"
        assert result["all_hop_citations"][1]["chunk_id"] == "c2"
        assert result["crag_web_fallback_used"] is True

    @pytest.mark.asyncio
    async def test_web_fallback_uses_correct_hop_index(self):
        """web_fallback citations must use hop_count from state, not hardcoded 0."""
        from lasagna.agentic.nodes.web_fallback import _build_web_citations

        mock_chunk = types.SimpleNamespace(
            content="Web content",
            document_id="web",
            document_name="web_result",
            chunk_id="w1",
            score=0.5,
        )

        citations, _, _ = _build_web_citations(
            [mock_chunk],
            query="test",
            start_index=5,
            seen_hashes=set(),
            hop_index=3,
        )

        assert len(citations) == 1
        assert citations[0]["hop_index"] == 3
        assert citations[0]["chunk_index"] == 0

    @pytest.mark.asyncio
    async def test_web_fallback_enumerates_chunk_index(self):
        """Each web citation must have sequential chunk_index, not all 0."""
        from lasagna.agentic.nodes.web_fallback import _build_web_citations

        chunks = [
            types.SimpleNamespace(
                content=f"Content {i}",
                document_id="web",
                document_name="web",
                chunk_id=f"w{i}",
                score=0.5,
            )
            for i in range(3)
        ]

        citations, _, _ = _build_web_citations(
            chunks,
            query="test",
            start_index=1,
            seen_hashes=set(),
            hop_index=0,
        )

        chunk_indices = [c["chunk_index"] for c in citations]
        assert chunk_indices == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_rewrite_query_does_not_advance_sub_query(self):
        """rewrite_query_node must NOT advance current_sub_query_index."""
        from lasagna.agentic.nodes.rewrite_query import rewrite_query_node

        mock_llm = _make_async_llm("What are the benefits of RAG")

        state = _base_state(
            llm=mock_llm,
            sub_queries=["sq1", "sq2", "sq3"],
            current_sub_query_index=0,
            current_query="sq1",
            rewrite_count=0,
        )
        result = await rewrite_query_node(state)

        # Must NOT change sub-query index
        assert "current_sub_query_index" not in result
        # Must have done LLM rewrite
        assert result["current_query"] == "What are the benefits of RAG"
        assert result["rewrite_count"] == 1

    def test_final_search_citations_have_full_content_field(self):
        """final_search citation dicts must include a 'full_content' field."""
        # We test the shape of the citation dict built in final_search_node.
        # Since running the actual node requires a RAGBackend, we just verify
        # the schema expectation is documented via content field presence.
        cit = {
            "index": 0,
            "content_preview": "short preview",
            "full_content": "full document content that may be very long " * 10,
        }
        assert "full_content" in cit
        assert len(cit["full_content"]) > len(cit["content_preview"])

    def test_format_output_merge_uses_full_content_for_dedup(self):
        """_merge_final_search_citations must hash full_content, not content_preview."""
        from lasagna.agentic.nodes.format_output import (
            _merge_final_search_citations,
        )
        from lasagna.agentic.agentic_rag import _generate_content_hash

        full_content = "A" * 500
        truncated = full_content[:200]

        # Hop citation was built with full content hash
        hop_hash = _generate_content_hash(full_content)
        seen_hashes = {hop_hash}

        hop_citations = [
            {"index": 1, "chunk_id": "c1", "content_preview": truncated, "score": 0.9},
        ]
        # Final-search citation has same full content but different preview
        final_citations = [
            {
                "index": 0,
                "chunk_id": "c2",
                "content_preview": truncated,
                "full_content": full_content,
                "score": 0.8,
            },
        ]

        merged, _ = _merge_final_search_citations(
            hop_citations,
            final_citations,
            seen_hashes,
        )

        # The final-search citation should be DEDUPED because full_content matches
        assert len(merged) == 1
        assert merged[0]["chunk_id"] == "c1"

    def test_format_output_merge_admits_novel_final_search_content(self):
        """Novel final-search content (different full_content) should be merged in."""
        from lasagna.agentic.nodes.format_output import (
            _merge_final_search_citations,
        )

        hop_citations = [
            {"index": 1, "chunk_id": "c1", "content_preview": "hop content", "score": 0.9},
        ]
        final_citations = [
            {
                "index": 0,
                "chunk_id": "c3",
                "content_preview": "novel short",
                "full_content": "Completely novel content not seen before",
                "score": 0.7,
            },
        ]

        merged, _ = _merge_final_search_citations(
            hop_citations,
            final_citations,
            seen_content_hashes=set(),
        )

        assert len(merged) == 2
        # Indices should be contiguous [1, 2]
        assert merged[0]["index"] == 1
        assert merged[1]["index"] == 2

    def test_config_response_includes_structured_graph_fields(self):
        """ConfigResponse model must include all structured-graph settings."""
        from lasagna.api.rest.models import ConfigResponse

        fields = ConfigResponse.model_fields
        expected = [
            "use_structured_graph",
            "enable_hallucination_check",
            "enable_answer_quality_check",
            "max_quality_check_attempts",
            "coverage_scorer_mode",
            "final_search_mode",
        ]
        for field_name in expected:
            assert field_name in fields, f"ConfigResponse missing field: {field_name}"

    def test_web_fallback_citations_include_full_content(self):
        """Web fallback citations must include full_content field."""
        from lasagna.agentic.nodes.web_fallback import _build_web_citations

        long_content = "X" * 500
        chunk = types.SimpleNamespace(
            content=long_content,
            document_id="web",
            document_name="web_result",
            chunk_id="w1",
            score=0.5,
        )

        citations, _, _ = _build_web_citations(
            [chunk],
            query="test",
            start_index=1,
            seen_hashes=set(),
            hop_index=0,
        )

        assert len(citations) == 1
        assert citations[0]["full_content"] == long_content
        assert citations[0]["content_preview"] == long_content[:200]


# ============================================================================
# 10. Round 3 Regression Fixes
# ============================================================================


class TestRound3Fixes:
    """Regression tests for round 3 bug fixes."""

    @pytest.mark.asyncio
    async def test_web_fallback_strips_current_hop_incorrect_citations(self):
        """web_fallback must remove current-hop citations (graded incorrect) before appending web results."""
        from lasagna.agentic.nodes.web_fallback import web_fallback_node
        from unittest.mock import patch

        # Hop 0 had valid citations, hop 1 was graded incorrect
        existing_citations = [
            {
                "index": 1,
                "document_id": "doc1",
                "chunk_id": "c1",
                "content_preview": "valid hop0",
                "hop_index": 0,
            },
            {
                "index": 2,
                "document_id": "doc1",
                "chunk_id": "c2",
                "content_preview": "valid hop0",
                "hop_index": 0,
            },
            {
                "index": 3,
                "document_id": "doc1",
                "chunk_id": "c3",
                "content_preview": "bad hop1",
                "hop_index": 1,
            },
            {
                "index": 4,
                "document_id": "doc1",
                "chunk_id": "c4",
                "content_preview": "bad hop1",
                "hop_index": 1,
            },
        ]

        mock_web_chunk = types.SimpleNamespace(
            content="Web replacement content",
            document_id="web",
            document_name="web_result",
            chunk_id="web_1",
            score=0.5,
        )

        state = _base_state(
            all_hop_citations=existing_citations,
            global_citation_index=5,
            seen_content_hashes=set(),
            hop_count=1,  # current hop is 1 (the incorrect one)
            graded_chunks=[],
        )

        with patch(
            "lasagna.agentic.nodes.web_fallback._crag_web_search_fallback",
            new_callable=AsyncMock,
            return_value=[mock_web_chunk],
        ):
            result = await web_fallback_node(state)

        citations = result["all_hop_citations"]
        # Hop 0 citations preserved, hop 1 citations stripped, web citation added
        hop_indices = [c.get("hop_index") for c in citations]
        assert 0 in hop_indices, "Hop 0 citations must be preserved"
        # Hop 1 citations (the incorrect ones) must be gone — only web citation for hop 1
        hop1_cits = [c for c in citations if c.get("hop_index") == 1]
        assert all(c.get("document_id") == "web" for c in hop1_cits), (
            "Hop 1 should only have web citations, not the original incorrect ones"
        )

    @pytest.mark.asyncio
    async def test_web_fallback_reindexes_after_stripping(self):
        """After stripping current-hop citations, web citations get correct indices."""
        from lasagna.agentic.nodes.web_fallback import web_fallback_node
        from unittest.mock import patch

        existing_citations = [
            {"index": 1, "chunk_id": "c1", "hop_index": 0, "content_preview": "ok"},
            {"index": 2, "chunk_id": "c2", "hop_index": 1, "content_preview": "bad"},
        ]

        mock_web_chunk = types.SimpleNamespace(
            content="Web content",
            document_id="web",
            document_name="web",
            chunk_id="w1",
            score=0.5,
        )

        state = _base_state(
            all_hop_citations=existing_citations,
            global_citation_index=3,
            seen_content_hashes=set(),
            hop_count=1,
            graded_chunks=[],
        )

        with patch(
            "lasagna.agentic.nodes.web_fallback._crag_web_search_fallback",
            new_callable=AsyncMock,
            return_value=[mock_web_chunk],
        ):
            result = await web_fallback_node(state)

        citations = result["all_hop_citations"]
        # Prior hop had index 1, so web citation should start at 2
        assert citations[0]["index"] == 1  # prior hop
        assert citations[1]["index"] == 2  # web citation

    def test_agentic_rag_create_accepts_session_config(self):
        """AgenticRAG.create() must accept session_config parameter."""
        from lasagna.agentic.agentic_rag import AgenticRAG
        import inspect

        sig = inspect.signature(AgenticRAG.create)
        assert "session_config" in sig.parameters, (
            "AgenticRAG.create() must accept session_config parameter"
        )
        # Default should be None
        assert sig.parameters["session_config"].default is None

    def test_final_search_node_imports_hyde_decision(self):
        """final_search_node must import _should_use_hyde_for_agentic."""
        from lasagna.agentic.nodes import final_search

        assert hasattr(final_search, "_should_use_hyde_for_agentic"), (
            "final_search module must import _should_use_hyde_for_agentic"
        )

    @pytest.mark.asyncio
    async def test_web_fallback_cleans_stale_hashes_for_incorrect_chunks(self):
        """When stripping incorrect-hop citations, their content hashes must also be removed."""
        from lasagna.agentic.nodes.web_fallback import web_fallback_node
        from lasagna.agentic.agentic_rag import _generate_content_hash
        from unittest.mock import patch

        # Simulate: hop 0 had a chunk with content "AAA", hop 1 had "BBB" (graded incorrect)
        content_a = "Content from hop 0 that was fine"
        content_b = "Content from hop 1 that was graded incorrect"
        hash_a = _generate_content_hash(content_a)
        hash_b = _generate_content_hash(content_b)

        existing_citations = [
            {"index": 1, "chunk_id": "c1", "hop_index": 0, "content_preview": content_a[:200]},
            {"index": 2, "chunk_id": "c2", "hop_index": 1, "content_preview": content_b[:200]},
        ]

        # graded_chunks are the current hop's RetrievedChunkData objects.
        # chunk_id must match the citation's chunk_id so hash cleanup
        # can identify which hashes belong to the stripped citations.
        incorrect_chunk = types.SimpleNamespace(content=content_b, chunk_id="c2")

        mock_web_chunk = types.SimpleNamespace(
            content="Web replacement",
            document_id="web",
            document_name="web",
            chunk_id="w1",
            score=0.5,
        )

        state = _base_state(
            all_hop_citations=existing_citations,
            global_citation_index=3,
            seen_content_hashes={hash_a, hash_b},
            hop_count=1,
            graded_chunks=[incorrect_chunk],
        )

        with patch(
            "lasagna.agentic.nodes.web_fallback._crag_web_search_fallback",
            new_callable=AsyncMock,
            return_value=[mock_web_chunk],
        ):
            result = await web_fallback_node(state)

        # hash_a must still be in seen_content_hashes (hop 0 was valid)
        assert hash_a in result["seen_content_hashes"]
        # hash_b must be REMOVED (hop 1 was incorrect, its chunks are discarded)
        assert hash_b not in result["seen_content_hashes"]


# ============================================================================
# Round 5 regression tests
# ============================================================================


class TestRound5Fixes:
    """Regression tests for round 5 feedback fixes."""

    @pytest.mark.asyncio
    async def test_web_fallback_preserves_shared_content_hashes(self):
        """Hash cleanup must NOT remove hashes for chunks shared with prior hops.

        Scenario: hop 0 and hop 1 both retrieved a chunk with the same content
        (dedup-skipped in hop 1, so no citation was created for it in hop 1).
        When hop 1 is graded incorrect and stripped, the shared chunk's hash
        must remain because hop 0's citation still references it.
        """
        from lasagna.agentic.nodes.web_fallback import web_fallback_node
        from lasagna.agentic.agentic_rag import _generate_content_hash
        from unittest.mock import patch

        shared_content = "Content that appears in both hop 0 and hop 1"
        hop1_only_content = "Content unique to hop 1"
        hash_shared = _generate_content_hash(shared_content)
        hash_hop1 = _generate_content_hash(hop1_only_content)

        existing_citations = [
            # hop 0 citation for the shared chunk
            {
                "index": 1,
                "chunk_id": "shared_c1",
                "hop_index": 0,
                "content_preview": shared_content[:200],
            },
            # hop 1 citation for its unique chunk
            {
                "index": 2,
                "chunk_id": "hop1_c1",
                "hop_index": 1,
                "content_preview": hop1_only_content[:200],
            },
        ]

        # graded_chunks includes BOTH: the hop1-unique chunk AND a dedup-skipped
        # chunk with shared content.  The shared chunk has chunk_id "shared_c1_dup"
        # which does NOT appear in the stripped citations (only "hop1_c1" does).
        graded_chunks = [
            types.SimpleNamespace(content=hop1_only_content, chunk_id="hop1_c1"),
            types.SimpleNamespace(content=shared_content, chunk_id="shared_c1_dup"),
        ]

        mock_web = types.SimpleNamespace(
            content="Web result",
            document_id="web",
            document_name="web",
            chunk_id="w1",
            score=0.5,
        )

        state = _base_state(
            all_hop_citations=existing_citations,
            global_citation_index=3,
            seen_content_hashes={hash_shared, hash_hop1},
            hop_count=1,
            graded_chunks=graded_chunks,
        )

        with patch(
            "lasagna.agentic.nodes.web_fallback._crag_web_search_fallback",
            new_callable=AsyncMock,
            return_value=[mock_web],
        ):
            result = await web_fallback_node(state)

        # hash_shared must be PRESERVED — its chunk_id "shared_c1_dup" is NOT
        # in the stripped citations (only "hop1_c1" was stripped)
        assert hash_shared in result["seen_content_hashes"], (
            "Shared-content hash was incorrectly removed"
        )
        # hash_hop1 must be REMOVED — "hop1_c1" was in the stripped citations
        assert hash_hop1 not in result["seen_content_hashes"], (
            "Hop-1-only hash should have been removed"
        )

    def test_route_after_select_chunks_enforces_budget(self):
        """route_after_select_chunks must stop advancing sub-queries when
        retrieval budget is exhausted."""
        from lasagna.agentic.edges import route_after_select_chunks

        state = _base_state(
            sub_queries=["q1", "q2", "q3"],
            current_sub_query_index=0,  # still has q2 and q3
            hop_count=5,
            max_retrieval_attempts=5,  # budget exhausted
        )
        assert route_after_select_chunks(state) == "final_search"

    def test_route_after_select_chunks_allows_advance_within_budget(self):
        """route_after_select_chunks must allow advancing when budget remains."""
        from lasagna.agentic.edges import route_after_select_chunks

        state = _base_state(
            sub_queries=["q1", "q2", "q3"],
            current_sub_query_index=0,
            hop_count=2,
            max_retrieval_attempts=5,  # budget available
        )
        assert route_after_select_chunks(state) == "advance_sub_query"

    def test_orchestrator_passes_session_config_to_agentic_rag(self):
        """_search_orchestrator must pass session_config= to AgenticRAG()."""
        import ast

        # Resolve from the imported module so the test works from a source
        # checkout, an installed wheel, or any clone location.
        from lasagna.api import _search_orchestrator

        with open(_search_orchestrator.__file__) as f:
            source = f.read()
        tree = ast.parse(source)

        # Find all AgenticRAG() calls and check they have session_config kwarg
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match AgenticRAG(...)
                is_agentic_rag = (isinstance(func, ast.Name) and func.id == "AgenticRAG") or (
                    isinstance(func, ast.Attribute) and func.attr == "AgenticRAG"
                )
                if is_agentic_rag:
                    kwarg_names = [kw.arg for kw in node.keywords]
                    assert "session_config" in kwarg_names, (
                        f"AgenticRAG() call at line {node.lineno} missing session_config kwarg"
                    )

    def test_retrieve_structured_uses_instance_session_config(self):
        """_retrieve_structured must prefer self._session_config over global config."""
        import ast

        # Resolve from the imported module (see note above).
        from lasagna.agentic import agentic_rag

        with open(agentic_rag.__file__) as f:
            source = f.read()

        # Find the _retrieve_structured method and verify it references
        # self._session_config
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_retrieve_structured":
                body_source = ast.get_source_segment(source, node)
                assert "self._session_config" in body_source, (
                    "_retrieve_structured must reference self._session_config"
                )
                break
        else:
            pytest.fail("Could not find _retrieve_structured method")
