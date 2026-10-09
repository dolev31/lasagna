"""Routing functions (conditional edges) for the structured RAG graph.

Each function receives the current ``RAGGraphState`` and returns a string
naming the next node.  These are consumed by
``StateGraph.add_conditional_edges()`` in ``graph_builder.py``.

Important invariants enforced here:
  - ``hop_count`` never exceeds ``max_retrieval_attempts``
  - ``quality_check_attempts`` never exceeds ``max_quality_check_attempts``
  - ``rewrite_count`` never exceeds ``max_rewrites``
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger


# ---------------------------------------------------------------------------
# After CRAG grading
# ---------------------------------------------------------------------------


def route_after_grading(state: Dict[str, Any]) -> str:
    """Decide what to do based on CRAG document grade.

    Returns one of: ``"generate"``, ``"web_fallback"``, ``"rewrite_query"``.
    """
    grade = state.get("overall_grade", "correct")
    attempts = state.get("retrieval_attempts", 0)
    max_attempts = state.get("max_retrieval_attempts", 3)
    session_config = state.get("session_config")

    if grade == "correct":
        return "generate"

    if grade == "incorrect":
        web_fallback = (
            getattr(session_config, "crag_web_search_fallback", False) if session_config else False
        )
        if web_fallback and state.get("crag_web_fallback_used", False):
            logger.info("[Edge] CRAG grade='incorrect' — routing to web_fallback")
            return "web_fallback"

    # ambiguous or incorrect without web fallback: rewrite if budget remains
    if attempts < max_attempts:
        logger.info(
            "[Edge] CRAG grade='{grade}', attempt {a}/{max_a} — routing to rewrite_query",
            grade=grade,
            a=attempts,
            max_a=max_attempts,
        )
        return "rewrite_query"

    # Budget exhausted — generate best-effort answer
    logger.info(
        "[Edge] CRAG grade='{grade}' but retrieval budget exhausted — routing to generate",
        grade=grade,
    )
    return "generate"


# ---------------------------------------------------------------------------
# After generation — check if quality gates are enabled
# ---------------------------------------------------------------------------


def route_after_generation(state: Dict[str, Any]) -> str:
    """Route to quality gates if enabled, otherwise directly to select_chunks.

    Returns one of: ``"check_hallucination"``, ``"check_answer_quality"``,
    ``"select_chunks"``.
    """
    session_config = state.get("session_config")
    if session_config is None:
        return "select_chunks"

    if getattr(session_config, "enable_hallucination_check", False):
        return "check_hallucination"
    if getattr(session_config, "enable_answer_quality_check", False):
        return "check_answer_quality"
    return "select_chunks"


# ---------------------------------------------------------------------------
# After hallucination check
# ---------------------------------------------------------------------------


def route_after_hallucination_check(state: Dict[str, Any]) -> str:
    """Route after hallucination grading.

    Returns one of: ``"check_answer_quality"``, ``"rewrite_query"``,
    ``"select_chunks"``.
    """
    session_config = state.get("session_config")
    label = state.get("hallucination_check", "supported")
    attempts = state.get("quality_check_attempts", 0)
    max_attempts = getattr(session_config, "max_quality_check_attempts", 2) if session_config else 2

    if label == "not_supported" and attempts < max_attempts:
        logger.info(
            "[Edge] Hallucination='not_supported', attempt {a}/{max_a} — rewrite_query",
            a=attempts,
            max_a=max_attempts,
        )
        return "rewrite_query"

    # Continue to answer quality check if enabled
    if session_config and getattr(session_config, "enable_answer_quality_check", False):
        return "check_answer_quality"
    return "select_chunks"


# ---------------------------------------------------------------------------
# After answer quality check
# ---------------------------------------------------------------------------


def route_after_quality_check(state: Dict[str, Any]) -> str:
    """Route after answer quality grading.

    Returns one of: ``"select_chunks"``, ``"rewrite_query"``, ``"generate"``.
    """
    session_config = state.get("session_config")
    quality = state.get("answer_quality", "addresses_question")
    hallucination = state.get("hallucination_check", "supported")
    attempts = state.get("quality_check_attempts", 0)
    max_attempts = getattr(session_config, "max_quality_check_attempts", 2) if session_config else 2

    # Both pass → proceed
    if hallucination in ("supported", "partially_supported") and quality == "addresses_question":
        return "select_chunks"

    # Budget exhausted → accept what we have
    if attempts >= max_attempts:
        logger.info("[Edge] Quality check budget exhausted — accepting current answer")
        return "select_chunks"

    # Hallucination issue → need better retrieval
    if hallucination == "not_supported":
        return "rewrite_query"

    # Answer quality issue → regenerate
    if quality == "does_not_address":
        return "generate"

    # Partially OK — accept
    return "select_chunks"


# ---------------------------------------------------------------------------
# After select_chunks — check for remaining sub-queries
# ---------------------------------------------------------------------------


def route_after_select_chunks(state: Dict[str, Any]) -> str:
    """Check if there are remaining sub-queries to process.

    Enforces the global retrieval budget (``max_retrieval_attempts``) across
    all sub-queries so that a large decomposition cannot bypass the iteration
    cap.

    Returns one of: ``"advance_sub_query"``, ``"final_search"``.
    """
    sub_queries = state.get("sub_queries", [])
    current_idx = state.get("current_sub_query_index", 0)
    hop_count = state.get("hop_count", 0)
    max_attempts = state.get("max_retrieval_attempts", 3)

    if current_idx < len(sub_queries) - 1:
        if hop_count >= max_attempts:
            logger.info(
                "[Edge] Sub-query {idx}/{total} — retrieval budget exhausted "
                "({hops}/{max_a} hops), skipping remaining sub-queries",
                idx=current_idx + 1,
                total=len(sub_queries),
                hops=hop_count,
                max_a=max_attempts,
            )
            return "final_search"
        logger.info(
            "[Edge] Sub-query {idx}/{total} — advancing to next sub-query",
            idx=current_idx + 1,
            total=len(sub_queries),
        )
        return "advance_sub_query"

    return "final_search"


# ---------------------------------------------------------------------------
# After rewrite_query — check if we have retrieval budget left
# ---------------------------------------------------------------------------


def route_after_rewrite(state: Dict[str, Any]) -> str:
    """Guard against infinite rewrite→retrieve loops.

    Returns one of: ``"retrieve"``, ``"generate"``.
    """
    rewrite_count = state.get("rewrite_count", 0)
    max_rewrites = state.get("max_rewrites", 3)

    if rewrite_count >= max_rewrites:
        logger.info(
            "[Edge] Max rewrites ({max}) reached — routing to generate",
            max=max_rewrites,
        )
        return "generate"
    return "retrieve"
