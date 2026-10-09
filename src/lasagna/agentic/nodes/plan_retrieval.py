"""Plan Retrieval Node — maps upstream routing to query planning state.

This node receives the already-computed ``routing_decision`` from state
(pre-computed upstream in ``_search_orchestrator.py``) and translates it into
the query-planning fields consumed by downstream retrieval nodes.

It does **NOT** re-run the ``QueryRouter`` — that would double-classify the
query and waste latency/cost.

Features:
  - LLM-based query decomposition for complex queries (when
    ``enable_query_decomposition`` is enabled in session config).
  - Adaptive iteration budgets are computed from routing tier complexity.

Owned state fields (write):
  ``query_complexity``, ``sub_queries``, ``current_sub_query_index``,
  ``current_query``, ``max_retrieval_attempts``
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from loguru import logger

from lasagna.agentic.graph_state import RAGGraphState
from lasagna.retrieval.routing.models import RoutingTier
from ..prompts import resolve_prompt

_DECOMPOSITION_PROMPT = """\
Break the following complex question into 2-4 independent sub-queries that \
can be searched separately. Each sub-query should be:
- Self-contained (understandable without the original question)
- 3-12 words long
- Focused on a specific aspect

Question: {question}

Return ONLY a numbered list (1. ..., 2. ..., etc.) with nothing else."""


async def _decompose_query(
    llm: Any,
    question: str,
    max_sub_queries: int = 4,
    session_config: Any = None,
) -> List[str]:
    """Use the LLM to break a complex question into independent sub-queries.

    Parameters
    ----------
    llm:
        A ``BaseChatModel`` (or compatible) instance already available in
        graph state.
    question:
        The original user question.
    max_sub_queries:
        Upper bound on the number of sub-queries returned.
    session_config:
        Session-scoped config, so a configured prompt override reaches this
        helper. Passed explicitly rather than read from a global: the node
        already has it, and a global would make the prompt depend on process
        state instead of the run being measured.

    Returns
    -------
    list[str]
        Between 2 and *max_sub_queries* sub-queries, or ``[question]`` as a
        fallback when decomposition fails.
    """
    try:
        prompt = resolve_prompt("plan_retrieval", _DECOMPOSITION_PROMPT, session_config).format(
            question=question
        )
        response = await llm.ainvoke(prompt)
        # response may be an AIMessage or a plain string
        text = response.content if hasattr(response, "content") else str(response)

        # Parse numbered list lines like "1. …", "2) …", etc.
        lines = re.findall(r"^\s*\d+[\.\)]\s*(.+)", text, re.MULTILINE)
        if not lines:
            logger.warning(
                "[PlanRetrieval] LLM decomposition returned no parseable sub-queries; "
                "falling back to original question"
            )
            return [question]

        # Trim and cap
        sub_queries = [line.strip() for line in lines[:max_sub_queries] if line.strip()]
        if len(sub_queries) < 2:
            # Need at least 2 sub-queries for decomposition to be useful
            return [question]

        logger.debug(
            "[PlanRetrieval] Decomposed into {n} sub-queries: {sqs}",
            n=len(sub_queries),
            sqs=sub_queries,
        )
        return sub_queries
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[PlanRetrieval] Query decomposition failed ({exc}); falling back to original question",
            exc=exc,
        )
        return [question]


# ---------------------------------------------------------------------------
# Tier → complexity mapping
# ---------------------------------------------------------------------------

_TIER_TO_COMPLEXITY: Dict[RoutingTier, str] = {
    RoutingTier.FAST: "simple",
    RoutingTier.STANDARD: "moderate",
    RoutingTier.COMPREHENSIVE: "complex",
}

# Adaptive iteration budget per complexity level.
# "simple" queries rarely need more than one search hop; "complex" queries are
# allowed the full budget supplied in ``max_retrieval_attempts``.
_COMPLEXITY_BUDGET: Dict[str, int] = {
    "simple": 1,
    "moderate": 2,
    # "complex" uses the existing max_retrieval_attempts (no override)
}


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def plan_retrieval_node(state: RAGGraphState) -> Dict[str, Any]:
    """Translate routing decision into query-planning state fields.

    Parameters
    ----------
    state:
        Current ``RAGGraphState``.  Reads ``routing_decision``,
        ``original_question``, ``max_retrieval_attempts``,
        ``session_config``, and ``llm``.

    Returns
    -------
    dict
        Updated fields: ``query_complexity``, ``sub_queries``,
        ``current_sub_query_index``, ``current_query``,
        ``max_retrieval_attempts``.
    """
    routing_decision = state.get("routing_decision")
    original_question: str = state.get("original_question", "")
    configured_max_attempts: int = state.get("max_retrieval_attempts", 3)

    # ── 1. Map routing tier → complexity string ────────────────────────
    complexity = "moderate"  # safe default when routing_decision is None
    if routing_decision is not None:
        try:
            tier: RoutingTier = routing_decision.tier
            complexity = _TIER_TO_COMPLEXITY.get(tier, "moderate")
            logger.debug(
                "[PlanRetrieval] Mapped routing tier {tier} → complexity '{complexity}'",
                tier=tier.value,
                complexity=complexity,
            )
        except AttributeError:
            # routing_decision exists but has no .tier — treat as moderate
            logger.warning(
                "[PlanRetrieval] routing_decision has no 'tier' attribute; defaulting to 'moderate'"
            )
    else:
        logger.debug("[PlanRetrieval] routing_decision is None; defaulting to 'moderate'")

    # ── 2. Sub-query decomposition ─────────────────────────────────────
    session_config = state.get("session_config")
    decomposition_enabled = getattr(session_config, "enable_query_decomposition", False)

    if complexity == "complex" and decomposition_enabled:
        llm = state.get("llm")
        if llm is not None:
            sub_queries = await _decompose_query(
                llm, original_question, session_config=state.get("session_config")
            )
            logger.debug(
                "[PlanRetrieval] Query decomposition produced {n} sub-queries",
                n=len(sub_queries),
            )
        else:
            logger.warning(
                "[PlanRetrieval] LLM not available in state; skipping query decomposition"
            )
            sub_queries = [original_question]
    else:
        sub_queries = [original_question]

    # ── 3. Adaptive iteration budget ──────────────────────────────────
    if complexity in _COMPLEXITY_BUDGET:
        max_attempts = _COMPLEXITY_BUDGET[complexity]
    else:
        # "complex" — honour the full budget passed from build_initial_state
        max_attempts = configured_max_attempts

    logger.info(
        "[PlanRetrieval] complexity='{complexity}', sub_queries={n_sq}, "
        "max_retrieval_attempts={max_attempts}",
        complexity=complexity,
        n_sq=len(sub_queries),
        max_attempts=max_attempts,
    )

    return {
        "query_complexity": complexity,
        "sub_queries": sub_queries,
        "current_sub_query_index": 0,
        "current_query": sub_queries[0],
        "max_retrieval_attempts": max_attempts,
    }
