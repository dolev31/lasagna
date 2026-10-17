"""Rewrite Query Node — query rewriting for retry loops.

When the graph decides that the current evidence is insufficient (coverage
gap, low quality, etc.) this node rewrites the active search query to
target the missing information.

Uses an LLM-based refinement prompt (identical to the monolithic
``QUERY_REFINEMENT_PROMPT``) that takes the original question, current
results summary, and ``uncovered_aspects`` to produce a focused
follow-up query.  Increments ``rewrite_count`` on every invocation.

**Note**: Sub-query advancement is handled exclusively by
``advance_sub_query_node``.  This node only rewrites the *current*
query — it never skips to the next sub-query.

Owned state fields (write):
    ``current_query``, ``rewrite_count``
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from loguru import logger

from lasagna.agentic.graph_state import RAGGraphState
from ..prompts import resolve_prompt


# ---------------------------------------------------------------------------
# Prompt — mirrors QUERY_REFINEMENT_PROMPT in agentic_rag.py (line 410)
# ---------------------------------------------------------------------------

_QUERY_REFINEMENT_PROMPT = """\
You are a search query optimizer. Given the user's original question and \
the current search results, propose a refined search query to find missing \
information.

## Original Question:
{original_question}

## Current Query:
{current_query}

## Current Search Results Summary:
{results_summary}

## Missing Information (what we still need):
{missing_info}

## Instructions:
1. If the current results are sufficient, respond with: SUFFICIENT
2. If you need to search for more specific information, propose a NEW search query
3. Keep the query SHORT (ideally 3-12 words)
4. Focus on specific entities, terms, or concepts not yet found
5. Avoid repeating the same query - try alternative phrasings

## Response Format:
Either "SUFFICIENT" or your new search query (nothing else)

New query:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_results_summary(state: RAGGraphState) -> str:
    """Build a concise summary of what has been retrieved so far."""
    hop_data: List[Any] = state.get("hop_data", [])
    if not hop_data:
        return "No results retrieved yet."

    parts: List[str] = []
    for hop in hop_data:
        n_chunks = len(hop.retrieved_chunks)
        query_preview = hop.query_used[:80]
        parts.append(f"- Hop {hop.hop_index}: '{query_preview}' -> {n_chunks} chunk(s)")
    return "\n".join(parts)


def _build_missing_info(state: RAGGraphState) -> str:
    """Describe what is still missing based on ``uncovered_aspects``."""
    uncovered: List[str] = state.get("uncovered_aspects", [])
    if uncovered:
        return "\n".join(f"- {aspect}" for aspect in uncovered)
    return "Unable to determine specific gaps — try alternative phrasing."


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def rewrite_query_node(state: RAGGraphState) -> Dict[str, Any]:
    """Rewrite the active search query using LLM-based refinement.

    Sub-query advancement is handled exclusively by ``advance_sub_query_node``.
    This node only rewrites the *current* query to target uncovered aspects.

    Parameters
    ----------
    state:
        Current ``RAGGraphState``.

    Returns
    -------
    dict
        Updated fields: ``current_query``, ``rewrite_count``.
    """
    rewrite_count: int = state.get("rewrite_count", 0)

    # ------------------------------------------------------------------
    # LLM-based query rewrite targeting uncovered aspects
    # ------------------------------------------------------------------
    llm = state.get("llm")
    original_question = state.get("original_question", "")
    current_query = state.get("current_query", original_question)

    results_summary = _build_results_summary(state)
    missing_info = _build_missing_info(state)

    prompt = resolve_prompt(
        "rewrite_query", _QUERY_REFINEMENT_PROMPT, state.get("session_config")
    ).format(
        original_question=original_question,
        current_query=current_query,
        results_summary=results_summary,
        missing_info=missing_info,
    )

    new_query = current_query  # safe fallback
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # If the LLM says the results are sufficient, keep the current query
        # unchanged — the routing edge will detect stabilisation.
        if raw.upper().startswith("SUFFICIENT"):
            logger.info("[RewriteQuery] LLM reports results are SUFFICIENT")
            new_query = current_query
        else:
            new_query = raw
            logger.debug(
                "[RewriteQuery] Rewritten query (attempt {n}): '{q}'",
                n=rewrite_count + 1,
                q=new_query[:80],
            )
    except Exception:
        logger.exception("[RewriteQuery] LLM rewrite failed; keeping current query")

    return {
        "current_query": new_query,
        "rewrite_count": rewrite_count + 1,
    }
