"""Advance Sub-Query Node — moves to the next decomposed sub-query.

When the graph has multiple sub-queries (from query decomposition), this
node increments ``current_sub_query_index`` and sets ``current_query`` to
the next sub-query before looping back to ``retrieve``.

Owned state fields (write):
    ``current_sub_query_index``, ``current_query``
"""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from lasagna.agentic.graph_state import RAGGraphState


async def advance_sub_query_node(state: RAGGraphState) -> Dict[str, Any]:
    """Advance to the next sub-query in the decomposition list.

    Parameters
    ----------
    state:
        Current ``RAGGraphState``.

    Returns
    -------
    dict
        Updated ``current_sub_query_index`` and ``current_query``.
    """
    sub_queries: List[str] = state.get("sub_queries", [])
    current_idx: int = state.get("current_sub_query_index", 0)

    next_idx = current_idx + 1
    next_query = sub_queries[next_idx]

    logger.info(
        "[AdvanceSubQuery] Moving to sub-query {idx}/{total}: '{q}'",
        idx=next_idx + 1,
        total=len(sub_queries),
        q=next_query[:60],
    )

    return {
        "current_sub_query_index": next_idx,
        "current_query": next_query,
    }
