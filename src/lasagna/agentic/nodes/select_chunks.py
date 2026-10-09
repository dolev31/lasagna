"""Select Chunks Node — citation filtering across ALL strategies.

This node applies the configured ``chunk_cite_retrieval`` strategy to
decide which accumulated hop citations make it into the final answer.
Supported strategies:

  - **agent_decides** — ask the LLM which chunks support the generation.
  - **fixed** — take the first ``fixed_size`` citations.
  - **above_threshold** — filter by ``threshold_score``, capped at ``max_chunks``.
  - **adaptive** — score-gap analysis via ``adaptive_k_selection()``.
  - *(default fallback)* — first 5 citations.

Owned state fields (write):
    ``filtered_citations``, ``marked_chunks``
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from loguru import logger

from lasagna.agentic.graph_state import RAGGraphState
from ..prompts import resolve_prompt


# ---------------------------------------------------------------------------
# LLM prompt for the ``agent_decides`` strategy
# ---------------------------------------------------------------------------

_AGENT_DECIDES_PROMPT = """\
You are a citation auditor.  Given an answer and a numbered list of source
chunks, determine which chunks actually support the claims made in the answer.

## Answer
{generation}

## Source Chunks
{chunks_text}

## Instructions
List ONLY the chunk numbers (e.g. 1, 3, 7) that directly support claims in the
answer.  Return the numbers as a comma-separated list and nothing else.

Relevant chunk numbers:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_citations_for_llm(citations: List[Dict[str, Any]]) -> str:
    """Render citations as numbered text blocks for the LLM prompt."""
    parts: List[str] = []
    for c in citations:
        idx = c.get("index", "?")
        preview = c.get("content_preview", "")
        parts.append(f"[{idx}]\n{preview}")
    return "\n\n".join(parts)


def _parse_chunk_numbers(text: str) -> List[int]:
    """Parse comma-separated integers from the LLM response."""
    return [int(n) for n in re.findall(r"\d+", text)]


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def select_chunks_node(state: RAGGraphState) -> Dict[str, Any]:
    """Filter accumulated hop citations according to the configured strategy.

    Parameters
    ----------
    state:
        Current ``RAGGraphState``.

    Returns
    -------
    dict
        Updated fields: ``filtered_citations``, ``marked_chunks``.
    """
    session_config = state.get("session_config")
    all_hop_citations: List[Dict[str, Any]] = state.get("all_hop_citations", [])

    if not all_hop_citations:
        logger.info("[SelectChunks] No hop citations to filter")
        return {"filtered_citations": [], "marked_chunks": []}

    # Resolve strategy from session config
    chunk_cite_retrieval: Dict[str, Any] = {}
    if session_config is not None:
        chunk_cite_retrieval = getattr(session_config, "chunk_cite_retrieval", {}) or {}
    strategy: str = chunk_cite_retrieval.get("strategy", "fixed")

    marked_chunks: List[int] = list(state.get("marked_chunks", []))
    filtered: List[Dict[str, Any]]

    # ------------------------------------------------------------------
    # Strategy: agent_decides
    # ------------------------------------------------------------------
    if strategy == "agent_decides":
        generation = state.get("generation", "")
        llm = state.get("llm")

        if llm and generation:
            chunks_text = _format_citations_for_llm(all_hop_citations)
            prompt = resolve_prompt(
                "select_chunks", _AGENT_DECIDES_PROMPT, state.get("session_config")
            ).format(
                generation=generation,
                chunks_text=chunks_text,
            )
            try:
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                selected_numbers = _parse_chunk_numbers(response.content)
                marked_chunks = selected_numbers
                logger.info(
                    "[SelectChunks] agent_decides selected {n} chunk(s): {nums}",
                    n=len(selected_numbers),
                    nums=selected_numbers,
                )
            except Exception:
                logger.exception(
                    "[SelectChunks] agent_decides LLM call failed; falling back to all citations"
                )

        if marked_chunks:
            marked_set = set(marked_chunks)
            filtered = [c for c in all_hop_citations if c.get("index") in marked_set]
        else:
            # Fallback when no chunks were marked
            max_chunks = chunk_cite_retrieval.get("max_chunks", 5)
            filtered = all_hop_citations[:max_chunks]
            logger.warning(
                "[SelectChunks] agent_decides produced no marks; using first {n} citations",
                n=len(filtered),
            )

    # ------------------------------------------------------------------
    # Strategy: fixed
    # ------------------------------------------------------------------
    elif strategy == "fixed":
        fixed_size: int = chunk_cite_retrieval.get("fixed_size", 5)
        filtered = all_hop_citations[:fixed_size]
        logger.info(
            "[SelectChunks] fixed strategy: selected first {n} citation(s)",
            n=len(filtered),
        )

    # ------------------------------------------------------------------
    # Strategy: above_threshold
    # ------------------------------------------------------------------
    elif strategy == "above_threshold":
        threshold: float = chunk_cite_retrieval.get("threshold_score", 0.7)
        max_chunks: int = chunk_cite_retrieval.get("max_chunks", 20)
        above = [c for c in all_hop_citations if c.get("score", 0) >= threshold]
        filtered = above[:max_chunks]
        logger.info(
            "[SelectChunks] above_threshold (>={thr}): {above}/{total} -> capped at {n}",
            thr=threshold,
            above=len(above),
            total=len(all_hop_citations),
            n=len(filtered),
        )

    # ------------------------------------------------------------------
    # Strategy: adaptive (score-gap analysis)
    # ------------------------------------------------------------------
    elif strategy == "adaptive":
        from lasagna.utils.normalization import adaptive_k_selection

        scores = [c.get("score", 0) for c in all_hop_citations]
        optimal_k = adaptive_k_selection(
            scores=scores,
            min_k=chunk_cite_retrieval.get("min_chunks", 2),
            max_k=chunk_cite_retrieval.get("max_chunks", 20),
            buffer=chunk_cite_retrieval.get("score_gap_buffer", 2),
            min_gap_ratio=chunk_cite_retrieval.get("min_gap_ratio", 0.15),
        )
        filtered = all_hop_citations[:optimal_k]
        logger.info(
            "[SelectChunks] adaptive strategy: {total} candidates -> {k} selected "
            "(scores: {scores})",
            total=len(all_hop_citations),
            k=optimal_k,
            scores=[round(s, 3) for s in scores[: optimal_k + 2]],
        )

    # ------------------------------------------------------------------
    # Default fallback
    # ------------------------------------------------------------------
    else:
        filtered = all_hop_citations[:5]
        logger.info(
            "[SelectChunks] Unknown strategy '{strat}'; using first 5 citations",
            strat=strategy,
        )

    return {
        "filtered_citations": filtered,
        "marked_chunks": marked_chunks,
    }
