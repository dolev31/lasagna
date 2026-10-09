"""CRAG document grading node for the structured RAG graph.

Extracts the Corrective RAG (CRAG) grading logic from ``agentic_rag.py``
(lines 1112-1178).  The node grades each refined chunk for relevance and
determines whether a web-search fallback should be triggered.

When CRAG is disabled (``session_config.enable_crag is False``) or there
are no refined chunks, all chunks pass through with ``grade="correct"``.
"""

from __future__ import annotations

import time
from typing import List

from loguru import logger

# -- Domain helpers -----------------------------------------------------------
from lasagna.agentic.agentic_rag import (
    RetrievedChunkData,
    _grade_retrieval_quality,
)


async def grade_documents_node(state: dict) -> dict:
    """Grade refined chunks using Corrective RAG (CRAG).

    Reads:
        refined_chunks, current_query, session_config

    Writes:
        overall_grade, graded_chunks, document_grades,
        crag_grading_latency_ms, crag_web_fallback_used
    """
    refined_chunks: List[RetrievedChunkData] = state.get("refined_chunks", [])
    query: str = state.get("current_query", state.get("original_question", ""))
    session_config = state.get("session_config")

    # -- Fast path: CRAG disabled or nothing to grade -------------------------
    enable_crag = getattr(session_config, "enable_crag", False) if session_config else False

    if not enable_crag or not refined_chunks:
        num = len(refined_chunks)
        return {
            "overall_grade": "correct",
            "graded_chunks": refined_chunks,
            "document_grades": ["correct"] * num,
            "crag_grading_latency_ms": 0.0,
            "crag_web_fallback_used": False,
        }

    # -- Grade retrieval quality ----------------------------------------------
    crag_start = time.time()
    crag_web_fallback_used = False

    try:
        overall_grade, filtered_chunks, per_chunk_grades = await _grade_retrieval_quality(
            query=query,
            retrieved_chunks=refined_chunks,
            session_config=session_config,
        )
        crag_grading_latency_ms = (time.time() - crag_start) * 1000

        hop_index = state.get("hop_count", 0)
        logger.info(
            f"[CRAG] Hop {hop_index}: overall_grade={overall_grade}, "
            f"filtered={len(filtered_chunks)}/{len(refined_chunks)}, "
            f"latency={crag_grading_latency_ms:.1f}ms"
        )

        # -- Route on grade ---------------------------------------------------
        if overall_grade == "incorrect":
            web_fallback_enabled = getattr(session_config, "crag_web_search_fallback", False)
            if web_fallback_enabled:
                logger.info(f"[CRAG] Hop {hop_index}: Triggering web search fallback")
                crag_web_fallback_used = True
                # The actual web search is performed by web_fallback_node;
                # we just signal the need here.
            else:
                logger.info(
                    f"[CRAG] Hop {hop_index}: Retrieval quality is 'incorrect' "
                    "but web fallback is disabled. Keeping original chunks."
                )
            # When grade is "incorrect", graded_chunks stays as the original
            # set so downstream nodes still have content to work with.
            graded_chunks = refined_chunks

        elif overall_grade in ("ambiguous", "correct") and filtered_chunks:
            # Use only the correct/ambiguous chunks (drops stray incorrects)
            graded_chunks = filtered_chunks
        else:
            graded_chunks = refined_chunks

    except Exception as exc:
        crag_grading_latency_ms = (time.time() - crag_start) * 1000
        logger.warning(
            f"[CRAG] Grading failed: {exc}. Proceeding with ungraded results.",
            exc_info=True,
        )
        overall_grade = "correct"
        per_chunk_grades = ["correct"] * len(refined_chunks)
        graded_chunks = refined_chunks

    # -- Backfill CRAG info onto the latest hop_data entry --------------------
    hop_data_list: list = list(state.get("hop_data", []))
    if hop_data_list:
        latest_hop = hop_data_list[-1]
        latest_hop.crag_grade = overall_grade
        latest_hop.crag_grades_detail = per_chunk_grades
        latest_hop.crag_web_fallback_used = crag_web_fallback_used

    return {
        "overall_grade": overall_grade,
        "graded_chunks": graded_chunks,
        "document_grades": per_chunk_grades,
        "crag_grading_latency_ms": crag_grading_latency_ms,
        "crag_web_fallback_used": crag_web_fallback_used,
        "hop_data": hop_data_list,
    }
