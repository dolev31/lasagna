"""Chunk refinement node for the structured RAG graph.

Preserves the exact chunk refinement behaviour from ``agentic_rag.py``
(lines 1085-1546) across all three refinement paths:

- **Vanilla** (``refinement_method == "vanilla"``): no LLM refinement; pass
  chunks through unchanged with deduplication and citation building.
- **Auto** (``refinement_method == "auto"``): heuristic gate via
  ``_should_auto_refine``; falls through to Refined when the gate fires,
  otherwise behaves identically to Vanilla.
- **Refined** (``refinement_method == "refined"``): parallel LLM-based chunk
  refinement with NOT_RELEVANT filtering, dedup, token-reduction metrics,
  and a fallback-to-top-chunk safety net.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Set

from loguru import logger

# -- Domain data structures ---------------------------------------------------
from lasagna.agentic.agentic_rag import (
    HopData,
    RetrievedChunkData,
    _generate_content_hash,
    _should_auto_refine,
    refine_chunk,
)

# -- Citation highlight extraction --------------------------------------------
from lasagna.agentic.citation_manager import extract_highlight_ranges

# -- LLM resolution ----------------------------------------------------------
from lasagna.providers import get_llm_provider
from lasagna.utils.model_utils import (
    get_default_model_settings,
    resolve_model_settings,
)

# -- Token estimation --------------------------------------------------------
from lasagna.utils.tokenization import estimate_tokens


# =============================================================================
# Internal helpers
# =============================================================================


def _build_citation(
    chunk: RetrievedChunkData,
    global_idx: int,
    hop_index: int,
    query: str,
    *,
    content_override: str | None = None,
) -> Dict[str, Any]:
    """Build a single citation dict matching the format produced by
    ``agentic_rag.py`` (vanilla & refined paths).

    Args:
        chunk: The chunk to cite.
        global_idx: Global citation index for ``[N]`` numbering.
        hop_index: Current hop number.
        query: The search query (used for highlight extraction).
        content_override: If supplied, use this text instead of
            ``chunk.content`` (used for refined content).

    Returns:
        Citation metadata dict.
    """
    content = content_override if content_override is not None else chunk.content
    highlight_ranges = extract_highlight_ranges(content, query) if content else None
    return {
        "index": global_idx,
        "document_id": chunk.document_id,
        "document_name": chunk.document_name,
        "chunk_id": chunk.chunk_id,
        "content_preview": content[:200],
        "full_content": content,
        "score": chunk.score,
        "chunk_index": chunk.chunk_index,
        "page_number": chunk.page_number,
        "hop_index": hop_index,
        "highlight_ranges": highlight_ranges,
        "line_start": chunk.metadata.get("line_start"),
        "line_end": chunk.metadata.get("line_end"),
        "auto_merged": chunk.metadata.get("auto_merged", False),
    }


def _format_chunk_line(
    chunk: RetrievedChunkData,
    global_idx: int,
    *,
    content_override: str | None = None,
) -> str:
    """Render a single ``[N] doc_name, page X\\ncontent`` line."""
    content = content_override if content_override is not None else chunk.content
    page_info = f", page {chunk.page_number}" if chunk.page_number else ""
    line_info = ""
    if not chunk.page_number and chunk.metadata.get("line_start"):
        ls = chunk.metadata["line_start"]
        le = chunk.metadata.get("line_end")
        line_info = f", lines {ls}-{le}" if le and le != ls else f", line {ls}"
    return f"[{global_idx}] {chunk.document_name}{page_info}{line_info}\n{content}\n"


def _vanilla_pass(
    retrieved_chunks: List[RetrievedChunkData],
    query: str,
    hop_index: int,
    citation_start_index: int,
    seen_content_hashes: Set[str],
    *,
    refinement_skipped: bool = False,
    refinement_skip_reason: str = "",
    search_latency_ms: float = 0.0,
    crag_grade: str = "",
    crag_grades_detail: List[str] | None = None,
    crag_web_fallback_used: bool = False,
    hyde_used: bool = False,
    hyde_reason: str = "",
) -> Dict[str, Any]:
    """Execute the vanilla (no-refinement) path.

    Shared by the ``vanilla`` method and the ``auto`` method when the
    auto-refine gate decides to skip.

    Returns:
        Partial state update dict.
    """
    formatted_results: List[str] = []
    hop_citations: List[Dict[str, Any]] = []
    current_idx = citation_start_index
    duplicates_skipped = 0

    for chunk in retrieved_chunks:
        content_hash = _generate_content_hash(chunk.content)
        if content_hash in seen_content_hashes:
            duplicates_skipped += 1
            continue
        seen_content_hashes.add(content_hash)

        global_idx = current_idx
        current_idx += 1

        chunk.refinement_kept = True
        formatted_results.append(_format_chunk_line(chunk, global_idx))
        hop_citations.append(_build_citation(chunk, global_idx, hop_index, query))

    if duplicates_skipped > 0:
        logger.info(
            f"[Agentic RAG] Hop {hop_index}: skipped {duplicates_skipped} duplicate chunk(s)"
        )

    hop_data = HopData(
        hop_index=hop_index,
        query_used=query,
        retrieved_chunks=retrieved_chunks,
        search_latency_ms=search_latency_ms,
        refinement_skipped=refinement_skipped,
        refinement_skip_reason=refinement_skip_reason,
        hyde_used=hyde_used,
        hyde_reason=hyde_reason,
        crag_grade=crag_grade,
        crag_grades_detail=crag_grades_detail or [],
        crag_web_fallback_used=crag_web_fallback_used,
    )

    return {
        "refined_chunks": retrieved_chunks,
        "global_citation_index": current_idx,
        "all_hop_citations": hop_citations,  # caller extends
        "seen_content_hashes": seen_content_hashes,
        "hop_data_entry": hop_data,
        "refinement_metrics": {
            "hop_index": hop_index,
            "tokens_before": 0,
            "tokens_after": 0,
            "chunks_filtered": 0,
            "reduction_pct": 0.0,
            "refinement_failures": 0,
            "refinement_skipped": refinement_skipped,
            "refinement_skip_reason": refinement_skip_reason,
        },
        "formatted_context": "\n---\n".join(formatted_results),
    }


# =============================================================================
# Node entry-point
# =============================================================================


async def refine_chunks_node(state: dict) -> dict:
    """Refine retrieved chunks according to the configured refinement method.

    This is the most critical node in the RAG graph: it faithfully reproduces
    the three refinement paths (vanilla / auto / refined) that were previously
    embedded inside ``search_knowledge_base`` in ``agentic_rag.py``.

    Reads:
        retrieved_chunks, refinement_method, current_query, session_config,
        hop_count, global_citation_index, all_hop_citations,
        seen_content_hashes, hop_data, rag_metrics_list, timing_spans

    Writes:
        refined_chunks, global_citation_index, all_hop_citations,
        seen_content_hashes, hop_data, refinement_metrics,
        rag_metrics_list, timing_spans
    """
    node_start = time.time()

    # -- Read from state ------------------------------------------------------
    retrieved_chunks: List[RetrievedChunkData] = state.get("retrieved_chunks", [])
    refinement_method: str = state.get("refinement_method", "auto")
    query: str = state.get("current_query", state.get("original_question", ""))
    session_config = state.get("session_config")
    hop_index: int = state.get("hop_count", 0)

    citation_start_index: int = state.get("global_citation_index", 1)
    all_hop_citations: List[Dict[str, Any]] = list(state.get("all_hop_citations", []))
    seen_content_hashes: Set[str] = set(state.get("seen_content_hashes", set()))
    hop_data_list: List[Any] = list(state.get("hop_data", []))
    rag_metrics_list: List[Dict[str, Any]] = list(state.get("rag_metrics_list", []))
    timing_spans: Dict[str, float] = dict(state.get("timing_spans", {}))

    # Upstream timing context (set by retrieve_node)
    search_latency_ms: float = timing_spans.get("last_search_ms", 0.0)
    hyde_used: bool = bool(timing_spans.get("last_hyde_used", False))
    hyde_reason: str = str(timing_spans.get("last_hyde_reason", ""))
    # Note: CRAG fields are NOT read here because grading runs AFTER
    # refinement in the graph (retrieve → refine_chunks → grade_documents).
    # HopData.crag_* fields will be backfilled by grade_documents_node.
    crag_grade: str = ""
    crag_grades_detail: List[str] = []
    crag_web_fallback_used: bool = False

    # -- Guard: nothing to refine ---------------------------------------------
    if not retrieved_chunks:
        logger.debug("[refine_chunks_node] No retrieved chunks to refine.")
        return {
            "refined_chunks": [],
            "refinement_metrics": {},
        }

    # =========================================================================
    # Path A: Vanilla — no refinement
    # =========================================================================
    if refinement_method == "vanilla":
        result = _vanilla_pass(
            retrieved_chunks,
            query,
            hop_index,
            citation_start_index,
            seen_content_hashes,
            search_latency_ms=search_latency_ms,
            crag_grade=crag_grade,
            crag_grades_detail=crag_grades_detail,
            crag_web_fallback_used=crag_web_fallback_used,
            hyde_used=hyde_used,
            hyde_reason=hyde_reason,
        )

        all_hop_citations.extend(result["all_hop_citations"])
        hop_data_list.append(result["hop_data_entry"])

        elapsed_ms = (time.time() - node_start) * 1000
        timing_spans["refinement_ms"] = elapsed_ms

        return {
            "refined_chunks": result["refined_chunks"],
            "global_citation_index": result["global_citation_index"],
            "all_hop_citations": all_hop_citations,
            "seen_content_hashes": result["seen_content_hashes"],
            "hop_data": hop_data_list,
            "refinement_metrics": result["refinement_metrics"],
            "rag_metrics_list": rag_metrics_list,
            "timing_spans": timing_spans,
        }

    # =========================================================================
    # Path B: Auto — decide whether to refine
    # =========================================================================
    should_refine = refinement_method == "refined"
    skip_reason = ""

    if refinement_method == "auto":
        should_refine, skip_reason = _should_auto_refine(
            chunks=retrieved_chunks,
            length_threshold=getattr(session_config, "auto_refine_avg_length_threshold", 1000),
            chunk_count_threshold=getattr(session_config, "auto_refine_chunk_count_threshold", 7),
            score_variance_threshold=0.3,
        )

    if not should_refine:
        # Auto decided to skip refinement — behave exactly as vanilla
        logger.info(f"[Agentic RAG] Hop {hop_index}: refinement skipped ({skip_reason})")
        result = _vanilla_pass(
            retrieved_chunks,
            query,
            hop_index,
            citation_start_index,
            seen_content_hashes,
            refinement_skipped=True,
            refinement_skip_reason=skip_reason,
            search_latency_ms=search_latency_ms,
            crag_grade=crag_grade,
            crag_grades_detail=crag_grades_detail,
            crag_web_fallback_used=crag_web_fallback_used,
            hyde_used=hyde_used,
            hyde_reason=hyde_reason,
        )

        all_hop_citations.extend(result["all_hop_citations"])
        hop_data_list.append(result["hop_data_entry"])

        elapsed_ms = (time.time() - node_start) * 1000
        timing_spans["refinement_ms"] = elapsed_ms

        return {
            "refined_chunks": result["refined_chunks"],
            "global_citation_index": result["global_citation_index"],
            "all_hop_citations": all_hop_citations,
            "seen_content_hashes": result["seen_content_hashes"],
            "hop_data": hop_data_list,
            "refinement_metrics": result["refinement_metrics"],
            "rag_metrics_list": rag_metrics_list,
            "timing_spans": timing_spans,
        }

    # =========================================================================
    # Path C: Refined — LLM-based chunk refinement
    # =========================================================================
    refinement_start = time.time()

    # Resolve the refinement LLM
    default_model = get_default_model_settings()
    max_tokens = default_model.get("max_tokens")
    model_settings = resolve_model_settings(
        model_ref=getattr(session_config, "refinement_model", None),
        default_settings=default_model,
        max_tokens=max_tokens,
        context="refinement_model",
    )
    refinement_llm = get_llm_provider().get_chat_model(model_settings)

    # Token count before refinement
    tokens_before = sum(estimate_tokens(c.content) for c in retrieved_chunks)

    # Prepare parallel refinement tasks
    refinement_tasks = [
        refine_chunk(
            llm=refinement_llm,
            query=query,
            chunk_content=chunk.content,
            source_name=chunk.document_name,
            chunk_index=i,
        )
        for i, chunk in enumerate(retrieved_chunks)
    ]

    # Execute all refinements concurrently
    refinement_results = await asyncio.gather(*refinement_tasks)
    refinement_latency_ms = (time.time() - refinement_start) * 1000

    # -- Process results ------------------------------------------------------
    formatted_results: List[str] = []
    hop_citations: List[Dict[str, Any]] = []
    chunks_filtered = 0
    tokens_after = 0
    refinement_failures = 0
    current_global_idx = citation_start_index
    duplicates_skipped = 0

    for chunk_idx, refined_content, original_content in refinement_results:
        chunk = retrieved_chunks[chunk_idx]

        if refined_content is None:
            # Chunk was marked as NOT_RELEVANT by the LLM
            chunks_filtered += 1
            chunk.refinement_kept = False
            continue

        # Deduplicate by content hash
        content_hash = _generate_content_hash(refined_content)
        if content_hash in seen_content_hashes:
            duplicates_skipped += 1
            chunk.refinement_kept = False
            continue
        seen_content_hashes.add(content_hash)

        # Track if refinement preserved content unchanged
        if refined_content == original_content:
            refinement_failures += 1

        # Store refined content while keeping original for citations
        chunk.original_content = chunk.content
        chunk.content = refined_content
        chunk.is_refined = True

        tokens_after += estimate_tokens(refined_content)

        formatted_results.append(
            _format_chunk_line(chunk, current_global_idx, content_override=refined_content)
        )
        hop_citations.append(
            _build_citation(
                chunk,
                current_global_idx,
                hop_index,
                query,
                content_override=refined_content,
            )
        )
        current_global_idx += 1

    if duplicates_skipped > 0:
        logger.info(
            f"[Agentic RAG] Hop {hop_index}: skipped {duplicates_skipped} duplicate chunk(s)"
        )

    # -- Fallback: if ALL chunks were filtered, use the top original chunk ----
    if not formatted_results and retrieved_chunks:
        logger.warning("All chunks filtered as NOT_RELEVANT. Using top original chunk as fallback.")
        top_chunk = retrieved_chunks[0]
        fallback_hash = _generate_content_hash(top_chunk.content)
        if fallback_hash not in seen_content_hashes:
            seen_content_hashes.add(fallback_hash)
            top_chunk.refinement_kept = True
            formatted_results.append(_format_chunk_line(top_chunk, citation_start_index))
            tokens_after = estimate_tokens(top_chunk.content)
            chunks_filtered = len(retrieved_chunks) - 1
            current_global_idx = citation_start_index + 1

            hop_citations.append(_build_citation(top_chunk, citation_start_index, hop_index, query))
        else:
            logger.info(
                f"[Agentic RAG] Hop {hop_index}: fallback chunk is also a duplicate, skipping"
            )

    # -- Accumulate citations -------------------------------------------------
    all_hop_citations.extend(hop_citations)

    # -- Log token reduction --------------------------------------------------
    reduction_pct = (
        ((tokens_before - tokens_after) / tokens_before * 100) if tokens_before > 0 else 0
    )
    logger.info(
        f"[Agentic RAG] Hop {hop_index} refinement: {len(retrieved_chunks)} chunks, "
        f"{tokens_before} -> {tokens_after} tokens ({reduction_pct:.1f}% reduction), "
        f"{chunks_filtered} chunk(s) filtered, citations {citation_start_index}-{current_global_idx - 1}"
    )

    # -- Record HopData -------------------------------------------------------
    hop_data = HopData(
        hop_index=hop_index,
        query_used=query,
        retrieved_chunks=retrieved_chunks,
        search_latency_ms=search_latency_ms,
        refinement_latency_ms=refinement_latency_ms,
        tokens_before_refinement=tokens_before,
        tokens_after_refinement=tokens_after,
        chunks_filtered=chunks_filtered,
        hyde_used=hyde_used,
        hyde_reason=hyde_reason,
        crag_grade=crag_grade,
        crag_grades_detail=crag_grades_detail or [],
        crag_web_fallback_used=crag_web_fallback_used,
    )
    hop_data_list.append(hop_data)

    # -- Per-hop metrics entry ------------------------------------------------
    metrics_entry = {
        "hop_index": hop_index,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "chunks_filtered": chunks_filtered,
        "reduction_pct": reduction_pct,
        "refinement_failures": refinement_failures,
    }
    rag_metrics_list.append(metrics_entry)

    # -- Timing ---------------------------------------------------------------
    elapsed_ms = (time.time() - node_start) * 1000
    timing_spans["refinement_ms"] = elapsed_ms

    # -- Collect surviving chunks for downstream nodes ------------------------
    refined_chunks_out = [c for c in retrieved_chunks if c.refinement_kept]

    return {
        "refined_chunks": refined_chunks_out,
        "global_citation_index": current_global_idx,
        "all_hop_citations": all_hop_citations,
        "seen_content_hashes": seen_content_hashes,
        "hop_data": hop_data_list,
        "refinement_metrics": metrics_entry,
        "rag_metrics_list": rag_metrics_list,
        "timing_spans": timing_spans,
    }
