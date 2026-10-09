"""Format Output Node — assembles the final output from graph state.

This is the terminal node of the RAG graph.  It reads the filtered
citations produced by ``select_chunks_node``, looks up full chunk content
from ``hop_data``, constructs the ``<knowledge_base>`` context block,
builds the final citation / chunk lists for the API response, and computes
``AgenticRAGMetrics``.

Owned state fields (write):
    ``formatted_context``, ``final_chunks``, ``final_citations``,
    ``final_metrics``, ``stop_reason``
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Set, Tuple

from loguru import logger

from lasagna.agentic.graph_state import RAGGraphState
from lasagna.agentic.agentic_rag import (
    AgenticRAGMetrics,
    _generate_content_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reindex_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Renumber citation indices to be contiguous ``[1, 2, 3, ...]``.

    The incoming citations keep their relative order; only the ``"index"``
    field is rewritten so that there are no gaps.

    Returns a *new* list (shallow copies of each dict) — the originals are
    not mutated.
    """
    reindexed: List[Dict[str, Any]] = []
    for new_idx, cit in enumerate(citations, start=1):
        updated = dict(cit)
        updated["index"] = new_idx
        reindexed.append(updated)
    return reindexed


def _merge_final_search_citations(
    hop_citations: List[Dict[str, Any]],
    final_search_citations: List[Dict[str, Any]],
    seen_content_hashes: Set[str],
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Merge final-search results into the accumulated hop citations.

    Parameters
    ----------
    hop_citations:
        Citations accumulated across retrieval hops (already indexed).
    final_search_citations:
        Citations produced by a final comprehensive search pass.
    seen_content_hashes:
        Content hashes already seen (from prior hops).  Updated in-place
        **and** returned so callers can keep the reference.

    Returns
    -------
    (merged_citations, updated_seen_hashes)
        ``merged_citations`` preserves all *hop_citations* first, followed
        by any **non-duplicate** final-search citations.  All indices are
        renumbered for contiguity via ``_reindex_citations``.
    """
    # Work on a copy of the set so we don't mutate the caller's original
    # unexpectedly if they hold a separate reference.
    updated_hashes: Set[str] = set(seen_content_hashes)

    # Determine the starting index for final-search citations.
    if hop_citations:
        merge_start_index = max(c["index"] for c in hop_citations) + 1
    else:
        merge_start_index = 1

    merged: List[Dict[str, Any]] = [dict(c) for c in hop_citations]
    current_idx = merge_start_index

    for cit in final_search_citations:
        # Use full_content for dedup hashing to match seen_content_hashes
        # which are built from full chunk content in refine_chunks_node.
        content = cit.get("full_content") or cit.get("content_preview", "")
        content_hash = _generate_content_hash(content) if content else ""

        if content_hash and content_hash in updated_hashes:
            # Duplicate — skip this citation
            continue

        if content_hash:
            updated_hashes.add(content_hash)

        new_cit = dict(cit)
        new_cit["index"] = current_idx
        current_idx += 1
        merged.append(new_cit)

    # Re-index for contiguity: [1, 2, 3, ...]
    merged = _reindex_citations(merged)

    return merged, updated_hashes


def _build_chunk_content_map(hop_data: List[Any]) -> Dict[str, str]:
    """Build a mapping ``{chunk_id: full_content}`` from all hops.

    Merged (parent) chunks take priority over individual children to avoid
    duplicating content.
    """
    content_map: Dict[str, str] = {}
    merged_ids: set = set()

    for hop in hop_data:
        for chunk in hop.retrieved_chunks:
            if not chunk.refinement_kept:
                continue
            cid = chunk.chunk_id
            if chunk.metadata.get("auto_merged"):
                content_map[cid] = chunk.content
                merged_ids.add(cid)
            elif cid not in merged_ids:
                content_map[cid] = chunk.content

    return content_map


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def format_output_node(state: RAGGraphState) -> Dict[str, Any]:
    """Assemble the final output payload from the accumulated graph state.

    Parameters
    ----------
    state:
        Current ``RAGGraphState``.

    Returns
    -------
    dict
        Updated fields: ``formatted_context``, ``final_chunks``,
        ``final_citations``, ``final_metrics``, ``stop_reason``.
    """
    filtered_citations: List[Dict[str, Any]] = state.get("filtered_citations", [])
    hop_data: List[Any] = state.get("hop_data", [])
    session_config = state.get("session_config")
    total_start_time: float = state.get("total_start_time", 0.0)
    rag_metrics_list: List[Dict[str, Any]] = state.get("rag_metrics_list", [])
    stop_reason: str = state.get("stop_reason", "")

    # -- 0. Final-search merge/replace logic --------------------------------
    use_final_search = getattr(session_config, "use_final_search", False) or getattr(
        session_config, "enable_final_search", False
    )
    final_search_mode = getattr(session_config, "final_search_mode", "merge")

    if use_final_search:
        final_search_results: List[Dict[str, Any]] = state.get("final_search_results", [])
        seen_content_hashes: Set[str] = set(state.get("seen_content_hashes", set()))

        if final_search_mode == "replace":
            # Legacy behaviour: discard all hop citations, keep only
            # final-search results.  The structured graph does not currently
            # invoke a final-search pass, so this path is a no-op placeholder
            # that preserves compatibility with the legacy ReAct code-path.
            # When final_search_results are provided they would fully replace
            # filtered_citations here.
            if final_search_results:
                filtered_citations = _reindex_citations(final_search_results)
                logger.info(
                    "[FormatOutput] final_search_mode='replace': using {n} "
                    "final-search citations (hop citations discarded)",
                    n=len(filtered_citations),
                )
        elif final_search_mode == "merge":
            # Merge: keep accumulated hop citations, append deduplicated
            # final-search results, then re-index for contiguity.
            if final_search_results:
                merged, seen_content_hashes = _merge_final_search_citations(
                    hop_citations=filtered_citations,
                    final_search_citations=final_search_results,
                    seen_content_hashes=seen_content_hashes,
                )
                logger.info(
                    "[FormatOutput] final_search_mode='merge': {hop} hop + "
                    "{fs} final-search -> {merged} merged citations",
                    hop=len(filtered_citations),
                    fs=len(final_search_results),
                    merged=len(merged),
                )
                filtered_citations = merged
        else:
            logger.warning(
                "[FormatOutput] Unknown final_search_mode='{mode}', "
                "falling back to default (no merge/replace)",
                mode=final_search_mode,
            )

    # -- 1. Build formatted_context in <knowledge_base> tags ----------------
    chunk_content_map = _build_chunk_content_map(hop_data)

    content_sections: List[str] = []
    final_citations: List[Dict[str, Any]] = []
    final_chunks: List[Dict[str, Any]] = []

    for citation in filtered_citations:
        chunk_id = citation.get("chunk_id", "")
        full_content = chunk_content_map.get(chunk_id, "")
        # For final-search / web-fallback citations that aren't in hop_data,
        # fall back to the full_content field stored on the citation dict.
        if not full_content:
            full_content = citation.get("full_content", "")
        content_preview = citation.get("content_preview", "")
        display_content = full_content if full_content else content_preview

        idx = citation.get("index", 0)
        content_sections.append(f"[{idx}]\n{display_content}")

        # -- 2. final_citations (preserving all metadata fields) ------------
        cit_entry: Dict[str, Any] = {
            "index": idx,
            "document_id": citation.get("document_id", ""),
            "document_name": citation.get("document_name", ""),
            "chunk_id": str(chunk_id),
            "content_preview": content_preview,
            "score": round(citation.get("score", 0.0), 3),
            "chunk_index": citation.get("chunk_index", 0),
            "page_number": citation.get("page_number"),
            "hop_index": citation.get("hop_index"),
        }
        if citation.get("highlight_ranges"):
            cit_entry["highlight_ranges"] = citation["highlight_ranges"]
        if citation.get("line_start"):
            cit_entry["line_start"] = citation["line_start"]
            cit_entry["line_end"] = citation.get("line_end")
        final_citations.append(cit_entry)

        # -- 3. final_chunks (full content for frontend) --------------------
        final_chunks.append(
            {
                "chunk_id": str(chunk_id),
                "content": display_content,
                "chunk_index": citation.get("chunk_index", 0),
            }
        )

    if content_sections:
        formatted_context = "<knowledge_base>\n{}\n</knowledge_base>".format(
            "\n\n".join(content_sections)
        )
    else:
        formatted_context = (
            "<knowledge_base>\n"
            "The agent searched the knowledge base but did not find specific relevant information.\n"
            "</knowledge_base>"
        )

    # -- 4. Build AgenticRAGMetrics -----------------------------------------
    hop_count = state.get("hop_count", len(hop_data))
    now = time.time()
    total_latency_ms = (now - total_start_time) * 1000 if total_start_time else 0.0

    hop_search_latencies = [h.search_latency_ms for h in hop_data]
    hop_refinement_latencies = [h.refinement_latency_ms for h in hop_data]

    total_tokens_before = sum(m.get("tokens_before", 0) for m in rag_metrics_list)
    total_tokens_after = sum(m.get("tokens_after", 0) for m in rag_metrics_list)
    total_chunks_filtered = sum(m.get("chunks_filtered", 0) for m in rag_metrics_list)
    total_refinement_failures = sum(m.get("refinement_failures", 0) for m in rag_metrics_list)
    reduction_pct = (
        ((total_tokens_before - total_tokens_after) / total_tokens_before * 100)
        if total_tokens_before > 0
        else 0.0
    )

    # Char metrics
    total_chars_before = 0
    total_chars_after = 0
    refinement_was_skipped = False
    refinement_skip_reasons: List[str] = []
    for hop in hop_data:
        if hop.refinement_skipped:
            refinement_was_skipped = True
            if hop.refinement_skip_reason:
                refinement_skip_reasons.append(f"hop{hop.hop_index}: {hop.refinement_skip_reason}")
        for chunk in hop.retrieved_chunks:
            if chunk.original_content:
                total_chars_before += len(chunk.original_content)
                if chunk.refinement_kept:
                    total_chars_after += len(chunk.content)
            else:
                total_chars_before += len(chunk.content)
                if chunk.refinement_kept:
                    total_chars_after += len(chunk.content)

    # Unique docs / pages
    all_doc_ids: set = set()
    all_page_numbers: set = set()
    for hop in hop_data:
        all_doc_ids.update(hop.unique_doc_ids)
        all_page_numbers.update(hop.unique_page_numbers)

    # Orchestration timing
    agent_orchestration_ms = total_latency_ms  # whole graph run
    total_hop_search_ms = sum(hop_search_latencies)
    total_hop_refinement_ms = sum(hop_refinement_latencies)
    agent_llm_calls_ms = max(
        0, agent_orchestration_ms - total_hop_search_ms - total_hop_refinement_ms
    )

    # Quality booster flags
    hybrid_search_enabled = getattr(session_config, "enable_hybrid_search", False)
    reranking_enabled = getattr(session_config, "enable_reranking", False)
    mmr_enabled = getattr(session_config, "enable_mmr", False)
    hyde_enabled = getattr(session_config, "enable_hyde", False)

    # CRAG
    crag_enabled = getattr(session_config, "enable_crag", False)
    crag_grades = [h.crag_grade for h in hop_data if h.crag_grade]
    crag_web_fallback_triggered = state.get("crag_web_fallback_used", False)
    crag_grading_latency_ms = state.get("crag_grading_latency_ms", 0.0)

    # -- 5. citation_missing_page_rate --------------------------------------
    citations_with_page = sum(1 for c in final_citations if c.get("page_number") is not None)
    citation_missing_page_rate = (
        1.0 - (citations_with_page / len(final_citations)) if final_citations else 0.0
    )

    # Derive stop_reason if not yet set
    if not stop_reason:
        stop_reason = "graph_complete"

    metrics = AgenticRAGMetrics(
        agentic_enabled=True,
        hop_count=hop_count,
        agent_iterations_used=hop_count,
        # Latency
        total_latency_ms=total_latency_ms,
        hop_search_latencies_ms=hop_search_latencies,
        hop_refinement_latencies_ms=hop_refinement_latencies,
        agent_orchestration_ms=agent_orchestration_ms,
        agent_llm_calls_ms=agent_llm_calls_ms,
        # Tokens / chars
        total_tokens_before=total_tokens_before,
        total_tokens_after=total_tokens_after,
        total_tokens_saved=total_tokens_before - total_tokens_after,
        reduction_percentage=reduction_pct,
        total_chars_before=total_chars_before,
        total_chars_after=total_chars_after,
        # Refinement
        total_chunks_filtered=total_chunks_filtered,
        refinement_failures_count=total_refinement_failures,
        refinement_skipped=refinement_was_skipped,
        refinement_skip_reason="; ".join(refinement_skip_reasons)
        if refinement_skip_reasons
        else "",
        # Coverage
        unique_docs_across_hops=len(all_doc_ids),
        unique_pages_across_hops=len(all_page_numbers),
        citation_count=len(final_citations),
        citation_missing_page_rate=citation_missing_page_rate,
        stop_reason=stop_reason,
        # Quality boosters
        hybrid_search_enabled=hybrid_search_enabled,
        reranking_enabled=reranking_enabled,
        mmr_enabled=mmr_enabled,
        hyde_enabled=hyde_enabled,
        # CRAG
        crag_enabled=crag_enabled,
        crag_grades=crag_grades,
        crag_web_fallback_triggered=crag_web_fallback_triggered,
        crag_grading_latency_ms=crag_grading_latency_ms,
    )

    logger.info(
        "[FormatOutput] Assembled output: {n_cites} citations, {n_chunks} chunks, "
        "stop_reason='{reason}', latency={ms:.1f} ms",
        n_cites=len(final_citations),
        n_chunks=len(final_chunks),
        reason=stop_reason,
        ms=total_latency_ms,
    )

    return {
        "formatted_context": formatted_context,
        "final_chunks": final_chunks,
        "final_citations": final_citations,
        "final_metrics": metrics,
        "stop_reason": stop_reason,
    }
