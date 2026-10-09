"""Web Fallback Node — CRAG web search fallback.

When Corrective RAG (CRAG) grading determines that the retrieved documents
are of poor quality (``overall_grade == "incorrect"``), this node performs
a web search via DuckDuckGo to supplement the evidence pool.

Behaviour:
  - If web results are found, ``graded_chunks`` is replaced with the web
    results, ``crag_web_fallback_used`` is set to ``True``, and
    ``all_hop_citations`` is rebuilt to reflect the web results (ensuring
    the citations shown to the user match the evidence the answer was
    generated from).
  - If no web results are returned, the original ``graded_chunks`` are
    kept unchanged (same as the monolithic implementation at line 1156).

Owned state fields (write):
    ``graded_chunks``, ``crag_web_fallback_used``, ``all_hop_citations``,
    ``global_citation_index``, ``seen_content_hashes``
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from loguru import logger

from lasagna.agentic.graph_state import RAGGraphState
from lasagna.agentic.agentic_rag import (
    _crag_web_search_fallback,
    _generate_content_hash,
)
from lasagna.agentic.citation_manager import extract_highlight_ranges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_web_citations(
    web_chunks: List[Any],
    query: str,
    start_index: int,
    seen_hashes: Set[str],
    hop_index: int = 0,
) -> tuple[List[Dict[str, Any]], int, Set[str]]:
    """Build citation entries for web fallback chunks.

    Returns (citations, next_global_index, updated_seen_hashes).
    """
    citations: List[Dict[str, Any]] = []
    current_idx = start_index
    updated_hashes = set(seen_hashes)

    for chunk_i, chunk in enumerate(web_chunks):
        content = getattr(chunk, "content", str(chunk))
        content_hash = _generate_content_hash(content) if content else ""
        if content_hash and content_hash in updated_hashes:
            continue
        if content_hash:
            updated_hashes.add(content_hash)

        highlight_ranges = extract_highlight_ranges(content, query) if content else None

        cit: Dict[str, Any] = {
            "index": current_idx,
            "document_id": getattr(chunk, "document_id", "web"),
            "document_name": getattr(chunk, "document_name", "web_result"),
            "chunk_id": getattr(chunk, "chunk_id", f"web_{current_idx}"),
            "content_preview": content[:200] if content else "",
            "full_content": content or "",
            "score": getattr(chunk, "score", 0.0),
            "chunk_index": chunk_i,
            "page_number": None,
            "hop_index": hop_index,
            "highlight_ranges": highlight_ranges,
            "auto_merged": False,
        }
        citations.append(cit)
        current_idx += 1

    return citations, current_idx, updated_hashes


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def web_fallback_node(state: RAGGraphState) -> Dict[str, Any]:
    """Perform a CRAG web search fallback when local retrieval quality is poor.

    Parameters
    ----------
    state:
        Current ``RAGGraphState``.  Reads ``current_query``,
        ``graded_chunks``, ``all_hop_citations``, ``global_citation_index``,
        and ``seen_content_hashes``.

    Returns
    -------
    dict
        Updated fields: ``graded_chunks``, ``crag_web_fallback_used``,
        ``all_hop_citations``, ``global_citation_index``,
        ``seen_content_hashes``.
    """
    query: str = state.get("current_query", state.get("original_question", ""))
    graded_chunks: List[Any] = state.get("graded_chunks", [])

    logger.info("[WebFallback] Initiating CRAG web search fallback for query")

    try:
        web_results = await _crag_web_search_fallback(query, max_results=5)
    except Exception:
        logger.exception("[WebFallback] Web search fallback raised an unexpected error")
        web_results = []

    if web_results:
        # The current hop was graded "incorrect" — strip its citations so
        # they don't leak into the final output.  Prior hops' citations
        # are preserved (they were graded correct/ambiguous on their own).
        existing_citations: List[Dict[str, Any]] = list(state.get("all_hop_citations", []))
        existing_hashes: Set[str] = set(state.get("seen_content_hashes", set()))
        hop_index: int = state.get("hop_count", 0)

        # Remove citations from the current (incorrect) hop
        removed_citations = [c for c in existing_citations if c.get("hop_index") == hop_index]
        prior_citations = [c for c in existing_citations if c.get("hop_index") != hop_index]
        removed_count = len(removed_citations)

        # Remove content hashes ONLY for chunks that had current-hop citations.
        # We key by chunk_id from the stripped citations rather than hashing
        # all graded_chunks, because graded_chunks may include dedup-skipped
        # chunks whose content is shared with valid prior hops.
        stripped_chunk_ids: Set[str] = {c.get("chunk_id", "") for c in removed_citations}
        stripped_chunk_ids.discard("")
        incorrect_hashes: Set[str] = set()
        for chunk in graded_chunks:
            chunk_id = getattr(chunk, "chunk_id", None)
            if chunk_id and chunk_id in stripped_chunk_ids:
                content = getattr(chunk, "content", "")
                if content:
                    incorrect_hashes.add(_generate_content_hash(content))
        cleaned_hashes = existing_hashes - incorrect_hashes

        # Recompute global_citation_index from prior citations
        if prior_citations:
            global_idx = max(c["index"] for c in prior_citations) + 1
        else:
            global_idx = 1

        logger.info(
            "[WebFallback] Web search returned {n} result(s); stripped {removed} "
            "incorrect-hop citation(s) and {hash_removed} stale hash(es), "
            "appending to {prior} prior citation(s)",
            n=len(web_results),
            removed=removed_count,
            hash_removed=len(incorrect_hashes),
            prior=len(prior_citations),
        )

        web_citations, next_idx, updated_hashes = _build_web_citations(
            web_results,
            query,
            start_index=global_idx,
            seen_hashes=cleaned_hashes,
            hop_index=hop_index,
        )

        merged_citations = prior_citations + web_citations

        return {
            "graded_chunks": web_results,
            "crag_web_fallback_used": True,
            "all_hop_citations": merged_citations,
            "global_citation_index": next_idx,
            "seen_content_hashes": updated_hashes,
        }

    # No web results — keep the original chunks as a last resort.
    logger.warning(
        "[WebFallback] Web search returned no results; keeping original {n} graded chunk(s)",
        n=len(graded_chunks),
    )
    return {
        "graded_chunks": graded_chunks,
        "crag_web_fallback_used": False,
    }
