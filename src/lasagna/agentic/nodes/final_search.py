"""Final Search Node — optional comprehensive search pass before output.

When ``enable_final_search`` is enabled in session config, this node
performs one last broad search using the original question and converts
the results into citation-format dicts that ``format_output_node`` can
merge or replace with the accumulated hop citations.

Owned state fields (write):
    ``final_search_results``
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from lasagna.agentic.graph_state import RAGGraphState
from lasagna.agentic.agentic_rag import (
    _should_use_hyde_for_agentic,
    arm_query_overrides,
)
from lasagna.agentic.citation_manager import extract_highlight_ranges


async def final_search_node(state: RAGGraphState) -> Dict[str, Any]:
    """Run a final comprehensive search using the original question.

    Parameters
    ----------
    state:
        Current ``RAGGraphState``.

    Returns
    -------
    dict
        Updated ``final_search_results`` — a list of citation-format dicts
        ready for ``_merge_final_search_citations`` in ``format_output_node``.
    """
    session_config = state.get("session_config")
    use_final_search = (
        (
            getattr(session_config, "use_final_search", False)
            or getattr(session_config, "enable_final_search", False)
        )
        if session_config
        else False
    )

    if not use_final_search:
        return {"final_search_results": []}

    original_question: str = state.get("original_question", "")
    session_id: str = state.get("session_id", "default")
    document_filter: Optional[List[str]] = state.get("document_filter")
    top_k: int = state.get("top_k", 10)
    trace_id: Optional[str] = state.get("trace_id")

    logger.info("[FinalSearch] Running final comprehensive search pass")

    # Use the same adaptive HyDE decision as the legacy final search path
    hop_data = state.get("hop_data", [])
    use_hyde_final = False
    try:
        use_hyde_final, hyde_reason = _should_use_hyde_for_agentic(
            query=original_question,
            hop_index=len(hop_data),
            hop_data_list=hop_data,
            session_config=session_config,
            is_final=True,
        )
        logger.debug(
            "[FinalSearch] HyDE decision: use_hyde={hyde}, reason={reason}",
            hyde=use_hyde_final,
            reason=hyde_reason,
        )
    except Exception:
        logger.debug("[FinalSearch] HyDE decision failed; defaulting to False")

    start = time.time()
    try:
        from lasagna.api.rag_backend import RAGBackend

        backend = RAGBackend()
        results = backend.search(
            query=original_question,
            session_id=session_id,
            top_k=top_k,
            document_ids=document_filter,
            use_hyde=use_hyde_final,
            trace_id=trace_id,
            _from_routing=True,
            query_config_overrides=arm_query_overrides(session_config),
        )
    except Exception:
        logger.exception("[FinalSearch] Final search failed")
        return {"final_search_results": []}

    elapsed_ms = (time.time() - start) * 1000

    # Convert results to citation-format dicts
    citation_dicts: List[Dict[str, Any]] = []
    for result in results:
        page_num = None
        if result.chunk.metadata:
            page_num = result.chunk.metadata.get("page_number")
        if page_num is None and hasattr(result.chunk, "page_number"):
            page_num = result.chunk.page_number

        content = result.chunk.content
        highlight_ranges = extract_highlight_ranges(content, original_question) if content else None

        cit: Dict[str, Any] = {
            "index": 0,  # will be reindexed by merge
            "document_id": result.chunk.document_id,
            "document_name": result.document.filename,
            "chunk_id": result.chunk.chunk_id,
            "content_preview": content[:200] if content else "",
            "full_content": content or "",
            "score": result.score,
            "chunk_index": result.chunk.chunk_index,
            "page_number": page_num,
            "hop_index": -1,  # sentinel for "final search"
            "highlight_ranges": highlight_ranges,
            "line_start": (
                result.chunk.metadata.get("line_start") if result.chunk.metadata else None
            ),
            "line_end": (result.chunk.metadata.get("line_end") if result.chunk.metadata else None),
            "auto_merged": False,
        }
        citation_dicts.append(cit)

    logger.info(
        "[FinalSearch] Final search returned {n} result(s) in {ms:.1f} ms",
        n=len(citation_dicts),
        ms=elapsed_ms,
    )

    return {"final_search_results": citation_dicts}
