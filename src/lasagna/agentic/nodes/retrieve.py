"""Retrieve Node — execute a single retrieval hop against the knowledge base.

This node extracts the core retrieval logic from the monolithic
``search_knowledge_base`` tool (``agentic_rag.py`` lines 924-1109) into a
self-contained graph node.  It performs retrieval **only** — no refinement,
no CRAG grading, and no citation assembly.

Responsibilities:
  1. Query deduplication (exact + semantic).
  2. HyDE decision (delegates to ``_should_use_hyde_for_agentic``).
  3. Backend search via ``RAGBackend().search()``.
  4. Coverage scoring for intelligent stopping.
  5. Construction of ``RetrievedChunkData`` list from raw results.
  6. Timing and metrics bookkeeping.

Owned state fields (write):
  ``retrieved_chunks``, ``hop_count``, ``previous_queries``,
  ``previous_query_embeddings``, ``coverage_ratio``,
  ``coverage_confidence``, ``uncovered_aspects``, ``suggested_followup``,
  ``timing_spans``, ``retrieval_attempts``, ``stop_reason``
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from loguru import logger

from lasagna.agentic.graph_state import RAGGraphState
from lasagna.agentic.agentic_rag import (
    RetrievedChunkData,
    _should_use_hyde_for_agentic,
    _find_semantically_similar_query,
    arm_query_overrides,
)
from lasagna.utils.tokenization import estimate_tokens
from lasagna.retrieval.enhancement.coverage import (
    get_coverage_scorer,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _embed_query_for_dedup(query: str) -> Optional[Any]:
    """Compute an embedding vector for *query* for semantic deduplication.

    Uses an **absolute** import path to avoid relative-import surprises
    when the node is invoked from different entry points.

    Returns ``None`` on any failure — callers must tolerate ``None``
    embeddings gracefully.
    """
    try:
        from lasagna.infrastructure.embedding.manager import (
            get_embedding_manager,
        )

        manager = get_embedding_manager()
        return manager.encode_query(query)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[Retrieve] Could not compute query embedding for dedup: {exc}",
            exc=exc,
        )
        return None


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------


async def retrieve_node(state: RAGGraphState) -> Dict[str, Any]:
    """Execute a single retrieval hop and return raw chunks.

    Parameters
    ----------
    state:
        Current ``RAGGraphState``.  Key read fields:

        * ``current_query`` — the active search query
        * ``session_id``, ``document_filter``, ``top_k``, ``trace_id``
        * ``previous_queries``, ``previous_query_embeddings``
        * ``hop_count``, ``hop_data``
        * ``session_config``
        * ``max_retrieval_attempts``, ``retrieval_attempts``
        * ``timing_spans``

    Returns
    -------
    dict
        Partial state update with retrieval results and bookkeeping.
    """
    # ── Unpack state ──────────────────────────────────────────────────
    query: str = state.get("current_query", "")
    session_id: str = state.get("session_id", "default")
    document_filter: Optional[List[str]] = state.get("document_filter")
    top_k: int = state.get("top_k", 10)
    trace_id: Optional[str] = state.get("trace_id")
    session_config = state.get("session_config")

    previous_queries: List[str] = list(state.get("previous_queries", []))
    previous_embeddings: List[Any] = list(state.get("previous_query_embeddings", []))

    hop_count: int = state.get("hop_count", 0)
    hop_data: List[Any] = list(state.get("hop_data", []))
    retrieval_attempts: int = state.get("retrieval_attempts", 0)

    timing_spans: Dict[str, float] = dict(state.get("timing_spans", {}))

    search_start = time.time()

    # ── 1. Exact-match deduplication ──────────────────────────────────
    if query in previous_queries:
        logger.info(
            "[Retrieve] Query stabilized (exact repeat), hash={h}",
            h=_safe_hash(query),
        )
        return {
            "stop_reason": "query_stabilized",
            "retrieved_chunks": state.get("retrieved_chunks", []),
        }

    # ── 2. Semantic deduplication ─────────────────────────────────────
    similar = _find_semantically_similar_query(
        query,
        previous_queries,
        previous_embeddings,
        similarity_threshold=0.92,
    )
    if similar:
        logger.info(
            "[Retrieve] Query stabilized (semantic match), hash={h}",
            h=_safe_hash(query),
        )
        return {
            "stop_reason": "query_stabilized",
            "retrieved_chunks": state.get("retrieved_chunks", []),
        }

    # Record query + embedding for future dedup checks
    previous_queries.append(query)
    query_embedding = _embed_query_for_dedup(query)
    previous_embeddings.append(query_embedding)

    # ── 3. HyDE decision ─────────────────────────────────────────────
    use_hyde = False
    hyde_reason = "not_evaluated"
    try:
        use_hyde, hyde_reason = _should_use_hyde_for_agentic(
            query=query,
            hop_index=hop_count,
            hop_data_list=hop_data,
            session_config=session_config,
        )
        logger.debug(
            "[Retrieve] HyDE decision: hop={hop}, use_hyde={use_hyde}, "
            "reason={reason}, query_tokens={tokens}",
            hop=hop_count,
            use_hyde=use_hyde,
            reason=hyde_reason,
            tokens=estimate_tokens(query),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[Retrieve] HyDE decision failed, defaulting to no-HyDE: {exc}",
            exc=exc,
        )

    # ── 4. Execute backend search ────────────────────────────────────
    try:
        from lasagna.api.rag_backend import RAGBackend

        backend = RAGBackend()
        results = backend.search(
            query=query,
            session_id=session_id,
            top_k=top_k,
            document_ids=document_filter,
            use_hyde=use_hyde,
            trace_id=trace_id,
            _from_routing=True,  # Prevent COMPREHENSIVE recursion
            query_config_overrides=arm_query_overrides(session_config),
        )
    except Exception as exc:  # noqa: BLE001
        search_latency_ms = (time.time() - search_start) * 1000
        logger.error(
            "[Retrieve] Backend search failed after {ms:.1f}ms: {exc}",
            ms=search_latency_ms,
            exc=exc,
        )
        timing_spans["search_ms"] = timing_spans.get("search_ms", 0) + search_latency_ms
        return {
            "retrieved_chunks": [],
            "hop_count": hop_count + 1,
            "previous_queries": previous_queries,
            "previous_query_embeddings": previous_embeddings,
            "retrieval_attempts": retrieval_attempts + 1,
            "timing_spans": timing_spans,
        }

    search_latency_ms = (time.time() - search_start) * 1000

    # ── 5. Coverage scoring ──────────────────────────────────────────
    coverage_ratio = 0.0
    coverage_confidence = 0.0
    uncovered_aspects: List[str] = []
    suggested_followup: Optional[str] = None

    coverage_mode = (
        getattr(session_config, "coverage_scorer_mode", "keyword") if session_config else "keyword"
    )
    try:
        chunks_for_scoring = [
            {"content": r.chunk.content, "chunk_id": r.chunk.chunk_id} for r in results
        ]

        if coverage_mode == "llm":
            from lasagna.retrieval.enhancement.llm_coverage import (
                LLMCoverageScorer,
            )

            llm = state.get("llm")
            if llm is not None:
                llm_scorer = LLMCoverageScorer()
                coverage_score = await llm_scorer.score_coverage(query, chunks_for_scoring, llm)
            else:
                logger.warning(
                    "[Retrieve] LLM coverage requested but no LLM in state; "
                    "falling back to keyword scorer"
                )
                scorer = get_coverage_scorer()
                coverage_score = scorer.score_coverage(query, chunks_for_scoring)
        else:
            scorer = get_coverage_scorer()
            coverage_score = scorer.score_coverage(query, chunks_for_scoring)

        coverage_ratio = coverage_score.coverage_ratio
        coverage_confidence = coverage_score.confidence
        uncovered_aspects = coverage_score.uncovered_aspects
        suggested_followup = coverage_score.suggested_followup

        logger.debug(
            "[Retrieve] Coverage ({mode}): ratio={ratio:.2f}, confidence={conf:.2f}, "
            "uncovered={n_uncov}",
            mode=coverage_mode,
            ratio=coverage_ratio,
            conf=coverage_confidence,
            n_uncov=len(uncovered_aspects),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[Retrieve] Coverage scoring failed (non-critical): {exc}",
            exc=exc,
        )

    # ── 6. Build RetrievedChunkData list ─────────────────────────────
    retrieved_chunks: List[RetrievedChunkData] = []

    if not results:
        logger.info(
            "[Retrieve] No results for hop {hop}, hash={h}",
            hop=hop_count,
            h=_safe_hash(query),
        )
    else:
        for result in results:
            # Resolve page number from metadata or direct attribute
            page_num = None
            if result.chunk.metadata:
                page_num = result.chunk.metadata.get("page_number")
            if page_num is None and hasattr(result.chunk, "page_number"):
                page_num = result.chunk.page_number

            chunk_data = RetrievedChunkData(
                chunk_id=result.chunk.chunk_id,
                document_id=result.chunk.document_id,
                document_name=result.document.filename,
                content=result.chunk.content,
                score=result.score,
                chunk_index=result.chunk.chunk_index,
                page_number=page_num,
                section_header=(
                    result.chunk.metadata.get("section_header") if result.chunk.metadata else None
                ),
                metadata=result.chunk.metadata or {},
            )
            retrieved_chunks.append(chunk_data)

        logger.info(
            "[Retrieve] Hop {hop}: {n_chunks} chunks retrieved in {ms:.1f}ms "
            "(HyDE={use_hyde}, reason={hyde_reason})",
            hop=hop_count,
            n_chunks=len(retrieved_chunks),
            ms=search_latency_ms,
            use_hyde=use_hyde,
            hyde_reason=hyde_reason,
        )

    # ── 7. Update timing spans ───────────────────────────────────────
    timing_spans["search_ms"] = timing_spans.get("search_ms", 0) + search_latency_ms
    # Store per-hop timing so refine_chunks_node can record accurate HopData
    timing_spans["last_search_ms"] = search_latency_ms
    timing_spans["last_hyde_used"] = use_hyde
    timing_spans["last_hyde_reason"] = hyde_reason

    return {
        "retrieved_chunks": retrieved_chunks,
        "hop_count": hop_count + 1,
        "previous_queries": previous_queries,
        "previous_query_embeddings": previous_embeddings,
        "coverage_ratio": coverage_ratio,
        "coverage_confidence": coverage_confidence,
        "uncovered_aspects": uncovered_aspects,
        "suggested_followup": suggested_followup,
        "timing_spans": timing_spans,
        "retrieval_attempts": retrieval_attempts + 1,
    }


# ---------------------------------------------------------------------------
# Tiny privacy helper (avoid logging raw queries at INFO level)
# ---------------------------------------------------------------------------


def _safe_hash(text: str) -> str:
    """Return an 8-char SHA-256 hash for privacy-safe logging."""
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:8]
