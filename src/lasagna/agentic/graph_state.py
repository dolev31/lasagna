"""State definition for the structured RAG graph.

This module defines ``RAGGraphState``, the typed state that flows through
every node in the structured agentic RAG ``StateGraph``.  Field groups are
documented inline; each node reads/writes only the fields it owns.

Design notes:
  - Uses ``TypedDict`` (not Pydantic BaseModel) because LangGraph StateGraph
    supports ``Annotated`` reducers with TypedDict.
  - Citation tracking fields are lifted verbatim from the current
    ``configurable`` dict in ``search_knowledge_base`` (agentic_rag.py:960-966)
    so that citation semantics are preserved exactly.
  - ``routing_decision`` is received FROM ``_search_orchestrator.py``; the graph
    never re-runs the QueryRouter (avoids double-classification cost).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, TypedDict


class RAGGraphState(TypedDict, total=False):
    """Typed state flowing through the structured RAG StateGraph.

    ``total=False`` makes every field optional at construction time so nodes
    can return partial updates.  The graph builder sets initial defaults in
    ``build_rag_graph``.
    """

    # ── Input (immutable after init) ────────────────────────────────────
    original_question: str
    session_id: str
    document_filter: Optional[List[str]]
    top_k: int
    trace_id: Optional[str]

    # ── Upstream routing (passed from _search_orchestrator, NOT recomputed) ─
    routing_decision: Any  # RoutingDecision | None — Any avoids circular import

    # ── Session config ──────────────────────────────────────────────────
    session_config: Any  # RAGConfig instance (session-scoped)

    # ── LLM reference ──────────────────────────────────────────────────
    llm: Any  # BaseChatModel — passed in so every node can use it

    # ── Query planning ──────────────────────────────────────────────────
    query_complexity: str  # "simple" | "moderate" | "complex"
    sub_queries: List[str]  # Decomposed sub-queries (or [original_question])
    current_sub_query_index: int
    current_query: str  # Active search query (may be rewritten)

    # ── Retrieval (per-hop, reset each iteration) ───────────────────────
    retrieved_chunks: List[Any]  # List[RetrievedChunkData] for current hop
    hop_data: List[Any]  # List[HopData] — accumulated across hops
    hop_count: int

    # ── Refinement ─────────────────────────────────────────────────────
    refined_chunks: List[Any]  # List[RetrievedChunkData] after refinement
    refinement_method: str  # "vanilla" | "refined" | "auto"
    refinement_metrics: Dict[str, Any]  # Per-hop: tokens_before/after, chunks_filtered, etc.

    # ── Citations (PRESERVED exactly from current system) ──────────────
    global_citation_index: int  # Monotonically increasing across hops
    all_hop_citations: List[Dict[str, Any]]  # Accumulated citation dicts
    seen_content_hashes: Set[str]  # Deduplication set

    # ── Grading (CRAG) ─────────────────────────────────────────────────
    document_grades: List[str]  # Per-chunk: "correct" | "ambiguous" | "incorrect"
    overall_grade: str  # Aggregate grade for routing
    graded_chunks: List[Any]  # List[RetrievedChunkData] after grading/filtering
    crag_web_fallback_used: bool
    crag_grading_latency_ms: float

    # ── Generation ─────────────────────────────────────────────────────
    generation: str  # LLM-generated answer text
    agent_summary: str  # Same field name as current AgenticRAGResult
    generation_citations: List[str]  # Extracted [N] refs from generation

    # ── Quality gates (opt-in) ─────────────────────────────────────────
    hallucination_check: str  # "supported" | "partially_supported" | "not_supported"
    answer_quality: str  # "addresses_question" | "partially_addresses" | "does_not_address"
    quality_check_attempts: int

    # ── Coverage scoring ───────────────────────────────────────────────
    coverage_ratio: float  # 0.0 to 1.0
    coverage_confidence: float  # 0.0 to 1.0
    uncovered_aspects: List[str]
    suggested_followup: Optional[str]  # Query suggestion for gap-filling

    # ── Chunk selection (all strategies incl. agent_decides) ───────────
    marked_chunks: List[int]  # Result numbers selected by LLM (agent_decides)
    filtered_citations: List[Dict[str, Any]]  # After strategy-based filtering

    # ── Flow control ───────────────────────────────────────────────────
    retrieval_attempts: int
    max_retrieval_attempts: int
    rewrite_count: int
    max_rewrites: int
    stop_reason: str  # "max_iterations" | "query_stabilized" | "sufficient_coverage" | ...

    # ── Metrics collection ─────────────────────────────────────────────
    timing_spans: Dict[str, float]  # Accumulated timing: search_ms, refinement_ms, etc.
    rag_metrics_list: List[Dict[str, Any]]  # Per-hop metrics (mirrors configurable["rag_metrics"])
    total_start_time: float  # time.time() at graph entry

    # ── Query deduplication ────────────────────────────────────────────
    previous_queries: List[str]
    previous_query_embeddings: List[Any]  # Embedding vectors | None

    # ── Final output (set by format_output_node) ───────────────────────
    formatted_context: str
    final_chunks: List[Dict[str, Any]]
    final_citations: List[Dict[str, Any]]
    final_metrics: Any  # AgenticRAGMetrics


def build_initial_state(
    question: str,
    session_id: str,
    top_k: int,
    session_config: Any,
    llm: Any,
    *,
    document_filter: Optional[List[str]] = None,
    trace_id: Optional[str] = None,
    routing_decision: Any = None,
    max_retrieval_attempts: int = 3,
    max_rewrites: int = 3,
) -> RAGGraphState:
    """Build the initial RAGGraphState dict for graph invocation.

    This centralises default-value logic so ``AgenticRAG.retrieve()`` and
    tests can construct state consistently.
    """
    return RAGGraphState(
        # Input
        original_question=question,
        session_id=session_id,
        document_filter=document_filter,
        top_k=top_k,
        trace_id=trace_id,
        # Upstream
        routing_decision=routing_decision,
        session_config=session_config,
        llm=llm,
        # Query planning
        query_complexity="moderate",
        sub_queries=[question],
        current_sub_query_index=0,
        current_query=question,
        # Retrieval
        retrieved_chunks=[],
        hop_data=[],
        hop_count=0,
        # Refinement
        refined_chunks=[],
        refinement_method=getattr(session_config, "chunk_refinement_method", "auto"),
        refinement_metrics={},
        # Citations
        global_citation_index=1,
        all_hop_citations=[],
        seen_content_hashes=set(),
        # Grading
        document_grades=[],
        overall_grade="",
        graded_chunks=[],
        crag_web_fallback_used=False,
        crag_grading_latency_ms=0.0,
        # Generation
        generation="",
        agent_summary="",
        generation_citations=[],
        # Quality gates
        hallucination_check="",
        answer_quality="",
        quality_check_attempts=0,
        # Coverage
        coverage_ratio=0.0,
        coverage_confidence=0.0,
        uncovered_aspects=[],
        suggested_followup=None,
        # Chunk selection
        marked_chunks=[],
        filtered_citations=[],
        # Flow control
        retrieval_attempts=0,
        max_retrieval_attempts=max_retrieval_attempts,
        rewrite_count=0,
        max_rewrites=max_rewrites,
        stop_reason="",
        # Metrics
        timing_spans={},
        rag_metrics_list=[],
        total_start_time=0.0,
        # Deduplication
        previous_queries=[],
        previous_query_embeddings=[],
        # Final output (populated by format_output_node)
        formatted_context="",
        final_chunks=[],
        final_citations=[],
        final_metrics=None,
    )
