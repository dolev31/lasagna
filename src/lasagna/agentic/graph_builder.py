"""Graph builder — constructs the structured RAG StateGraph.

``build_rag_graph()`` wires all nodes from ``nodes/`` with conditional edges
from ``edges.py`` and returns a compiled LangGraph ``CompiledGraph`` ready
for invocation.

Usage:
    graph = build_rag_graph(session_config)
    result = await graph.ainvoke(initial_state)

The graph layout mirrors the plan:

    plan_retrieval → retrieve → refine_chunks → grade_documents
        → { generate | rewrite_query | web_fallback }
        → [ check_hallucination → check_answer_quality ] (opt-in)
        → select_chunks → { format_output | retrieve (sub-query loop) }

Custom node overrides:
    Pass ``node_overrides={"retrieve": my_custom_fn}`` to swap specific
    nodes while keeping the rest of the graph intact.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from langgraph.graph import END, START, StateGraph
from loguru import logger

from lasagna.agentic.graph_state import RAGGraphState
from lasagna.core.base_components import BaseAgenticPipeline, AgenticPipelineInput

if TYPE_CHECKING:
    # Resolves the return annotation on BuiltinStructuredPipeline.run for type
    # checkers. The runtime import stays inside that method: agentic_rag is a
    # large module and importing it here would make every graph_builder import
    # pay for it.
    #
    # This does not make `typing.get_type_hints()` work on that method — the
    # name is absent from module globals at runtime by design. Nothing in the
    # codebase introspects it; if something ever needs to, the import has to
    # move to module scope.
    from lasagna.agentic.agentic_rag import AgenticRAGResult

# ── Node imports ──────────────────────────────────────────────────────────
from lasagna.agentic.nodes.plan_retrieval import plan_retrieval_node
from lasagna.agentic.nodes.retrieve import retrieve_node
from lasagna.agentic.nodes.refine_chunks import refine_chunks_node
from lasagna.agentic.nodes.grade_documents import grade_documents_node
from lasagna.agentic.nodes.generate import generate_node
from lasagna.agentic.nodes.rewrite_query import rewrite_query_node
from lasagna.agentic.nodes.select_chunks import select_chunks_node
from lasagna.agentic.nodes.format_output import format_output_node
from lasagna.agentic.nodes.final_search import final_search_node
from lasagna.agentic.nodes.web_fallback import web_fallback_node
from lasagna.agentic.nodes.advance_sub_query import advance_sub_query_node
from lasagna.agentic.nodes.quality_gates import (
    check_hallucination_node,
    check_answer_quality_node,
)

# ── Edge routing functions ────────────────────────────────────────────────
from lasagna.agentic.edges import (
    route_after_grading,
    route_after_generation,
    route_after_hallucination_check,
    route_after_quality_check,
    route_after_select_chunks,
    route_after_rewrite,
)

# ── Valid node names (whitelist for override validation) ──────────────────
VALID_NODE_NAMES = frozenset(
    {
        "plan_retrieval",
        "retrieve",
        "refine_chunks",
        "grade_documents",
        "generate",
        "rewrite_query",
        "select_chunks",
        "format_output",
        "final_search",
        "web_fallback",
        "advance_sub_query",
        "check_hallucination",
        "check_answer_quality",
    }
)

# Default node function mapping
_DEFAULT_NODES: Dict[str, Callable] = {
    "plan_retrieval": plan_retrieval_node,
    "retrieve": retrieve_node,
    "refine_chunks": refine_chunks_node,
    "grade_documents": grade_documents_node,
    "generate": generate_node,
    "rewrite_query": rewrite_query_node,
    "select_chunks": select_chunks_node,
    "format_output": format_output_node,
    "final_search": final_search_node,
    "web_fallback": web_fallback_node,
    "advance_sub_query": advance_sub_query_node,
    "check_hallucination": check_hallucination_node,
    "check_answer_quality": check_answer_quality_node,
}


def _validate_node_overrides(overrides: Dict[str, Callable]) -> None:
    """Validate node override dict: known names + callable check."""
    unknown = set(overrides) - VALID_NODE_NAMES
    if unknown:
        raise ValueError(
            f"Unknown node names in overrides: {sorted(unknown)}. "
            f"Valid node names: {sorted(VALID_NODE_NAMES)}"
        )
    for name, fn in overrides.items():
        if not callable(fn):
            raise TypeError(f"Node override '{name}' must be callable, got {type(fn).__name__}")
        # Require async — LangGraph nodes must be async coroutine functions
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(
                f"Node override '{name}' ({getattr(fn, '__name__', repr(fn))}) "
                f"must be an async function (coroutinefunction). "
                f"LangGraph nodes are invoked with `await`."
            )


def build_rag_graph(
    session_config: Any = None,
    node_overrides: Optional[Dict[str, Callable]] = None,
) -> Any:
    """Build and compile the structured RAG StateGraph.

    Parameters
    ----------
    session_config:
        ``RAGConfig`` instance.  Used to decide which quality-gate nodes to
        wire.  If ``None``, quality gates are disabled.
    node_overrides:
        Optional dict mapping node names to replacement callables.
        Example: ``{"retrieve": my_custom_retrieve_node}``.
        Unknown names raise ``ValueError``.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph ready for ``await graph.ainvoke(state)``.
    """
    overrides = node_overrides or {}
    if overrides:
        _validate_node_overrides(overrides)

    graph = StateGraph(RAGGraphState)

    # ── Register nodes (use override if provided, else default) ───────────
    for name, default_fn in _DEFAULT_NODES.items():
        graph.add_node(name, overrides.get(name, default_fn))

    # ── Unconditional edges ───────────────────────────────────────────────
    graph.add_edge(START, "plan_retrieval")
    graph.add_edge("plan_retrieval", "retrieve")
    graph.add_edge("retrieve", "refine_chunks")
    graph.add_edge("refine_chunks", "grade_documents")
    graph.add_edge("web_fallback", "generate")
    graph.add_edge("format_output", END)

    # ── Conditional edges ─────────────────────────────────────────────────

    # After CRAG grading: correct → generate, incorrect → web_fallback or rewrite, ambiguous → rewrite or generate
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "web_fallback": "web_fallback",
            "rewrite_query": "rewrite_query",
        },
    )

    # After generation: quality gates (if enabled) or straight to select_chunks
    graph.add_conditional_edges(
        "generate",
        route_after_generation,
        {
            "check_hallucination": "check_hallucination",
            "check_answer_quality": "check_answer_quality",
            "select_chunks": "select_chunks",
        },
    )

    # After hallucination check: answer quality check, rewrite, or accept
    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {
            "check_answer_quality": "check_answer_quality",
            "rewrite_query": "rewrite_query",
            "select_chunks": "select_chunks",
        },
    )

    # After answer quality check: accept, rewrite, or regenerate
    graph.add_conditional_edges(
        "check_answer_quality",
        route_after_quality_check,
        {
            "select_chunks": "select_chunks",
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        },
    )

    # After select_chunks: more sub-queries → advance index, else → final_search
    graph.add_conditional_edges(
        "select_chunks",
        route_after_select_chunks,
        {
            "advance_sub_query": "advance_sub_query",
            "final_search": "final_search",
        },
    )

    # final_search always goes to format_output
    graph.add_edge("final_search", "format_output")

    # advance_sub_query always goes back to retrieve
    graph.add_edge("advance_sub_query", "retrieve")

    # After rewrite_query: budget check → retrieve or generate
    graph.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    # ── Compile ───────────────────────────────────────────────────────────
    compiled = graph.compile()

    logger.info(
        "[GraphBuilder] RAG graph compiled with {n} nodes",
        n=len(graph.nodes),
    )

    return compiled


# =========================================================================
# BuiltinStructuredPipeline — wraps the graph as a BaseAgenticPipeline
# =========================================================================


class BuiltinStructuredPipeline(BaseAgenticPipeline):
    """Built-in structured RAG pipeline (13-node LangGraph StateGraph).

    Subclasses ``BaseAgenticPipeline`` and supports node-level
    customization via ``node_overrides``.

    Registered in the component registry as ``"default"``.
    """

    def __init__(
        self,
        session_config: Any = None,
        node_overrides: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ):
        self._session_config = session_config
        self._node_overrides = node_overrides or {}

    async def run(self, input: AgenticPipelineInput) -> "AgenticRAGResult":
        """Execute the structured RAG graph and return AgenticRAGResult."""
        from lasagna.agentic.agentic_rag import AgenticRAGResult
        from lasagna.agentic.graph_state import build_initial_state
        from lasagna.core.config import rag_config

        config = input.session_config or self._session_config or rag_config
        total_start_time = time.time()

        graph = build_rag_graph(
            session_config=config,
            node_overrides=self._node_overrides,
        )

        max_iters = getattr(config, "max_agent_iterations", 3)
        initial_state = build_initial_state(
            question=input.question,
            session_id=input.session_id,
            top_k=input.top_k,
            session_config=config,
            llm=input.llm,
            document_filter=input.document_filter,
            trace_id=input.trace_id,
            routing_decision=input.routing_decision,
            max_retrieval_attempts=max_iters,
            max_rewrites=max_iters,
        )
        initial_state["total_start_time"] = total_start_time

        final_state = await graph.ainvoke(initial_state)

        # Extract outputs from graph state — same logic as
        # AgenticRAG._retrieve_structured()
        rag_metrics_list = final_state.get("rag_metrics_list", [])
        total_tokens_before = sum(m.get("tokens_before", 0) for m in rag_metrics_list)
        total_tokens_after = sum(m.get("tokens_after", 0) for m in rag_metrics_list)
        total_chunks_filtered = sum(m.get("chunks_filtered", 0) for m in rag_metrics_list)
        total_reduction_pct = (
            ((total_tokens_before - total_tokens_after) / total_tokens_before * 100)
            if total_tokens_before > 0
            else 0.0
        )

        return AgenticRAGResult(
            formatted_context=final_state.get("formatted_context", ""),
            chunks=final_state.get("final_chunks", []),
            citations=final_state.get("final_citations", []),
            agent_summary=final_state.get("agent_summary", ""),
            search_count=final_state.get("hop_count", 0),
            tokens_before=total_tokens_before,
            tokens_after=total_tokens_after,
            tokens_saved=total_tokens_before - total_tokens_after,
            reduction_percentage=total_reduction_pct,
            chunks_filtered=total_chunks_filtered,
            hops=final_state.get("hop_data", []),
            metrics=final_state.get("final_metrics"),
        )

    @property
    def name(self) -> str:
        return "BuiltinStructuredPipeline"
