"""Agentic layer - Multi-hop retrieval, citations, and document awareness.

This layer provides advanced RAG capabilities:
- AgenticRAG: Multi-hop retrieval with query refinement (ReAct + Structured Graph)
- CitationManager: Stable citation tracking
- DocumentAwareness: Context-aware document understanding
- Structured Graph: LangGraph StateGraph with 13 nodes and quality gates
"""

from .agentic_rag import (
    AgenticRAG,
    AgenticRAGResult,
    AgenticRAGMetrics,
    HopData,
    RetrievedChunkData,
    search_knowledge_base,
    mark_chunk_for_citation,
    estimate_tokens,
    refine_chunk,
)

from .citation_manager import (
    CitationManager,
    Citation,
    get_citation_manager,
    extract_highlight_ranges,
)

from .document_awareness import (
    DocumentInfo,
    DocumentSnippet,
    get_document_awareness_context,
)

from .graph_builder import build_rag_graph
from .graph_state import RAGGraphState, build_initial_state

__all__ = [
    # Agentic RAG
    "AgenticRAG",
    "AgenticRAGResult",
    "AgenticRAGMetrics",
    "HopData",
    "RetrievedChunkData",
    "search_knowledge_base",
    "mark_chunk_for_citation",
    "estimate_tokens",
    "refine_chunk",
    # Structured Graph
    "build_rag_graph",
    "RAGGraphState",
    "build_initial_state",
    # Citations
    "CitationManager",
    "Citation",
    "get_citation_manager",
    "extract_highlight_ranges",
    # Document Awareness
    "DocumentInfo",
    "DocumentSnippet",
    "get_document_awareness_context",
]
