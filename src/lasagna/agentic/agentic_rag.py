"""
Agentic RAG Implementation

This module provides a ReAct-based agent for intelligent RAG retrieval.
The agent can decompose user questions, perform multiple refined searches,
and synthesize results for better context retrieval.

Features:
- Multi-hop query refinement with learning from previous hops
- Bounded iterations with multiple stopping conditions
- Per-hop citation tracking with full metadata preservation
- Chunk refinement with source mapping (never lose original citations)
- Comprehensive metrics and structured logging
"""

import asyncio
import os
import time
import json
import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from loguru import logger

# Core configuration
from lasagna.core.config import rag_config, RAGConfig
from lasagna.core.base_components import BaseAgenticPipeline, AgenticPipelineInput
from lasagna.utils.model_utils import (
    resolve_model_settings,
    get_default_model_settings,
    is_langfuse_tracing_enabled,
    message_text,
)
from lasagna.providers import get_llm_provider

# Agentic components
from lasagna.agentic.citation_manager import extract_highlight_ranges

# Retrieval enhancement
from lasagna.retrieval.enhancement.coverage import get_coverage_scorer


# =============================================================================
# PRIVACY & LOGGING HELPERS
# =============================================================================


def _hash_query(query: str) -> str:
    """Generate a short hash of a query for privacy-safe logging.

    Args:
        query: The raw query text

    Returns:
        8-character hash suitable for log correlation
    """
    return hashlib.sha256(query.encode()).hexdigest()[:8]


def _safe_query_for_log(query: str, max_len: int = 0) -> dict:
    """Create privacy-safe query representation for INFO-level logs.

    Args:
        query: The raw query text
        max_len: If > 0, include truncated preview (for DEBUG only)

    Returns:
        Dict with query_hash, query_len, and optionally query_preview
    """
    result = {
        "query_hash": _hash_query(query),
        "query_len": len(query),
    }
    if max_len > 0:
        result["query_preview"] = query[:max_len] + "..." if len(query) > max_len else query
    return result


def _generate_content_hash(content: str) -> str:
    """Generate a stable content hash for deduplication.

    Uses the same approach as CitationManager for consistency.
    Normalizes whitespace before hashing to catch near-duplicates.

    Args:
        content: The chunk content to hash

    Returns:
        16-character hex hash of the normalized content
    """
    # Normalize whitespace for robust deduplication
    normalized = " ".join(content.split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


# =============================================================================
# HYDE DECISION HELPERS
# =============================================================================

DEFAULT_AGENTIC_HYDE_STRATEGY = "adaptive"

# Maximum concurrent LLM calls when CRAG grades retrieved chunks.
#
# Grading issues one call per chunk, so an unbounded fan-out is as wide as the
# retrieval (easily ~100). That overruns typical hosted-endpoint concurrency
# limits, and a grade that times out falls back to "correct" -- meaning the CRAG
# decision gets made from answers that were never computed. Raise this only if
# your endpoint is known to tolerate it.
CRAG_GRADING_CONCURRENCY = max(1, int(os.environ.get("RAG_CRAG_GRADING_CONCURRENCY", "4")))


def _normalize_agentic_hyde_strategy(value: Optional[str]) -> str:
    """Normalize agentic HyDE strategy values to a known set."""
    if not value:
        return DEFAULT_AGENTIC_HYDE_STRATEGY
    normalized = str(value).strip().lower()
    if normalized in {"off", "disabled", "false", "no", "none"}:
        return "off"
    if normalized in {"always", "on"}:
        return "always"
    if normalized in {"first", "first_hop", "first-hop"}:
        return "first_hop"
    if normalized in {"adaptive", "auto"}:
        return "adaptive"
    logger.debug(
        f"Unknown agentic_hyde_strategy='{value}', defaulting to {DEFAULT_AGENTIC_HYDE_STRATEGY}"
    )
    return DEFAULT_AGENTIC_HYDE_STRATEGY


def _should_use_hyde_for_agentic(
    query: str,
    hop_index: int,
    hop_data_list: List["HopData"],
    session_config: RAGConfig,
    is_final: bool = False,
) -> Tuple[bool, str]:
    """Decide whether to apply HyDE for agentic RAG searches."""
    if not session_config.enable_hyde:
        return False, "hyde_disabled"

    strategy = _normalize_agentic_hyde_strategy(session_config.agentic_hyde_strategy)
    if strategy == "off":
        return False, "strategy_off"
    if strategy == "always":
        return True, "strategy_always"
    if strategy == "first_hop":
        if is_final:
            return False, "strategy_first_hop_final_skip"
        return (
            hop_index == 0
        ), "strategy_first_hop" if hop_index == 0 else "strategy_first_hop_skip"

    # Adaptive strategy: use HyDE for short queries or when evidence is insufficient.
    query_tokens = estimate_tokens(query)
    max_tokens = session_config.max_query_length or 12
    if query_tokens <= max_tokens and (hop_index == 0 or is_final):
        return True, f"short_query(tokens={query_tokens}<=max_query_length={max_tokens})"

    if hop_data_list:
        last_hop = hop_data_list[-1]
        min_chunks = session_config.chunk_cite_retrieval.get("min_chunks", 2)
        min_docs = 1
        if len(last_hop.retrieved_chunks) < min_chunks or len(last_hop.unique_doc_ids) < min_docs:
            return True, (
                "insufficient_evidence("
                f"chunks={len(last_hop.retrieved_chunks)}, docs={len(last_hop.unique_doc_ids)})"
            )

    if is_final:
        return False, "adaptive_final_skip"
    return False, "adaptive_skip"


# =============================================================================
# SCORE NORMALIZATION HELPERS
# =============================================================================

# Import from shared module
from lasagna.utils.normalization import normalize_score_relative as _normalize_score  # noqa: E402


def _should_auto_refine(
    chunks: List["RetrievedChunkData"],
    length_threshold: int = 1000,
    chunk_count_threshold: int = 7,
    score_variance_threshold: float = 0.3,
) -> Tuple[bool, str]:
    """Decide whether to auto-refine based on chunk quality metrics.

    Uses rank/quantile-based conditions rather than absolute score thresholds
    to handle different scoring schemes (including negative scores).

    Args:
        chunks: List of retrieved chunks
        length_threshold: Average character length threshold
        chunk_count_threshold: Number of chunks threshold
        score_variance_threshold: Score variance threshold (normalized)

    Returns:
        Tuple of (should_refine: bool, reason: str)
        reason is empty string if should_refine=True, otherwise explains why not
    """
    if not chunks:
        return False, "no_chunks"

    # Condition 1: Too many chunks
    if len(chunks) > chunk_count_threshold:
        return True, ""

    # Condition 2: Chunks are too long on average
    avg_length = sum(len(c.content) for c in chunks) / len(chunks)
    if avg_length > length_threshold:
        return True, ""

    # Condition 3: High score variance suggests uncertain relevance
    # Use normalized scores to handle different scoring schemes
    scores = [c.score for c in chunks]
    if len(scores) > 1:
        normalized = [_normalize_score(s, scores) for s in scores]
        variance = sum((s - sum(normalized) / len(normalized)) ** 2 for s in normalized) / len(
            normalized
        )
        if variance > score_variance_threshold:
            return True, ""

    # No conditions met - explain why refinement was skipped
    skip_reasons = []
    skip_reasons.append(f"chunk_count={len(chunks)}<={chunk_count_threshold}")
    skip_reasons.append(f"avg_length={avg_length:.0f}<={length_threshold}")
    if len(scores) > 1:
        skip_reasons.append(f"variance={variance:.3f}<={score_variance_threshold}")
    return False, "; ".join(skip_reasons)


def arm_query_overrides(session_config: Any) -> Optional[Dict[str, Any]]:
    """The per-query overrides an agentic search must carry, or None.

    The orchestrator hands an agentic pipeline the session config with the
    arm's per-query overrides applied. A search the pipeline makes with
    ``_from_routing=True`` re-reads the config for its session, so without
    these it searched under the shared session store, which concurrent arms
    of a sweep overwrite, rather than under its own arm's settings.
    """
    overrides = getattr(session_config, "_query_overrides", None)
    return dict(overrides) if overrides else None


# =============================================================================
# DATA STRUCTURES FOR MULTI-HOP TRACKING
# =============================================================================


@dataclass
class RetrievedChunkData:
    """Data for a single retrieved chunk with full citation metadata."""

    chunk_id: str
    document_id: str
    document_name: str
    content: str
    score: float
    chunk_index: int
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For refined chunks - pointer to original
    original_content: Optional[str] = None
    is_refined: bool = False
    refinement_kept: bool = True  # False if chunk was filtered as NOT_RELEVANT


@dataclass
class HopData:
    """Data structure for a single hop in multi-hop RAG.

    Each hop represents one search iteration with its query,
    retrieved chunks, and timing information.
    """

    hop_index: int
    query_used: str
    retrieved_chunks: List[RetrievedChunkData]
    search_latency_ms: float = 0.0
    refinement_latency_ms: float = 0.0
    tokens_before_refinement: int = 0
    tokens_after_refinement: int = 0
    chunks_filtered: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Refinement decision tracking
    refinement_skipped: bool = False
    refinement_skip_reason: str = ""  # e.g., "low_variance", "few_chunks", "short_content"
    hyde_used: bool = False
    hyde_reason: str = ""
    # CRAG (Corrective RAG) grading
    crag_grade: str = ""  # "correct", "ambiguous", "incorrect", or "" if not evaluated
    crag_grades_detail: List[str] = field(default_factory=list)  # Per-chunk grades
    crag_web_fallback_used: bool = False

    @property
    def unique_doc_ids(self) -> set:
        return {c.document_id for c in self.retrieved_chunks}

    @property
    def unique_page_numbers(self) -> set:
        return {c.page_number for c in self.retrieved_chunks if c.page_number is not None}


@dataclass
class AgenticRAGMetrics:
    """Comprehensive metrics for agentic RAG execution."""

    # Basic flags
    agentic_enabled: bool = True
    final_search_used: bool = False

    # Iteration tracking
    agent_iterations_used: int = 0
    hop_count: int = 0
    max_iterations_configured: int = 3

    # Timing breakdown (all in milliseconds)
    total_latency_ms: float = 0.0
    hop_search_latencies_ms: List[float] = field(default_factory=list)
    hop_refinement_latencies_ms: List[float] = field(default_factory=list)

    # NEW: Agent orchestration timing (the "missing" latency)
    agent_orchestration_ms: float = 0.0  # LangGraph routing, tool dispatch
    agent_llm_calls_ms: float = 0.0  # Time spent in LLM calls (query planning, synthesis)
    final_search_ms: float = 0.0  # Time for final comprehensive search (if enabled)
    context_formatting_ms: float = 0.0  # Time for formatting context for LLM

    # NEW: Latency breakdown accountability
    accounted_latency_ms: float = 0.0  # Sum of all measured components
    unaccounted_latency_ms: float = 0.0  # total - accounted (should be small)

    # Token metrics (estimated)
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    total_tokens_saved: int = 0
    reduction_percentage: float = 0.0

    # Char metrics (stable, no tokenizer dependency)
    total_chars_before: int = 0
    total_chars_after: int = 0

    # Refinement tracking
    total_chunks_filtered: int = 0
    refinement_failures_count: int = 0
    refinement_skipped: bool = False  # True if auto-refinement decided not to refine
    refinement_skip_reason: str = ""  # Why refinement was skipped

    # Document/page coverage across all hops
    unique_docs_across_hops: int = 0
    unique_pages_across_hops: int = 0

    # Citation quality
    citation_count: int = 0
    citation_missing_page_rate: float = 0.0

    # Stopping condition
    stop_reason: str = (
        ""  # "max_iterations", "query_stabilized", "no_new_relevant_chunks", "sufficient_context"
    )

    # Minimum evidence tracking (for sufficient_context validation)
    min_chunks_for_sufficient: int = 2
    min_unique_docs_for_sufficient: int = 1

    # NEW: Quality booster status (surfaced from RAGBackend)
    hybrid_search_enabled: bool = False
    reranking_enabled: bool = False
    mmr_enabled: bool = False
    hyde_enabled: bool = False

    # CRAG (Corrective RAG) metrics
    crag_enabled: bool = False
    crag_grades: List[str] = field(default_factory=list)  # Per-hop overall grades
    crag_web_fallback_triggered: bool = False
    crag_grading_latency_ms: float = 0.0

    # What a published loop (lasagna.agentic.loops) did on this question: its
    # queries, its stopping reason and, for FLARE, which confidence signal ran.
    loop_trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/API responses."""
        return {
            "agentic_enabled": self.agentic_enabled,
            "final_search_used": self.final_search_used,
            "agent_iterations_used": self.agent_iterations_used,
            "hop_count": self.hop_count,
            "max_iterations_configured": self.max_iterations_configured,
            # Latency breakdown
            "total_latency_ms": round(self.total_latency_ms, 2),
            "hop_search_latencies_ms": [round(ms, 2) for ms in self.hop_search_latencies_ms],
            "hop_refinement_latencies_ms": [
                round(ms, 2) for ms in self.hop_refinement_latencies_ms
            ],
            "agent_orchestration_ms": round(self.agent_orchestration_ms, 2),
            "agent_llm_calls_ms": round(self.agent_llm_calls_ms, 2),
            "final_search_ms": round(self.final_search_ms, 2),
            "context_formatting_ms": round(self.context_formatting_ms, 2),
            "accounted_latency_ms": round(self.accounted_latency_ms, 2),
            "unaccounted_latency_ms": round(self.unaccounted_latency_ms, 2),
            # Token/char metrics
            "total_tokens_before": self.total_tokens_before,
            "total_tokens_after": self.total_tokens_after,
            "total_tokens_saved": self.total_tokens_saved,
            "reduction_percentage": round(self.reduction_percentage, 2),
            "total_chars_before": self.total_chars_before,
            "total_chars_after": self.total_chars_after,
            # Refinement tracking
            "total_chunks_filtered": self.total_chunks_filtered,
            "refinement_failures_count": self.refinement_failures_count,
            "refinement_skipped": self.refinement_skipped,
            "refinement_skip_reason": self.refinement_skip_reason,
            # Coverage
            "unique_docs_across_hops": self.unique_docs_across_hops,
            "unique_pages_across_hops": self.unique_pages_across_hops,
            "citation_count": self.citation_count,
            "citation_missing_page_rate": round(self.citation_missing_page_rate, 4),
            "stop_reason": self.stop_reason,
            # Quality boosters status
            "quality_boosters": {
                "hybrid_search_enabled": self.hybrid_search_enabled,
                "reranking_enabled": self.reranking_enabled,
                "mmr_enabled": self.mmr_enabled,
                "hyde_enabled": self.hyde_enabled,
            },
            # CRAG metrics
            "crag": {
                "enabled": self.crag_enabled,
                "grades": self.crag_grades,
                "web_fallback_triggered": self.crag_web_fallback_triggered,
                "grading_latency_ms": round(self.crag_grading_latency_ms, 2),
            },
            **({"loop_trace": self.loop_trace} if self.loop_trace else {}),
        }


# =============================================================================
# QUERY REFINEMENT PROMPT
# =============================================================================

QUERY_REFINEMENT_PROMPT = """You are a search query optimizer. Given the user's original question and the current search results, propose a refined search query to find missing information.

## Original Question:
{original_question}

## Current Query:
{current_query}

## Current Search Results Summary:
{results_summary}

## Missing Information (what we still need):
{missing_info}

## Instructions:
1. If the current results are sufficient, respond with: SUFFICIENT
2. If you need to search for more specific information, propose a NEW search query
3. Keep the query SHORT (ideally 3-12 words)
4. Focus on specific entities, terms, or concepts not yet found
5. Avoid repeating the same query - try alternative phrasings

## Response Format:
Either "SUFFICIENT" or your new search query (nothing else)

New query:"""


"""
## Multi-Hop Search Strategy:
For complex questions that require multiple pieces of information:
- First hop: Search for the primary concept/entity
- If you find references to related entities or constraints, search for those next
- Learn from each search: note what's missing and target it in the next query
- Keep queries SHORT (3-12 words) and focused on specific concepts
"""


# System prompt for the RAG agent
AGENTIC_RAG_SYSTEM_PROMPT = """You are a knowledge retrieval specialist. Your task is to find relevant information from the user's uploaded documents to help answer their question.

## Process:
1. Analyze the user's question to understand their intent
2. Identify key concepts and search terms
3. Use the search_knowledge_base tool to find relevant information
4. If results are insufficient or too broad, refine your query and search again
5. Synthesize the most relevant findings

## Guidelines:
- Break complex questions into focused search queries
- If initial search returns poor results, try alternative phrasings
- Prefer specific technical terms when appropriate
- Stop searching when you have sufficient context (typically 2-4 good sources)
- Maximum of {max_iterations} search attempts
- DO NOT repeat the same query - try different terms if previous search was insufficient

{strategy_instructions}

## Final Output:
After gathering sufficient information, provide a comprehensive summary that:
- INCLUDES actual quotes and specific details from the documents (not just descriptions)
- References sources using ONLY the result number in square brackets: [1], [2], [3], etc.
- DO NOT add any other characters, symbols, or text inside the brackets - ONLY the number
- Place citations RIGHT AFTER the sentence or phrase that uses that information
- CORRECT examples: "The cost was $1,370 [1]. The workshops included three sessions [2]."
- INCORRECT examples: "[Result 1]", "[Source 1]", "[Document 1]", "[1: filename.pdf]"
- Provides enough context and detail for someone to fully understand the answer
- If you couldn't find relevant information, clearly state that WITHOUT any citations

IMPORTANT: Your summary should contain the actual information from the documents, not just meta-descriptions like "I found information about X". Include the specific facts, numbers, quotes, and details. Only cite sources that actually contain the information you're presenting - do NOT cite sources when saying information was not found.
"""

# Strategy-specific instructions for agent_decides mode
AGENT_DECIDES_INSTRUCTIONS = """
## Chunk Selection (agent_decides mode):
After performing searches, you MUST use the mark_chunk_for_citation tool to select which chunks should be included in the final context.
- Review all search results carefully
- Use mark_chunk_for_citation(result_number) to mark each relevant chunk
- Only mark chunks that are directly relevant and useful for answering the question
- You can mark multiple chunks by calling the tool multiple times
- The result_number corresponds to the [N] number in the search results
- Example: If [2] is relevant, call mark_chunk_for_citation(2)

## Citation Format in Your Summary:
When referencing information in your summary, cite sources using ONLY the result number in square brackets:
- CORRECT: "The cost was $1,370 [1]. The workshops included three sessions [2]."
- Use ONLY numbers inside brackets: [1], [2], [3], etc.
- Place citations immediately after the relevant sentence or phrase
"""


# Prompt for chunk refinement - extracts only relevant information
CHUNK_REFINEMENT_PROMPT = """Given the search query and a retrieved text chunk, extract only the information that is directly relevant to answering the query. Remove any irrelevant sentences or paragraphs.

Search Query: {query}

Retrieved Chunk from "{source_name}":
{chunk_content}

Instructions:
- Keep sentences and paragraphs that directly relate to the search query
- Remove sentences that are off-topic or not useful for answering the query
- If the ENTIRE chunk is irrelevant to the query, respond with exactly: NOT_RELEVANT
- Preserve the original wording of relevant content (do not summarize or paraphrase)
- Keep section headers or context markers if they help understand the content

Relevant content:"""


# =============================================================================
# CORRECTIVE RAG (CRAG) GRADING PROMPT
# =============================================================================

CRAG_GRADING_PROMPT = """You are a relevance grader. Given a user question and a retrieved document, \
determine if the document contains information relevant to answering the question.

Question: {query}
Document: {document_text}

Grade the document as one of:
- "correct" - the document is relevant and helpful for answering the question
- "ambiguous" - the document may be partially relevant
- "incorrect" - the document is not relevant to the question

Respond with ONLY one word: correct, ambiguous, or incorrect."""


# Import accurate token estimation from utils
from lasagna.utils.tokenization import estimate_tokens  # noqa: E402


def _find_semantically_similar_query(
    query: str,
    previous_queries: List[str],
    previous_embeddings: List[Any],
    similarity_threshold: float = 0.92,
) -> Optional[str]:
    """Find if a query is semantically similar to any previous queries.

    Uses cosine similarity between query embeddings to detect when the agent
    is essentially repeating a search with rephrased words.

    Args:
        query: The new query to check
        previous_queries: List of previous query strings
        previous_embeddings: List of previous query embeddings (numpy arrays)
        similarity_threshold: Minimum similarity to consider queries equivalent

    Returns:
        The similar previous query if found, None otherwise
    """
    if not previous_queries or not previous_embeddings:
        return None

    if len(previous_queries) != len(previous_embeddings):
        logger.warning("Query/embedding count mismatch in semantic deduplication")
        return None

    try:
        # Import numpy for similarity calculation
        import numpy as np

        # Get embedding for current query
        from lasagna.infrastructure.embedding.manager import get_embedding_manager

        embedding_manager = get_embedding_manager()
        query_embedding = embedding_manager.encode_query(query)

        # Compare with previous embeddings
        for prev_query, prev_embedding in zip(previous_queries, previous_embeddings):
            if prev_embedding is None:
                continue

            # Cosine similarity
            similarity = np.dot(query_embedding, prev_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(prev_embedding)
            )

            if similarity >= similarity_threshold:
                logger.debug(
                    f"Semantic similarity detected: {similarity:.3f} "
                    f"(threshold={similarity_threshold})"
                )
                return prev_query

    except Exception as e:
        logger.debug(f"Semantic deduplication check failed: {e}")

    return None


def _is_irrelevant_response(response: str) -> bool:
    """Check if a refinement response indicates the chunk is irrelevant.

    Handles various LLM response formats that indicate irrelevance:
    - "NOT_RELEVANT" (exact match)
    - "not relevant" (case insensitive)
    - "This chunk is not relevant to the query"
    - "N/A - irrelevant content"
    - "The content does not relate to..."
    - Empty or whitespace-only responses

    Args:
        response: The LLM refinement response

    Returns:
        True if the response indicates irrelevance, False otherwise
    """
    import re

    if not response or not response.strip():
        return True

    # Normalize for comparison
    normalized = response.strip().upper()

    # Check exact match first
    if normalized == "NOT_RELEVANT":
        return True

    # Check if starts with NOT_RELEVANT (e.g., "NOT_RELEVANT - ...")
    if normalized.startswith("NOT_RELEVANT"):
        return True

    # Check common irrelevance patterns (case-insensitive)
    lower_response = response.lower().strip()

    # Patterns that indicate the chunk is not relevant
    irrelevance_patterns = [
        r"^not\s+relevant",  # "not relevant to the query"
        r"^n/?a\b",  # "N/A" or "NA"
        r"^none\b",  # "None" or "None applicable"
        r"^irrelevant\b",  # "Irrelevant content"
        r"^this\s+(chunk|content|text)\s+is\s+not\s+relevant",  # "This chunk is not relevant"
        r"^the\s+(chunk|content|text)\s+is\s+not\s+relevant",  # "The content is not relevant"
        r"does\s+not\s+(relate|pertain)\s+to",  # "does not relate to"
        r"no\s+relevant\s+(information|content)",  # "no relevant information"
        r"entirely\s+(irrelevant|unrelated)",  # "entirely irrelevant"
        r"nothing\s+(relevant|useful)",  # "nothing relevant"
    ]

    for pattern in irrelevance_patterns:
        if re.search(pattern, lower_response):
            return True

    # Check if the response is very short and contains "irrelevant"
    if len(lower_response) < 50 and "irrelevant" in lower_response:
        return True

    return False


async def refine_chunk(
    llm: BaseChatModel, query: str, chunk_content: str, source_name: str, chunk_index: int
) -> Tuple[int, Optional[str], str]:
    """Refine a single chunk by removing irrelevant information.

    Args:
        llm: Language model for refinement
        query: The search query for context
        chunk_content: The raw chunk content to refine
        source_name: Name of the source document (for prompt context)
        chunk_index: Index of this chunk (for logging)

    Returns:
        Tuple of (original_chunk_index, refined_content or None if irrelevant, original_content)
    """
    try:
        prompt = CHUNK_REFINEMENT_PROMPT.format(
            query=query, source_name=source_name, chunk_content=chunk_content
        )

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        refined_content = message_text(response).strip()

        # Check if chunk was marked as irrelevant
        if _is_irrelevant_response(refined_content):
            logger.debug(f"Chunk {chunk_index} from '{source_name}' marked as NOT_RELEVANT")
            return (chunk_index, None, chunk_content)

        return (chunk_index, refined_content, chunk_content)

    except Exception as e:
        logger.warning(f"Chunk refinement failed for chunk {chunk_index}: {e}. Using original.")
        # On error, return original content
        return (chunk_index, chunk_content, chunk_content)


# =============================================================================
# CORRECTIVE RAG (CRAG) — RETRIEVAL QUALITY GRADING
# =============================================================================


async def _grade_single_document(
    llm: BaseChatModel,
    query: str,
    document_text: str,
    doc_index: int,
) -> Tuple[int, str]:
    """Grade a single retrieved document for relevance using the LLM.

    Args:
        llm: Language model for grading
        query: The user's search query
        document_text: The document/chunk text to grade
        doc_index: Index of this document (for result ordering)

    Returns:
        Tuple of (doc_index, grade) where grade is "correct", "ambiguous", or "incorrect"
    """
    try:
        prompt = CRAG_GRADING_PROMPT.format(
            query=query,
            document_text=document_text[:2000],  # Truncate very long documents
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        grade = message_text(response).strip().lower()

        # Normalize the grade to one of the three valid values
        if grade in ("correct", "ambiguous", "incorrect"):
            return (doc_index, grade)

        # Handle edge cases where LLM outputs extra text
        for valid_grade in ("correct", "ambiguous", "incorrect"):
            if valid_grade in grade:
                logger.debug(
                    f"CRAG: Normalized grade '{grade}' -> '{valid_grade}' for doc {doc_index}"
                )
                return (doc_index, valid_grade)

        # Default to ambiguous if we can't parse the response
        logger.warning(
            f"CRAG: Unparseable grade '{grade}' for doc {doc_index}, defaulting to 'ambiguous'"
        )
        return (doc_index, "ambiguous")

    except Exception as e:
        # Include the exception type: a grade that fails here still counts
        # toward the CRAG decision as 'ambiguous', so a systematic failure
        # (rate limiting, a stale event loop) would otherwise quietly shift
        # every grade without any indication of why.
        logger.warning(
            f"CRAG: Grading failed for doc {doc_index} "
            f"[{type(e).__name__}]: {e}. Defaulting to 'ambiguous'."
        )
        logger.exception(f"CRAG grading failed for doc {doc_index}")
        return (doc_index, "ambiguous")


async def _grade_retrieval_quality(
    query: str,
    retrieved_chunks: List[RetrievedChunkData],
    session_config: "RAGConfig",
) -> Tuple[str, List[RetrievedChunkData], List[str]]:
    """Grade retrieval quality using Corrective RAG (CRAG).

    Uses the LLM to grade each retrieved document as "correct", "ambiguous",
    or "incorrect". Returns an overall grade and filtered results.

    Args:
        query: The user's search query
        retrieved_chunks: List of retrieved chunk data to grade
        session_config: Session-aware RAG configuration (for LLM settings)

    Returns:
        Tuple of:
            - overall_grade: "correct", "ambiguous", or "incorrect"
            - filtered_chunks: Chunks that are "correct" or "ambiguous" (excludes "incorrect")
            - per_chunk_grades: List of grades in same order as input chunks
    """
    if not retrieved_chunks:
        return ("incorrect", [], [])

    # Get LLM for grading (reuse refinement model settings for consistency)
    default_model = get_default_model_settings()
    max_tokens = default_model.get("max_tokens")
    model_settings = resolve_model_settings(
        model_ref=session_config.refinement_model,
        default_settings=default_model,
        max_tokens=max_tokens,
        context="crag_grading_model",
    )
    grading_llm = get_llm_provider().get_chat_model(model_settings, temperature=0.0)

    # Grade chunks in parallel, but bounded.
    #
    # This used to fan out one LLM call per chunk with no limit. On a large
    # retrieval that is ~100 simultaneous requests, which overruns typical
    # hosted-endpoint concurrency limits: some calls time out, _grade_single_document
    # catches the failure and falls back to "correct", and the CRAG decision is
    # then made from grades that were never actually computed. Capping the
    # fan-out keeps every grade a real answer.
    _grade_semaphore = asyncio.Semaphore(CRAG_GRADING_CONCURRENCY)

    async def _grade_bounded(doc_index: int, chunk):
        async with _grade_semaphore:
            return await _grade_single_document(
                llm=grading_llm,
                query=query,
                document_text=chunk.content,
                doc_index=doc_index,
            )

    grading_tasks = [_grade_bounded(i, chunk) for i, chunk in enumerate(retrieved_chunks)]
    grading_results = await asyncio.gather(*grading_tasks)

    # Sort by original index to maintain order
    grading_results.sort(key=lambda x: x[0])
    per_chunk_grades = [grade for _, grade in grading_results]

    # Count grades
    correct_count = per_chunk_grades.count("correct")
    ambiguous_count = per_chunk_grades.count("ambiguous")
    incorrect_count = per_chunk_grades.count("incorrect")
    total = len(per_chunk_grades)

    logger.info(
        f"[CRAG] Grading results: correct={correct_count}, ambiguous={ambiguous_count}, "
        f"incorrect={incorrect_count} (total={total})"
    )

    # Filter out incorrect chunks, keep correct and ambiguous
    filtered_chunks = [
        chunk for chunk, grade in zip(retrieved_chunks, per_chunk_grades) if grade != "incorrect"
    ]

    # Determine overall grade based on majority
    if incorrect_count > total / 2:
        overall_grade = "incorrect"
        # When majority are incorrect, return empty list to trigger fallback
        filtered_chunks = []
    elif correct_count > total / 2:
        overall_grade = "correct"
    else:
        overall_grade = "ambiguous"

    return (overall_grade, filtered_chunks, per_chunk_grades)


async def _crag_web_search_fallback(query: str, max_results: int = 5) -> List[RetrievedChunkData]:
    """Perform a web search fallback when CRAG determines retrieval is poor.

    Uses DuckDuckGo (same as online_rag) to find web results and converts
    them into RetrievedChunkData, the form the existing pipeline takes.

    Args:
        query: The search query to use for web search
        max_results: Maximum number of web results to fetch

    Returns:
        List of RetrievedChunkData constructed from web search results
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))

        if not results:
            logger.info("[CRAG] Web search fallback returned no results")
            return []

        web_chunks = []
        for i, result in enumerate(results):
            title = result.get("title", "Web Result")
            url = result.get("href", "")
            body = result.get("body", "")

            # Construct content combining title, URL, and body
            content = f"{title}\nSource: {url}\n\n{body}"

            chunk = RetrievedChunkData(
                chunk_id=f"web_{i}",
                document_id=f"web_search_{i}",
                document_name=f"Web: {title[:80]}",
                content=content,
                score=0.0,  # Web results don't have a vector similarity score
                chunk_index=i,
                page_number=None,
                section_header=title,
                metadata={
                    "source": "crag_web_fallback",
                    "url": url,
                    "title": title,
                },
            )
            web_chunks.append(chunk)

        logger.info(f"[CRAG] Web search fallback returned {len(web_chunks)} results")
        return web_chunks

    except Exception as e:
        logger.error(f"[CRAG] Web search fallback failed: {e}", exc_info=True)
        return []


@tool
async def mark_chunk_for_citation(result_number: int, config: RunnableConfig) -> str:
    """Mark a specific search result chunk to be used for citation.

    Use this tool to select which chunks from your search results should be included
    in the final context for answering the user's question. Only mark chunks that are
    directly relevant and useful for answering the question.

    Args:
        result_number: The result number from the search results (e.g., 1 for [1])

    Returns:
        Confirmation message
    """
    # Get configuration from RunnableConfig
    configurable = config.get("configurable", {})

    # Initialize marked_chunks list if not exists
    if "marked_chunks" not in configurable:
        configurable["marked_chunks"] = []

    # Add the result number to marked chunks (avoid duplicates)
    if result_number not in configurable["marked_chunks"]:
        configurable["marked_chunks"].append(result_number)
        logger.info(f"Agent marked chunk {result_number} for citation")
        return f"Successfully marked Result {result_number} for citation. Total marked: {len(configurable['marked_chunks'])}"
    else:
        return f"Result {result_number} was already marked for citation."


@tool
async def search_knowledge_base(query: str, config: RunnableConfig) -> str:
    """Search the knowledge base for relevant information.

    Use this tool to search through the user's uploaded documents.
    You can refine your search by using different query terms.

    Args:
        query: The search query - can be a refined/specific search term

    Returns:
        Formatted search results with content and source information
    """
    # Import here to avoid circular imports
    from lasagna.api.rag_backend import RAGBackend

    # Get configuration from RunnableConfig
    configurable = config.get("configurable", {})
    session_id = configurable.get("session_id")
    if not session_id:
        logger.warning("[AgenticRAG] No session_id in configurable: falling back to 'default'")
        session_id = "default"
    document_filter = configurable.get("document_filter")
    top_k = configurable.get("top_k", 10)
    trace_id = configurable.get("trace_id")

    # Initialize hop tracking
    if "hop_data" not in configurable:
        configurable["hop_data"] = []
    if "previous_queries" not in configurable:
        configurable["previous_queries"] = []

    hop_index = len(configurable["hop_data"])
    search_start = time.time()
    use_hyde = False
    hyde_reason = "not_evaluated"

    # Get global citation index for this hop
    if "global_citation_index" not in configurable:
        configurable["global_citation_index"] = 1
    if "all_hop_citations" not in configurable:
        configurable["all_hop_citations"] = []
    if "seen_content_hashes" not in configurable:
        configurable["seen_content_hashes"] = set()

    citation_start_index = configurable["global_citation_index"]

    # Check for repeated or semantically similar query (stopping condition)
    # First check exact match
    if query in configurable["previous_queries"]:
        logger.debug(f"Query stabilized (exact repeat): '{query[:50]}...'")
        configurable["stop_reason"] = "query_stabilized"
        return "You've already searched with this exact query. Please synthesize the results you have or try a different search approach."

    # Check for semantically similar queries (if we have previous queries and embeddings)
    similar_query = _find_semantically_similar_query(
        query,
        configurable.get("previous_queries", []),
        configurable.get("previous_query_embeddings", []),
        similarity_threshold=0.92,  # High threshold to avoid false positives
    )
    if similar_query:
        logger.debug(f"Query stabilized (semantically similar to: '{similar_query[:50]}...')")
        configurable["stop_reason"] = "query_stabilized"
        return "Your query is very similar to a previous search. Please synthesize the results you have or try a significantly different search approach."

    configurable["previous_queries"].append(query)

    # Store embedding for semantic deduplication in future hops
    try:
        from lasagna.infrastructure.embedding.manager import get_embedding_manager

        embedding_manager = get_embedding_manager()
        query_embedding = embedding_manager.encode_query(query)
        if "previous_query_embeddings" not in configurable:
            configurable["previous_query_embeddings"] = []
        configurable["previous_query_embeddings"].append(query_embedding)
    except Exception as e:
        logger.debug(f"Could not store query embedding for deduplication: {e}")
        if "previous_query_embeddings" not in configurable:
            configurable["previous_query_embeddings"] = []
        configurable["previous_query_embeddings"].append(None)

    # The query's text at DEBUG only: server logs are kept and read by others.
    logger.info(f"Agentic RAG hop {hop_index}: session={session_id}")
    logger.debug(f"Agentic RAG hop {hop_index}: query='{query[:50]}...'")

    try:
        backend = RAGBackend()
        session_config = rag_config.with_session(session_id)
        use_hyde, hyde_reason = _should_use_hyde_for_agentic(
            query=query,
            hop_index=hop_index,
            hop_data_list=configurable.get("hop_data", []),
            session_config=session_config,
        )
        logger.debug(
            f"[Agentic RAG] HyDE decision: hop={hop_index}, use_hyde={use_hyde}, "
            f"reason={hyde_reason}, query_tokens={estimate_tokens(query)}"
        )
        results = backend.search(
            query=query,
            session_id=session_id,
            top_k=top_k,
            # Use backend default (-100) to allow negative cross-encoder logits.
            # Raw rerank scores are unbounded and often negative; filtering
            # should use normalized_score, not raw rerank_score.
            document_ids=document_filter,
            use_hyde=use_hyde,
            trace_id=trace_id,
            _from_routing=True,  # Prevent COMPREHENSIVE recursion
        )

        search_latency_ms = (time.time() - search_start) * 1000

        # Coverage scoring for intelligent stopping condition
        # Check how well the retrieved chunks cover the query aspects
        coverage_score = None
        try:
            scorer = get_coverage_scorer()
            chunks_for_scoring = [
                {"content": r.chunk.content, "chunk_id": r.chunk.chunk_id} for r in results
            ]
            coverage_score = scorer.score_coverage(query, chunks_for_scoring)

            # Log coverage metrics
            logger.debug(
                f"[Agentic RAG] Coverage score: ratio={coverage_score.coverage_ratio:.2f}, "
                f"confidence={coverage_score.confidence:.2f}, "
                f"uncovered={len(coverage_score.uncovered_aspects)}"
            )

            # Store coverage in configurable for later analysis
            if "coverage_scores" not in configurable:
                configurable["coverage_scores"] = []
            configurable["coverage_scores"].append(
                {
                    "hop_index": hop_index,
                    "coverage_ratio": coverage_score.coverage_ratio,
                    "confidence": coverage_score.confidence,
                    "uncovered_aspects": coverage_score.uncovered_aspects,
                }
            )

            # Check if we have high enough coverage to suggest stopping
            if coverage_score.coverage_ratio >= 0.8 and coverage_score.confidence >= 0.7:
                configurable["coverage_sufficient"] = True
                logger.info(
                    f"[Agentic RAG] High coverage achieved (ratio={coverage_score.coverage_ratio:.2f}, "
                    f"confidence={coverage_score.confidence:.2f})"
                )
        except Exception as e:
            logger.debug(f"Coverage scoring failed (non-critical): {e}")

        if not results:
            # Record empty hop
            hop_data = HopData(
                hop_index=hop_index,
                query_used=query,
                retrieved_chunks=[],
                search_latency_ms=search_latency_ms,
                hyde_used=use_hyde,
                hyde_reason=hyde_reason,
            )
            configurable["hop_data"].append(hop_data)
            return "No relevant documents found for this query. Try rephrasing or using different search terms."

        # Get chunk refinement method from session-aware config
        refinement_method = session_config.chunk_refinement_method

        # Build retrieved chunk data with full metadata including page numbers
        retrieved_chunks: List[RetrievedChunkData] = []
        for i, result in enumerate(results):
            page_num = None
            if result.chunk.metadata:
                page_num = result.chunk.metadata.get("page_number")
            # Also try direct attribute
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
                section_header=result.chunk.metadata.get("section_header")
                if result.chunk.metadata
                else None,
                metadata=result.chunk.metadata or {},
            )
            retrieved_chunks.append(chunk_data)

        # =====================================================================
        # CORRECTIVE RAG (CRAG) — Grade retrieval quality post-search
        # =====================================================================
        crag_grade = ""
        crag_grades_detail: List[str] = []
        crag_web_fallback_used = False
        crag_grading_latency_ms = 0.0

        if session_config.enable_crag and retrieved_chunks:
            crag_start = time.time()
            try:
                crag_grade, filtered_chunks, crag_grades_detail = await _grade_retrieval_quality(
                    query=query,
                    retrieved_chunks=retrieved_chunks,
                    session_config=session_config,
                )
                crag_grading_latency_ms = (time.time() - crag_start) * 1000

                logger.info(
                    f"[CRAG] Hop {hop_index}: overall_grade={crag_grade}, "
                    f"filtered={len(filtered_chunks)}/{len(retrieved_chunks)}, "
                    f"latency={crag_grading_latency_ms:.1f}ms"
                )

                # Track CRAG grading across hops
                if "crag_grades" not in configurable:
                    configurable["crag_grades"] = []
                configurable["crag_grades"].append(crag_grade)
                if "crag_total_grading_latency_ms" not in configurable:
                    configurable["crag_total_grading_latency_ms"] = 0.0
                configurable["crag_total_grading_latency_ms"] += crag_grading_latency_ms

                if crag_grade == "incorrect":
                    # All or most documents are irrelevant
                    if session_config.crag_web_search_fallback:
                        logger.info(f"[CRAG] Hop {hop_index}: Triggering web search fallback")
                        web_chunks = await _crag_web_search_fallback(query, max_results=5)
                        if web_chunks:
                            retrieved_chunks = web_chunks
                            crag_web_fallback_used = True
                            if "crag_web_fallback_triggered" not in configurable:
                                configurable["crag_web_fallback_triggered"] = False
                            configurable["crag_web_fallback_triggered"] = True
                        else:
                            # Web search also failed — keep original chunks as last resort
                            logger.warning(
                                f"[CRAG] Hop {hop_index}: Web fallback returned no results, "
                                "keeping original chunks"
                            )
                    else:
                        logger.info(
                            f"[CRAG] Hop {hop_index}: Retrieval quality is 'incorrect' but "
                            "web fallback is disabled. Keeping original chunks."
                        )
                elif crag_grade == "ambiguous" and filtered_chunks:
                    # Mixed quality — use only the correct/ambiguous chunks
                    retrieved_chunks = filtered_chunks
                elif crag_grade == "correct" and filtered_chunks:
                    # Good quality — use only correct/ambiguous chunks (drops any stray incorrects)
                    retrieved_chunks = filtered_chunks

            except Exception as e:
                crag_grading_latency_ms = (time.time() - crag_start) * 1000
                logger.warning(
                    f"[CRAG] Grading failed for hop {hop_index}: {e}. "
                    "Proceeding with ungraded results.",
                    exc_info=True,
                )

        # =====================================================================
        # VANILLA SEARCH (original implementation without refinement)
        # =====================================================================
        if refinement_method == "vanilla":
            formatted_results = []
            hop_citations = []  # Track citations created in this hop
            seen_content_hashes = configurable["seen_content_hashes"]
            current_idx = citation_start_index
            duplicates_skipped = 0

            for chunk in retrieved_chunks:
                # Deduplicate by content hash to avoid showing identical citations
                content_hash = _generate_content_hash(chunk.content)
                if content_hash in seen_content_hashes:
                    duplicates_skipped += 1
                    continue
                seen_content_hashes.add(content_hash)

                # Use global index for citation numbering (only for non-duplicates)
                global_idx = current_idx
                current_idx += 1

                page_info = f", page {chunk.page_number}" if chunk.page_number else ""
                line_info = ""
                if not chunk.page_number and chunk.metadata.get("line_start"):
                    ls = chunk.metadata["line_start"]
                    le = chunk.metadata.get("line_end")
                    line_info = f", lines {ls}-{le}" if le and le != ls else f", line {ls}"
                # Use clean [N] format with GLOBAL index - no "Result" prefix
                chunk_info = (
                    f"[{global_idx}] {chunk.document_name}{page_info}{line_info}\n{chunk.content}\n"
                )
                formatted_results.append(chunk_info)

                # Build citation metadata for this hop with highlight extraction
                highlight_ranges = (
                    extract_highlight_ranges(chunk.content, query) if chunk.content else None
                )
                hop_citations.append(
                    {
                        "index": global_idx,
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "chunk_id": chunk.chunk_id,
                        "content_preview": chunk.content[:200],
                        "score": chunk.score,
                        "chunk_index": chunk.chunk_index,
                        "page_number": chunk.page_number,
                        "hop_index": hop_index,
                        "highlight_ranges": highlight_ranges,
                        "line_start": chunk.metadata.get("line_start"),
                        "line_end": chunk.metadata.get("line_end"),
                        "auto_merged": chunk.metadata.get("auto_merged", False),
                    }
                )

            if duplicates_skipped > 0:
                logger.info(
                    f"[Agentic RAG] Hop {hop_index}: skipped {duplicates_skipped} duplicate chunk(s)"
                )

            # Update global citation index for next hop
            configurable["global_citation_index"] = current_idx

            # Accumulate citations across hops
            configurable["all_hop_citations"].extend(hop_citations)

            # Record hop data (vanilla path)
            hop_data = HopData(
                hop_index=hop_index,
                query_used=query,
                retrieved_chunks=retrieved_chunks,
                search_latency_ms=search_latency_ms,
                hyde_used=use_hyde,
                hyde_reason=hyde_reason,
                crag_grade=crag_grade,
                crag_grades_detail=crag_grades_detail,
                crag_web_fallback_used=crag_web_fallback_used,
            )
            configurable["hop_data"].append(hop_data)

            return "\n---\n".join(formatted_results)
        # =====================================================================
        # END VANILLA SEARCH
        # =====================================================================

        # =====================================================================
        # REFINED SEARCH (with LLM-based chunk refinement)
        # =====================================================================
        elif refinement_method == "refined" or refinement_method == "auto":
            refinement_start = time.time()

            # For 'auto', decide whether to refine based on chunk quality
            # Uses rank/quantile-based conditions to handle different scoring schemes
            should_refine = refinement_method == "refined"
            skip_reason = ""
            if refinement_method == "auto":
                should_refine, skip_reason = _should_auto_refine(
                    chunks=retrieved_chunks,
                    length_threshold=session_config.auto_refine_avg_length_threshold,
                    chunk_count_threshold=session_config.auto_refine_chunk_count_threshold,
                    score_variance_threshold=0.3,  # High variance = uncertain relevance
                )

            if not should_refine:
                # Skip refinement, format as vanilla
                # NOTE: This is why tokens_saved=0 when refinement is skipped
                logger.info(f"[Agentic RAG] Hop {hop_index}: refinement skipped ({skip_reason})")

                formatted_results = []
                hop_citations = []  # Track citations created in this hop
                seen_content_hashes = configurable["seen_content_hashes"]
                current_idx = citation_start_index
                duplicates_skipped = 0

                for chunk in retrieved_chunks:
                    # Deduplicate by content hash to avoid showing identical citations
                    content_hash = _generate_content_hash(chunk.content)
                    if content_hash in seen_content_hashes:
                        duplicates_skipped += 1
                        continue
                    seen_content_hashes.add(content_hash)

                    # Use global index for citation numbering (only for non-duplicates)
                    global_idx = current_idx
                    current_idx += 1

                    page_info = f", page {chunk.page_number}" if chunk.page_number else ""
                    line_info = ""
                    if not chunk.page_number and chunk.metadata.get("line_start"):
                        ls = chunk.metadata["line_start"]
                        le = chunk.metadata.get("line_end")
                        line_info = f", lines {ls}-{le}" if le and le != ls else f", line {ls}"
                    # Use clean [N] format with GLOBAL index
                    chunk_info = (
                        f"[{global_idx}] {chunk.document_name}{page_info}{line_info}\n"
                        f"{chunk.content}\n"
                    )
                    formatted_results.append(chunk_info)

                    # Build citation metadata for this hop with highlight extraction
                    highlight_ranges = (
                        extract_highlight_ranges(chunk.content, query) if chunk.content else None
                    )
                    hop_citations.append(
                        {
                            "index": global_idx,
                            "document_id": chunk.document_id,
                            "document_name": chunk.document_name,
                            "chunk_id": chunk.chunk_id,
                            "content_preview": chunk.content[:200],
                            "score": chunk.score,
                            "chunk_index": chunk.chunk_index,
                            "page_number": chunk.page_number,
                            "hop_index": hop_index,
                            "highlight_ranges": highlight_ranges,
                            "line_start": chunk.metadata.get("line_start"),
                            "line_end": chunk.metadata.get("line_end"),
                            "auto_merged": chunk.metadata.get("auto_merged", False),
                        }
                    )

                if duplicates_skipped > 0:
                    logger.info(
                        f"[Agentic RAG] Hop {hop_index}: skipped {duplicates_skipped} duplicate chunk(s)"
                    )

                # Update global citation index for next hop
                configurable["global_citation_index"] = current_idx

                # Accumulate citations across hops
                configurable["all_hop_citations"].extend(hop_citations)

                hop_data = HopData(
                    hop_index=hop_index,
                    query_used=query,
                    retrieved_chunks=retrieved_chunks,
                    search_latency_ms=search_latency_ms,
                    refinement_skipped=True,
                    refinement_skip_reason=skip_reason,
                    hyde_used=use_hyde,
                    hyde_reason=hyde_reason,
                    crag_grade=crag_grade,
                    crag_grades_detail=crag_grades_detail,
                    crag_web_fallback_used=crag_web_fallback_used,
                )
                configurable["hop_data"].append(hop_data)

                return "\n---\n".join(formatted_results)

            # Get LLM for refinement
            default_model = get_default_model_settings()
            max_tokens = default_model.get("max_tokens")
            model_settings = resolve_model_settings(
                model_ref=session_config.refinement_model,
                default_settings=default_model,
                max_tokens=max_tokens,
                context="refinement_model",
            )
            refinement_llm = get_llm_provider().get_chat_model(model_settings)

            # Calculate tokens before refinement
            tokens_before = sum(estimate_tokens(c.content) for c in retrieved_chunks)

            # Prepare refinement tasks for parallel execution
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

            # Execute all refinements in parallel
            refinement_results = await asyncio.gather(*refinement_tasks)
            refinement_latency_ms = (time.time() - refinement_start) * 1000

            # Process refinement results and build formatted output
            formatted_results = []
            hop_citations = []  # Track citations created in this hop
            chunks_filtered = 0
            tokens_after = 0
            refinement_failures = 0
            current_global_idx = citation_start_index
            seen_content_hashes = configurable["seen_content_hashes"]
            duplicates_skipped = 0

            for chunk_idx, refined_content, original_content in refinement_results:
                chunk = retrieved_chunks[chunk_idx]

                if refined_content is None:
                    # Chunk was marked as NOT_RELEVANT
                    chunks_filtered += 1
                    chunk.refinement_kept = False
                    continue

                # Deduplicate by content hash to avoid showing identical citations
                content_hash = _generate_content_hash(refined_content)
                if content_hash in seen_content_hashes:
                    duplicates_skipped += 1
                    chunk.refinement_kept = False
                    continue
                seen_content_hashes.add(content_hash)

                # Track if refinement preserved content
                if refined_content == original_content:
                    refinement_failures += 1

                # Use refined content but KEEP original for citations
                chunk.original_content = chunk.content
                chunk.content = refined_content
                chunk.is_refined = True

                tokens_after += estimate_tokens(refined_content)

                page_info = f", page {chunk.page_number}" if chunk.page_number else ""
                line_info = ""
                if not chunk.page_number and chunk.metadata.get("line_start"):
                    ls = chunk.metadata["line_start"]
                    le = chunk.metadata.get("line_end")
                    line_info = f", lines {ls}-{le}" if le and le != ls else f", line {ls}"
                # Use clean [N] format with GLOBAL index
                chunk_info = (
                    f"[{current_global_idx}] {chunk.document_name}{page_info}{line_info}\n"
                    f"{refined_content}\n"
                )
                formatted_results.append(chunk_info)

                # Build citation metadata for this hop with highlight extraction
                highlight_ranges = (
                    extract_highlight_ranges(refined_content, query) if refined_content else None
                )
                hop_citations.append(
                    {
                        "index": current_global_idx,
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "chunk_id": chunk.chunk_id,
                        "content_preview": refined_content[:200],
                        "score": chunk.score,
                        "chunk_index": chunk.chunk_index,
                        "page_number": chunk.page_number,
                        "hop_index": hop_index,
                        "highlight_ranges": highlight_ranges,
                        "line_start": chunk.metadata.get("line_start"),
                        "line_end": chunk.metadata.get("line_end"),
                        "auto_merged": chunk.metadata.get("auto_merged", False),
                    }
                )
                current_global_idx += 1

            if duplicates_skipped > 0:
                logger.info(
                    f"[Agentic RAG] Hop {hop_index}: skipped {duplicates_skipped} duplicate chunk(s)"
                )

            # Fallback: if all chunks were filtered, return the top original chunk
            if not formatted_results and retrieved_chunks:
                logger.warning(
                    "All chunks filtered as NOT_RELEVANT. Using top original chunk as fallback."
                )
                top_chunk = retrieved_chunks[0]
                # Check if fallback is also a duplicate
                fallback_hash = _generate_content_hash(top_chunk.content)
                if fallback_hash not in seen_content_hashes:
                    seen_content_hashes.add(fallback_hash)
                    top_chunk.refinement_kept = True
                    page_info = f", page {top_chunk.page_number}" if top_chunk.page_number else ""
                    # Use clean [N] format with global index
                    chunk_info = (
                        f"[{citation_start_index}] {top_chunk.document_name}{page_info}\n"
                        f"{top_chunk.content}\n"
                    )
                    formatted_results.append(chunk_info)
                    tokens_after = estimate_tokens(top_chunk.content)
                    chunks_filtered = len(retrieved_chunks) - 1
                    current_global_idx = citation_start_index + 1

                    # Add fallback citation with highlight extraction
                    highlight_ranges = (
                        extract_highlight_ranges(top_chunk.content, query)
                        if top_chunk.content
                        else None
                    )
                    hop_citations.append(
                        {
                            "index": citation_start_index,
                            "document_id": top_chunk.document_id,
                            "document_name": top_chunk.document_name,
                            "chunk_id": top_chunk.chunk_id,
                            "content_preview": top_chunk.content[:200],
                            "score": top_chunk.score,
                            "chunk_index": top_chunk.chunk_index,
                            "page_number": top_chunk.page_number,
                            "hop_index": hop_index,
                            "highlight_ranges": highlight_ranges,
                            "line_start": top_chunk.metadata.get("line_start"),
                            "line_end": top_chunk.metadata.get("line_end"),
                            "auto_merged": top_chunk.metadata.get("auto_merged", False),
                        }
                    )
                else:
                    logger.info(
                        f"[Agentic RAG] Hop {hop_index}: fallback chunk is also a duplicate, skipping"
                    )

            # Update global citation index for next hop
            configurable["global_citation_index"] = current_global_idx

            # Accumulate citations across hops
            configurable["all_hop_citations"].extend(hop_citations)

            # Log token reduction
            reduction_pct = (
                ((tokens_before - tokens_after) / tokens_before * 100) if tokens_before > 0 else 0
            )
            logger.info(
                f"[Agentic RAG] Hop {hop_index} refinement: {len(retrieved_chunks)} chunks, "
                f"{tokens_before} → {tokens_after} tokens ({reduction_pct:.1f}% reduction), "
                f"{chunks_filtered} chunk(s) filtered, citations {citation_start_index}-{current_global_idx - 1}"
            )

            # Record hop data with refinement info
            hop_data = HopData(
                hop_index=hop_index,
                query_used=query,
                retrieved_chunks=retrieved_chunks,
                search_latency_ms=search_latency_ms,
                refinement_latency_ms=refinement_latency_ms,
                tokens_before_refinement=tokens_before,
                tokens_after_refinement=tokens_after,
                chunks_filtered=chunks_filtered,
                hyde_used=use_hyde,
                hyde_reason=hyde_reason,
                crag_grade=crag_grade,
                crag_grades_detail=crag_grades_detail,
                crag_web_fallback_used=crag_web_fallback_used,
            )
            configurable["hop_data"].append(hop_data)

            # Store metrics in config for later retrieval
            if "rag_metrics" not in configurable:
                configurable["rag_metrics"] = []
            configurable["rag_metrics"].append(
                {
                    "hop_index": hop_index,
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "chunks_filtered": chunks_filtered,
                    "reduction_pct": reduction_pct,
                    "refinement_failures": refinement_failures,
                }
            )

            return "\n---\n".join(formatted_results)
        # =====================================================================
        # END REFINED SEARCH
        # =====================================================================
        else:
            # Fallback to vanilla if unknown method
            logger.warning(f"Unknown chunk_refinement_method: {refinement_method}. Using vanilla.")
            formatted_results = []
            hop_citations = []  # Track citations created in this hop
            seen_content_hashes = configurable["seen_content_hashes"]
            current_idx = citation_start_index
            duplicates_skipped = 0

            for chunk in retrieved_chunks:
                # Deduplicate by content hash to avoid showing identical citations
                content_hash = _generate_content_hash(chunk.content)
                if content_hash in seen_content_hashes:
                    duplicates_skipped += 1
                    continue
                seen_content_hashes.add(content_hash)

                # Use global index for citation numbering (only for non-duplicates)
                global_idx = current_idx
                current_idx += 1

                page_info = f", page {chunk.page_number}" if chunk.page_number else ""
                line_info = ""
                if not chunk.page_number and chunk.metadata.get("line_start"):
                    ls = chunk.metadata["line_start"]
                    le = chunk.metadata.get("line_end")
                    line_info = f", lines {ls}-{le}" if le and le != ls else f", line {ls}"
                # Use clean [N] format with GLOBAL index
                chunk_info = (
                    f"[{global_idx}] {chunk.document_name}{page_info}{line_info}\n{chunk.content}\n"
                )
                formatted_results.append(chunk_info)

                # Build citation metadata for this hop with highlight extraction (fallback path)
                highlight_ranges = (
                    extract_highlight_ranges(chunk.content, query) if chunk.content else None
                )
                hop_citations.append(
                    {
                        "index": global_idx,
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "chunk_id": chunk.chunk_id,
                        "content_preview": chunk.content[:200],
                        "score": chunk.score,
                        "chunk_index": chunk.chunk_index,
                        "page_number": chunk.page_number,
                        "hop_index": hop_index,
                        "highlight_ranges": highlight_ranges,
                        "line_start": chunk.metadata.get("line_start"),
                        "line_end": chunk.metadata.get("line_end"),
                        "auto_merged": chunk.metadata.get("auto_merged", False),
                    }
                )

            if duplicates_skipped > 0:
                logger.info(
                    f"[Agentic RAG] Hop {hop_index}: skipped {duplicates_skipped} duplicate chunk(s)"
                )

            # Update global citation index for next hop
            configurable["global_citation_index"] = current_idx

            # Accumulate citations across hops
            configurable["all_hop_citations"].extend(hop_citations)

            hop_data = HopData(
                hop_index=hop_index,
                query_used=query,
                retrieved_chunks=retrieved_chunks,
                search_latency_ms=search_latency_ms,
                hyde_used=use_hyde,
                hyde_reason=hyde_reason,
                crag_grade=crag_grade,
                crag_grades_detail=crag_grades_detail,
                crag_web_fallback_used=crag_web_fallback_used,
            )
            configurable["hop_data"].append(hop_data)

            return "\n---\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        # Record failed hop
        if "hop_data" in configurable:
            hop_data = HopData(
                hop_index=hop_index,
                query_used=query,
                retrieved_chunks=[],
                search_latency_ms=(time.time() - search_start) * 1000,
                hyde_used=use_hyde,
                hyde_reason=hyde_reason,
            )
            configurable["hop_data"].append(hop_data)
        return f"Search failed: {str(e)}"


@dataclass
class AgenticRAGResult:
    """Result from agentic RAG retrieval."""

    formatted_context: str
    chunks: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    agent_summary: str
    search_count: int
    # Token reduction metrics
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    reduction_percentage: float = 0.0
    chunks_filtered: int = 0
    # Multi-hop tracking
    hops: List[HopData] = field(default_factory=list)
    metrics: Optional[AgenticRAGMetrics] = None

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary for logging/API responses."""
        if self.metrics:
            return self.metrics.to_dict()
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "reduction_percentage": self.reduction_percentage,
            "search_count": self.search_count,
            "chunks_filtered": self.chunks_filtered,
        }


class AgenticRAG:
    """ReAct agent for intelligent RAG retrieval.

    Supports two execution modes:
      - **Legacy ReAct** (default): flat agent with a monolithic search tool.
      - **Structured Graph**: explicit LangGraph StateGraph with quality gates,
        CRAG routing, and query decomposition.  Enabled via
        ``session_config.use_structured_graph``.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_iterations: int = 3,
        enable_final_search: bool = True,
        session_config: Any = None,
        _pipeline_override: Any = None,
        _node_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the AgenticRAG.

        Args:
            llm: Language model to use for the agent
            max_iterations: Maximum number of search iterations
            enable_final_search: Whether to perform final comprehensive search for citations (default: True)
            session_config: Session-scoped RAG config. When provided, takes
                precedence over the global ``rag_config`` for feature flags
                like ``use_structured_graph``.
            _pipeline_override: A ``BaseAgenticPipeline`` instance to use
                instead of the built-in pipeline.  Takes highest precedence.
            _node_overrides: Dict mapping node names to callables.  Only
                applies when using the built-in structured graph pipeline.
        """
        self.llm = llm
        self.max_iterations = max_iterations
        self.enable_final_search = enable_final_search

        # Store the explicitly-provided session config (or None).  Keeping
        # None lets _retrieve_structured fall back to rag_config.with_session()
        # which applies per-session overrides.  We only use the global
        # rag_config here for the _use_structured_graph decision.
        self._session_config = session_config
        effective_config = session_config if session_config is not None else rag_config

        # ── Custom pipeline override (highest precedence) ────────────
        self._custom_pipeline = None
        if _pipeline_override is not None:
            self._custom_pipeline = _pipeline_override
            self._use_structured_graph = False  # irrelevant when using custom pipeline
            logger.info(
                f"AgenticRAG initialized with CUSTOM PIPELINE "
                f"'{getattr(_pipeline_override, 'name', type(_pipeline_override).__name__)}' "
                f"(max_iterations={max_iterations})"
            )
            return

        # ── Resolve pipeline from config ─────────────────────────────
        pipeline_name = getattr(effective_config, "agentic_pipeline", None)

        if pipeline_name and pipeline_name not in ("legacy_react",):
            # Custom or built-in named pipeline from registry
            try:
                from lasagna.core.registry import registry

                pipeline_cls = registry.get_class("agentic_pipeline", pipeline_name)
                self._custom_pipeline = pipeline_cls(
                    session_config=effective_config,
                    node_overrides=_node_overrides,
                )
                self._use_structured_graph = False
                logger.info(
                    f"AgenticRAG initialized with registered pipeline "
                    f"'{pipeline_name}' (max_iterations={max_iterations})"
                )
                return
            except KeyError:
                raise KeyError(
                    f"Agentic pipeline '{pipeline_name}' is not registered. "
                    f"Available pipelines can be listed via registry.list_pipelines()."
                )

        # ── Built-in paths ───────────────────────────────────────────
        self._use_structured_graph = getattr(effective_config, "use_structured_graph", False)

        if self._use_structured_graph:
            # ── Structured graph path ────────────────────────────────────
            from lasagna.agentic.graph_builder import build_rag_graph

            self.graph = build_rag_graph(
                session_config=effective_config,
                node_overrides=_node_overrides,
            )
            logger.info(
                f"AgenticRAG initialized with STRUCTURED GRAPH (max_iterations={max_iterations})"
            )
        else:
            # ── Legacy ReAct path ────────────────────────────────────────
            # Get chunk retrieval strategy from effective config
            chunk_cite_retrieval = effective_config.chunk_cite_retrieval
            strategy = chunk_cite_retrieval.get("strategy", "fixed")

            # Base tools always include search
            self.tools = [search_knowledge_base]

            # Add mark_chunk_for_citation tool only for agent_decides strategy
            if strategy == "agent_decides":
                self.tools.append(mark_chunk_for_citation)
                logger.info(
                    "AgenticRAG initialized with 'agent_decides' strategy - mark_chunk_for_citation tool enabled"
                )

            # Create the system prompt with max_iterations and strategy-specific instructions
            strategy_instructions = (
                AGENT_DECIDES_INSTRUCTIONS if strategy == "agent_decides" else ""
            )
            system_prompt = AGENTIC_RAG_SYSTEM_PROMPT.format(
                max_iterations=max_iterations, strategy_instructions=strategy_instructions
            )
            self.graph = create_agent(
                llm,
                tools=self.tools,
                system_prompt=system_prompt,
            )

    async def retrieve(
        self,
        question: str,
        session_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        trace_id: Optional[str] = None,
        routing_decision: Any = None,
    ) -> AgenticRAGResult:
        """Perform agentic retrieval for a user question.

        Args:
            question: The user's question
            session_id: Session ID for document filtering
            document_ids: Optional list of specific document IDs to search
            top_k: Number of results per search
            trace_id: Optional trace ID for observability
            routing_decision: Pre-computed RoutingDecision from upstream
                orchestrator.  Passed through to the structured graph's
                ``plan_retrieval`` node so routing is NOT re-computed.

        Returns:
            AgenticRAGResult with formatted context, chunks, citations, and comprehensive metrics
        """
        # ── Custom pipeline path ─────────────────────────────────────────
        if self._custom_pipeline is not None:
            from lasagna.core.base_components import AgenticPipelineInput

            pipeline_input = AgenticPipelineInput(
                question=question,
                session_id=session_id,
                top_k=top_k,
                document_filter=document_ids,
                trace_id=trace_id,
                routing_decision=routing_decision,
                session_config=self._session_config,
                llm=self.llm,
            )
            return await self._custom_pipeline.run(pipeline_input)

        # ── Structured graph path ────────────────────────────────────────
        if self._use_structured_graph:
            return await self._retrieve_structured(
                question=question,
                session_id=session_id,
                document_ids=document_ids,
                top_k=top_k,
                trace_id=trace_id,
                routing_decision=routing_decision,
            )

        # ── Legacy ReAct path (unchanged) ────────────────────────────────
        # Import here to avoid circular imports
        from lasagna.api.rag_backend import RAGBackend

        total_start_time = time.time()

        inputs = {"messages": [("user", question)]}

        config = {
            "configurable": {
                "session_id": session_id,
                "document_filter": document_ids,
                "top_k": top_k,
                "trace_id": trace_id,
                "hop_data": [],  # Initialize hop tracking
                "previous_queries": [],  # Track queries for deduplication
                "global_citation_index": 1,  # Track citation index across hops
                "all_hop_citations": [],  # Accumulate citations from all hops
                "seen_content_hashes": set(),  # Track content hashes for deduplication
            }
        }

        # Track search iterations and collect tool responses
        search_count = 0
        all_chunks = []
        all_citations = []
        collected_search_results = []  # Store all search results from tool calls

        # Timing spans for latency breakdown
        agent_orchestration_start = time.time()

        try:
            # Run the agent with streaming
            final_message = None
            async for event in self.graph.astream(inputs, config=config, stream_mode="values"):
                messages = event.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    # Count tool calls
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        search_count += len(last_message.tool_calls)

                    # Collect tool responses (search results)
                    if hasattr(last_message, "type") and last_message.type == "tool":
                        # This is a tool response message
                        if hasattr(last_message, "content") and last_message.content:
                            collected_search_results.append(last_message.content)

                    final_message = last_message

            # Agent orchestration time = total LangGraph execution (includes LLM calls + tool routing)
            agent_orchestration_ms = (time.time() - agent_orchestration_start) * 1000

            # Extract agent's final summary
            agent_summary = ""
            if isinstance(final_message, AIMessage):
                agent_summary = message_text(final_message)

            # Get hop data from config
            hop_data_list: List[HopData] = config.get("configurable", {}).get("hop_data", [])

            # Aggregate token metrics from all search iterations
            rag_metrics = config.get("configurable", {}).get("rag_metrics", [])
            total_tokens_before = sum(m.get("tokens_before", 0) for m in rag_metrics)
            total_tokens_after = sum(m.get("tokens_after", 0) for m in rag_metrics)
            total_chunks_filtered = sum(m.get("chunks_filtered", 0) for m in rag_metrics)
            total_refinement_failures = sum(m.get("refinement_failures", 0) for m in rag_metrics)
            total_reduction_pct = (
                ((total_tokens_before - total_tokens_after) / total_tokens_before * 100)
                if total_tokens_before > 0
                else 0.0
            )

            # Compute unique docs and pages across all hops
            all_doc_ids = set()
            all_page_numbers = set()
            total_chunks_retrieved = 0
            for hop in hop_data_list:
                all_doc_ids.update(hop.unique_doc_ids)
                all_page_numbers.update(hop.unique_page_numbers)
                total_chunks_retrieved += len(hop.retrieved_chunks)

            # Minimum evidence thresholds for "sufficient_context"
            MIN_CHUNKS_FOR_SUFFICIENT = 2
            MIN_UNIQUE_DOCS_FOR_SUFFICIENT = 1

            # Use coverage scoring for more intelligent stop reason
            coverage_sufficient = config.get("configurable", {}).get("coverage_sufficient", False)
            coverage_scores = config.get("configurable", {}).get("coverage_scores", [])
            final_coverage_ratio = coverage_scores[-1]["coverage_ratio"] if coverage_scores else 0.0
            final_coverage_confidence = (
                coverage_scores[-1]["confidence"] if coverage_scores else 0.0
            )

            # Get stop reason with minimum evidence validation
            stop_reason = config.get("configurable", {}).get("stop_reason", "")
            if not stop_reason:
                if len(hop_data_list) >= self.max_iterations:
                    stop_reason = "max_iterations"
                elif coverage_sufficient:
                    # High coverage achieved - confident we have good results
                    stop_reason = "sufficient_coverage"
                    logger.info(
                        f"Stop reason: sufficient_coverage "
                        f"(ratio={final_coverage_ratio:.2f}, confidence={final_coverage_confidence:.2f})"
                    )
                elif (
                    total_chunks_retrieved >= MIN_CHUNKS_FOR_SUFFICIENT
                    and len(all_doc_ids) >= MIN_UNIQUE_DOCS_FOR_SUFFICIENT
                ):
                    # Only mark as sufficient_context if we have minimum evidence
                    stop_reason = "sufficient_context"
                else:
                    # Agent stopped but without minimum evidence - treat as early stop
                    stop_reason = "early_stop_insufficient_evidence"
                    logger.warning(
                        f"Agent stopped with insufficient evidence: "
                        f"{total_chunks_retrieved} chunks (min: {MIN_CHUNKS_FOR_SUFFICIENT}), "
                        f"{len(all_doc_ids)} docs (min: {MIN_UNIQUE_DOCS_FOR_SUFFICIENT}), "
                        f"coverage={final_coverage_ratio:.2f}"
                    )

            # Get session-aware config for UI overrides
            session_config = rag_config.with_session(session_id)

            # Extract marked chunks from config (only for agent_decides strategy)
            chunk_cite_retrieval = session_config.chunk_cite_retrieval
            strategy = chunk_cite_retrieval.get("strategy", "fixed")
            marked_chunks = []
            if strategy == "agent_decides":
                marked_chunks = config.get("configurable", {}).get("marked_chunks", [])
                if marked_chunks:
                    logger.info(
                        f"Agent marked {len(marked_chunks)} chunks for citation: {marked_chunks}"
                    )

            # Conditional: perform final search only if enabled
            # IMPORTANT: When final_search is enabled, citations come ONLY from
            # the final search call. Earlier hop citations are NOT merged.
            final_search_ms = 0.0
            context_formatting_ms = 0.0

            if self.enable_final_search:
                # Clear any accumulated citations from earlier hops
                # Citations must come exclusively from the final search
                all_chunks = []  # Reset - only final search chunks
                all_citations = []  # Reset - only final search citations

                # Now do a final comprehensive search to get structured results
                final_search_start = time.time()
                backend = RAGBackend()
                use_hyde_final, hyde_final_reason = _should_use_hyde_for_agentic(
                    query=question,
                    hop_index=len(hop_data_list),
                    hop_data_list=hop_data_list,
                    session_config=session_config,
                    is_final=True,
                )
                logger.debug(
                    f"[Agentic RAG] HyDE decision (final search): use_hyde={use_hyde_final}, "
                    f"reason={hyde_final_reason}, query_tokens={estimate_tokens(question)}"
                )
                final_results = backend.search(
                    query=question,
                    session_id=session_id,
                    top_k=top_k,
                    # Use backend default (-100) to allow negative cross-encoder logits
                    document_ids=document_ids,
                    use_hyde=use_hyde_final,
                    trace_id=trace_id,
                    _from_routing=True,  # Prevent COMPREHENSIVE recursion
                )
                final_search_ms = (time.time() - final_search_start) * 1000

                # Build chunks and citations ONLY from final results
                for i, result in enumerate(final_results):
                    page_num = None
                    if result.chunk.metadata:
                        page_num = result.chunk.metadata.get("page_number")
                    if page_num is None and hasattr(result.chunk, "page_number"):
                        page_num = result.chunk.page_number

                    chunk_data = {
                        "chunk_id": result.chunk.chunk_id,
                        "document_id": result.chunk.document_id,
                        "document_name": result.document.filename,
                        "content": result.chunk.content,
                        "score": result.score,
                        "chunk_index": result.chunk.chunk_index,
                        "page_number": page_num,
                    }
                    all_chunks.append(chunk_data)

                    # Extract highlight ranges for the citation
                    highlight_ranges = (
                        extract_highlight_ranges(result.chunk.content, question)
                        if result.chunk.content
                        else None
                    )

                    citation = {
                        "index": i + 1,
                        "document_id": result.document.document_id,
                        "document_name": result.document.filename,
                        "chunk_id": str(result.chunk.chunk_id),
                        "content_preview": result.chunk.content[:200],
                        "score": round(result.score, 3),
                        "chunk_index": result.chunk.chunk_index,
                        "page_number": page_num,
                    }
                    if highlight_ranges:
                        citation["highlight_ranges"] = highlight_ranges
                    all_citations.append(citation)

                # Format context for LLM with citations
                formatting_start = time.time()
                formatted_context = self._format_context(
                    final_results, all_citations, agent_summary
                )
                context_formatting_ms = (time.time() - formatting_start) * 1000
            else:
                # Skip final search - use accumulated hop citations directly
                formatting_start = time.time()

                # Get all hop citations accumulated during search
                all_hop_citations = config.get("configurable", {}).get("all_hop_citations", [])

                # Use accumulated citations instead of re-parsing search results
                if all_hop_citations:
                    formatted_context, all_citations, all_chunks = (
                        self._format_context_from_hop_citations(
                            agent_summary=agent_summary,
                            hop_citations=all_hop_citations,
                            marked_chunks=marked_chunks,
                            hop_data_list=hop_data_list,
                            session_config=session_config,
                        )
                    )
                else:
                    # Fallback to parsing search results (legacy behavior)
                    formatted_context, all_citations, all_chunks = (
                        self._format_context_without_citations(
                            agent_summary=agent_summary,
                            search_results=collected_search_results,
                            marked_chunks=marked_chunks,
                            hop_data_list=hop_data_list,
                            session_config=session_config,
                        )
                    )
                context_formatting_ms = (time.time() - formatting_start) * 1000

            # Calculate citation quality metrics
            citations_with_page = sum(1 for c in all_citations if c.get("page_number") is not None)
            citation_missing_page_rate = (
                1.0 - (citations_with_page / len(all_citations)) if all_citations else 0.0
            )

            # Calculate char metrics (stable, no tokenizer dependency)
            total_chars_before = 0
            total_chars_after = 0
            refinement_was_skipped = False
            refinement_skip_reasons = []
            for hop in hop_data_list:
                # Track refinement skip reasons
                if hop.refinement_skipped:
                    refinement_was_skipped = True
                    if hop.refinement_skip_reason:
                        refinement_skip_reasons.append(
                            f"hop{hop.hop_index}: {hop.refinement_skip_reason}"
                        )

                for chunk in hop.retrieved_chunks:
                    if chunk.original_content:
                        # Refined chunk - count original for "before", refined for "after"
                        total_chars_before += len(chunk.original_content)
                        if chunk.refinement_kept:
                            total_chars_after += len(chunk.content)
                    else:
                        # Non-refined chunk - count same for both
                        total_chars_before += len(chunk.content)
                        if chunk.refinement_kept:
                            total_chars_after += len(chunk.content)

            # Calculate latency breakdown
            total_latency_ms = (time.time() - total_start_time) * 1000

            # Sum of all hop search/refinement latencies
            total_hop_search_ms = sum(h.search_latency_ms for h in hop_data_list)
            total_hop_refinement_ms = sum(h.refinement_latency_ms for h in hop_data_list)

            # Agent LLM time = orchestration time minus search/refinement time
            # This approximates time spent in LLM calls (query planning, tool selection, synthesis)
            agent_llm_calls_ms = max(
                0, agent_orchestration_ms - total_hop_search_ms - total_hop_refinement_ms
            )

            # Accounted latency = all measured components
            accounted_latency_ms = agent_orchestration_ms + final_search_ms + context_formatting_ms

            # Unaccounted = total - accounted (should be small, represents Python overhead)
            unaccounted_latency_ms = max(0, total_latency_ms - accounted_latency_ms)

            # Build comprehensive metrics
            metrics = AgenticRAGMetrics(
                agentic_enabled=True,
                final_search_used=self.enable_final_search,
                agent_iterations_used=search_count,
                hop_count=len(hop_data_list),
                max_iterations_configured=self.max_iterations,
                # Latency breakdown
                total_latency_ms=total_latency_ms,
                hop_search_latencies_ms=[h.search_latency_ms for h in hop_data_list],
                hop_refinement_latencies_ms=[h.refinement_latency_ms for h in hop_data_list],
                agent_orchestration_ms=agent_orchestration_ms,
                agent_llm_calls_ms=agent_llm_calls_ms,
                final_search_ms=final_search_ms,
                context_formatting_ms=context_formatting_ms,
                accounted_latency_ms=accounted_latency_ms,
                unaccounted_latency_ms=unaccounted_latency_ms,
                # Token/char metrics
                total_tokens_before=total_tokens_before,
                total_tokens_after=total_tokens_after,
                total_tokens_saved=total_tokens_before - total_tokens_after,
                reduction_percentage=total_reduction_pct,
                total_chars_before=total_chars_before,
                total_chars_after=total_chars_after,
                # Refinement tracking
                total_chunks_filtered=total_chunks_filtered,
                refinement_failures_count=total_refinement_failures,
                refinement_skipped=refinement_was_skipped,
                refinement_skip_reason="; ".join(refinement_skip_reasons)
                if refinement_skip_reasons
                else "",
                # Coverage
                unique_docs_across_hops=len(all_doc_ids),
                unique_pages_across_hops=len(all_page_numbers),
                citation_count=len(all_citations),
                citation_missing_page_rate=citation_missing_page_rate,
                stop_reason=stop_reason,
                min_chunks_for_sufficient=MIN_CHUNKS_FOR_SUFFICIENT,
                min_unique_docs_for_sufficient=MIN_UNIQUE_DOCS_FOR_SUFFICIENT,
                # Quality boosters status (from session-aware config)
                hybrid_search_enabled=session_config.enable_hybrid_search,
                reranking_enabled=session_config.enable_reranking,
                mmr_enabled=session_config.enable_mmr,
                hyde_enabled=session_config.enable_hyde,
                # CRAG metrics (aggregated from hops)
                crag_enabled=session_config.enable_crag,
                crag_grades=[h.crag_grade for h in hop_data_list if h.crag_grade],
                crag_web_fallback_triggered=config.get("configurable", {}).get(
                    "crag_web_fallback_triggered", False
                ),
                crag_grading_latency_ms=config.get("configurable", {}).get(
                    "crag_total_grading_latency_ms", 0.0
                ),
            )

            # Log structured summary (wrapped to prevent logging failures from breaking retrieval)
            try:
                self._log_agentic_rag_summary(metrics, hop_data_list)
            except Exception:
                logger.exception("Failed to log agentic RAG summary (non-fatal)")

            return AgenticRAGResult(
                formatted_context=formatted_context,
                chunks=all_chunks,
                citations=all_citations,
                agent_summary=agent_summary,
                search_count=search_count,
                tokens_before=total_tokens_before,
                tokens_after=total_tokens_after,
                tokens_saved=total_tokens_before - total_tokens_after,
                reduction_percentage=total_reduction_pct,
                chunks_filtered=total_chunks_filtered,
                hops=hop_data_list,
                metrics=metrics,
            )

        except Exception:
            # logger.exception() captures the current exception and logs at
            # ERROR level without re-formatting the exception's string
            # representation (which can contain dict literals like {'error':...}
            # that loguru would mistake for named format fields).
            logger.exception("Agentic RAG failed")
            raise

    async def _retrieve_structured(
        self,
        question: str,
        session_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 5,
        trace_id: Optional[str] = None,
        routing_decision: Any = None,
    ) -> AgenticRAGResult:
        """Execute retrieval via the structured StateGraph.

        This method builds the initial state, invokes the compiled graph,
        and extracts an ``AgenticRAGResult`` from the final state — producing
        the same output shape as the legacy ReAct path.
        """
        from lasagna.agentic.graph_state import build_initial_state

        session_config = self._session_config or rag_config.with_session(session_id)
        total_start_time = time.time()

        initial_state = build_initial_state(
            question=question,
            session_id=session_id,
            top_k=top_k,
            session_config=session_config,
            llm=self.llm,
            document_filter=document_ids,
            trace_id=trace_id,
            routing_decision=routing_decision,
            max_retrieval_attempts=self.max_iterations,
            max_rewrites=self.max_iterations,
        )
        initial_state["total_start_time"] = total_start_time

        try:
            final_state = await self.graph.ainvoke(initial_state)

            # Extract outputs from graph state
            formatted_context = final_state.get("formatted_context", "")
            final_chunks = final_state.get("final_chunks", [])
            final_citations = final_state.get("final_citations", [])
            agent_summary = final_state.get("agent_summary", "")
            metrics = final_state.get("final_metrics")
            hop_data_list = final_state.get("hop_data", [])

            # Aggregate token metrics
            rag_metrics_list = final_state.get("rag_metrics_list", [])
            total_tokens_before = sum(m.get("tokens_before", 0) for m in rag_metrics_list)
            total_tokens_after = sum(m.get("tokens_after", 0) for m in rag_metrics_list)
            total_chunks_filtered = sum(m.get("chunks_filtered", 0) for m in rag_metrics_list)
            total_reduction_pct = (
                ((total_tokens_before - total_tokens_after) / total_tokens_before * 100)
                if total_tokens_before > 0
                else 0.0
            )

            search_count = final_state.get("hop_count", 0)

            # Log summary
            try:
                if metrics:
                    self._log_agentic_rag_summary(metrics, hop_data_list)
            except Exception:
                logger.exception("Failed to log structured RAG summary (non-fatal)")

            return AgenticRAGResult(
                formatted_context=formatted_context,
                chunks=final_chunks,
                citations=final_citations,
                agent_summary=agent_summary,
                search_count=search_count,
                tokens_before=total_tokens_before,
                tokens_after=total_tokens_after,
                tokens_saved=total_tokens_before - total_tokens_after,
                reduction_percentage=total_reduction_pct,
                chunks_filtered=total_chunks_filtered,
                hops=hop_data_list,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Structured RAG graph failed: {e}", exc_info=True)
            raise

    def _log_agentic_rag_summary(self, metrics: AgenticRAGMetrics, hop_data_list: List[HopData]):
        """Log a structured summary of the agentic RAG execution.

        PRIVACY NOTE: INFO-level logs use query_hash/query_len only.
        Raw query text is logged at DEBUG level only.

        Args:
            metrics: Comprehensive metrics object
            hop_data_list: List of hop data from the execution
        """
        # Build hop summary (privacy-safe: no raw queries at INFO level)
        hop_summaries = []
        for hop in hop_data_list:
            hop_summary = {
                "hop": hop.hop_index,
                **_safe_query_for_log(hop.query_used),  # query_hash, query_len only
                "chunks": len(hop.retrieved_chunks),
                "search_ms": round(hop.search_latency_ms, 1),
                "refine_ms": round(hop.refinement_latency_ms, 1)
                if hop.refinement_latency_ms
                else 0,
                "hyde_used": hop.hyde_used,
                "hyde_reason": hop.hyde_reason if hop.hyde_used else None,
            }
            hop_summaries.append(hop_summary)

        summary = {
            "event": "agentic_rag_complete",
            "latency_breakdown_ms": {
                "total": round(metrics.total_latency_ms, 1),
                "agent_orchestration": round(metrics.agent_orchestration_ms, 1),
                "agent_llm_calls": round(metrics.agent_llm_calls_ms, 1),
                "hop_searches": round(sum(metrics.hop_search_latencies_ms), 1),
                "hop_refinements": round(sum(metrics.hop_refinement_latencies_ms), 1),
                "final_search": round(metrics.final_search_ms, 1),
                "context_formatting": round(metrics.context_formatting_ms, 1),
                "accounted": round(metrics.accounted_latency_ms, 1),
                "unaccounted": round(metrics.unaccounted_latency_ms, 1),
            },
            "metrics": {
                "hop_count": metrics.hop_count,
                "tokens_saved": metrics.total_tokens_saved,
                "chars_before": metrics.total_chars_before,
                "chars_after": metrics.total_chars_after,
                "reduction_pct": round(metrics.reduction_percentage, 1),
                "refinement_skipped": metrics.refinement_skipped,
                "refinement_skip_reason": metrics.refinement_skip_reason
                if metrics.refinement_skipped
                else None,
                "citation_count": metrics.citation_count,
                "missing_page_rate": round(metrics.citation_missing_page_rate, 3),
                "stop_reason": metrics.stop_reason,
            },
            "quality_boosters": {
                "hybrid_search": metrics.hybrid_search_enabled,
                "reranking": metrics.reranking_enabled,
                "mmr": metrics.mmr_enabled,
                "hyde": metrics.hyde_enabled,
            },
            "hops": hop_summaries,
        }

        # INFO level: privacy-safe (no raw queries)
        logger.info(f"[Agentic RAG Summary] {json.dumps(summary, indent=None)}")

        # DEBUG level: include query previews for debugging
        # Let loguru handle filtering - don't inspect logger internals
        debug_hops = []
        for hop in hop_data_list:
            debug_hops.append(
                {
                    "hop": hop.hop_index,
                    "query_preview": hop.query_used[:80] + "..."
                    if len(hop.query_used) > 80
                    else hop.query_used,
                }
            )
        logger.debug(f"[Agentic RAG Debug] Queries: {json.dumps(debug_hops)}")

    def _format_context(
        self, results: List, citations: List[Dict[str, Any]], agent_summary: str
    ) -> str:
        """Format retrieved context for injection into LLM prompt.

        Args:
            results: Search results
            citations: Citation objects
            agent_summary: Agent's synthesis of findings

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        # Build content sections with simple numbering
        content_sections = []
        for i, result in enumerate(results, 1):
            section = f"[{i}]\n{result.chunk.content}"
            content_sections.append(section)

        content = "\n\n".join(content_sections)

        # NOTE: Agent analysis and citation instructions are intentionally EXCLUDED
        # from the knowledge_base block. Only grounded evidence (chunks) should be here.
        # Citation instructions belong in the system prompt, not mixed with evidence.
        # The agent_summary is returned separately in AgenticRAGResult for logging/debugging.

        return f"""<knowledge_base>
{content}
</knowledge_base>"""

    def _format_context_without_citations(
        self,
        agent_summary: str,
        search_results: List[str] = None,
        marked_chunks: Optional[List[int]] = None,
        hop_data_list: Optional[List[HopData]] = None,
        session_config: Optional[RAGConfig] = None,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Format context with citations, using agent summary and filtered search results.

        Unlike _format_context, this method doesn't perform a final search but uses the
        search results collected during agent execution, applying strategy-based filtering.
        Preserves page numbers and document metadata from hop data.

        Args:
            agent_summary: Agent synthesis of findings
            search_results: List of raw search results from tool calls
            marked_chunks: List of result numbers marked by agent (for agent_decides strategy)
            hop_data_list: List of HopData from multi-hop retrieval (for metadata)
            session_config: Session-aware RAG configuration (optional)

        Returns:
            Tuple of (formatted context string, list of citation dicts, list of chunk dicts with full content)
        """
        # Use session config if provided, otherwise fall back to global config
        config = session_config if session_config else rag_config
        if not agent_summary and not search_results:
            return (
                """<knowledge_base>
The agent searched the knowledge base but did not find specific relevant information.
</knowledge_base>""",
                [],
                [],
            )

        # Build a mapping from hop_data to get full metadata
        chunk_metadata_map: Dict[str, RetrievedChunkData] = {}
        if hop_data_list:
            for hop in hop_data_list:
                for chunk in hop.retrieved_chunks:
                    if chunk.refinement_kept:
                        # Key by content preview to match parsed results
                        key = chunk.content[:100]
                        chunk_metadata_map[key] = chunk

        # Parse search results to extract content and metadata
        # New format: "[N] filename.pdf, page X\ncontent..."
        parsed_results = []
        for result_text in search_results or []:
            # Extract result number (format: "[N]" at start of line)
            result_num_match = re.search(r"^\[(\d+)\]", result_text.strip())
            result_num = int(result_num_match.group(1)) if result_num_match else None

            # Extract header line (format: "[N] filename.pdf, page X")
            header_match = re.search(r"^\[(\d+)\]\s*([^\n]+)", result_text.strip())
            header_info = header_match.group(2).strip() if header_match else ""

            # Parse page number from header if present (format: ", page N")
            page_match = re.search(r",\s*page\s+(\d+)", header_info)
            page_number = int(page_match.group(1)) if page_match else None

            # Extract document name (before ", page")
            doc_name = re.sub(r",\s*page\s+\d+", "", header_info).strip()

            # Extract content (everything after the first line)
            lines = result_text.strip().split("\n", 1)
            content = lines[1].strip() if len(lines) > 1 else ""

            # Try to get full metadata from hop_data
            chunk_key = content[:100]
            full_chunk = chunk_metadata_map.get(chunk_key)

            # Score is no longer in output - get from hop_data if available
            score = full_chunk.score if full_chunk else 0.0

            parsed_results.append(
                {
                    "content": content,
                    "score": score,
                    "result_num": result_num,
                    "document_name": full_chunk.document_name if full_chunk else doc_name,
                    "page_number": full_chunk.page_number if full_chunk else page_number,
                    "chunk_id": full_chunk.chunk_id if full_chunk else 0,
                    "document_id": full_chunk.document_id if full_chunk else "",
                    "chunk_index": full_chunk.chunk_index if full_chunk else 0,
                }
            )

        # Get chunk retrieval strategy from config
        chunk_cite_retrieval = config.chunk_cite_retrieval
        strategy = chunk_cite_retrieval.get("strategy", "fixed")

        # Apply filtering strategy
        if strategy == "agent_decides":
            # Use agent-marked chunks if available (only for agent_decides strategy)
            if marked_chunks and len(marked_chunks) > 0:
                # Filter to only include chunks marked by the agent
                filtered_results = [r for r in parsed_results if r["result_num"] in marked_chunks]
                logger.info(
                    f"Using agent_decides strategy: {len(filtered_results)} chunks selected by agent"
                )
            else:
                # Fallback: if agent didn't mark any chunks, use all results up to max_chunks
                max_chunks = chunk_cite_retrieval.get("max_chunks", 5)
                filtered_results = parsed_results[:max_chunks]
                logger.warning(
                    f"agent_decides strategy but no chunks marked. Using top {len(filtered_results)} chunks as fallback"
                )
        elif strategy == "fixed":
            # Use fixed_size parameter
            fixed_size = chunk_cite_retrieval.get("fixed_size", 5)
            filtered_results = parsed_results[:fixed_size]
        elif strategy == "above_threshold":
            # Filter by threshold and limit by max_chunks
            threshold = chunk_cite_retrieval.get("threshold_score", 0.7)
            max_chunks = chunk_cite_retrieval.get("max_chunks", 5)

            # Filter results above threshold
            above_threshold = [r for r in parsed_results if r["score"] >= threshold]

            # Take minimum between max_chunks and filtered results
            filtered_results = above_threshold[:max_chunks]
        else:
            # Default fallback to first 5 results
            filtered_results = parsed_results[:5]

        if not filtered_results:
            return (
                """<knowledge_base>
The agent searched the knowledge base but did not find specific relevant information.
</knowledge_base>""",
                [],
                [],
            )

        # Build citations and chunks from filtered results (CRITICAL: preserve page numbers)
        citations = []
        chunks = []  # NEW: Build chunks with full content for frontend
        for i, r in enumerate(filtered_results):
            # Extract highlight ranges from content using any available query context
            highlight_ranges = (
                extract_highlight_ranges(r["content"], agent_summary)
                if r["content"] and agent_summary
                else None
            )
            citation_data = {
                "index": i + 1,
                "document_id": r["document_id"],
                "document_name": r["document_name"],
                "chunk_id": str(r["chunk_id"]),
                "content_preview": r["content"][:200],
                "score": round(r["score"], 3),
                "chunk_index": r["chunk_index"],
                "page_number": r["page_number"],  # PRESERVED from hop data
            }
            if highlight_ranges:
                citation_data["highlight_ranges"] = highlight_ranges
            citations.append(citation_data)

            # Build chunk with FULL content for frontend display
            chunk_data = {
                "chunk_id": str(r["chunk_id"]),
                "content": r["content"],  # FULL content, not just preview
                "chunk_index": r["chunk_index"],
            }
            chunks.append(chunk_data)

        # Build content sections with simple numbering
        content_sections = []
        for i, r in enumerate(filtered_results, 1):
            section = f"[{i}]\n{r['content']}"
            content_sections.append(section)

        content = "\n\n".join(content_sections)

        # NOTE: Agent analysis and citation instructions are intentionally EXCLUDED
        # from the knowledge_base block. Only grounded evidence (chunks) should be here.
        # Citation instructions belong in the system prompt, not mixed with evidence.
        # The agent_summary is returned separately in AgenticRAGResult for logging/debugging.

        formatted_context = f"""<knowledge_base>
{content}
</knowledge_base>"""

        return formatted_context, citations, chunks

    def _format_context_from_hop_citations(
        self,
        agent_summary: str,
        hop_citations: List[Dict[str, Any]],
        marked_chunks: Optional[List[int]] = None,
        hop_data_list: Optional[List[HopData]] = None,
        session_config: Optional[RAGConfig] = None,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Format context using pre-accumulated hop citations with global indexing.

        This method uses citations that were accumulated during search_knowledge_base calls
        with proper global indexing, avoiding the need to re-parse search results.

        Args:
            agent_summary: Agent synthesis of findings
            hop_citations: List of citation dicts accumulated across all hops
            marked_chunks: List of result numbers marked by agent (for agent_decides strategy)
            hop_data_list: List of HopData from multi-hop retrieval (for content lookup)
            session_config: Session-aware RAG configuration (optional)

        Returns:
            Tuple of (formatted context string, list of citation dicts, list of chunk dicts with full content)
        """
        config = session_config if session_config else rag_config

        if not hop_citations:
            return (
                """<knowledge_base>
The agent searched the knowledge base but did not find specific relevant information.
</knowledge_base>""",
                [],
                [],
            )

        # Build a content lookup from hop_data for full content retrieval
        # Merged (parent) content takes priority over individual child content
        chunk_content_map: Dict[int, str] = {}  # chunk_id -> content
        merged_ids: set = set()
        if hop_data_list:
            for hop in hop_data_list:
                for chunk in hop.retrieved_chunks:
                    if chunk.refinement_kept:
                        if chunk.metadata.get("auto_merged"):
                            chunk_content_map[chunk.chunk_id] = chunk.content
                            merged_ids.add(chunk.chunk_id)
                        elif chunk.chunk_id not in merged_ids:
                            chunk_content_map[chunk.chunk_id] = chunk.content

        # Get chunk retrieval strategy from config
        chunk_cite_retrieval = config.chunk_cite_retrieval
        strategy = chunk_cite_retrieval.get("strategy", "fixed")

        # Apply filtering strategy based on the GLOBAL indices
        if strategy == "agent_decides":
            if marked_chunks and len(marked_chunks) > 0:
                # Filter to only include chunks marked by the agent
                filtered_citations = [c for c in hop_citations if c["index"] in marked_chunks]
                logger.info(
                    f"Using agent_decides strategy with global indexing: {len(filtered_citations)} chunks selected"
                )
            else:
                max_chunks = chunk_cite_retrieval.get("max_chunks", 5)
                filtered_citations = hop_citations[:max_chunks]
                logger.warning(
                    f"agent_decides strategy but no chunks marked. Using first {len(filtered_citations)} chunks"
                )
        elif strategy == "fixed":
            fixed_size = chunk_cite_retrieval.get("fixed_size", 5)
            filtered_citations = hop_citations[:fixed_size]
        elif strategy == "above_threshold":
            threshold = chunk_cite_retrieval.get("threshold_score", 0.7)
            max_chunks = chunk_cite_retrieval.get("max_chunks", 20)
            above_threshold = [c for c in hop_citations if c.get("score", 0) >= threshold]
            filtered_citations = above_threshold[:max_chunks]
        elif strategy == "adaptive":
            from lasagna.utils.normalization import adaptive_k_selection

            scores = [c.get("score", 0) for c in hop_citations]
            optimal_k = adaptive_k_selection(
                scores=scores,
                min_k=chunk_cite_retrieval.get("min_chunks", 2),
                max_k=chunk_cite_retrieval.get("max_chunks", 20),
                buffer=chunk_cite_retrieval.get("score_gap_buffer", 2),
                min_gap_ratio=chunk_cite_retrieval.get("min_gap_ratio", 0.15),
            )
            filtered_citations = hop_citations[:optimal_k]
            logger.info(
                f"Adaptive chunk retrieval: {len(hop_citations)} candidates -> "
                f"{optimal_k} selected (scores: {[round(s, 3) for s in scores[: optimal_k + 2]]})"
            )
        else:
            filtered_citations = hop_citations[:5]

        if not filtered_citations:
            return (
                """<knowledge_base>
The agent searched the knowledge base but did not find specific relevant information.
</knowledge_base>""",
                [],
                [],
            )

        # Build final citations list - PRESERVE global indices
        # This is the key difference: we keep the original global indices
        final_citations = []
        final_chunks = []  # NEW: Build chunks with full content for frontend
        content_sections = []

        for citation in filtered_citations:
            # Get full content from hop_data if available
            chunk_id = citation.get("chunk_id", 0)
            full_content = chunk_content_map.get(chunk_id, "")
            content_preview = citation.get("content_preview", "")

            # Build citation with PRESERVED global index and highlight ranges
            citation_data = {
                "index": citation["index"],  # Preserve global index!
                "document_id": citation.get("document_id", ""),
                "document_name": citation.get("document_name", ""),
                "chunk_id": str(citation.get("chunk_id", "")),
                "content_preview": content_preview,
                "score": round(citation.get("score", 0.0), 3),
                "chunk_index": citation.get("chunk_index", 0),
                "page_number": citation.get("page_number"),
                "hop_index": citation.get("hop_index"),
            }
            # Preserve highlight ranges if present
            if citation.get("highlight_ranges"):
                citation_data["highlight_ranges"] = citation["highlight_ranges"]
            # Preserve line range metadata for citations
            if citation.get("line_start"):
                citation_data["line_start"] = citation["line_start"]
                citation_data["line_end"] = citation.get("line_end")
            final_citations.append(citation_data)

            # Build chunk with FULL content for frontend display
            # This ensures SourceDetailModal can show complete content, not just preview
            chunk_data = {
                "chunk_id": str(citation.get("chunk_id", "")),
                "content": full_content if full_content else content_preview,
                "chunk_index": citation.get("chunk_index", 0),
            }
            final_chunks.append(chunk_data)

            # Build content section with the global index
            section_content = full_content if full_content else content_preview
            section = f"[{citation['index']}]\n{section_content}"
            content_sections.append(section)

        content = "\n\n".join(content_sections)

        formatted_context = f"""<knowledge_base>
{content}
</knowledge_base>"""

        logger.info(
            f"Formatted context from hop citations: {len(final_citations)} citations "
            f"(indices: {[c['index'] for c in final_citations]}), {len(final_chunks)} chunks with full content"
        )

        return formatted_context, final_citations, final_chunks

    @staticmethod
    def create(
        max_iterations: int = 3, enable_final_search: bool = True, session_config: Any = None
    ) -> "AgenticRAG":
        """Create an AgenticRAG instance with default configuration.

        Args:
            max_iterations: Maximum number of search iterations
            enable_final_search: Whether to perform final comprehensive search for citations (default: True)
            session_config: Session-scoped RAG config for per-session feature flags

        Returns:
            Configured AgenticRAG instance
        """
        default_model = get_default_model_settings()
        llm = get_llm_provider().get_chat_model(default_model)

        langfuse_handler = None
        if is_langfuse_tracing_enabled():
            try:
                from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
            except ImportError:
                try:
                    from langfuse.callback.langchain import (
                        LangchainCallbackHandler as LangfuseCallbackHandler,
                    )
                except ImportError:
                    LangfuseCallbackHandler = None

            if LangfuseCallbackHandler is not None:
                langfuse_handler = LangfuseCallbackHandler()

        if langfuse_handler is not None:
            if hasattr(llm, "bind"):
                llm = llm.bind(callbacks=[langfuse_handler])
            else:
                try:
                    llm.callbacks = [langfuse_handler]
                except Exception:
                    logger.debug("Failed to attach Langfuse callbacks to AgenticRAG LLM")
        return AgenticRAG(
            llm=llm,
            max_iterations=max_iterations,
            enable_final_search=enable_final_search,
            session_config=session_config,
        )


# =========================================================================
# LegacyReActPipeline — wraps AgenticRAG as a BaseAgenticPipeline
# =========================================================================


class LegacyReActPipeline(BaseAgenticPipeline):
    """Legacy ReAct agent wrapped as a pluggable pipeline.

    Subclasses ``BaseAgenticPipeline``.
    Delegates to ``AgenticRAG`` with ``use_structured_graph=False``.
    Registered in the component registry as ``"legacy_react"``.
    """

    def __init__(self, session_config=None, **kwargs):
        self._session_config = session_config

    async def run(self, input: AgenticPipelineInput) -> AgenticRAGResult:
        """Execute the legacy ReAct agent pipeline."""
        config = input.session_config or self._session_config or rag_config
        llm = input.llm
        if llm is None:
            from lasagna.providers import get_llm_provider

            llm = get_llm_provider().get_llm()

        max_iterations = getattr(config, "max_agent_iterations", 3)

        # Force legacy mode by temporarily ensuring structured graph is off
        agent = AgenticRAG(
            llm=llm,
            max_iterations=max_iterations,
            session_config=config,
        )
        # Override to ensure legacy path is used even if config says structured
        agent._use_structured_graph = False

        return await agent.retrieve(
            question=input.question,
            session_id=input.session_id,
            document_ids=input.document_filter,
            top_k=input.top_k,
            trace_id=input.trace_id,
            routing_decision=input.routing_decision,
        )

    @property
    def name(self) -> str:
        return "LegacyReActPipeline"
