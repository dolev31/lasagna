"""Signal extraction for query routing.

Extracts lightweight signals from the query text and existing
QueryClassification to inform the routing decision. No LLM calls.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from ..enhancement.classifier import QueryClassification, QueryType
from .models import RoutingSignals

# Compiled patterns for multi-hop query detection (immutable tuple).
# These look for conjunctions joining distinct topics or explicit
# multi-step reasoning indicators.
_MULTI_HOP_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # Conjunctions joining distinct information needs
        r"\b(and also|and then|additionally|furthermore|moreover)\b",
        # Explicit multi-step phrasing
        r"\b(step by step|first .+ then|based on .+ what|using .+ explain)\b",
        # Cross-document references
        r"\b(across (all |multiple |different )?documents|from each|in every)\b",
        # Chains of reasoning
        r"\b(if .+ then .+ what|given that .+ how|assuming .+ what)\b",
        # Aggregation across sources
        r"\b(summarize all|combine .+ from|synthesize|aggregate)\b",
    )
)


def extract_signals(
    query: str,
    document_ids: Optional[Sequence[str]],
    classification: Optional[QueryClassification],
) -> RoutingSignals:
    """Extract routing signals from a query and its classification.

    Args:
        query: The raw query text.
        document_ids: Optional document filter from the caller.
        classification: Pre-computed QueryClassification (may be None
            if query classification is disabled in config).

    Returns:
        RoutingSignals populated from the query and classification.
    """
    words = query.split()
    word_count = len(words)

    # Reuse classifier output when available, otherwise use safe defaults
    if classification is not None:
        query_type = classification.query_type.value
        query_type_confidence = classification.confidence
    else:
        query_type = QueryType.UNKNOWN.value
        query_type_confidence = 0.0

    has_multi_hop = _detect_multi_hop(query)
    has_doc_filter = bool(document_ids)

    estimated_complexity = _estimate_complexity(
        query_type=query_type,
        query_type_confidence=query_type_confidence,
        word_count=word_count,
        has_multi_hop=has_multi_hop,
    )

    return RoutingSignals(
        query_text=query,
        word_count=word_count,
        query_type=query_type,
        query_type_confidence=query_type_confidence,
        has_multi_hop_indicators=has_multi_hop,
        has_specific_document_filter=has_doc_filter,
        estimated_complexity=estimated_complexity,
    )


def _detect_multi_hop(query: str) -> bool:
    """Check if the query contains multi-hop reasoning indicators."""
    return any(p.search(query) for p in _MULTI_HOP_PATTERNS)


def _estimate_complexity(
    query_type: str,
    query_type_confidence: float,
    word_count: int,
    has_multi_hop: bool,
) -> str:
    """Estimate query complexity as simple/moderate/complex.

    Rules:
        simple:   (FACTUAL or NAVIGATIONAL) AND word_count <= 8 AND confidence > 0.5
        complex:  has_multi_hop_indicators OR (word_count > 20 AND CONCEPTUAL)
        moderate: everything else
    """
    if has_multi_hop:
        return "complex"

    if word_count > 20 and query_type == QueryType.CONCEPTUAL.value:
        return "complex"

    if query_type in (QueryType.FACTUAL.value, QueryType.NAVIGATIONAL.value):
        if word_count <= 8 and query_type_confidence > 0.5:
            return "simple"

    return "moderate"
