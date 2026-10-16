"""Query Type Classification for Adaptive RAG.

This module classifies queries into types to optimize retrieval parameters
dynamically. Different query types benefit from different retrieval strategies:

- FACTUAL: Specific facts, entities, numbers (favor BM25 for keyword matching)
- CONCEPTUAL: Abstract ideas, explanations (favor dense + HyDE)
- NAVIGATIONAL: Finding specific documents/sections (balanced approach)
- COMPARATIVE: Comparing multiple items (use query expansion)

This is a technique for adaptive RAG that improves retrieval quality
by matching retrieval strategy to query intent.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re


class QueryType(Enum):
    """Types of queries for adaptive retrieval."""

    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    NAVIGATIONAL = "navigational"
    COMPARATIVE = "comparative"
    UNKNOWN = "unknown"


@dataclass
class QueryClassification:
    """Result of query classification with recommended parameters."""

    query_type: QueryType
    confidence: float  # 0.0 to 1.0
    recommended_alpha: float  # Hybrid search alpha (dense weight)
    use_hyde: bool
    hyde_strategy: str  # 'single', 'multi', 'off'
    use_query_expansion: bool
    rationale: str

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        return {
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "recommended_alpha": self.recommended_alpha,
            "use_hyde": self.use_hyde,
            "hyde_strategy": self.hyde_strategy,
            "use_query_expansion": self.use_query_expansion,
            "rationale": self.rationale,
        }


# =============================================================================
# Pattern Definitions for Classification
# =============================================================================

# FACTUAL: Questions about specific facts, entities, dates, numbers
FACTUAL_PATTERNS = [
    r"\b(what is|what are|what\'s|whats)\b",
    r"\b(who is|who are|who\'s|whos)\b",
    r"\b(when did|when was|when were|when is)\b",
    r"\b(where is|where are|where was|where were)\b",
    r"\b(how many|how much|how old|how long|how far)\b",
    r"\b(define|definition of|meaning of)\b",
    r"\b(name of|list of|number of|amount of)\b",
    r"\b(date|year|price|cost|size|length|height|weight|age)\b",
    r"\b(tell me (the|a)?\s*(name|date|number|fact))\b",
]

# CONCEPTUAL: Questions about concepts, explanations, mechanisms
CONCEPTUAL_PATTERNS = [
    r"\b(explain|describe|elaborate)\b",
    r"\b(why|why is|why are|why do|why does)\b",
    r"\b(how does|how do|how can|how is|how are)\b",
    r"\b(what happens|what would happen)\b",
    r"\b(concept|theory|principle|mechanism|process)\b",
    r"\b(understand|understanding|meaning|significance|importance)\b",
    r"\b(relationship between|connection between)\b",
    r"\b(impact of|effect of|cause of|reason for)\b",
    r"\b(in what way|in which way)\b",
]

# NAVIGATIONAL: Questions about finding specific documents or sections
NAVIGATIONAL_PATTERNS = [
    r"\b(find|locate|search for|look for|looking for)\b",
    r"\b(where in|which section|which chapter|which page|which part)\b",
    r"\b(document|file|chapter|section|appendix|paragraph)\b",
    r"\b(show me|take me to|go to|navigate to)\b",
    r"\b(find me|get me|retrieve)\b",
    r"\b(mentioned in|stated in|written in|found in)\b",
]

# COMPARATIVE: Questions comparing multiple things
COMPARATIVE_PATTERNS = [
    r"\b(compare|comparison|comparing)\b",
    r"\b(contrast|contrasting|difference between|differences between)\b",
    r"\b(versus|vs\.?|v\.s\.?)\b",
    r"\b(better than|worse than|more than|less than)\b",
    r"\b(advantages|disadvantages|pros and cons|pros|cons)\b",
    r"\b(similar to|different from|same as)\b",
    r"\b(which is better|which one|which should)\b",
    r"\b(or\b.*\bor\b)",  # Multiple "or" suggesting comparison
]


class QueryClassifier:
    """Classifies queries for adaptive retrieval parameter selection.

    This classifier uses rule-based pattern matching for efficiency.
    For production systems requiring higher accuracy, consider:
    - Fine-tuned BERT classifier
    - Zero-shot classification with a small LLM
    - Ensemble of multiple classification methods
    """

    def __init__(self):
        """Initialize the query classifier with compiled patterns."""
        self._compiled_patterns = {
            QueryType.FACTUAL: [re.compile(p, re.IGNORECASE) for p in FACTUAL_PATTERNS],
            QueryType.CONCEPTUAL: [re.compile(p, re.IGNORECASE) for p in CONCEPTUAL_PATTERNS],
            QueryType.NAVIGATIONAL: [re.compile(p, re.IGNORECASE) for p in NAVIGATIONAL_PATTERNS],
            QueryType.COMPARATIVE: [re.compile(p, re.IGNORECASE) for p in COMPARATIVE_PATTERNS],
        }

        # Parameter recommendations by query type
        # These are tuned based on RAG research and empirical testing
        self._type_params = {
            QueryType.FACTUAL: {
                "alpha": 0.5,  # More weight to BM25 for precise keyword matching
                "use_hyde": False,  # HyDE can hallucinate facts
                "hyde_strategy": "off",
                "use_expansion": False,  # Stick to exact terms
            },
            QueryType.CONCEPTUAL: {
                "alpha": 0.8,  # More weight to dense for semantic understanding
                "use_hyde": True,
                "hyde_strategy": "multi",  # Multi-doc HyDE for broader coverage
                "use_expansion": True,  # Expand to capture related concepts
            },
            QueryType.NAVIGATIONAL: {
                "alpha": 0.6,  # Balanced - need both semantic and keyword
                "use_hyde": False,  # Want to find exact location
                "hyde_strategy": "off",
                "use_expansion": False,  # Don't expand navigation queries
            },
            QueryType.COMPARATIVE: {
                "alpha": 0.75,  # Lean toward semantic
                "use_hyde": True,
                "hyde_strategy": "single",  # Single HyDE sufficient
                "use_expansion": True,  # Expand to cover both sides of comparison
            },
            QueryType.UNKNOWN: {
                "alpha": 0.7,  # Default balanced
                "use_hyde": True,
                "hyde_strategy": "adaptive",  # Let system decide
                "use_expansion": False,
            },
        }

    def classify(self, query: str) -> QueryClassification:
        """Classify a query and return recommended retrieval parameters.

        Args:
            query: The search query to classify

        Returns:
            QueryClassification with type and parameter recommendations
        """
        if not query or not query.strip():
            return self._build_classification(QueryType.UNKNOWN, 0.0, query, "Empty query")

        query_lower = query.lower().strip()

        # Count pattern matches for each type
        scores = {qt: 0.0 for qt in QueryType if qt != QueryType.UNKNOWN}
        match_details = {qt: [] for qt in QueryType if qt != QueryType.UNKNOWN}

        for query_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(query_lower)
                if match:
                    scores[query_type] += 1.0
                    match_details[query_type].append(match.group())

        # Normalize and find winner
        total_matches = sum(scores.values())

        if total_matches == 0:
            # No patterns matched - use heuristics
            return self._classify_by_heuristics(query)

        # Find type with highest score
        best_type = max(scores, key=scores.get)
        confidence = min(scores[best_type] / max(total_matches, 1), 1.0)

        # Boost confidence if multiple patterns match for the same type
        if scores[best_type] >= 2:
            confidence = min(confidence + 0.1, 1.0)

        rationale = f"Matched patterns: {match_details[best_type][:3]}"  # Top 3 matches

        return self._build_classification(best_type, confidence, query, rationale)

    def _classify_by_heuristics(self, query: str) -> QueryClassification:
        """Fallback classification using query characteristics when no patterns match."""
        words = query.split()
        query_lower = query.lower()

        # Short queries (< 4 words) are often factual lookups
        if len(words) < 4:
            return self._build_classification(
                QueryType.FACTUAL, 0.4, query, "Short query - likely factual lookup"
            )

        # Questions starting with "why" or "how" (but not "how many/much") are conceptual
        if query_lower.startswith(("why", "how")) and not any(
            query_lower.startswith(p) for p in ["how many", "how much", "how old", "how long"]
        ):
            return self._build_classification(
                QueryType.CONCEPTUAL, 0.5, query, "Starts with why/how - likely conceptual"
            )

        # Questions ending with ? that start with what/who/where/when are often factual
        if query.endswith("?") and query_lower.startswith(("what", "who", "where", "when")):
            return self._build_classification(
                QueryType.FACTUAL, 0.5, query, "Question form suggests factual query"
            )

        # Long queries (> 10 words) are often conceptual
        if len(words) > 10:
            return self._build_classification(
                QueryType.CONCEPTUAL, 0.4, query, "Long query - likely conceptual/complex"
            )

        # Default to unknown
        return self._build_classification(
            QueryType.UNKNOWN, 0.3, query, "No strong signals - using default parameters"
        )

    def _build_classification(
        self, query_type: QueryType, confidence: float, query: str, rationale: str
    ) -> QueryClassification:
        """Build classification with recommended parameters."""
        params = self._type_params[query_type]

        return QueryClassification(
            query_type=query_type,
            confidence=round(confidence, 3),
            recommended_alpha=params["alpha"],
            use_hyde=params["use_hyde"],
            hyde_strategy=params["hyde_strategy"],
            use_query_expansion=params["use_expansion"],
            rationale=rationale,
        )

    def get_adjusted_alpha(
        self, query: str, base_alpha: float, classification: Optional[QueryClassification] = None
    ) -> float:
        """Get adjusted alpha based on query classification.

        This method provides a smoother transition by blending the
        recommended alpha with the base alpha based on confidence.

        Args:
            query: The query string
            base_alpha: The default/configured alpha
            classification: Pre-computed classification (optional)

        Returns:
            Adjusted alpha value
        """
        if classification is None:
            classification = self.classify(query)

        # Blend based on confidence
        # High confidence -> use recommended alpha
        # Low confidence -> stay closer to base alpha
        blend_factor = classification.confidence
        adjusted = blend_factor * classification.recommended_alpha + (1 - blend_factor) * base_alpha

        return round(adjusted, 3)


# =============================================================================
# Module-level Singleton
# =============================================================================

import threading  # noqa: E402

_classifier: Optional[QueryClassifier] = None
_classifier_lock = threading.Lock()


def get_query_classifier() -> QueryClassifier:
    """Get or create the query classifier singleton.

    Thread-safe implementation to prevent race conditions during initialization.

    Returns:
        QueryClassifier instance
    """
    global _classifier
    # Add thread safety to singleton initialization
    if _classifier is None:
        with _classifier_lock:
            # Double-check locking pattern
            if _classifier is None:
                _classifier = QueryClassifier()
    return _classifier


def classify_query(query: str) -> QueryClassification:
    """Convenience function to classify a query.

    Args:
        query: The search query

    Returns:
        QueryClassification with type and recommendations
    """
    return get_query_classifier().classify(query)


def get_adjusted_alpha(query: str, base_alpha: float) -> float:
    """Get adjusted hybrid search alpha for a query.

    Args:
        query: The search query
        base_alpha: The default/configured alpha

    Returns:
        Adjusted alpha value based on query type
    """
    return get_query_classifier().get_adjusted_alpha(query, base_alpha)


def reset_query_classifier():
    """Reset the query classifier singleton."""
    global _classifier
    with _classifier_lock:
        _classifier = None
