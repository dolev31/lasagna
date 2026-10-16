"""Query Aspect Coverage Scoring for RAG System.

This module tracks how well retrieved chunks cover different aspects
of complex queries. It is used for:

1. Confidence scoring - knowing when we have sufficient coverage
2. Gap identification - knowing what aspects need more retrieval
3. Result quality assessment - quantifying retrieval completeness

The coverage score helps the agentic RAG decide when to stop searching
and provides feedback for query refinement.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
import re
import threading


@dataclass
class QueryAspect:
    """A single aspect/component of a query."""

    text: str
    keywords: Set[str]
    is_covered: bool = False
    covering_chunk_ids: List[str] = field(default_factory=list)
    coverage_strength: float = 0.0  # How well this aspect is covered (0-1)
    # Pre-compiled regex patterns for efficient matching
    # This avoids recompiling patterns in the inner loop (O(n³) -> O(n²))
    keyword_patterns: Dict[str, "re.Pattern"] = field(default_factory=dict)


@dataclass
class CoverageScore:
    """Coverage analysis result for a query."""

    aspects: List[QueryAspect]
    coverage_ratio: float  # 0.0 to 1.0 - fraction of aspects covered
    weighted_coverage: float  # 0.0 to 1.0 - weighted by coverage strength
    uncovered_aspects: List[str]
    confidence: float  # Overall confidence score
    suggested_followup: Optional[str]  # Suggested query for missing aspects

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "coverage_ratio": round(self.coverage_ratio, 3),
            "weighted_coverage": round(self.weighted_coverage, 3),
            "confidence": round(self.confidence, 3),
            "aspects_total": len(self.aspects),
            "aspects_covered": len([a for a in self.aspects if a.is_covered]),
            "uncovered_aspects": self.uncovered_aspects,
            "suggested_followup": self.suggested_followup,
        }


class CoverageScorer:
    """Analyzes query aspect coverage by retrieved chunks.

    This scorer extracts key aspects from a query and checks whether
    the retrieved chunks adequately cover each aspect.
    """

    # Common stop words to exclude from aspect keywords
    STOP_WORDS = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "about",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "this",
        "that",
        "these",
        "those",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "so",
        "because",
        "while",
        "it",
        "its",
        "they",
        "them",
        "their",
        "he",
        "she",
        "his",
        "her",
    }

    def __init__(self, min_keyword_length: int = 3):
        """Initialize coverage scorer.

        Args:
            min_keyword_length: Minimum length for aspect keywords
        """
        self.min_keyword_length = min_keyword_length

    def extract_aspects(self, query: str) -> List[QueryAspect]:
        """Extract aspects from a query.

        Uses simple heuristics to identify distinct aspects:
        - Splits on conjunctions (and, or, comma)
        - Extracts key phrases
        - Filters stop words

        Args:
            query: The search query

        Returns:
            List of QueryAspect objects
        """
        if not query or not query.strip():
            return []

        # Split on conjunctions and commas
        parts = re.split(r"\s+and\s+|\s*,\s*|\s+or\s+", query, flags=re.IGNORECASE)

        aspects = []
        for part in parts:
            part = part.strip()
            if len(part) < self.min_keyword_length:
                continue

            # Extract keywords (excluding stop words)
            words = re.findall(r"\b\w+\b", part.lower())
            keywords = {
                w for w in words if len(w) >= self.min_keyword_length and w not in self.STOP_WORDS
            }

            if keywords:
                # Pre-compile regex patterns for each keyword
                keyword_patterns = {kw: re.compile(rf"\b{re.escape(kw)}\b") for kw in keywords}
                aspects.append(
                    QueryAspect(text=part, keywords=keywords, keyword_patterns=keyword_patterns)
                )

        # If no aspects found, treat whole query as single aspect
        if not aspects:
            words = re.findall(r"\b\w+\b", query.lower())
            keywords = {
                w for w in words if len(w) >= self.min_keyword_length and w not in self.STOP_WORDS
            }
            if keywords:
                # Pre-compile regex patterns
                keyword_patterns = {kw: re.compile(rf"\b{re.escape(kw)}\b") for kw in keywords}
                aspects.append(
                    QueryAspect(text=query, keywords=keywords, keyword_patterns=keyword_patterns)
                )

        return aspects

    def score_coverage(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        keyword_threshold: float = 0.4,
    ) -> CoverageScore:
        """Score how well chunks cover query aspects.

        Args:
            query: The search query
            chunks: Retrieved chunks with 'content' field
            keyword_threshold: Fraction of keywords needed for coverage

        Returns:
            CoverageScore with detailed analysis
        """
        aspects = self.extract_aspects(query)

        if not aspects:
            return CoverageScore(
                aspects=[],
                coverage_ratio=1.0,  # No aspects to cover
                weighted_coverage=1.0,
                uncovered_aspects=[],
                confidence=0.5,  # Low confidence due to no aspects
                suggested_followup=None,
            )

        # Check each chunk against each aspect
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            chunk_id = chunk.get("chunk_id", chunk.get("id", 0))

            for aspect in aspects:
                # Use word boundary matching instead of substring matching
                # This prevents false positives like "test" matching "testing", "contest"
                # Use pre-compiled patterns for O(1) lookup vs O(n) recompilation
                matches = 0
                for kw, pattern in aspect.keyword_patterns.items():
                    if pattern.search(content):
                        matches += 1
                match_ratio = matches / len(aspect.keywords) if aspect.keywords else 0

                if match_ratio >= keyword_threshold:
                    aspect.is_covered = True
                    aspect.covering_chunk_ids.append(chunk_id)
                    # Update coverage strength (keep max)
                    aspect.coverage_strength = max(aspect.coverage_strength, match_ratio)

        # Calculate metrics
        covered_count = sum(1 for a in aspects if a.is_covered)
        coverage_ratio = covered_count / len(aspects)

        # Weighted coverage considers strength of coverage
        total_strength = sum(a.coverage_strength for a in aspects)
        weighted_coverage = total_strength / len(aspects)

        uncovered = [a.text for a in aspects if not a.is_covered]

        # Confidence combines coverage and retrieval quality
        # Base confidence on coverage, boost if multiple chunks cover aspects
        avg_chunks_per_aspect = sum(len(a.covering_chunk_ids) for a in aspects) / len(aspects)
        redundancy_boost = min(0.2, avg_chunks_per_aspect * 0.05)  # Boost for redundant coverage
        confidence = min(1.0, weighted_coverage * 0.8 + coverage_ratio * 0.2 + redundancy_boost)

        # Generate suggested followup for uncovered aspects
        suggested_followup = None
        if uncovered and coverage_ratio < 0.8:
            suggested_followup = f"Find information about: {', '.join(uncovered[:3])}"

        return CoverageScore(
            aspects=aspects,
            coverage_ratio=coverage_ratio,
            weighted_coverage=weighted_coverage,
            uncovered_aspects=uncovered,
            confidence=confidence,
            suggested_followup=suggested_followup,
        )

    def should_continue_retrieval(
        self,
        coverage: CoverageScore,
        min_coverage: float = 0.6,
        min_confidence: float = 0.5,
    ) -> Tuple[bool, str]:
        """Determine if more retrieval is needed based on coverage.

        Args:
            coverage: CoverageScore from score_coverage()
            min_coverage: Minimum coverage ratio to be satisfied
            min_confidence: Minimum confidence to be satisfied

        Returns:
            Tuple of (should_continue, reason)
        """
        if coverage.coverage_ratio >= min_coverage and coverage.confidence >= min_confidence:
            return False, "Sufficient coverage achieved"

        if coverage.coverage_ratio < min_coverage:
            return True, f"Coverage too low ({coverage.coverage_ratio:.2f} < {min_coverage})"

        if coverage.confidence < min_confidence:
            return True, f"Confidence too low ({coverage.confidence:.2f} < {min_confidence})"

        return False, "Coverage adequate"


# =============================================================================
# Module Singleton
# =============================================================================

_scorer: Optional[CoverageScorer] = None
_scorer_lock = threading.Lock()


def get_coverage_scorer() -> CoverageScorer:
    """Get or create the coverage scorer singleton (thread-safe)."""
    global _scorer
    if _scorer is None:
        with _scorer_lock:
            if _scorer is None:
                _scorer = CoverageScorer()
    return _scorer


def score_query_coverage(query: str, chunks: List[Dict[str, Any]]) -> CoverageScore:
    """Convenience function to score query coverage.

    Args:
        query: The search query
        chunks: Retrieved chunks

    Returns:
        CoverageScore with analysis
    """
    return get_coverage_scorer().score_coverage(query, chunks)


def reset_coverage_scorer():
    """Reset the coverage scorer singleton."""
    global _scorer
    with _scorer_lock:
        _scorer = None
