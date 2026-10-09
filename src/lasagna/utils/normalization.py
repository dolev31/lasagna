"""Score normalization utilities for RAG system.

This module provides centralized score normalization functions used across
the RAG system for consistent score handling. Cross-encoders and different
vector backends produce scores on different scales (often negative), so
normalization is critical for:

1. User display (confidence scores should be 0-1)
2. MMR diversity calculation (requires comparable scales)
3. Citation ranking (consistent ordering across backends)

IMPORTANT: All RAG components should import from this module instead of
duplicating normalization logic.
"""

import math
from typing import List, Optional


def normalize_score_sigmoid(score: float) -> float:
    """Normalize a score using sigmoid function.

    Maps any real number to [0, 1] range. Useful for cross-encoder
    outputs which can be negative unbounded logits.

    Properties:
    - sigmoid(0) = 0.5
    - sigmoid(-inf) → 0
    - sigmoid(+inf) → 1
    - Preserves ordering (monotonic)

    Args:
        score: Raw score (can be negative)

    Returns:
        Normalized score in [0, 1]
    """
    try:
        return 1.0 / (1.0 + math.exp(-score))
    except OverflowError:
        # Handle extreme values
        return 0.0 if score < 0 else 1.0


def normalize_scores_minmax(scores: List[float]) -> List[float]:
    """Normalize scores using min-max scaling within a batch.

    Maps the minimum score to 0 and maximum to 1. Useful when
    comparing results within the same retrieval batch.

    Args:
        scores: List of raw scores

    Returns:
        List of normalized scores in [0, 1]
    """
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        # All scores equal - return 1.0 for all (maximum relevance)
        return [1.0] * len(scores)

    return [(s - min_s) / (max_s - min_s) for s in scores]


def normalize_scores_for_mmr(scores: List[float], strategy: str = "sigmoid_minmax") -> List[float]:
    """Normalize rerank scores to [0, 1] range for MMR computation.

    This is critical for MMR correctness: the MMR formula mixes relevance
    scores with cosine similarity (which is bounded to [0, 1]). Using raw
    cross-encoder logits (which can be negative and unbounded) causes
    mathematically incorrect diversity calculations.

    Args:
        scores: Raw rerank scores (potentially negative cross-encoder logits)
        strategy: Normalization strategy:
            - 'sigmoid': Apply sigmoid to each score (preserves absolute meaning)
            - 'minmax': Apply min-max scaling (preserves relative ordering)
            - 'sigmoid_minmax': Apply sigmoid then minmax (best of both)
            - 'none': No normalization (legacy behavior, not recommended)

    Returns:
        Normalized scores in [0, 1] range, comparable to cosine similarity
    """
    if not scores:
        return []

    if strategy == "none":
        return scores

    if strategy == "sigmoid":
        return [normalize_score_sigmoid(s) for s in scores]

    if strategy == "minmax":
        return normalize_scores_minmax(scores)

    # Default: sigmoid_minmax - apply sigmoid first, then minmax for batch consistency
    sigmoid_scores = [normalize_score_sigmoid(s) for s in scores]

    # Apply minmax on top to ensure full [0, 1] utilization within batch
    min_s = min(sigmoid_scores)
    max_s = max(sigmoid_scores)

    if max_s == min_s:
        return [0.5] * len(sigmoid_scores)  # All equal, use midpoint

    return [(s - min_s) / (max_s - min_s) for s in sigmoid_scores]


def normalize_score_for_display(
    score: float, all_scores: Optional[List[float]] = None, method: str = "sigmoid"
) -> float:
    """Normalize a score for user-facing display.

    Args:
        score: Raw score to normalize
        all_scores: Optional list of all scores for relative normalization
        method: Normalization method ("sigmoid", "minmax", "clamp")

    Returns:
        Normalized score in [0, 1] suitable for display
    """
    if method == "sigmoid":
        return normalize_score_sigmoid(score)
    elif method == "minmax" and all_scores:
        idx = all_scores.index(score) if score in all_scores else -1
        normalized = normalize_scores_minmax(all_scores)
        return normalized[idx] if idx >= 0 else normalize_score_sigmoid(score)
    else:
        # Clamp to [0, 1] - assumes score is already roughly in range
        return max(0.0, min(1.0, score))


def adaptive_k_selection(
    scores: List[float],
    min_k: int = 2,
    max_k: int = 20,
    buffer: int = 2,
    min_gap_ratio: float = 0.15,
) -> int:
    """Determine optimal number of results using score-gap analysis.

    Based on the Adaptive-k approach (EMNLP 2025): sorts relevance scores
    in descending order and finds the largest relative gap in the score
    distribution. Chunks above the gap are "clearly relevant"; chunks below
    are not. A buffer is added past the gap to catch borderline-relevant items.

    This replaces fixed top-k retrieval, which either retrieves too many
    irrelevant chunks or too few relevant ones depending on the query.

    Args:
        scores: Relevance scores (normalized to [0, 1], descending order expected
                but will be sorted internally). Must not be empty.
        min_k: Minimum number of results to return (floor).
        max_k: Maximum number of results to return (ceiling).
        buffer: Extra chunks to include past the score gap as safety margin.
        min_gap_ratio: Minimum relative gap size to trigger cutoff.
                       Prevents cutting on noise when scores are tightly clustered.

    Returns:
        Optimal k value clamped to [min_k, max_k].
    """
    if not scores:
        return min_k

    n = len(scores)
    if n <= min_k:
        return min(n, min_k)

    # Ensure descending order
    sorted_scores = sorted(scores, reverse=True)

    # Find the largest relative gap that exceeds min_gap_ratio
    best_gap = 0.0
    best_gap_idx = -1

    for i in range(n - 1):
        if sorted_scores[i] <= 0:
            # Avoid division by zero or negative scores
            break
        relative_gap = (sorted_scores[i] - sorted_scores[i + 1]) / sorted_scores[i]
        if relative_gap >= min_gap_ratio and relative_gap > best_gap:
            best_gap = relative_gap
            best_gap_idx = i

    if best_gap_idx >= 0:
        # Cut after the gap position + buffer
        k = best_gap_idx + 1 + buffer
    else:
        # No significant gap found — fall back to all scores above 50% of max
        threshold = sorted_scores[0] * 0.5
        k = sum(1 for s in sorted_scores if s >= threshold)
        k = max(k, min_k)

    return max(min_k, min(k, max_k, n))


def normalize_score_relative(score: float, all_scores: List[float]) -> float:
    """Normalize a score relative to a batch using min-max scaling.

    Handles negative scores and various scoring schemes by using
    min-max normalization within the current result set.

    Args:
        score: The score to normalize
        all_scores: All scores in the current result set

    Returns:
        Normalized score in [0, 1] range
    """
    if not all_scores:
        return 0.5

    min_score = min(all_scores)
    max_score = max(all_scores)

    if max_score == min_score:
        return 1.0  # All scores equal, treat as maximum relevance

    return (score - min_score) / (max_score - min_score)
