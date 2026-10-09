"""IR evaluation helpers with chunk-to-document deduplication.

Wraps knowledge_evaluation_sdk/ir_metrics.py functions with proper
handling of chunk-level API results: the REST API returns SearchResultResponse
per-chunk (same document_id can appear multiple times), so we deduplicate
to document-level ranking before computing IR metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List

from lasagna.knowledge_evaluation_sdk.ir_metrics import (
    corpus_metrics,
    f1_at_k,
    hit_rate_at_k,
    map_score,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def deduplicate_to_doc_ranking(results: List[Dict[str, Any]]) -> List[str]:
    """Extract unique document_ids from chunk-level search results.

    Preserves first-occurrence order (highest-scored chunk per doc wins
    the ranking position). This is necessary because the REST API returns
    SearchResultResponse at chunk granularity — the same document_id can
    appear multiple times.

    Args:
        results: List of search result dicts with 'document_id' key
                 (from SearchResultResponse JSON).

    Returns:
        Ordered list of unique document IDs.
    """
    seen: set = set()
    ranking: List[str] = []
    for r in results:
        doc_id = r.get("document_id", "")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ranking.append(doc_id)
    return ranking


def evaluate_search_results(
    results: List[Dict[str, Any]],
    ground_truth: Dict[str, int],
    k: int = 5,
) -> Dict[str, float]:
    """Compute IR metrics from chunk-level API results.

    1. Deduplicates chunk results to document-level ranking
    2. Computes standard IR metrics using canonical ir_metrics functions

    Args:
        results: List of SearchResultResponse dicts from the API.
        ground_truth: Mapping of document_id → relevance grade (0, 1, or 2).
        k: Cut-off rank.

    Returns:
        Dict with metric names and values.
    """
    doc_ranking = deduplicate_to_doc_ranking(results)

    return {
        "ndcg@k": ndcg_at_k(ground_truth, doc_ranking, k),
        "precision@k": precision_at_k(ground_truth, doc_ranking, k),
        "recall@k": recall_at_k(ground_truth, doc_ranking, k),
        "mrr": mrr(ground_truth, doc_ranking),
        "map": map_score(ground_truth, doc_ranking, k),
        "f1@k": f1_at_k(ground_truth, doc_ranking, k),
        "hit_rate@k": hit_rate_at_k(ground_truth, doc_ranking, k),
    }


def evaluate_corpus(
    all_results: List[List[Dict[str, Any]]],
    all_ground_truths: List[Dict[str, int]],
    k: int = 5,
) -> Dict[str, float]:
    """Compute corpus-level IR metrics across multiple queries.

    Args:
        all_results: List of per-query search result lists.
        all_ground_truths: List of per-query relevance maps.
        k: Cut-off rank.

    Returns:
        Averaged IR metrics across all queries.
    """
    if len(all_results) != len(all_ground_truths):
        raise ValueError(
            f"Mismatched lengths: {len(all_results)} results vs "
            f"{len(all_ground_truths)} ground truths"
        )

    # Deduplicate each query's results to doc-level
    doc_rankings = [deduplicate_to_doc_ranking(r) for r in all_results]

    return corpus_metrics(all_ground_truths, doc_rankings, k)
