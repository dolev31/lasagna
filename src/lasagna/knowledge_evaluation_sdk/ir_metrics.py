"""Canonical IR retrieval metrics for RAG evaluation.

Pure-numpy implementations of standard Information Retrieval metrics.
These are the building blocks for both the SDK evaluator and the
ablation framework — having one canonical module prevents drift and
ensures consistent computation.

All functions follow the same convention:
- ``relevance`` is a mapping from document ID to graded relevance
  (binary {0, 1} or graded {0, 1, 2, 3}).
- ``retrieved`` is an ordered list of document IDs as returned by
  the retrieval system.

Example:
    from lasagna.knowledge_evaluation_sdk.ir_metrics import (
        ndcg_at_k, precision_at_k, recall_at_k, mrr, map_score,
    )

    relevance = {"doc1": 2, "doc2": 1, "doc3": 1}
    retrieved = ["doc4", "doc1", "doc3", "doc5", "doc2"]

    print(f"NDCG@5:      {ndcg_at_k(relevance, retrieved, k=5):.4f}")
    print(f"Precision@3: {precision_at_k(relevance, retrieved, k=3):.4f}")
    print(f"Recall@5:    {recall_at_k(relevance, retrieved, k=5):.4f}")
    print(f"MRR:         {mrr(relevance, retrieved):.4f}")
    print(f"MAP:         {map_score(relevance, retrieved):.4f}")
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Union

# We avoid importing numpy at module level so the module stays importable
# without heavy deps. Functions that need vectorised operations import
# numpy lazily — but most of the logic here uses plain Python for
# clarity and zero-dep simplicity.

RelevanceMap = Dict[str, Union[int, float]]



def _unique(retrieved: List[str]) -> List[str]:
    """Keep the first occurrence of each document ID, in rank order.

    A ranked list of chunks names a document once per retrieved chunk. Scoring
    the repeats counted one relevant document several times: three chunks of
    it gave NDCG@10 2.13 and Recall@10 3.0. trec_eval ignores repeated IDs;
    so do these functions.
    """
    seen: set = set()
    out: List[str] = []
    for doc_id in retrieved:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out

def ndcg_at_k(
    relevance: RelevanceMap,
    retrieved: Sequence[str],
    k: int = 10,
    exponential_gain: bool = False,
) -> float:
    """Normalised Discounted Cumulative Gain at rank *k*.

    ``DCG_k = sum_{i=1..k} rel_i / log2(i + 1)``, normalised by the ideal DCG.

    **Linear gain, matching ``trec_eval``** — which is what BEIR delegates to,
    and therefore what every published BEIR number is computed with. This used
    ``2^rel - 1``, the Burges/RankNet convention: identical on binary
    judgements, and different wherever relevance is graded. Of the registered
    BEIR datasets that is ``nfcorpus``, ``trec-covid`` and ``dbpedia-entity``,
    whose numbers did not match the literature and could not be checked against
    it, because no reference implementation is a dependency.

    Pass ``exponential_gain=True`` for the Burges form if you are comparing
    against work that uses it. The choice is recorded, not silent.

    Args:
        relevance: Ground-truth relevance map (doc_id → relevance score).
        retrieved: Ordered list of retrieved document IDs.
        k: Cut-off rank.
        exponential_gain: Use ``2^rel - 1`` instead of ``rel``.

    Returns:
        NDCG@k in [0, 1]. Returns 0.0 when there are no relevant documents.
    """
    if not relevance or k <= 0:
        return 0.0

    def _gain(rel: float) -> float:
        return (2**rel - 1) if exponential_gain else rel

    # DCG of the retrieved list
    dcg = 0.0
    for i, doc_id in enumerate(_unique(retrieved)[:k]):
        rel = relevance.get(doc_id, 0)
        dcg += _gain(rel) / math.log2(i + 2)  # i+2 because rank is 1-indexed

    # Ideal DCG: sort all relevance scores descending and compute
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += _gain(rel) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(
    relevance: RelevanceMap,
    retrieved: Sequence[str],
    k: int = 10,
) -> float:
    """Precision at rank *k* (binary relevance: rel > 0 is relevant).

    Args:
        relevance: Ground-truth relevance map.
        retrieved: Ordered list of retrieved document IDs.
        k: Cut-off rank.

    Returns:
        P@k in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = _unique(retrieved)[:k]
    relevant_count = sum(1 for doc_id in top_k if relevance.get(doc_id, 0) > 0)
    return relevant_count / k


def recall_at_k(
    relevance: RelevanceMap,
    retrieved: Sequence[str],
    k: int = 10,
) -> float:
    """Recall at rank *k* (binary relevance: rel > 0 is relevant).

    Args:
        relevance: Ground-truth relevance map.
        retrieved: Ordered list of retrieved document IDs.
        k: Cut-off rank.

    Returns:
        R@k in [0, 1]. Returns 0.0 when there are no relevant documents.
    """
    total_relevant = sum(1 for v in relevance.values() if v > 0)
    if total_relevant == 0 or k <= 0:
        return 0.0
    top_k = _unique(retrieved)[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if relevance.get(doc_id, 0) > 0)
    return relevant_retrieved / total_relevant


def mrr(
    relevance: RelevanceMap,
    retrieved: Sequence[str],
    k: int | None = None,
) -> float:
    """Mean Reciprocal Rank (single-query version).

    Returns the reciprocal of the rank of the first relevant document.
    For a corpus-level MRR, average the per-query values.

    Args:
        relevance: Ground-truth relevance map.
        retrieved: Ordered list of retrieved document IDs.
        k: Optional cut-off rank. If provided, only the top-k documents
            are considered. If ``None``, the entire list is scanned.

    Returns:
        1/rank of first relevant document, or 0.0 if none found.
    """
    eval_list = _unique(retrieved)[:k] if k is not None else _unique(retrieved)
    for rank, doc_id in enumerate(eval_list, 1):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def map_score(
    relevance: RelevanceMap,
    retrieved: Sequence[str],
    k: int | None = None,
) -> float:
    """Average Precision (single-query version of MAP).

    AP = (1/R) * sum_{k=1}^{n} P(k) * rel(k)
    where R is the number of relevant documents and P(k) is precision
    at rank k, computed only when the document at rank k is relevant.

    Args:
        relevance: Ground-truth relevance map.
        retrieved: Ordered list of retrieved document IDs.
        k: Optional cut-off rank. If provided, only the top-k documents
            are considered. If ``None``, the entire list is evaluated.

    Returns:
        Average Precision in [0, 1].
    """
    total_relevant = sum(1 for v in relevance.values() if v > 0)
    if total_relevant == 0:
        return 0.0

    eval_list = _unique(retrieved)[:k] if k is not None else _unique(retrieved)
    ap_sum = 0.0
    relevant_so_far = 0
    for rank, doc_id in enumerate(eval_list, 1):
        if relevance.get(doc_id, 0) > 0:
            relevant_so_far += 1
            ap_sum += relevant_so_far / rank

    return ap_sum / total_relevant


def f1_at_k(
    relevance: RelevanceMap,
    retrieved: Sequence[str],
    k: int = 10,
) -> float:
    """F1 score at rank *k* (harmonic mean of P@k and R@k).

    Args:
        relevance: Ground-truth relevance map.
        retrieved: Ordered list of retrieved document IDs.
        k: Cut-off rank.

    Returns:
        F1@k in [0, 1].
    """
    p = precision_at_k(relevance, retrieved, k)
    r = recall_at_k(relevance, retrieved, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def hit_rate_at_k(
    relevance: RelevanceMap,
    retrieved: Sequence[str],
    k: int = 10,
) -> float:
    """Hit rate (Success@k): 1.0 if any relevant doc is in top-k, else 0.0.

    Args:
        relevance: Ground-truth relevance map.
        retrieved: Ordered list of retrieved document IDs.
        k: Cut-off rank.

    Returns:
        1.0 or 0.0.
    """
    if k <= 0:
        return 0.0
    for doc_id in _unique(retrieved)[:k]:
        if relevance.get(doc_id, 0) > 0:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Batch (corpus-level) helpers
# ---------------------------------------------------------------------------


def corpus_metrics(
    relevances: List[RelevanceMap],
    retrieved_lists: List[Sequence[str]],
    k: int = 10,
) -> Dict[str, float]:
    """Compute all standard IR metrics averaged across queries.

    Convenience function that returns a dict of corpus-level metrics
    for easy integration with the ablation framework.

    Args:
        relevances: List of per-query relevance maps.
        retrieved_lists: List of per-query retrieved ID lists.
        k: Cut-off rank for P@k, R@k, NDCG@k, F1@k, Hit@k.

    Returns:
        Dict with keys: ``ndcg@{k}``, ``precision@{k}``, ``recall@{k}``,
        ``f1@{k}``, ``mrr@{k}``, ``map@{k}``, ``hit_rate@{k}``.
    """
    n = len(relevances)
    if n == 0:
        return {}
    if len(retrieved_lists) != n:
        raise ValueError(
            f"relevances and retrieved_lists must have the same length "
            f"({n} vs {len(retrieved_lists)})"
        )

    sums = {
        f"ndcg@{k}": 0.0,
        f"precision@{k}": 0.0,
        f"recall@{k}": 0.0,
        f"f1@{k}": 0.0,
        f"mrr@{k}": 0.0,
        f"map@{k}": 0.0,
        f"hit_rate@{k}": 0.0,
    }

    for rel, ret in zip(relevances, retrieved_lists):
        sums[f"ndcg@{k}"] += ndcg_at_k(rel, ret, k)
        sums[f"precision@{k}"] += precision_at_k(rel, ret, k)
        sums[f"recall@{k}"] += recall_at_k(rel, ret, k)
        sums[f"f1@{k}"] += f1_at_k(rel, ret, k)
        sums[f"mrr@{k}"] += mrr(rel, ret, k)
        sums[f"map@{k}"] += map_score(rel, ret, k)
        sums[f"hit_rate@{k}"] += hit_rate_at_k(rel, ret, k)

    return {metric: total / n for metric, total in sums.items()}
