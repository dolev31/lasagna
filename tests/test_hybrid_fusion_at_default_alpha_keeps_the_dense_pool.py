"""At the default weights, hybrid fusion reorders the dense candidates and adds none.

The README's 2x2 (hybrid search x reranking on 100 SciFact queries) found the
reranked cells identical on every query, which read like hybrid search being
skipped once a reranker is on. It is not skipped. ``enable_hybrid_search``
fuses by weighted rank, ``alpha / (rrf_k + dense_rank) + (1 - alpha) /
(rrf_k + bm25_rank)``, and cuts the fused list at the dense list's length. At
``hybrid_alpha`` 0.7 and ``rrf_k`` 60 a chunk only BM25 finds scores at most
0.3/60, below the last of 40 dense chunks (0.7/99), so the reranker is handed
the dense 40 in another order and, scoring each chunk on its own, returns the
same list. The README now says so; these tests hold it to the code.
"""

from __future__ import annotations

import pytest

from lasagna.retrieval.search.hybrid import HybridSearcher

DEPTH = 40


def _dense(n=DEPTH):
    return [
        {"id": f"d{i}", "score": 1.0 - i / 100, "entity": {"document_id": f"doc{i}", "content": ""}}
        for i in range(n)
    ]


def _bm25_only(n=2 * DEPTH):
    """BM25 hits none of which the dense list has, best first."""
    docs = [{"chunk_id": f"b{i}", "document_id": f"bdoc{i}", "content": ""} for i in range(n)]
    return [(i, 10.0 - i / 10) for i in range(n)], docs


def _fused_pool(alpha, dense_n=DEPTH):
    searcher = HybridSearcher.__new__(HybridSearcher)
    searcher.rrf_k = 60
    sparse, docs = _bm25_only()
    fused = searcher._reciprocal_rank_fusion(_dense(dense_n), sparse, docs, alpha=alpha, rrf_k=60)
    fused.sort(key=lambda r: r.combined_score, reverse=True)
    return {r.chunk_id for r in fused[:DEPTH]}


def test_at_alpha_0_7_the_fused_pool_is_the_dense_pool():
    assert _fused_pool(0.7) == {f"d{i}" for i in range(DEPTH)}


def test_the_readme_threshold_is_where_bm25_starts_adding_chunks():
    assert _fused_pool(0.63) == {f"d{i}" for i in range(DEPTH)}
    assert "b0" in _fused_pool(0.62)


def test_a_short_dense_list_leaves_room_for_bm25():
    assert {"b0", "b1"} <= _fused_pool(0.7, dense_n=DEPTH - 2)


@pytest.mark.parametrize("alpha", [0.7, 0.5])
def test_the_fusion_reorders_when_both_lists_agree(alpha):
    """Hybrid still acts at 0.7: a dense chunk BM25 ranks first moves up."""
    searcher = HybridSearcher.__new__(HybridSearcher)
    searcher.rrf_k = 60
    docs = [{"chunk_id": "d39", "document_id": "doc39", "content": ""}]
    fused = searcher._reciprocal_rank_fusion(_dense(), [(0, 9.0)], docs, alpha=alpha, rrf_k=60)
    fused.sort(key=lambda r: r.combined_score, reverse=True)
    assert [r.chunk_id for r in fused].index("d39") < 39
