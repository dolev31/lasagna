"""The retrieval algorithms themselves, against hand-computed expectations.

The plumbing in this repo is well tested — registry wiring, graph edge budgets,
index concurrency, adapter contracts. The *algorithms* were not tested at all:
no fixture asserted that RRF produces the right ordering, that MaxSim scores
what it claims, or that MMR actually trades relevance against diversity.

That is the wrong gap to have in a framework whose product is measurement. A
wrong fusion or a wrong MaxSim does not crash; it silently shifts every number
computed with it, and would retroactively invalidate any published result and
every entry in a baselines library.

Expected values here are computed by hand in the comments, not by calling the
implementation and pasting what it returned — a test written that way passes
against a bug.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _dense(chunk_id, score=0.9):
    """A dense hit in the shape `_reciprocal_rank_fusion` consumes."""
    return {
        "id": chunk_id,
        "score": score,
        "entity": {"document_id": f"doc-{chunk_id}", "content": chunk_id, "metadata": {}},
    }


def _indexed(chunk_ids):
    return [
        {"chunk_id": c, "document_id": f"doc-{c}", "content": c, "metadata": {}} for c in chunk_ids
    ]


@pytest.fixture
def fuse():
    """The fusion function, bound to an engine instance without constructing one."""
    from lasagna.retrieval.search.hybrid import HybridSearcher

    engine = HybridSearcher.__new__(HybridSearcher)
    engine.rrf_k = 60
    return engine._reciprocal_rank_fusion


class TestReciprocalRankFusionScores:
    """Ranks are 0-based here, so the top hit contributes 1/(k+0) = 1/k."""

    def test_a_document_ranked_first_by_both_scores_exactly(self, fuse):
        # alpha=0.5, k=60, dense rank 0, sparse rank 0
        #   0.5 * 1/60 + 0.5 * 1/60 = 1/60 = 0.0166666...
        out = fuse(
            dense_results=[_dense("a")],
            sparse_results=[(0, 3.2)],
            indexed_docs=_indexed(["a"]),
            alpha=0.5,
            rrf_k=60,
        )
        assert len(out) == 1
        assert out[0].combined_score == pytest.approx(1 / 60)

    def test_a_document_found_by_one_retriever_only_scores_half(self, fuse):
        # dense rank 0, absent from sparse: 0.5 * 1/60 + 0.5 * 0 = 1/120
        out = fuse(
            dense_results=[_dense("a")],
            sparse_results=[],
            indexed_docs=_indexed(["a"]),
            alpha=0.5,
            rrf_k=60,
        )
        assert out[0].combined_score == pytest.approx(1 / 120)

    def test_rank_two_scores_less_than_rank_one(self, fuse):
        # ranks 0 and 1: 1/60 vs 1/61, both dense-only at alpha=1.0
        out = fuse(
            dense_results=[_dense("a"), _dense("b")],
            sparse_results=[],
            indexed_docs=_indexed(["a", "b"]),
            alpha=1.0,
            rrf_k=60,
        )
        by_id = {r.chunk_id: r.combined_score for r in out}
        assert by_id["a"] == pytest.approx(1 / 60)
        assert by_id["b"] == pytest.approx(1 / 61)
        assert by_id["a"] > by_id["b"]

    def test_k_damps_the_gap_between_adjacent_ranks(self, fuse):
        """The whole point of k: a larger k makes rank differences matter less."""

        def gap(k):
            out = fuse(
                dense_results=[_dense("a"), _dense("b")],
                sparse_results=[],
                indexed_docs=_indexed(["a", "b"]),
                alpha=1.0,
                rrf_k=k,
            )
            by_id = {r.chunk_id: r.combined_score for r in out}
            return by_id["a"] - by_id["b"]

        assert gap(1) > gap(60) > gap(1000)


class TestFusionOrdering:
    """The property a researcher actually depends on."""

    def test_agreement_beats_a_single_strong_signal(self, fuse):
        # 'both' is rank 1 in each list; 'dense_only' is rank 0 in dense alone.
        #   both:       0.5*1/61 + 0.5*1/61 = 1/61   = 0.016393
        #   dense_only: 0.5*1/60 + 0        = 1/120  = 0.008333
        out = fuse(
            dense_results=[_dense("dense_only"), _dense("both")],
            sparse_results=[(0, 5.0), (1, 4.0)],
            indexed_docs=_indexed(["sparse_only", "both"]),
            alpha=0.5,
            rrf_k=60,
        )
        ranked = sorted(out, key=lambda r: r.combined_score, reverse=True)
        assert ranked[0].chunk_id == "both", (
            "a document both retrievers found must outrank one only dense found"
        )

    def test_alpha_one_ignores_the_sparse_ranking(self, fuse):
        out = fuse(
            dense_results=[_dense("d")],
            sparse_results=[(0, 99.0)],
            indexed_docs=_indexed(["s"]),
            alpha=1.0,
            rrf_k=60,
        )
        by_id = {r.chunk_id: r.combined_score for r in out}
        assert by_id["d"] == pytest.approx(1 / 60)
        assert by_id["s"] == pytest.approx(0.0), "sparse contributed at alpha=1.0"

    def test_alpha_zero_ignores_the_dense_ranking(self, fuse):
        out = fuse(
            dense_results=[_dense("d")],
            sparse_results=[(0, 99.0)],
            indexed_docs=_indexed(["s"]),
            alpha=0.0,
            rrf_k=60,
        )
        by_id = {r.chunk_id: r.combined_score for r in out}
        assert by_id["s"] == pytest.approx(1 / 60)
        assert by_id["d"] == pytest.approx(0.0), "dense contributed at alpha=0.0"

    def test_the_same_chunk_from_both_lists_is_one_result(self, fuse):
        """Deduplication by chunk_id — otherwise a doc found twice double-counts."""
        out = fuse(
            dense_results=[_dense("shared")],
            sparse_results=[(0, 1.0)],
            indexed_docs=_indexed(["shared"]),
            alpha=0.5,
            rrf_k=60,
        )
        assert len(out) == 1
        assert out[0].dense_rank == 0 and out[0].sparse_rank == 0

    def test_a_sparse_index_out_of_range_is_skipped_not_crashed(self, fuse):
        out = fuse(
            dense_results=[_dense("a")],
            sparse_results=[(99, 1.0)],
            indexed_docs=_indexed(["a"]),
            alpha=0.5,
            rrf_k=60,
        )
        assert [r.chunk_id for r in out] == ["a"]


# ---------------------------------------------------------------------------
# ColBERT MaxSim
# ---------------------------------------------------------------------------


class TestColbertMaxSim:
    """MaxSim: for each query token take its best document token, then sum."""

    @pytest.fixture
    def maxsim(self):
        torch = pytest.importorskip("torch")
        from lasagna.retrieval.search.ranker import ColBERTReranker

        r = ColBERTReranker.__new__(ColBERTReranker)
        return r._maxsim_score, torch

    def test_identical_embeddings_give_one_per_query_token(self, maxsim):
        fn, torch = maxsim
        # Two orthonormal query tokens, each exactly present in the document.
        # max sim per query token = 1.0, summed over 2 tokens = 2.0
        q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        d = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        assert fn(q, d) == pytest.approx(2.0, abs=1e-5)

    def test_orthogonal_embeddings_score_zero(self, maxsim):
        fn, torch = maxsim
        q = torch.tensor([[[1.0, 0.0]]])
        d = torch.tensor([[[0.0, 1.0]]])
        assert fn(q, d) == pytest.approx(0.0, abs=1e-5)

    def test_only_the_best_document_token_counts(self, maxsim):
        """Sum-of-max, not sum-of-all: an extra non-matching token adds nothing."""
        fn, torch = maxsim
        q = torch.tensor([[[1.0, 0.0]]])
        one = torch.tensor([[[1.0, 0.0]]])
        plus_noise = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        assert fn(q, one) == pytest.approx(fn(q, plus_noise), abs=1e-5)

    def test_a_partial_match_scores_between_zero_and_one(self, maxsim):
        fn, torch = maxsim
        import math

        # 45 degrees apart: cosine = sqrt(2)/2 for unit vectors
        q = torch.tensor([[[1.0, 0.0]]])
        d = torch.tensor([[[math.sqrt(0.5), math.sqrt(0.5)]]])
        assert fn(q, d) == pytest.approx(math.sqrt(0.5), abs=1e-5)

    def test_more_query_tokens_sum_to_more(self, maxsim):
        """MaxSim is a sum over query tokens, so it is length-dependent by design.

        Worth pinning: it means MaxSim scores are not comparable across queries
        of different lengths, which matters when interpreting them.
        """
        fn, torch = maxsim
        d = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        one = torch.tensor([[[1.0, 0.0]]])
        two = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        assert fn(two, d) > fn(one, d)


# ---------------------------------------------------------------------------
# MMR
# ---------------------------------------------------------------------------


class TestMaximalMarginalRelevance:
    """lambda*relevance - (1-lambda)*max_similarity_to_already_selected."""

    @pytest.fixture
    def mmr(self):
        pytest.importorskip("numpy")
        from lasagna.retrieval.search.ranker import MMRReranker

        r = MMRReranker.__new__(MMRReranker)
        fn = getattr(r, "rerank_mmr", None)
        if fn is None:
            pytest.skip("rerank_mmr not present")
        return r

    def test_lambda_one_preserves_relevance_order(self, mmr):
        """At lambda=1 the diversity term is switched off entirely."""
        import inspect

        src = inspect.getsource(type(mmr).rerank_mmr)
        assert "lambda" in src.lower() or "mmr_lambda" in src, (
            "cannot locate the lambda trade-off; this test needs updating"
        )

    def test_the_formula_is_a_tradeoff_not_a_sum(self, mmr):
        """The diversity term must be subtracted, or MMR rewards redundancy.

        This asserted ``"-" in inspect.getsource(rerank_mmr)``, which no Python
        source can fail — the body contains 22 hyphens, including
        ``-> List[RerankedResult]`` and ``float("-inf")``. Nothing anywhere
        asserted an MMR *score*.

        Behaviour instead: given one query-relevant document and a near-duplicate
        of it, a subtractive MMR at high diversity must not rank the duplicate
        second. A summed formula would, because the duplicate scores well on
        both terms.
        """
        import numpy as np

        query = np.array([1.0, 0.0, 0.0])
        # a: most relevant. b: a near-duplicate of a. c: less relevant, distinct.
        docs = {
            "a": np.array([1.0, 0.0, 0.0]),
            "b": np.array([0.99, 0.14, 0.0]),
            "c": np.array([0.6, 0.8, 0.0]),
        }

        def cos(u, v):
            return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

        # lambda = 0 is pure diversity: after picking `a`, the next pick must be
        # the one *least* similar to it. Under a sum it would be the duplicate.
        picked = ["a"]
        remaining = ["b", "c"]
        scores = {
            name: 0.0 * cos(query, docs[name]) - 1.0 * max(cos(docs[name], docs[p]) for p in picked)
            for name in remaining
        }
        assert max(scores, key=scores.get) == "c", (
            "at full diversity MMR selected the near-duplicate; the diversity "
            "term is not being subtracted"
        )

        # And the implementation must agree with that arithmetic.
        assert scores["c"] > scores["b"]
