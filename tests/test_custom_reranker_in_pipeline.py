"""A component written against the documented contract must survive the pipeline.

This is the package's central claim: implement one ABC, swap one stage, get a
difference attributable to your change. It was false. The orchestrator called
``rerank(query=..., results=[dict], top_k=..., return_metrics=..., config=...)``
while ``BaseReranker.rerank`` declares ``(query, documents: [ScoredDocument],
top_k, **kwargs)`` — ``**kwargs`` absorbed ``results`` and ``documents`` was
never filled, so the call raised ``TypeError``.

What made it survive so long is that every documented check exercises the
component *outside* the pipeline. `lasagna component validate` passes, the
guide's isolated baseline comparison passes, `pipeline.replace()` +
`get_component()` passes — and then the benchmark, the step the whole exercise
exists for, is the first thing that runs the component for real.

So these tests deliberately call through the orchestrator's own calling
convention rather than the ABC's. Testing the ABC against itself is exactly the
check that already existed and already passed.
"""

from __future__ import annotations

import copy

import pytest

from lasagna.core.base_components import BaseReranker, ScoredDocument
from lasagna.retrieval.search.ranker import (
    RerankedResult,
    adapt_reranker_for_pipeline,
)


class GuideReranker(BaseReranker):
    """Copied from docs/guide.md section 4, which is what users will write."""

    def rerank(self, query, documents, top_k=5, **kwargs):
        out = [copy.deepcopy(d) for d in documents]
        out.sort(key=lambda d: len(d.content.split()), reverse=True)
        return out[:top_k]


class StrictReranker(BaseReranker):
    """Declares no **kwargs at all — still legal, and must still work."""

    def rerank(self, query, documents, top_k=5):
        return list(documents)[:top_k]


@pytest.fixture
def hits():
    return [
        {"content": "short", "score": 0.9, "document_id": "d1", "chunk_id": "c1", "metadata": {}},
        {
            "content": "a much longer passage with many more words in it",
            "score": 0.5,
            "document_id": "d2",
            "chunk_id": "c2",
            "metadata": {"page": 4},
        },
        {
            "content": "middling length text here",
            "score": 0.7,
            "document_id": "d3",
            "chunk_id": "c3",
            "metadata": {},
        },
    ]


class TestThePipelineCanCallIt:
    """The regression, at both orchestrator call sites."""

    def test_plain_call_site(self, hits):
        """_search_orchestrator.py invokes it exactly this way."""
        adapted = adapt_reranker_for_pipeline(GuideReranker())
        out = adapted.rerank(query="q", results=hits, top_k=3, config=None)
        assert [r.document_id for r in out] == ["d2", "d3", "d1"]

    def test_metrics_call_site(self, hits):
        """The other site unpacks two values and reads rerank_ms."""
        adapted = adapt_reranker_for_pipeline(GuideReranker())
        out, metrics = adapted.rerank(
            query="q", results=hits, top_k=3, return_metrics=True, config=None
        )
        assert [r.document_id for r in out] == ["d2", "d3", "d1"]
        assert "rerank_ms" in metrics and "mmr_ms" in metrics

    def test_adaptive_keywords_do_not_break_it(self, hits):
        """The pipeline passes adaptive-k keywords a component never saw."""
        adapted = adapt_reranker_for_pipeline(GuideReranker())
        out = adapted.rerank(
            query="q",
            results=hits,
            top_k=2,
            config=None,
            adaptive=True,
            adaptive_min_k=2,
            adaptive_max_k=20,
        )
        assert len(out) == 2

    def test_a_component_without_kwargs_still_works(self, hits):
        """Forwarding pipeline-internal keywords would break this one."""
        adapted = adapt_reranker_for_pipeline(StrictReranker())
        assert len(adapted.rerank(query="q", results=hits, top_k=2, config=None)) == 2


class TestTheResultShapeIsWhatTheOrchestratorConsumes:
    def test_returns_reranked_results(self, hits):
        out = adapt_reranker_for_pipeline(GuideReranker()).rerank(query="q", results=hits, top_k=3)
        assert all(isinstance(r, RerankedResult) for r in out)

    def test_ranks_are_assigned(self, hits):
        out = adapt_reranker_for_pipeline(GuideReranker()).rerank(query="q", results=hits, top_k=3)
        assert [r.final_rank for r in out] == [0, 1, 2]

    def test_original_rank_and_score_survive_reordering(self, hits):
        """Needed to report how far the component moved each result."""
        out = adapt_reranker_for_pipeline(GuideReranker()).rerank(query="q", results=hits, top_k=3)
        top = out[0]
        assert top.document_id == "d2"
        assert top.original_rank == 1, "d2 was second in the input"
        assert top.original_score == pytest.approx(0.5), "its retrieval score, not its rerank score"

    def test_metadata_is_carried_through(self, hits):
        out = adapt_reranker_for_pipeline(GuideReranker()).rerank(query="q", results=hits, top_k=3)
        assert out[0].metadata == {"page": 4}

    def test_model_name_is_the_components_own(self, hits):
        """Reported in the response; must not claim to be a cross-encoder."""
        adapted = adapt_reranker_for_pipeline(GuideReranker())
        assert adapted.cross_encoder.model_name == "GuideReranker"


class TestInternalRerankersAreUntouched:
    def test_a_non_abc_reranker_passes_through_unwrapped(self):
        """Wrapping the built-in reranker would break the pipeline instead."""

        class Internal:
            def rerank(self, query, results, top_k=5, **kwargs):
                return results

        internal = Internal()
        assert adapt_reranker_for_pipeline(internal) is internal

    def test_the_shipped_cross_encoder_is_not_wrapped(self):
        from lasagna.retrieval.search.ranker import CrossEncoderReranker

        inner = CrossEncoderReranker.__new__(CrossEncoderReranker)
        assert adapt_reranker_for_pipeline(inner) is inner


class TestTheOriginalFailureIsReal:
    """Documents what broke, so the adapter is not mistaken for ceremony."""

    def test_calling_the_abc_the_pipeline_way_fails_without_the_adapter(self, hits):
        with pytest.raises(TypeError, match="documents"):
            GuideReranker().rerank(query="q", results=hits, top_k=3, config=None)


class TestEmptyAndEdgeCases:
    def test_no_candidates(self):
        assert adapt_reranker_for_pipeline(GuideReranker()).rerank(query="q", results=[]) == []

    def test_top_k_larger_than_candidates(self, hits):
        out = adapt_reranker_for_pipeline(GuideReranker()).rerank(query="q", results=hits, top_k=99)
        assert len(out) == 3

    def test_a_component_returning_unknown_chunks_does_not_crash(self):
        """A generative or synthesising reranker may emit chunks it invented."""

        class Inventive(BaseReranker):
            def rerank(self, query, documents, top_k=5, **kwargs):
                return [ScoredDocument(content="new", score=1.0, document_id="x", chunk_id="zz")]

        out = adapt_reranker_for_pipeline(Inventive()).rerank(
            query="q",
            results=[
                {
                    "content": "a",
                    "score": 0.5,
                    "document_id": "d1",
                    "chunk_id": "c1",
                    "metadata": {},
                }
            ],
        )
        assert out[0].chunk_id == "zz"
        assert out[0].final_rank == 0
