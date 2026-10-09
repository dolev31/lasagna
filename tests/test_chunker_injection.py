"""A registered chunker must actually chunk the document.

`KnowledgePipeline(chunker=...)` was accepted, stored, and serialized — and
never reached ingestion. `chunk_document` took a *config* and built its own
`SemanticChunker`, so a chunking sweep produced byte-identical arms and a
difference of exactly zero.

That made chunking research impossible while appearing to support it, which is
the worst combination: the sweep runs, the numbers come out, and the conclusion
is confidently wrong.

These tests exercise `chunk_document` directly with a component, because that is
the seam. A test that only checked `pipeline.get_component("chunker")` would
have passed the entire time the feature did not work.
"""

from __future__ import annotations

import pytest

from lasagna.core.base_components import BaseChunker, Chunk
from lasagna.processing.chunker.semantic import adapt_component_chunks, chunk_document


class FixedSizeChunker(BaseChunker):
    """Splits every `size` characters. Deliberately unlike the built-ins."""

    def __init__(self, size: int = 10):
        self.size = size

    def chunk(self, text: str, **kwargs):
        return [
            Chunk(content=text[i : i + self.size], chunk_index=n)
            for n, i in enumerate(range(0, len(text), self.size))
        ]


class SentinelChunker(BaseChunker):
    """Returns one recognisable chunk, so its output is unmistakable."""

    def chunk(self, text: str, **kwargs):
        return [Chunk(content="SENTINEL-CHUNK", chunk_index=0)]


TEXT = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."


class TestTheComponentActuallyRuns:
    def test_injected_chunks_replace_the_built_in_ones(self):
        out = chunk_document(
            text=TEXT,
            document_id="d1",
            session_id="s1",
            filename="f.txt",
            chunker=SentinelChunker(),
        )
        assert [c for c, _ in out] == ["SENTINEL-CHUNK"]

    def test_two_chunkers_produce_different_chunks(self):
        """The ablation-is-real test. This was false before the fix."""
        a = chunk_document(
            text=TEXT,
            document_id="d",
            session_id="s",
            filename="f",
            chunker=FixedSizeChunker(size=10),
        )
        b = chunk_document(
            text=TEXT,
            document_id="d",
            session_id="s",
            filename="f",
            chunker=FixedSizeChunker(size=25),
        )
        assert [c for c, _ in a] != [c for c, _ in b]
        assert len(a) > len(b)

    def test_without_a_component_the_built_in_path_still_runs(self):
        """Injecting a knob must not change behaviour for anyone not using it."""
        out = chunk_document(text=TEXT, document_id="d", session_id="s", filename="f")
        assert out, "the default chunking path produced nothing"
        assert "SENTINEL-CHUNK" not in [c for c, _ in out]


class TestTheContractIsEnforced:
    def test_a_non_basechunker_is_rejected(self):
        class NotAChunker:
            def chunk(self, text, **kwargs):
                return []

        with pytest.raises(TypeError, match="BaseChunker"):
            chunk_document(
                text=TEXT,
                document_id="d",
                session_id="s",
                filename="f",
                chunker=NotAChunker(),
            )

    def test_the_error_points_at_the_validator(self):
        class NotAChunker:
            pass

        with pytest.raises(TypeError) as exc:
            chunk_document(
                text=TEXT,
                document_id="d",
                session_id="s",
                filename="f",
                chunker=NotAChunker(),
            )
        assert "component validate" in str(exc.value)

    def test_a_chunk_without_content_is_rejected_by_name(self):
        with pytest.raises(TypeError, match="content"):
            adapt_component_chunks([object()], TEXT)


class TestMetadataIngestDependsOn:
    """Downstream needs stable ids and character spans, not just text."""

    def _meta(self, chunker):
        out = chunk_document(
            text=TEXT,
            document_id="d1",
            session_id="s1",
            filename="f.txt",
            chunker=chunker,
        )
        return [m for _, m in out]

    def test_every_chunk_gets_a_stable_id(self):
        """Citation tracking and de-duplication both key on it."""
        metas = self._meta(FixedSizeChunker(size=12))
        assert all(m["stable_id"] for m in metas)

    def test_stable_ids_are_content_derived_and_reproducible(self):
        a = self._meta(FixedSizeChunker(size=12))
        b = self._meta(FixedSizeChunker(size=12))
        assert [m["stable_id"] for m in a] == [m["stable_id"] for m in b]

    def test_character_spans_are_filled_in_when_a_component_omits_them(self):
        """Citations carry char spans; a component returning only text would
        otherwise produce chunks that cannot be highlighted."""
        metas = self._meta(FixedSizeChunker(size=10))
        first = metas[0]
        assert first["end_char"] > first["start_char"]
        assert TEXT[first["start_char"] : first["end_char"]] == TEXT[:10]

    def test_spans_advance_rather_than_all_matching_the_first_occurrence(self):
        """Repeated text must map to successive positions, not position 0."""

        class RepeatChunker(BaseChunker):
            def chunk(self, text, **kwargs):
                # "eta" occurs in "beta", "zeta", "Eta", "theta" — so a
                # find-from-zero would map all four to the same offset.
                return [Chunk(content="eta") for _ in range(3)]

        metas = self._meta(RepeatChunker())
        starts = [m["start_char"] for m in metas]
        assert len(set(starts)) == 3, f"chunks share an offset: {starts}"
        assert starts == sorted(starts), "offsets should advance through the text"

    def test_a_chunk_not_present_in_the_source_gets_no_span(self):
        """A summarising chunker rewrites text, so there is no span to report.

        Inventing (0, len) would point a citation at unrelated text near the
        top of the document and highlight it confidently.
        """

        class SummarisingChunker(BaseChunker):
            def chunk(self, text, **kwargs):
                return [Chunk(content="A summary in words not present above.")]

        meta = self._meta(SummarisingChunker())[0]
        assert (meta["start_char"], meta["end_char"]) == (0, 0)

    def test_document_and_session_are_carried(self):
        meta = self._meta(SentinelChunker())[0]
        assert meta["document_id"] == "d1"
        assert meta["session_id"] == "s1"
        assert meta["filename"] == "f.txt"

    def test_a_component_supplied_span_is_respected(self):
        class SpanChunker(BaseChunker):
            def chunk(self, text, **kwargs):
                return [Chunk(content="gamma", start_char=11, end_char=16)]

        meta = self._meta(SpanChunker())[0]
        assert (meta["start_char"], meta["end_char"]) == (11, 16)


class TestTheStageListsStayTruthful:
    def test_chunker_has_moved_out_of_the_not_yet_injected_list(self):
        from lasagna.core.pipeline import (
            INGEST_TIME_STAGES,
            INGEST_TIME_STAGES_NOT_YET_INJECTED,
        )

        assert "chunker" in INGEST_TIME_STAGES
        assert "chunker" not in INGEST_TIME_STAGES_NOT_YET_INJECTED

    def test_the_stage_lists_partition_the_registry(self):
        from lasagna.core.pipeline import (
            ANSWER_TIME_STAGES,
            INGEST_TIME_STAGES,
            INGEST_TIME_STAGES_NOT_YET_INJECTED,
            QUERY_TIME_STAGES,
        )
        from lasagna.core.registry import COMPONENT_TYPES

        lists = [
            QUERY_TIME_STAGES,
            INGEST_TIME_STAGES,
            ANSWER_TIME_STAGES,
            INGEST_TIME_STAGES_NOT_YET_INJECTED,
        ]
        union = set().union(*(set(x) for x in lists))
        assert union == set(COMPONENT_TYPES)
        total = sum(len(x) for x in lists)
        assert total == len(COMPONENT_TYPES), "a stage appears in two lists"

    def test_ingest_passes_the_component_through(self):
        """The seam itself: `ingest` must forward what it resolved."""
        import inspect

        from lasagna.core.pipeline import KnowledgePipeline

        src = inspect.getsource(KnowledgePipeline.ingest)
        assert 'get_component("chunker")' in src
        assert "_chunker_override=chunker" in src


class TestRetrieverInjection:
    """`retriever` owns how candidates are found — the stage under test.

    It replaces the vector-store call outright rather than layering on top of
    it. Imposing the backend's filters and search params on someone else's
    algorithm would mean the component being measured is not the component that
    ran.
    """

    def test_scored_documents_adapt_to_backend_row_shape(self):
        from lasagna.api._search_orchestrator import _adapt_retriever_results
        from lasagna.core.base_components import ScoredDocument

        rows = _adapt_retriever_results(
            [
                ScoredDocument(
                    content="hello",
                    score=0.8,
                    document_id="d1",
                    chunk_id="c1",
                    metadata={"page": 2},
                )
            ]
        )
        assert rows[0].id == "c1"
        assert rows[0].score == pytest.approx(0.8)
        assert rows[0].entity["document_id"] == "d1"
        assert rows[0].entity["content"] == "hello"
        assert rows[0].entity["metadata"] == {"page": 2}

    def test_distance_mirrors_score_rather_than_being_invented(self):
        """A component reports similarity; it never computed a distance."""
        from lasagna.api._search_orchestrator import _adapt_retriever_results
        from lasagna.core.base_components import ScoredDocument

        row = _adapt_retriever_results([ScoredDocument(content="x", score=0.42)])[0]
        assert row.distance == pytest.approx(0.42)

    def test_a_wrong_return_type_is_rejected_by_name(self):
        from lasagna.api._search_orchestrator import _adapt_retriever_results

        with pytest.raises(TypeError, match="ScoredDocument"):
            _adapt_retriever_results([{"content": "not a ScoredDocument"}])

    def test_empty_results_are_fine(self):
        from lasagna.api._search_orchestrator import _adapt_retriever_results

        assert _adapt_retriever_results([]) == []
        assert _adapt_retriever_results(None) == []

    def test_the_orchestrator_branches_on_the_override(self):
        import inspect

        from lasagna.api._search_orchestrator import SearchOrchestrator

        src = inspect.getsource(SearchOrchestrator)
        assert "_retriever_override is not None" in src
        assert "_adapt_retriever_results(" in src


class TestDocumentProcessorInjection:
    def test_the_pipeline_branches_on_the_override(self):
        import inspect

        from lasagna.api._document_pipeline import DocumentPipeline

        src = inspect.getsource(DocumentPipeline)
        assert "_document_processor_override is not None" in src
        assert "_extract_text_with_ocr" in src, "the built-in path must survive"

    def test_a_non_processor_is_rejected(self):
        """The type check must name the ABC and the validator."""
        import inspect

        from lasagna.api._document_pipeline import DocumentPipeline

        src = inspect.getsource(DocumentPipeline)
        assert "BaseDocumentProcessor" in src
        assert "component validate" in src

    def test_a_processor_returning_no_text_is_rejected(self):
        import inspect

        from lasagna.api._document_pipeline import DocumentPipeline

        src = inspect.getsource(DocumentPipeline)
        assert "`text` attribute" in src


class TestEveryStageNowInjects:
    def test_nothing_is_left_uninjected(self):
        from lasagna.core.pipeline import INGEST_TIME_STAGES_NOT_YET_INJECTED

        assert INGEST_TIME_STAGES_NOT_YET_INJECTED == ()

    def test_the_pipeline_resolves_every_stage(self):
        import inspect

        from lasagna.core.pipeline import KnowledgePipeline
        from lasagna.core.registry import COMPONENT_TYPES

        src = inspect.getsource(KnowledgePipeline)
        # `agentic_pipeline` resolves through `_resolve_agentic_pipeline()`
        # rather than `get_component`, so accept either accessor. Asserting one
        # mechanism would have failed a stage that is in fact injected.
        missing = [
            c for c in COMPONENT_TYPES if f'get_component("{c}")' not in src and f'"{c}"' not in src
        ]
        assert not missing, f"stages the pipeline never resolves: {missing}"

    def test_each_stage_reaches_an_override_or_an_ingest_argument(self):
        """Resolution is not injection; check the value is actually passed on."""
        import inspect

        from lasagna.core.pipeline import (
            INGEST_TIME_STAGES,
            QUERY_TIME_STAGES,
            KnowledgePipeline,
        )

        src = inspect.getsource(KnowledgePipeline)

        # The kwarg each stage is actually injected under. `query_enhancer` uses
        # `_enhancer_override`, not `_query_enhancer_override`, and that is why
        # this loop used to carry `or "_agentic_pipeline_override" in src` --
        # a disjunct with no `stage` in it, so it was true on every iteration
        # and the loop could not fail for any stage. It was masking the one
        # stage whose name does not follow the pattern.
        OVERRIDE_KWARGS = {
            "reranker": "_reranker_override",
            "query_enhancer": "_enhancer_override",
            "embedder": "_embedder_override",
            "agentic_pipeline": "_agentic_pipeline_override",
            "retriever": "_retriever_override",
        }
        assert set(OVERRIDE_KWARGS) == set(QUERY_TIME_STAGES), (
            "a query-time stage was added or removed; name its override kwarg here"
        )
        for stage in QUERY_TIME_STAGES:
            kwarg = OVERRIDE_KWARGS[stage]
            assert f'"{kwarg}"' in src, (
                f"{stage} is resolved but never passed to the search path (looked for {kwarg!r})"
            )
        for stage in INGEST_TIME_STAGES:
            assert f"_{stage}_override=" in src, f"{stage} is resolved but never passed to ingest"
