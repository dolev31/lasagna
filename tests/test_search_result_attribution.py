"""Retrieved chunks must not vanish between the vector store and the caller.

The search orchestrator attaches a parent `Document` to every retrieved chunk.
When a chunk's document has no row in the metadata database, the result used to
be dropped with no `else` branch, no log and no counter.

That produced the worst possible output: dense search returned 20 candidates,
reranking and MMR ranked them correctly, and the caller received an empty list.
A benchmark run on such a corpus reported `ndcg@10 = 0.0` — a well-formed,
precise, entirely wrong zero that looks exactly like "your retriever found
nothing" rather than "the harness threw the answers away".

Two guards, because the bug needed both to happen:
  1. the ingest path used by benchmarks registers its documents, and
  2. dropping results for lack of metadata is loud, whoever ingested them.
"""

from __future__ import annotations

import pytest

from tests.test_event_loop_safety import executable_source


class TestDroppedResultsAreReported:
    """The orchestrator must never discard a ranked result in silence."""

    def test_missing_metadata_is_logged_not_swallowed(self):
        from lasagna.api import _search_orchestrator as orch

        src = executable_source(orch.SearchOrchestrator.search)
        assert "missing_metadata_ids" in src, (
            "the orchestrator must count results dropped for missing document "
            "metadata; without it, discarding every result is indistinguishable "
            "from retrieving nothing"
        )
        assert "logger.error" in src or "level = " in src, (
            "dropping the entire result set must be logged at error level"
        )

    def test_the_drop_has_an_else_branch(self):
        """The literal shape of the original bug: `if doc_metadata:` with no else."""
        from lasagna.api import _search_orchestrator as orch

        src = executable_source(orch.SearchOrchestrator.search)
        idx = src.find("doc_metadata_map.get(")
        assert idx != -1, "expected a document-metadata lookup in the search path"
        # The branch that consumes the lookup must handle the miss.
        assert "else:" in src[idx : idx + 1200], (
            "the `if doc_metadata:` branch must have an `else` that records the miss"
        )


class TestBenchmarkIngestRegistersDocuments:
    """Chunks in the vector store are useless if the documents are unknown."""

    def test_fast_ingest_saves_document_rows(self):
        from lasagna.knowledge_evaluation_sdk import _fast_ingest

        src = executable_source(_fast_ingest.fast_ingest_corpus)
        assert "save_document" in src, (
            "fast_ingest_corpus must register the documents it ingests, or every "
            "search over that corpus returns zero results"
        )

    def test_registration_failure_is_an_error_not_a_warning(self):
        from lasagna.knowledge_evaluation_sdk import _fast_ingest

        src = executable_source(_fast_ingest.fast_ingest_corpus)
        after = src[src.find("save_document") :]
        assert "logger.error" in after, (
            "a failed registration silently turns every subsequent metric into "
            "0.0; it cannot be a warning"
        )


class TestSmokeBenchmarkSupportsLiveMode:
    """The benchmark that makes the above testable without a download."""

    def test_smoke_provides_a_corpus(self):
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        corpus = benchmark_registry.get("smoke").load_corpus()
        assert corpus, "smoke must provide a corpus; it is the only hermetic live-mode path"
        assert len(corpus) >= 12

    def test_every_qrel_names_a_document_in_the_corpus(self):
        """A qrel pointing at a missing document silently caps recall below 1.0
        and would make a genuine regression indistinguishable from bad data."""
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        adapter = benchmark_registry.get("smoke")
        corpus = adapter.load_corpus()
        dangling = {
            doc_id
            for sample in adapter.load()
            for doc_id in sample.expected_docs
            if doc_id not in corpus
        }
        assert not dangling, f"qrels reference documents absent from the corpus: {sorted(dangling)}"

    def test_samples_carry_both_qrels_and_gold_answers(self):
        """Which is what lets one benchmark serve retrieval and generation
        modes — the property that makes it usable as the quickstart."""
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        samples = benchmark_registry.get("smoke").load()
        assert all(s.expected_docs for s in samples), "every sample needs qrels for IR metrics"
        assert all(s.ground_truth for s in samples), "every sample needs an answer for generation"
        assert all(s.context for s in samples), "gold context is needed for generation-only mode"

    def test_sampling_is_deterministic(self):
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        a = benchmark_registry.get("smoke")
        first = [s.query_id for s in a.load(n_samples=5, seed=7)]
        assert first == [s.query_id for s in a.load(n_samples=5, seed=7)]

    def test_unknown_split_is_refused(self):
        """No silent fallback to the one split it has."""
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        with pytest.raises(ValueError, match="only a 'test' split"):
            benchmark_registry.get("smoke").load(split="train")
