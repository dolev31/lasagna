"""Tests a plugin author should copy along with the code.

The first is the one that matters: `ComponentTestHarness` is the same check
`lasagna component validate` runs, so passing it here means the component is
genuinely interchangeable — not merely that it registered.
"""

from __future__ import annotations

import pytest

from lasagna.core.base_components import ScoredDocument
from lasagna.core.testing import ComponentTestHarness

from lasagna_plugin_template.benchmark import TinyBenchmark
from lasagna_plugin_template.reranker import LengthReranker


def _docs():
    return [
        ScoredDocument(content="tiny", score=0.9, document_id="a", chunk_id="a1"),
        ScoredDocument(
            content="a good deal more words here", score=0.5, document_id="b", chunk_id="b1"
        ),
        ScoredDocument(content="middling length text", score=0.7, document_id="c", chunk_id="c1"),
    ]


class TestComponentContract:
    def test_satisfies_the_stage_contract(self):
        report = ComponentTestHarness().validate(LengthReranker())
        assert report.passed, f"contract failures: {report.failures}"

    def test_does_not_mutate_its_input(self):
        """The property that makes a comparison attributable: a component that
        edits what it was handed changes results for whatever runs after it."""
        docs = _docs()
        before = [(d.document_id, d.score) for d in docs]
        LengthReranker().rerank("q", docs)
        assert [(d.document_id, d.score) for d in docs] == before

    def test_honours_top_k(self):
        assert len(LengthReranker().rerank("q", _docs(), top_k=2)) == 2

    def test_actually_reorders(self):
        """A component indistinguishable from the control cannot be shown to
        have run at all."""
        order = [d.document_id for d in LengthReranker().rerank("q", _docs())]
        assert order == ["b", "c", "a"]

    def test_constructible_with_no_arguments(self):
        """The pipeline builds components from a config name, so it cannot
        supply constructor values."""
        assert LengthReranker() is not None


class TestBenchmarkContract:
    def test_info_and_load(self):
        adapter = TinyBenchmark()
        assert adapter.info().name == "template_tiny"
        samples = adapter.load()
        assert len(samples) == 3
        assert all(s.expected_docs for s in samples), "qrels are needed for IR metrics"
        assert all(s.ground_truth for s in samples), "answers are needed for generation"

    def test_corpus_supports_live_mode(self):
        assert TinyBenchmark().load_corpus(), "without a corpus, live mode is unavailable"

    def test_every_qrel_names_a_document_that_exists(self):
        """A dangling qrel silently caps recall and looks like a bad retriever."""
        adapter = TinyBenchmark()
        corpus = adapter.load_corpus()
        dangling = {d for s in adapter.load() for d in s.expected_docs if d not in corpus}
        assert not dangling, f"qrels point at missing documents: {dangling}"

    def test_sampling_is_deterministic(self):
        a = TinyBenchmark()
        assert [s.query_id for s in a.load(n_samples=2, seed=1)] == [
            s.query_id for s in a.load(n_samples=2, seed=1)
        ]

    def test_unknown_split_is_refused(self):
        with pytest.raises(ValueError, match="only a 'test' split"):
            TinyBenchmark().load(split="train")


class TestEntryPointsAreDeclared:
    def test_pyproject_declares_both_groups(self):
        try:
            import tomllib
        except ModuleNotFoundError:  # Python 3.10, where tomli may not be installed
            tomllib = pytest.importorskip("tomli")
        from pathlib import Path

        cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
        eps = cfg["project"]["entry-points"]
        assert "lasagna_components" in eps, "the component would never be discovered"
        assert "lasagna_benchmarks" in eps, "the benchmark would never be discovered"
        # "{stage}.{name}" — a malformed name is skipped at discovery.
        assert all("." in k for k in eps["lasagna_components"])
