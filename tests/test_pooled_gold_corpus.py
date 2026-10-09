"""Most of these benchmarks ship gold context and nothing to search.

Eight of the thirty-one public benchmarks carry the answer's supporting
passages inline and no corpus — SQuAD v2, HotpotQA, MuSiQue and the five
RAGBench domains — so retrieval could not be measured on any of them. Ten more
carry nothing to retrieve at all and still cannot be.

Pooling those passages across the whole sample makes a searchable corpus: each
query has to find its own passages among every other query's. Pooling per query
would score 1.0 by construction and measure nothing.

**It is an easier task than the published benchmark, and that is the risk.**
HotpotQA and MuSiQue discard their distractor paragraphs at conversion, so the
pool has no hard negatives — only other questions' answers. Recall against it
reads higher than against a real corpus of the same size. A note in the docs
would not stop someone putting the two numbers in one table; a different
comparability key does, which is why `corpus_provenance` is in the key rather
than beside it.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk.benchmarks import BenchmarkSample, benchmark_registry
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BaseBenchmarkAdapter


class _Pooling(BaseBenchmarkAdapter):
    pools_gold_context = True

    def info(self):  # pragma: no cover - not exercised
        raise NotImplementedError

    def load(self, **kwargs):  # pragma: no cover - not exercised
        raise NotImplementedError


def _samples(contexts):
    return [
        BenchmarkSample(query_id=f"q{i}", question=f"question {i}", ground_truth="", context=ctx)
        for i, ctx in enumerate(contexts)
    ]


class TestThePoolIsARealRetrievalTask:
    def test_passages_from_every_sample_end_up_in_one_corpus(self):
        """The property that makes it non-trivial.

        With one query's passages only, the retriever picks from a corpus of
        size 1 and recall@10 is 1.0 whatever it does.
        """
        corpus = _Pooling().build_pooled_gold_corpus(_samples(["alpha", "beta", "gamma"]))
        assert len(corpus) == 3
        assert set(corpus.values()) == {"alpha", "beta", "gamma"}

    def test_a_multi_passage_context_is_split(self):
        corpus = _Pooling().build_pooled_gold_corpus(_samples(["first\n\nsecond\n\nthird"]))
        assert sorted(corpus.values()) == ["first", "second", "third"]

    def test_doc_ids_identify_their_source_query(self):
        """So a per-query row can say which passage was supposed to be found."""
        corpus = _Pooling().build_pooled_gold_corpus(_samples(["a\n\nb"]))
        assert set(corpus) == {"gold:q0:0", "gold:q0:1"}

    def test_blank_passages_are_dropped(self):
        corpus = _Pooling().build_pooled_gold_corpus(_samples(["a\n\n\n\n   \n\nb"]))
        assert sorted(corpus.values()) == ["a", "b"]

    def test_expected_docs_are_minted_alongside(self):
        """Without qrels every IR metric scores zero, which reads as a broken
        retriever rather than as missing labels."""
        samples = _samples(["a\n\nb", "c"])
        qrels = _Pooling().pooled_gold_expected_docs(samples)
        assert qrels["q0"] == {"gold:q0:0": 1, "gold:q0:1": 1}
        assert qrels["q1"] == {"gold:q1:0": 1}

    def test_every_qrel_names_a_document_that_exists(self):
        """A qrel pointing at a passage not in the corpus deflates the metric
        silently."""
        samples = _samples(["a\n\nb", "c", "d\n\ne\n\nf"])
        adapter = _Pooling()
        corpus = adapter.build_pooled_gold_corpus(samples)
        for docs in adapter.pooled_gold_expected_docs(samples).values():
            assert set(docs) <= set(corpus)


class TestBenchmarksWithNothingToRetrieveStaySilent:
    def test_no_context_returns_none_not_an_empty_corpus(self):
        """Natural Questions and TriviaQA ship a question and an answer string.

        An empty corpus would ingest cleanly and then score zero on every
        query, which looks like a retrieval failure rather than like a
        benchmark that cannot measure retrieval.
        """
        assert _Pooling().build_pooled_gold_corpus(_samples(["", "   "])) is None

    def test_an_empty_sample_list_returns_none(self):
        assert _Pooling().build_pooled_gold_corpus([]) is None

    def test_an_adapter_that_has_not_opted_in_pools_nothing(self):
        class _NotPooling(_Pooling):
            pools_gold_context = False

        assert _NotPooling().corpus_provenance == "native"


class TestProvenanceIsLoadBearing:
    def test_a_pooling_adapter_declares_itself(self):
        assert _Pooling().corpus_provenance == "pooled_gold"

    def test_provenance_is_part_of_the_comparability_key(self):
        """The enforcement. Documentation would not stop the two numbers
        landing in one leaderboard row."""
        from lasagna.knowledge_evaluation_sdk.bundle import (
            HELD_FIXED_FIELDS,
            comparability_key,
        )

        assert "corpus_provenance" in HELD_FIXED_FIELDS
        base = {f: "x" for f in HELD_FIXED_FIELDS}
        native = comparability_key({**base, "corpus_provenance": "native"})
        pooled = comparability_key({**base, "corpus_provenance": "pooled_gold"})
        assert native != pooled

    def test_it_reaches_the_result_and_the_bundle(self):
        from lasagna.knowledge_evaluation_sdk.runner import TaskResult

        row = TaskResult(
            task_key="t",
            benchmark_name="b",
            config_name="c",
            run_index=0,
            corpus_provenance="pooled_gold",
        ).to_dict()
        assert row["corpus_provenance"] == "pooled_gold"

    def test_the_default_is_native(self):
        from lasagna.knowledge_evaluation_sdk.runner import TaskResult

        result = TaskResult(task_key="t", benchmark_name="b", config_name="c", run_index=0)
        assert result.corpus_provenance == "native"


class TestWhichBenchmarksGain:
    """The claim is specific, so it is checked against the registry."""

    #: Samples carry inline gold context that can be pooled.
    POOLS = ["squad_v2", "hotpotqa", "musique"]

    #: Question and answer string only — nothing to retrieve, at any price.
    CANNOT = ["natural_questions", "triviaqa"]

    @pytest.mark.parametrize("name", POOLS)
    def test_these_can_now_measure_retrieval(self, name):
        adapter = benchmark_registry.get(name)
        assert adapter.pools_gold_context is True
        assert adapter.corpus_provenance == "pooled_gold"

    @pytest.mark.parametrize("name", CANNOT)
    def test_these_still_cannot_and_must_not_pretend_to(self, name):
        """Opting one of these in would produce an empty corpus and a run that
        scores zero on every retrieval metric."""
        adapter = benchmark_registry.get(name)
        assert adapter.pools_gold_context is False
        assert adapter.build_pooled_gold_corpus(_samples(["", ""])) is None

    @pytest.mark.parametrize("name", POOLS)
    def test_mode_resolution_now_says_live_and_says_why(self, name):
        from lasagna.knowledge_evaluation_sdk.experiment import resolve_mode

        mode, reason = resolve_mode("auto", name)
        assert mode == "live"
        assert "pooled" in reason and "easier" in reason

    @pytest.mark.parametrize("name", CANNOT)
    def test_mode_resolution_still_says_generation_only(self, name):
        from lasagna.knowledge_evaluation_sdk.experiment import resolve_mode

        assert resolve_mode("auto", name)[0] == "generation_only"

    def test_the_counts_in_the_docs_are_the_counts_in_the_registry(self):
        """The docstrings above and in `_base.py` name specific numbers.

        A count written in prose is a claim about the code, and this one had to
        be corrected once already — "roughly two-thirds" was written before
        anyone counted, and the answer was eight of thirty-one public benchmarks.
        """
        from lasagna.knowledge_evaluation_sdk.experiment import resolve_mode

        pooled = generation_only = 0
        for name in benchmark_registry.names():
            adapter = benchmark_registry.get(name)
            mode, _ = resolve_mode("auto", name)
            if mode != "live":
                generation_only += 1
            elif getattr(adapter, "pools_gold_context", False):
                # getattr, not attribute access: the private_eval adapters are
                # registered without subclassing BaseBenchmarkAdapter, so they
                # carry none of its class attributes. The harness reads this the
                # same way for the same reason.
                pooled += 1

        assert pooled == 8, f"docs say eight benchmarks pool, registry says {pooled}"
        assert generation_only == 10, (
            f"docs say ten benchmarks still cannot measure retrieval, "
            f"registry says {generation_only}"
        )

    def test_a_native_corpus_benchmark_is_unaffected(self):
        """Guards against pooling quietly taking over from a real corpus."""
        from lasagna.knowledge_evaluation_sdk.experiment import resolve_mode

        mode, reason = resolve_mode("auto", "beir/scifact")
        assert mode == "live"
        assert "pooled" not in reason
