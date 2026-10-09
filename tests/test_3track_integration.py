"""Integration tests for 3-track evaluation pipeline.

Tests every untested risk area identified during implementation review:
1. E2e live dual-config query logic (IR-forced + gen-config)
2. RAGAS per-sample merge indexing with _gen_ps_offset
3. Per-sample score split for e2e samples
4. Ingestion signature uniqueness across profiles
5. Gold passage ID mapping roundtrip through corpus_to_valid
6. n_samples deduplication for e2e
7. Cost computation with model-aware pricing
8. answer_metrics edge cases
9. Checkpoint resume with mixed old/new schema
10. Adapter deprecation redirect
11. Builder 3-track split correctness
12. Corpus assembler gold passage injection
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest


def _private_eval_dataset_dir() -> str:
    """Where the private_eval adapter looks for its built dataset.

    Asked of the adapter rather than spelled as a path relative to this file,
    which pointed outside the repository after the tests moved, so these two
    tests always skipped whether or not the dataset had been built.
    """
    from lasagna.knowledge_evaluation_sdk.private_eval.adapter import _default_dataset_dir

    return _default_dataset_dir()


# ---------------------------------------------------------------------------
# 1. answer_metrics comprehensive tests
# ---------------------------------------------------------------------------


class TestAnswerMetrics:
    def setup_method(self):
        from lasagna.knowledge_evaluation_sdk.answer_metrics import (
            exact_match,
            token_f1,
            is_abstention,
            normalize_answer,
        )

        self.em = exact_match
        self.f1 = token_f1
        self.abstain = is_abstention
        self.norm = normalize_answer

    def test_normalize_strips_articles_and_punctuation(self):
        assert self.norm("The Eiffel Tower!") == "eiffel tower"
        assert self.norm("  a  big   cat  ") == "big cat"
        assert self.norm("An Apple.") == "apple"

    def test_em_exact_match(self):
        assert self.em("Paris", "Paris") == 1.0
        assert self.em("the paris", "Paris") == 1.0  # article stripped
        assert self.em("London", "Paris") == 0.0

    def test_em_with_aliases(self):
        assert self.em("Tour Eiffel", "Eiffel Tower", aliases=["Tour Eiffel"]) == 1.0
        assert self.em("wrong", "Eiffel Tower", aliases=["Tour Eiffel"]) == 0.0

    def test_em_empty_references(self):
        """Regression: max() on empty generator would crash."""
        assert self.em("Paris", "", aliases=["", ""]) == 0.0
        assert self.em("Paris", "", aliases=[]) == 0.0

    def test_em_no_answer_policy(self):
        assert self.em("", "") == 1.0  # both empty = correct abstention
        assert self.em("I don't know", "") == 1.0  # abstention + empty GT
        assert self.em("", "Paris") == 0.0  # empty pred, non-empty GT
        assert self.em("Paris", "") == 0.0  # non-empty pred, empty GT

    def test_f1_partial_overlap(self):
        f1 = self.f1("the big cat sat", "big cat")
        assert 0.5 < f1 <= 1.0

    def test_f1_no_overlap(self):
        assert self.f1("completely wrong", "totally different") == 0.0

    def test_f1_empty_policy(self):
        assert self.f1("", "") == 1.0
        assert self.f1("", "answer") == 0.0
        assert self.f1("answer", "") == 0.0

    def test_f1_with_aliases_takes_max(self):
        f1_main = self.f1("Paris France", "City of Light")
        f1_alias = self.f1("Paris France", "City of Light", aliases=["Paris"])
        assert f1_alias >= f1_main  # alias "Paris" gives better F1

    def test_abstention_detection(self):
        assert self.abstain("") is True
        assert self.abstain("  ") is True
        assert self.abstain("I don't know") is True
        assert self.abstain("I dont know") is True
        assert self.abstain("The provided context does not contain the answer") is True
        assert self.abstain("Unable to determine from the given passages") is True
        assert self.abstain("The answer is 42") is False
        assert self.abstain("Paris is the capital of France") is False

    def test_f1_multiword_token_counting(self):
        """Token F1 should count token occurrences, not just set intersection."""
        # "cat cat" vs "cat" — precision = 1/2, recall = 1/1, F1 = 2/3
        f1 = self.f1("cat cat dog", "cat")
        assert abs(f1 - (2 * (1 / 3) * 1.0) / (1 / 3 + 1.0)) < 0.01


# ---------------------------------------------------------------------------
# 2. Ingestion signature tests
# ---------------------------------------------------------------------------


class TestIngestionSignature:
    def test_profiles_produce_unique_signatures(self):
        from lasagna.knowledge_evaluation_sdk.experiment import _INGESTION_TIME_KEYS
        from lasagna.core.session_config._profiles import resolve_ingestion_defaults

        sigs = {}
        for profile in ["speed", "standard", "balanced", "max_quality"]:
            defaults = resolve_ingestion_defaults(profile)
            ing = {k: v for k, v in defaults.items() if k in _INGESTION_TIME_KEYS}
            sig = hashlib.sha256(json.dumps(ing, sort_keys=True).encode()).hexdigest()[:16]
            sigs[profile] = sig

        assert len(set(sigs.values())) == 4, f"Signature collision: {sigs}"

    def test_query_time_changes_dont_affect_signature(self):
        from lasagna.knowledge_evaluation_sdk.experiment import _INGESTION_TIME_KEYS
        from lasagna.core.session_config._profiles import resolve_ingestion_defaults

        defaults = resolve_ingestion_defaults("standard")
        ing = {k: v for k, v in defaults.items() if k in _INGESTION_TIME_KEYS}
        sig1 = hashlib.sha256(json.dumps(ing, sort_keys=True).encode()).hexdigest()[:16]

        # Adding query-time params should NOT change the signature
        query_time_overrides = {"hybrid_alpha": 0.3, "enable_reranking": False, "enable_mmr": True}
        merged = {**ing}  # only ingestion keys
        for k, v in query_time_overrides.items():
            if k not in _INGESTION_TIME_KEYS:
                continue  # correctly excluded
            merged[k] = v
        sig2 = hashlib.sha256(json.dumps(merged, sort_keys=True).encode()).hexdigest()[:16]

        assert sig1 == sig2, "Query-time overrides leaked into ingestion signature"

    def test_ingestion_time_changes_alter_signature(self):
        from lasagna.knowledge_evaluation_sdk.experiment import _INGESTION_TIME_KEYS
        from lasagna.core.session_config._profiles import resolve_ingestion_defaults

        defaults = resolve_ingestion_defaults("standard")
        ing1 = {k: v for k, v in defaults.items() if k in _INGESTION_TIME_KEYS}
        sig1 = hashlib.sha256(json.dumps(ing1, sort_keys=True).encode()).hexdigest()[:16]

        ing2 = dict(ing1)
        ing2["embedding_model"] = "custom-model"
        sig2 = hashlib.sha256(json.dumps(ing2, sort_keys=True).encode()).hexdigest()[:16]

        assert sig1 != sig2, "Embedding model change didn't alter signature"

    def test_unknown_profile_returns_empty(self):
        from lasagna.core.session_config._profiles import resolve_ingestion_defaults

        result = resolve_ingestion_defaults("nonexistent_profile")
        assert result == {}


# ---------------------------------------------------------------------------
# 3. Gold passage ID mapping roundtrip
# ---------------------------------------------------------------------------


class TestGoldPassageMapping:
    def test_gold_ids_survive_md5_mapping(self):
        """Gold passage doc_ids must be present in corpus and map correctly."""
        dd = _private_eval_dataset_dir()
        if not os.path.exists(dd):
            pytest.skip("Dataset not built")

        # Load e2e corpus
        corpus_path = os.path.join(dd, "corpus/corpus_e2e.jsonl")
        corpus_ids = set()
        with open(corpus_path) as f:
            for line in f:
                corpus_ids.add(json.loads(line)["doc_id"])

        # Build corpus_to_valid mapping (same as experiment.py)
        corpus_to_valid = {}
        for cid in corpus_ids:
            h = hashlib.md5(cid.encode()).hexdigest()[:16]
            corpus_to_valid[cid] = f"doc_{h}"

        # Load e2e samples and verify all expected_docs map
        e2e_path = os.path.join(dd, "e2e_generation_eval.jsonl")
        total_expected = 0
        total_mapped = 0
        with open(e2e_path) as f:
            for line in f:
                d = json.loads(line)
                for doc_id in d.get("expected_docs", {}):
                    total_expected += 1
                    if doc_id in corpus_to_valid:
                        total_mapped += 1

        assert total_expected > 0, "No expected_docs found in e2e samples"
        assert total_mapped == total_expected, (
            f"{total_expected - total_mapped} expected_docs not in corpus"
        )

    def test_gold_ids_not_in_retrieval_corpus(self):
        dd = _private_eval_dataset_dir()
        if not os.path.exists(dd):
            pytest.skip("Dataset not built")

        ret_path = os.path.join(dd, "corpus/corpus_retrieval.jsonl")
        gold_count = 0
        with open(ret_path) as f:
            for line in f:
                if json.loads(line)["doc_id"].startswith("gold:"):
                    gold_count += 1
        assert gold_count == 0, f"Found {gold_count} gold passages in retrieval corpus"


# ---------------------------------------------------------------------------
# 4. Sample routing and per_sample_scores structure
# ---------------------------------------------------------------------------


class TestSampleRouting:
    """Test the dual-config query routing for e2e samples."""

    def test_e2e_samples_appear_in_both_lists(self):
        """E2e samples (expected_docs + ground_truth) must be in both IR and gen lists."""

        @dataclass
        class FakeSample:
            query_id: str
            expected_docs: Dict[str, int]
            ground_truth: str

        samples = [
            FakeSample("ir_only", {"doc1": 1}, ""),
            FakeSample("e2e_1", {"gold1": 1}, "answer1"),
            FakeSample("gen_only", {}, "answer2"),
        ]

        ir_samples = [s for s in samples if s.expected_docs]
        gen_samples = [s for s in samples if s.ground_truth]

        assert len(ir_samples) == 2  # ir_only + e2e_1
        assert len(gen_samples) == 2  # e2e_1 + gen_only
        assert "e2e_1" in {s.query_id for s in ir_samples}
        assert "e2e_1" in {s.query_id for s in gen_samples}

    def test_n_samples_dedup(self):
        @dataclass
        class FakeSample:
            query_id: str
            expected_docs: Dict[str, int]
            ground_truth: str

        samples = [
            FakeSample("ir1", {"d": 1}, ""),
            FakeSample("e2e1", {"g": 1}, "ans"),
            FakeSample("e2e2", {"g": 1}, "ans"),
            FakeSample("gen1", {}, "ans"),
        ]
        ir_samples = [s for s in samples if s.expected_docs]
        gen_samples = [s for s in samples if s.ground_truth]

        n_old = len(ir_samples) + len(gen_samples)  # 3 + 3 = 6 (wrong)
        n_new = len({s.query_id for s in ir_samples} | {s.query_id for s in gen_samples})

        assert n_old == 6
        assert n_new == 4  # ir1, e2e1, e2e2, gen1

    def test_ragas_merge_offset_correct(self):
        """RAGAS per-sample scores must merge at correct offset for e2e."""
        # Simulate: 3 IR entries + 2 gen entries
        per_sample_scores = [
            {"_sample_id": "ir1", "ndcg@10": 0.5},
            {"_sample_id": "ir2", "ndcg@10": 0.8},
            {"_sample_id": "e2e1", "ndcg@10": 1.0},  # IR entry for e2e
            {"_sample_id": "e2e1", "exact_match": 0.0, "latency_ms": 100},  # gen entry
            {"_sample_id": "gen1", "exact_match": 0.5, "latency_ms": 200},
        ]

        ir_samples_count = 3  # ir1, ir2, e2e1
        _gen_ps_offset = ir_samples_count

        # RAGAS returns 2 scores (for gen_samples: e2e1, gen1)
        ragas_results = [
            {"faithfulness": 0.9},
            {"faithfulness": 0.7},
        ]

        for i, ps in enumerate(ragas_results):
            target_idx = _gen_ps_offset + i
            if target_idx < len(per_sample_scores):
                per_sample_scores[target_idx].update(ps)

        # e2e1 gen entry should have faithfulness
        assert per_sample_scores[3]["_sample_id"] == "e2e1"
        assert per_sample_scores[3]["faithfulness"] == 0.9

        # gen1 entry should have faithfulness
        assert per_sample_scores[4]["_sample_id"] == "gen1"
        assert per_sample_scores[4]["faithfulness"] == 0.7

        # IR entries should NOT have faithfulness
        assert "faithfulness" not in per_sample_scores[0]
        assert "faithfulness" not in per_sample_scores[1]
        assert "faithfulness" not in per_sample_scores[2]


# ---------------------------------------------------------------------------
# 5. Token accounting
# ---------------------------------------------------------------------------


class TestTokenAccounting:
    """These were tests of a price table that no longer exists.

    They asserted `_get_cost_rates("gpt-4o") == {"input": 2.5, ...}` — the
    table's contents, not any behaviour of the package — and passed for as long
    as the table sat there converting token counts that were permanently zero.
    A green suite alongside `$0.00000` in every report.

    What replaces them tests the thing that was actually broken: whether tokens
    are counted at all.
    """

    def test_task_result_token_aggregation(self):
        from lasagna.knowledge_evaluation_sdk.runner import TaskResult

        tr = TaskResult(task_key="t", benchmark_name="b", config_name="c", run_index=0)
        tr.per_sample_scores = [
            {"latency_ms": 100, "input_tokens": 300, "output_tokens": 50, "llm_calls": 1},
            {"latency_ms": 200, "input_tokens": 700, "output_tokens": 90, "llm_calls": 2},
        ]
        tr.compute_latency_stats()
        assert tr.system_input_tokens == 1000
        assert tr.system_output_tokens == 140
        assert tr.system_llm_calls == 3
        assert tr.avg_latency_ms == 150.0

    def test_system_and_eval_are_kept_apart(self):
        """Folding them together makes an expensive judge read as an expensive
        pipeline, which reverses the conclusion a spend comparison is drawn for.
        """
        from lasagna.knowledge_evaluation_sdk.runner import TaskResult

        tr = TaskResult(
            task_key="t",
            benchmark_name="b",
            config_name="c",
            run_index=0,
            system_input_tokens=100,
            eval_input_tokens=9000,
        )
        assert tr.system_input_tokens == 100
        assert tr.total_input_tokens == 9100

    def test_no_dollar_field_survives_anywhere(self):
        """A stale rate in a published bundle is a fabricated number."""
        from lasagna.knowledge_evaluation_sdk.runner import TaskResult

        row = TaskResult(task_key="t", benchmark_name="b", config_name="c", run_index=0).to_dict()
        assert not [k for k in row if "usd" in k.lower() or "cost" in k.lower()], row

    def test_zero_tokens_is_distinguishable_from_unmeasured(self):
        from lasagna.knowledge_evaluation_sdk.runner import TaskResult

        silent = TaskResult(task_key="t", benchmark_name="b", config_name="c", run_index=0)
        counted = TaskResult(
            task_key="t",
            benchmark_name="b",
            config_name="c",
            run_index=0,
            system_llm_calls=1,
        )
        assert not silent.tokens_measured
        assert counted.tokens_measured


# ---------------------------------------------------------------------------
# 6. Adapter deprecation and routing
# ---------------------------------------------------------------------------


class TestAdapter:
    def test_corpus_file_routing(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.adapter import _CORPUS_FILE_MAP

        assert _CORPUS_FILE_MAP["retrieval"] == "corpus_retrieval.jsonl"
        assert _CORPUS_FILE_MAP["e2e_generation"] == "corpus_e2e.jsonl"
        assert _CORPUS_FILE_MAP["hallucination"] == "corpus_retrieval.jsonl"

    def test_deprecated_adapter_warns(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.adapter import (
            _DeprecatedGenerationAdapter,
            PrivateEvalAdapter,
        )

        delegate = MagicMock(spec=PrivateEvalAdapter)
        deprecated = _DeprecatedGenerationAdapter(delegate)

        with patch("lasagna.knowledge_evaluation_sdk.private_eval.adapter.logger") as mock_logger:
            deprecated.info()
            mock_logger.warning.assert_called_once()
            assert "deprecated" in mock_logger.warning.call_args[0][0].lower()

        # Second call should not warn again
        with patch("lasagna.knowledge_evaluation_sdk.private_eval.adapter.logger") as mock_logger:
            deprecated.info()
            mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Builder 3-track structure
# ---------------------------------------------------------------------------


class TestBuilderStructure:
    def test_extract_gold_passages(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.builder import DatasetBuilder

        samples = [
            {
                "sample_id": "s1",
                "context": "passage A\n\npassage B",
                "source_benchmark": "test",
                "domain": "general",
            },
            {"sample_id": "s2", "context": "", "source_benchmark": "test", "domain": "general"},
            {
                "sample_id": "s3",
                "context": "single passage",
                "source_benchmark": "test",
                "domain": "general",
            },
        ]

        gold = DatasetBuilder._extract_gold_passages(samples)

        assert len(gold) == 3  # s1 has 2 passages, s3 has 1
        assert "gold:test:s1:0" in gold
        assert "gold:test:s1:1" in gold
        assert "gold:test:s3:0" in gold
        assert gold["gold:test:s1:0"]["text"] == "passage A"
        assert gold["gold:test:s1:1"]["text"] == "passage B"

    def test_strip_expected_docs_deep_copy(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.builder import DatasetBuilder

        original = {
            "sample_id": "s1",
            "expected_docs": {"gold:test:s1:0": 1},
            "ground_truth_answers": ["ans1", "ans2"],
        }
        stripped = DatasetBuilder._strip_expected_docs(original)

        assert "expected_docs" not in stripped
        assert "expected_docs" in original  # original unchanged
        # Deep copy: mutating stripped list shouldn't affect original
        stripped["ground_truth_answers"].append("ans3")
        assert len(original["ground_truth_answers"]) == 2

    def test_add_gold_expected_docs(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.builder import DatasetBuilder

        samples = [
            {"sample_id": "s1", "has_gold_context": True},
            {"sample_id": "s2", "has_gold_context": False},
        ]
        gold_passages = {
            "gold:test:s1:0": {"source_sample_id": "s1", "text": "p"},
            "gold:test:s1:1": {"source_sample_id": "s1", "text": "q"},
        }

        DatasetBuilder._add_gold_expected_docs(samples, gold_passages)

        assert samples[0]["expected_docs"] == {
            "gold:test:s1:0": 1,
            "gold:test:s1:1": 1,
        }
        assert "expected_docs" not in samples[1]  # no gold context


# ---------------------------------------------------------------------------
# 8. Corpus assembler gold passage injection
# ---------------------------------------------------------------------------


class TestCorpusAssembler:
    def test_add_gold_passages(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.corpus_assembler import (
            CorpusAssembler,
            CorpusData,
        )

        corpus = CorpusData()
        corpus.documents = {
            "beir/nfcorpus:doc1": {
                "text": "hello world",
                "title": "",
                "source_benchmark": "beir/nfcorpus",
                "domain": "bio",
            },
        }
        corpus.total_chars = 11

        gold = {
            "gold:test:s1:0": {
                "text": "gold passage",
                "source_benchmark": "test",
                "domain": "general",
            },
        }

        assembler = CorpusAssembler()
        assembler.add_gold_passages(corpus, gold)

        assert len(corpus.documents) == 2
        assert "gold:test:s1:0" in corpus.documents
        assert corpus.benchmark_doc_counts["gold_passages"] == 1
        assert corpus.total_chars == 11 + 12  # "gold passage" = 12 chars

    def test_add_gold_no_duplicates(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.corpus_assembler import (
            CorpusAssembler,
            CorpusData,
        )

        corpus = CorpusData()
        corpus.documents = {
            "existing_id": {"text": "x", "title": "", "source_benchmark": "", "domain": ""}
        }

        gold = {"existing_id": {"text": "new text"}}  # same ID
        assembler = CorpusAssembler()
        assembler.add_gold_passages(corpus, gold)

        assert len(corpus.documents) == 1  # not duplicated
        assert corpus.documents["existing_id"]["text"] == "x"  # original preserved


# ---------------------------------------------------------------------------
# 9. Manifest leakage check
# ---------------------------------------------------------------------------


class TestManifest:
    def test_leakage_check_ignores_cross_track(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.manifest import (
            DatasetManifest,
            SplitStats,
        )

        m = DatasetManifest()
        m.splits = {
            "e2e_generation_eval": SplitStats(count=2, sample_ids=["s1", "s2"]),
            "e2e_generation_test": SplitStats(count=2, sample_ids=["s3", "s4"]),
            "hallucination_eval": SplitStats(count=2, sample_ids=["s1", "s2"]),  # same as e2e
        }
        leakage = m.check_leakage()
        assert len(leakage) == 0  # cross-track overlap is expected

    def test_leakage_check_catches_within_track(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.manifest import (
            DatasetManifest,
            SplitStats,
        )

        m = DatasetManifest()
        m.splits = {
            "retrieval_train": SplitStats(count=2, sample_ids=["s1", "s2"]),
            "retrieval_eval": SplitStats(count=2, sample_ids=["s2", "s3"]),  # s2 leaked!
        }
        leakage = m.check_leakage()
        assert len(leakage) == 1
        assert "s2" in leakage[0]


# ---------------------------------------------------------------------------
# 10. Domain-stratified significance test
# ---------------------------------------------------------------------------


class TestDomainStratifiedBootstrap:
    def test_pairs_by_sample_id_not_order(self):
        """Verify pairing uses _sample_id, not positional zip."""
        from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults
        from lasagna.knowledge_evaluation_sdk.runner import TaskResult

        # Config A: samples in order [s1, s2, s3]
        tr_a = TaskResult(
            task_key="t",
            benchmark_name="bench",
            config_name="config_a",
            run_index=0,
            per_sample_scores=[
                {"_sample_id": "s1", "_source_benchmark": "d1", "ndcg@10": 0.5},
                {"_sample_id": "s2", "_source_benchmark": "d1", "ndcg@10": 0.8},
                {"_sample_id": "s3", "_source_benchmark": "d2", "ndcg@10": 0.3},
            ],
        )
        # Config B: samples in REVERSED order [s3, s2, s1]
        tr_b = TaskResult(
            task_key="t",
            benchmark_name="bench",
            config_name="config_b",
            run_index=0,
            per_sample_scores=[
                {"_sample_id": "s3", "_source_benchmark": "d2", "ndcg@10": 0.4},
                {"_sample_id": "s2", "_source_benchmark": "d1", "ndcg@10": 0.9},
                {"_sample_id": "s1", "_source_benchmark": "d1", "ndcg@10": 0.6},
            ],
        )

        results = BenchmarkResults(task_results=[tr_a, tr_b])
        result = results.domain_stratified_significance_test(
            "config_a",
            "config_b",
            "ndcg@10",
            "bench",
            n_bootstrap=100,
        )

        # Should work without error (old zip-based would mismatch domains)
        assert "n_domains" in result
        assert result["n_domains"] == 2
        assert set(result["domains"]) == {"d1", "d2"}


# ---------------------------------------------------------------------------
# 11. Shared corpus key mapping
# ---------------------------------------------------------------------------


class TestSharedCorpusKey:
    def test_3track_corpus_mapping(self):
        from lasagna.knowledge_evaluation_sdk.experiment import (
            _SHARED_CORPUS_KEY,
            _CORPUS_ID_MAP,
        )

        # Retrieval and hallucination share retrieval corpus
        assert _SHARED_CORPUS_KEY["private_eval/retrieval"] == "private_eval/retrieval_corpus"
        assert _SHARED_CORPUS_KEY["private_eval/hallucination"] == "private_eval/retrieval_corpus"

        # E2e has its own corpus
        assert _SHARED_CORPUS_KEY["private_eval/e2e_generation"] == "private_eval/e2e_corpus"

        # CORPUS_ID_MAP matches
        assert _CORPUS_ID_MAP["private_eval/retrieval"] == "retrieval_corpus"
        assert _CORPUS_ID_MAP["private_eval/hallucination"] == "retrieval_corpus"
        assert _CORPUS_ID_MAP["private_eval/e2e_generation"] == "e2e_corpus"

    def test_deprecated_generation_mapped(self):
        from lasagna.knowledge_evaluation_sdk.experiment import _SHARED_CORPUS_KEY

        assert "private_eval/generation" in _SHARED_CORPUS_KEY


# ---------------------------------------------------------------------------
# 12. BenchmarkTask corpus_fingerprint field
# ---------------------------------------------------------------------------


class TestBenchmarkTask:
    def test_has_corpus_fingerprint(self):
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

        task = BenchmarkTask(
            benchmark_name="test",
            config_name="cfg",
            config_overrides={},
            run_index=0,
        )
        assert hasattr(task, "corpus_fingerprint")
        assert task.corpus_fingerprint == ""

    def test_task_result_separates_system_and_eval_spend(self):
        """Was `test_task_result_has_cost_fields`, asserting the presence of
        `system_cost_usd` and `eval_cost_usd` — two fields that existed, were
        serialised, and were permanently 0.0. Presence was never the property
        worth testing.
        """
        from lasagna.knowledge_evaluation_sdk.runner import TaskResult

        tr = TaskResult(
            task_key="t",
            benchmark_name="b",
            config_name="c",
            run_index=0,
            system_input_tokens=10,
            eval_input_tokens=90,
        )
        d = tr.to_dict()
        assert d["system_input_tokens"] == 10
        assert d["eval_input_tokens"] == 90
        assert d["total_input_tokens"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
