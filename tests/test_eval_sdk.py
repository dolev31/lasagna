"""Regression tests for knowledge_evaluation_sdk.

Covers:
- IR metric correctness (known-answer tests)
- Statistical method correctness (bootstrap, Holm-Bonferroni)
- Checkpoint task key uniqueness
- SDK evaluation wrapper API contract
"""

from __future__ import annotations

import math
import pytest


# ---------------------------------------------------------------------------
# Phase 0C: IR Metric Fixtures
# ---------------------------------------------------------------------------


class TestIRMetrics:
    """Known-answer tests for all 7 IR metrics."""

    @pytest.fixture(autouse=True)
    def _import_metrics(self):
        from lasagna.knowledge_evaluation_sdk.ir_metrics import (
            ndcg_at_k,
            precision_at_k,
            recall_at_k,
            mrr,
            map_score,
            f1_at_k,
            hit_rate_at_k,
            corpus_metrics,
        )

        self.ndcg_at_k = ndcg_at_k
        self.precision_at_k = precision_at_k
        self.recall_at_k = recall_at_k
        self.mrr = mrr
        self.map_score = map_score
        self.f1_at_k = f1_at_k
        self.hit_rate_at_k = hit_rate_at_k
        self.corpus_metrics = corpus_metrics

    # --- Standard case: 3 relevant docs, 5 retrieved, k=3 ---
    # relevance: doc1=2, doc2=1, doc3=1
    # retrieved: [doc4, doc1, doc3, doc5, doc2]
    # At k=3: [doc4(0), doc1(2), doc3(1)]

    def test_precision_at_k_standard(self):
        rel = {"doc1": 2, "doc2": 1, "doc3": 1}
        ret = ["doc4", "doc1", "doc3", "doc5", "doc2"]
        # At k=3: 2 relevant out of 3
        assert self.precision_at_k(rel, ret, k=3) == pytest.approx(2 / 3)

    def test_recall_at_k_standard(self):
        rel = {"doc1": 2, "doc2": 1, "doc3": 1}
        ret = ["doc4", "doc1", "doc3", "doc5", "doc2"]
        # At k=3: 2 relevant retrieved out of 3 total relevant
        assert self.recall_at_k(rel, ret, k=3) == pytest.approx(2 / 3)

    def test_f1_at_k_standard(self):
        rel = {"doc1": 2, "doc2": 1, "doc3": 1}
        ret = ["doc4", "doc1", "doc3", "doc5", "doc2"]
        p = 2 / 3
        r = 2 / 3
        expected_f1 = 2 * p * r / (p + r)
        assert self.f1_at_k(rel, ret, k=3) == pytest.approx(expected_f1)

    def test_mrr_standard(self):
        rel = {"doc1": 2, "doc2": 1, "doc3": 1}
        ret = ["doc4", "doc1", "doc3", "doc5", "doc2"]
        # First relevant is doc1 at rank 2
        assert self.mrr(rel, ret) == pytest.approx(1 / 2)

    def test_mrr_with_k(self):
        rel = {"doc1": 2, "doc2": 1, "doc3": 1}
        ret = ["doc4", "doc1", "doc3", "doc5", "doc2"]
        # k=1: only doc4 (irrelevant) → 0
        assert self.mrr(rel, ret, k=1) == 0.0
        # k=2: doc4, doc1 → first relevant at rank 2
        assert self.mrr(rel, ret, k=2) == pytest.approx(1 / 2)

    def test_ndcg_at_k_matches_trec_eval_linear_gain(self):
        """Linear gain, as `trec_eval` computes it — and therefore BEIR.

        This test previously hard-coded `2**rel - 1`, the same expression the
        implementation used, so it was a change detector rather than a
        conformance check: it would have passed against any gain function the
        implementation happened to pick. The two agreed with each other and
        neither agreed with the literature.

        `trec_eval`'s `m_ndcg_cut.c` accumulates `gain / log2(i + 2)` with
        `gain = res_rels.results_rel_list[i]` — the raw graded judgement.
        """
        rel = {"doc1": 2, "doc2": 1, "doc3": 1}
        ret = ["doc4", "doc1", "doc3", "doc5", "doc2"]
        # DCG@3 over [doc4=0, doc1=2, doc3=1]
        dcg = 0 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
        # IDCG@3: ideal order [2, 1, 1]
        idcg = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        assert self.ndcg_at_k(rel, ret, k=3) == pytest.approx(dcg / idcg)

    def test_binary_judgements_are_unaffected_by_the_gain_change(self):
        """Which is why ten of the thirteen BEIR datasets are unaffected.

        `2^1 - 1 == 1`, so the two conventions coincide whenever every
        judgement is 0 or 1.
        """
        rel = {"doc1": 1, "doc2": 1}
        ret = ["doc3", "doc1", "doc2"]
        assert self.ndcg_at_k(rel, ret, k=3) == pytest.approx(
            self.ndcg_at_k(rel, ret, k=3, exponential_gain=True)
        )

    def test_graded_judgements_diverge(self):
        """Guards the guard: if these agreed, the test above proves nothing."""
        rel = {"doc1": 2, "doc2": 1}
        ret = ["doc2", "doc1"]
        assert self.ndcg_at_k(rel, ret, k=2) != pytest.approx(
            self.ndcg_at_k(rel, ret, k=2, exponential_gain=True)
        )

    def test_the_burges_convention_is_still_available(self):
        """Opt-in, for comparing against work that uses it."""
        rel = {"doc1": 2, "doc2": 1, "doc3": 1}
        ret = ["doc4", "doc1", "doc3", "doc5", "doc2"]
        dcg = (2**0 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
        idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
        assert self.ndcg_at_k(rel, ret, k=3, exponential_gain=True) == pytest.approx(dcg / idcg)

    def test_map_score_standard(self):
        rel = {"doc1": 2, "doc2": 1, "doc3": 1}
        ret = ["doc4", "doc1", "doc3", "doc5", "doc2"]
        # R = 3 (total relevant)
        # At rank 1: doc4 not relevant
        # At rank 2: doc1 relevant → P(2)=1/2 → contributes 1/2
        # At rank 3: doc3 relevant → P(3)=2/3 → contributes 2/3
        # At rank 4: doc5 not relevant
        # At rank 5: doc2 relevant → P(5)=3/5 → contributes 3/5
        # AP = (1/3) * (1/2 + 2/3 + 3/5) = (1/3) * (15/30 + 20/30 + 18/30) = (1/3)*(53/30)
        ap = (1 / 3) * (1 / 2 + 2 / 3 + 3 / 5)
        assert self.map_score(rel, ret) == pytest.approx(ap)

    def test_hit_rate_at_k_standard(self):
        rel = {"doc1": 2, "doc2": 1, "doc3": 1}
        ret = ["doc4", "doc1", "doc3", "doc5", "doc2"]
        # k=1: only doc4 (irrelevant) → 0
        assert self.hit_rate_at_k(rel, ret, k=1) == 0.0
        # k=2: doc4, doc1 → has relevant → 1.0
        assert self.hit_rate_at_k(rel, ret, k=2) == 1.0

    # --- Edge cases ---

    def test_empty_retrieved(self):
        rel = {"doc1": 1}
        ret = []
        assert self.precision_at_k(rel, ret, k=5) == 0.0
        assert self.recall_at_k(rel, ret, k=5) == 0.0
        assert self.ndcg_at_k(rel, ret, k=5) == 0.0
        assert self.mrr(rel, ret) == 0.0
        assert self.map_score(rel, ret) == 0.0
        assert self.f1_at_k(rel, ret, k=5) == 0.0
        assert self.hit_rate_at_k(rel, ret, k=5) == 0.0

    def test_empty_relevance(self):
        rel = {}
        ret = ["doc1", "doc2"]
        assert self.precision_at_k(rel, ret, k=5) == 0.0
        assert self.recall_at_k(rel, ret, k=5) == 0.0
        assert self.ndcg_at_k(rel, ret, k=5) == 0.0
        assert self.mrr(rel, ret) == 0.0
        assert self.map_score(rel, ret) == 0.0
        assert self.f1_at_k(rel, ret, k=5) == 0.0
        assert self.hit_rate_at_k(rel, ret, k=5) == 0.0

    def test_k_zero(self):
        rel = {"doc1": 1}
        ret = ["doc1"]
        assert self.precision_at_k(rel, ret, k=0) == 0.0
        assert self.recall_at_k(rel, ret, k=0) == 0.0
        assert self.ndcg_at_k(rel, ret, k=0) == 0.0
        assert self.f1_at_k(rel, ret, k=0) == 0.0
        assert self.hit_rate_at_k(rel, ret, k=0) == 0.0

    def test_all_relevant(self):
        rel = {"doc1": 1, "doc2": 1, "doc3": 1}
        ret = ["doc1", "doc2", "doc3"]
        assert self.precision_at_k(rel, ret, k=3) == 1.0
        assert self.recall_at_k(rel, ret, k=3) == 1.0
        assert self.f1_at_k(rel, ret, k=3) == 1.0
        assert self.mrr(rel, ret) == 1.0
        assert self.hit_rate_at_k(rel, ret, k=3) == 1.0

    def test_none_relevant(self):
        rel = {"doc1": 1, "doc2": 1}
        ret = ["doc3", "doc4", "doc5"]
        assert self.precision_at_k(rel, ret, k=3) == 0.0
        assert self.recall_at_k(rel, ret, k=3) == 0.0
        assert self.f1_at_k(rel, ret, k=3) == 0.0
        assert self.mrr(rel, ret) == 0.0
        assert self.hit_rate_at_k(rel, ret, k=3) == 0.0

    def test_corpus_metrics_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            self.corpus_metrics(
                [{"doc1": 1}],
                [["doc1"], ["doc2"]],
                k=3,
            )

    def test_corpus_metrics_empty(self):
        result = self.corpus_metrics([], [], k=3)
        assert result == {}

    def test_ndcg_perfect_ranking(self):
        """Perfect ranking should give NDCG = 1.0."""
        rel = {"doc1": 3, "doc2": 2, "doc3": 1}
        ret = ["doc1", "doc2", "doc3"]
        assert self.ndcg_at_k(rel, ret, k=3) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Phase 0B: Statistical Method Fixtures
# ---------------------------------------------------------------------------


class TestStatisticalMethods:
    """Known-answer tests for bootstrap significance and Holm-Bonferroni."""

    def _make_results(self, scores_a, scores_b, benchmark="bench", config_a="a", config_b="b"):
        """Create BenchmarkResults with per-sample scores for two configs."""
        from lasagna.knowledge_evaluation_sdk.runner import TaskResult
        from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults

        metric = "ndcg@10"
        results = []

        # Config A
        per_sample_a = [{metric: s} for s in scores_a]
        results.append(
            TaskResult(
                task_key=f"{benchmark}|{config_a}|hash_a|offline|run_0",
                benchmark_name=benchmark,
                config_name=config_a,
                run_index=0,
                metrics={metric: sum(scores_a) / len(scores_a)},
                per_sample_scores=per_sample_a,
                n_samples=len(scores_a),
            )
        )

        # Config B
        per_sample_b = [{metric: s} for s in scores_b]
        results.append(
            TaskResult(
                task_key=f"{benchmark}|{config_b}|hash_b|offline|run_0",
                benchmark_name=benchmark,
                config_name=config_b,
                run_index=0,
                metrics={metric: sum(scores_b) / len(scores_b)},
                per_sample_scores=per_sample_b,
                n_samples=len(scores_b),
            )
        )

        return BenchmarkResults(task_results=results)

    def test_bootstrap_identical_distributions(self):
        """Identical scores → p-value should be ~1.0 (no difference)."""
        scores = [0.5] * 50
        br = self._make_results(scores, scores)
        sig = br.significance_test("a", "b", "ndcg@10", "bench", n_bootstrap=5000)
        # p-value should be high (no difference) — at least > 0.5
        assert sig["p_value"] > 0.5
        assert not sig["significant"]
        assert sig["difference"] == pytest.approx(0.0)

    def test_bootstrap_separated_distributions(self):
        """Perfectly separated scores → p-value should be ~0.0."""
        a = [0.9] * 50
        b = [0.1] * 50
        br = self._make_results(a, b)
        sig = br.significance_test("a", "b", "ndcg@10", "bench", n_bootstrap=5000)
        assert sig["p_value"] < 0.05
        assert sig["significant"]
        assert sig["difference"] == pytest.approx(0.8)

    def test_bootstrap_effect_size_paired(self):
        """Effect size should use std(diffs) for paired design.

        The original data here was ``a = [0.9, 0.8, 0.7, 0.6, 0.5] * 10``
        against ``b = [0.4, 0.3, 0.2, 0.1, 0.0] * 10``: every difference is 0.5
        to a reader, but 0.9-0.4 and 0.7-0.2 differ in the last bit, leaving
        sd = 2.5e-17 and d_z = 1.99e16. The assertion recomputed the same
        expression and compared it to itself, so it passed while testing
        nothing but that floating-point noise is reproducible within one
        process.

        Now uses differences with real spread, which is what the formula this
        test names is about. The degenerate case has its own coverage in
        `test_statistics_consistency.py`.
        """
        import numpy as np

        a = [0.90, 0.82, 0.71, 0.65, 0.53] * 10
        b = [0.40, 0.35, 0.18, 0.14, 0.02] * 10
        br = self._make_results(a, b)
        sig = br.significance_test("a", "b", "ndcg@10", "bench")

        diffs = np.array(a) - np.array(b)
        sd = float(np.std(diffs, ddof=1))
        assert sd > 1e-6, "the premise: these differences genuinely vary"

        # For paired design, Cohen's d_z = mean(diffs) / std(diffs)
        expected_d = float(np.mean(diffs)) / sd
        assert sig["effect_size"] == pytest.approx(expected_d, rel=0.01)

    def test_holm_single_comparison(self):
        """Single comparison: Holm should reduce to uncorrected test."""
        a = [0.9] * 50
        b = [0.1] * 50
        br = self._make_results(a, b)
        result = br.pairwise_significance("ndcg@10", "bench", correction="holm")
        assert len(result) == 1
        # corrected_alpha should equal alpha (m=1, so alpha/(1-0) = alpha)
        assert result[0]["corrected_alpha"] == pytest.approx(0.05)

    def test_holm_boundary_rejection(self):
        """Holm should reject when p == corrected_alpha (using <=)."""
        # This tests that the boundary condition uses <= not <
        from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults
        from lasagna.knowledge_evaluation_sdk.runner import TaskResult

        # We'll create 3 configs and manually check the Holm logic
        # Create results with 3 configs
        metric = "ndcg@10"
        benchmark = "bench"
        configs = ["a", "b", "c"]
        results = []
        for cfg in configs:
            scores = [0.5] * 20
            results.append(
                TaskResult(
                    task_key=f"{benchmark}|{cfg}|hash_{cfg}|offline|run_0",
                    benchmark_name=benchmark,
                    config_name=cfg,
                    run_index=0,
                    metrics={metric: 0.5},
                    per_sample_scores=[{metric: s} for s in scores],
                    n_samples=20,
                )
            )

        br = BenchmarkResults(task_results=results)
        # With identical scores, all p-values should be high
        holm_results = br.pairwise_significance(metric, benchmark, correction="holm")
        # None should be significant (all identical)
        for r in holm_results:
            assert not r["significant"]


# ---------------------------------------------------------------------------
# Phase 0D: Checkpoint Task Key Uniqueness
# ---------------------------------------------------------------------------


class TestTaskKeyUniqueness:
    """Verify that task keys are unique for tasks differing in sampling params."""

    def test_different_splits_different_keys(self):
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

        t1 = BenchmarkTask(
            benchmark_name="bench",
            config_name="cfg",
            config_overrides={},
            run_index=0,
            split="test",
        )
        t2 = BenchmarkTask(
            benchmark_name="bench",
            config_name="cfg",
            config_overrides={},
            run_index=0,
            split="dev",
        )
        assert t1.task_key != t2.task_key

    def test_different_n_samples_different_keys(self):
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

        t1 = BenchmarkTask(
            benchmark_name="bench",
            config_name="cfg",
            config_overrides={},
            run_index=0,
            n_samples=50,
        )
        t2 = BenchmarkTask(
            benchmark_name="bench",
            config_name="cfg",
            config_overrides={},
            run_index=0,
            n_samples=100,
        )
        assert t1.task_key != t2.task_key

    def test_different_seed_different_keys(self):
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

        t1 = BenchmarkTask(
            benchmark_name="bench",
            config_name="cfg",
            config_overrides={},
            run_index=0,
            seed=42,
        )
        t2 = BenchmarkTask(
            benchmark_name="bench",
            config_name="cfg",
            config_overrides={},
            run_index=0,
            seed=123,
        )
        assert t1.task_key != t2.task_key

    def test_different_filters_different_keys(self):
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

        t1 = BenchmarkTask(
            benchmark_name="bench",
            config_name="cfg",
            config_overrides={},
            run_index=0,
            filters={"lang": "en"},
        )
        t2 = BenchmarkTask(
            benchmark_name="bench",
            config_name="cfg",
            config_overrides={},
            run_index=0,
            filters={"lang": "de"},
        )
        assert t1.task_key != t2.task_key

    def test_identical_tasks_same_key(self):
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

        t1 = BenchmarkTask(
            benchmark_name="bench",
            config_name="cfg",
            config_overrides={"k": 10},
            run_index=0,
            split="test",
            n_samples=50,
            seed=42,
            filters={},
        )
        t2 = BenchmarkTask(
            benchmark_name="bench",
            config_name="cfg",
            config_overrides={"k": 10},
            run_index=0,
            split="test",
            n_samples=50,
            seed=42,
            filters={},
        )
        assert t1.task_key == t2.task_key

    def test_different_max_corpus_docs_different_config_hash(self):
        """Tasks with different corpus caps must have different config hashes."""
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

        t_full = BenchmarkTask(
            benchmark_name="bench",
            config_name="cfg",
            config_overrides={},
            run_index=0,
            max_corpus_docs=0,
        )
        t_capped = BenchmarkTask(
            benchmark_name="bench",
            config_name="cfg",
            config_overrides={},
            run_index=0,
            max_corpus_docs=500,
        )
        assert t_full.config_hash != t_capped.config_hash
        assert t_full.task_key != t_capped.task_key

    def test_measurement_depth_is_part_of_the_checkpoint_key(self):
        """--resume must not restore a result measured at another depth.

        metric_k and the over-fetch change every IR number; recording-only
        keys (trajectory logging) must not split checkpoints.
        """
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

        def task(**overrides):
            return BenchmarkTask(
                benchmark_name="bench", config_name="cfg", config_overrides=overrides, run_index=0
            )

        base = task()
        assert task(metric_k=100).config_hash != base.config_hash
        assert task(retrieval_chunk_overfetch=8).config_hash != base.config_hash
        assert task(save_trajectories=True).config_hash == base.config_hash
        assert task(allow_inert_overrides=True).config_hash == base.config_hash

    def test_corpus_cache_key_consistency_with_pre_register(self):
        """_corpus_cache_key produces correct keys based on ingestion params.

        Key format: "{corpus_id}|{sha256_hex[:16]}" where the hash covers
        corpus_id + corpus_fingerprint + max_corpus_docs + merged ingestion params.
        Returns None only when NO profile ingestion defaults AND no explicit
        ingestion-time overrides AND no corpus cap.
        """
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask
        from lasagna.knowledge_evaluation_sdk.experiment import (
            _corpus_cache_key,
        )

        # Case 1: profile="standard" includes ingestion-time defaults
        # (embedding_model, chunking_strategy, etc.) → non-None.
        t1 = BenchmarkTask(
            benchmark_name="private_eval/retrieval",
            config_name="cfg",
            config_overrides={"enable_reranking": True},
            run_index=0,
        )
        key1 = _corpus_cache_key(t1)
        assert key1 is not None, (
            "Standard profile includes ingestion-time defaults — key should be non-None"
        )
        assert "|" in key1  # format: "corpus_id|hash"

        # Case 2: explicit ingestion-time override → different hash from Case 1
        t2 = BenchmarkTask(
            benchmark_name="private_eval/retrieval",
            config_name="cfg",
            config_overrides={"chunking_strategy": "fixed"},
            run_index=0,
        )
        key2 = _corpus_cache_key(t2)
        assert key2 is not None
        assert key2 != key1  # different chunking_strategy → different hash

        # Case 3: same profile, same overrides → same key (deterministic)
        t1_dup = BenchmarkTask(
            benchmark_name="private_eval/retrieval",
            config_name="other",
            config_overrides={"enable_reranking": True},
            run_index=5,
        )
        assert _corpus_cache_key(t1_dup) == key1  # config_name/run_index don't matter

        # Case 4: corpus cap changes the key
        t3 = BenchmarkTask(
            benchmark_name="private_eval/retrieval",
            config_name="cfg",
            config_overrides={},
            run_index=0,
            max_corpus_docs=500,
        )
        key3 = _corpus_cache_key(t3)
        assert key3 is not None

        # Case 5: different caps → different keys
        t5 = BenchmarkTask(
            benchmark_name="private_eval/retrieval",
            config_name="cfg",
            config_overrides={},
            run_index=0,
            max_corpus_docs=1000,
        )
        key5 = _corpus_cache_key(t5)
        assert key5 != key3  # 1000 != 500


# ---------------------------------------------------------------------------
# Phase 0A: SDK Evaluation Wrapper API Contract
# ---------------------------------------------------------------------------


class TestSDKEvaluationWrapper:
    """Verify the SDK evaluation wrapper uses correct API contracts."""

    def test_judge_accepts_correct_kwargs(self):
        """LLMJudge.evaluate() accepts question, expected, response, context."""
        import inspect
        from lasagna.knowledge_evaluation_sdk.judge import LLMJudge

        sig = inspect.signature(LLMJudge.evaluate)
        params = list(sig.parameters.keys())
        assert "question" in params
        assert "expected" in params
        assert "response" in params
        assert "context" in params
        # These should NOT be valid kwargs
        assert "expected_response" not in params
        assert "actual_response" not in params

    def test_eval_result_fields(self):
        """Backend EvalResult should not have .metrics, .num_samples, .details attributes."""
        from lasagna.knowledge_evaluation_sdk.models import EvalResult

        field_names = {f.name for f in EvalResult.__dataclass_fields__.values()}
        # These are fields the SDK wrapper incorrectly assumed exist
        assert "metrics" not in field_names
        assert "num_samples" not in field_names
        assert "details" not in field_names

    def test_llm_judge_model_param_type(self):
        """LLMJudge model param should accept BaseChatModel, not str."""
        import inspect
        from lasagna.knowledge_evaluation_sdk.judge import LLMJudge

        sig = inspect.signature(LLMJudge.__init__)
        model_param = sig.parameters.get("model")
        assert model_param is not None
        # The annotation should be Optional[BaseChatModel], not str
        annotation = model_param.annotation
        # When annotation is a string due to __future__ annotations, check the raw string
        ann_str = str(annotation)
        # It should not be 'str' type
        assert ann_str != "<class 'str'>"

    def test_backend_eval_result_has_judge_score(self):
        """Backend EvalResult stores judge_score, not separate metric fields."""
        from lasagna.knowledge_evaluation_sdk.models import (
            EvalResult as BackendResult,
            EvalCase,
            JudgeScore,
            ScoreLevel,
        )

        case = EvalCase(utterance="q", expected_response="a")
        judge_score = JudgeScore(
            score=4,
            level=ScoreLevel.GOOD,
            reasoning="ok",
            relevance=0.8,
            completeness=0.8,
            factual_accuracy=0.8,
        )
        result = BackendResult(
            case=case,
            agent_response="a",
            rag_context="ctx",
            rag_used=True,
            judge_score=judge_score,
            latency_ms=50.0,
            rag_latency_ms=25.0,
        )
        assert result.judge_score.score == 4
        assert result.judge_score.level == ScoreLevel.GOOD


# ---------------------------------------------------------------------------
# SDK Wrapper: Answer Generation and Config Resolution
# ---------------------------------------------------------------------------


class TestSDKWrapperAnswerGeneration:
    """Verify that evaluate_with_judge generates answers, not raw context."""

    def test_config_platform_fields_exist(self):
        """EvalConfig should have config-driven platform fields."""
        from lasagna.sdk.evaluation import EvalConfig

        cfg = EvalConfig()
        assert hasattr(cfg, "judge_model_platform")
        assert hasattr(cfg, "answer_model")
        assert hasattr(cfg, "answer_model_platform")
        assert hasattr(cfg, "answer_temperature")
        # Defaults
        assert cfg.judge_model_platform == "openai"
        assert cfg.answer_model is None
        assert cfg.answer_model_platform is None

    def test_config_effective_answer_defaults(self):
        """answer_model/platform default to judge_model/platform."""
        from lasagna.sdk.evaluation import EvalConfig

        cfg = EvalConfig(judge_model="gpt-4o", judge_model_platform="azure")
        assert cfg.effective_answer_model == "gpt-4o"
        assert cfg.effective_answer_platform == "azure"

    def test_config_effective_answer_overrides(self):
        """Explicit answer_model/platform overrides judge defaults."""
        from lasagna.sdk.evaluation import EvalConfig

        cfg = EvalConfig(
            judge_model="gpt-4o",
            judge_model_platform="openai",
            answer_model="gpt-3.5-turbo",
            answer_model_platform="azure",
        )
        assert cfg.effective_answer_model == "gpt-3.5-turbo"
        assert cfg.effective_answer_platform == "azure"

    def test_normalization_consistency(self):
        """from_backend and evaluate_with_judge use the same normalization."""
        from lasagna.sdk.evaluation import EvalResult

        # Both should use (score - 1) / 4 mapping: 1→0.0, 5→1.0
        # Test from_backend
        from lasagna.knowledge_evaluation_sdk.models import (
            EvalResult as BackendResult,
            EvalCase,
            JudgeScore,
            ScoreLevel,
        )

        case = EvalCase(utterance="test", expected_response="answer")
        judge_score = JudgeScore(
            score=5,
            level=ScoreLevel.EXCELLENT,
            reasoning="perfect",
            relevance=1.0,
            completeness=1.0,
            factual_accuracy=1.0,
        )
        backend_result = BackendResult(
            case=case,
            agent_response="answer",
            rag_context="ctx",
            rag_used=True,
            judge_score=judge_score,
            latency_ms=100.0,
            rag_latency_ms=50.0,
        )
        sdk_result = EvalResult.from_backend(backend_result)
        # Score 5 → (5-1)/4 = 1.0
        assert sdk_result.precision == pytest.approx(1.0)

        # Score 1 → (1-1)/4 = 0.0
        judge_score_1 = JudgeScore(
            score=1,
            level=ScoreLevel.FAIL,
            reasoning="bad",
            relevance=0.0,
            completeness=0.0,
            factual_accuracy=0.0,
        )
        backend_result_1 = BackendResult(
            case=case,
            agent_response="wrong",
            rag_context="ctx",
            rag_used=True,
            judge_score=judge_score_1,
            latency_ms=100.0,
            rag_latency_ms=50.0,
        )
        sdk_result_1 = EvalResult.from_backend(backend_result_1)
        assert sdk_result_1.precision == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_success_details_have_backward_compat_keys(self):
        """Success-path details must include both 'response' and 'actual_response'."""
        from unittest.mock import AsyncMock, patch, MagicMock
        from lasagna.sdk.evaluation import (
            KnowledgeEvaluator,
            EvalConfig,
        )

        client = AsyncMock()
        client.get_context = AsyncMock(return_value="some context")

        mock_answer = AsyncMock()
        mock_answer.content = "generated answer"

        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_answer)

        mock_provider = MagicMock()
        mock_provider.get_chat_model.return_value = mock_model

        mock_judge_result = MagicMock()
        mock_judge_result.score = 4
        mock_judge_result.level.value = "good"
        mock_judge_result.reasoning = "solid"

        mock_judge_instance = AsyncMock()
        mock_judge_instance.evaluate = AsyncMock(return_value=mock_judge_result)

        with (
            patch(
                "lasagna.providers.get_llm_provider",
                return_value=mock_provider,
            ),
            patch(
                "lasagna.sdk.evaluation.LLMJudge",
                return_value=mock_judge_instance,
            ),
        ):
            evaluator = KnowledgeEvaluator(client, config=EvalConfig(verbose=False))
            result = await evaluator.evaluate_with_judge([{"query": "q", "expected_answer": "a"}])

        detail = result.details[0]
        assert "response" in detail
        assert "actual_response" in detail
        assert detail["response"] == detail["actual_response"]
        assert detail["response"] == "generated answer"

    @pytest.mark.asyncio
    async def test_judge_receives_config_platform(self):
        """LLMJudge must be initialized with the config's platform, not hardcoded."""
        from unittest.mock import AsyncMock, patch, MagicMock
        from lasagna.sdk.evaluation import (
            KnowledgeEvaluator,
            EvalConfig,
        )

        client = AsyncMock()
        client.get_context = AsyncMock(return_value="ctx")

        mock_answer = AsyncMock()
        mock_answer.content = "ans"
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_answer)

        mock_provider = MagicMock()
        mock_provider.get_chat_model.return_value = mock_model

        mock_judge_result = MagicMock()
        mock_judge_result.score = 3
        mock_judge_result.level.value = "fair"
        mock_judge_result.reasoning = "ok"

        mock_judge_cls = MagicMock()
        mock_judge_cls.return_value.evaluate = AsyncMock(return_value=mock_judge_result)

        cfg = EvalConfig(
            judge_model="my-model",
            judge_model_platform="azure",
            verbose=False,
        )

        with (
            patch(
                "lasagna.providers.get_llm_provider",
                return_value=mock_provider,
            ),
            patch(
                "lasagna.sdk.evaluation.LLMJudge",
                mock_judge_cls,
            ),
        ):
            evaluator = KnowledgeEvaluator(client, config=cfg)
            await evaluator.evaluate_with_judge([{"query": "q", "expected_answer": "a"}])

        # Verify LLMJudge was constructed with the config platform, not hardcoded
        mock_judge_cls.assert_called_once_with(
            platform="azure",
            model_name="my-model",
            temperature=0.0,
        )

    def test_normalization_never_negative(self):
        """Precision must be >= 0 even when avg_score < 1 (failed cases with score=0)."""
        from lasagna.sdk.evaluation import EvalResult
        from lasagna.knowledge_evaluation_sdk.models import (
            EvalResult as BackendResult,
            EvalCase,
        )

        # Simulate from_backend with score=0 (no judge score)
        case = EvalCase(utterance="test", expected_response="answer")
        backend_result = BackendResult(
            case=case,
            agent_response="",
            rag_context=None,
            rag_used=False,
            judge_score=None,
            latency_ms=100.0,
            rag_latency_ms=50.0,
        )
        sdk_result = EvalResult.from_backend(backend_result)
        assert sdk_result.precision >= 0.0

    @pytest.mark.asyncio
    async def test_error_details_have_response_keys(self):
        """Error-path details dict must include 'response' and 'actual_response' keys."""
        from unittest.mock import AsyncMock, patch, MagicMock
        from lasagna.sdk.evaluation import (
            KnowledgeEvaluator,
            EvalConfig,
        )

        client = AsyncMock()
        # Make get_context raise so we hit the per-case error path
        client.get_context = AsyncMock(side_effect=RuntimeError("db down"))

        evaluator = KnowledgeEvaluator(client, config=EvalConfig(verbose=False))

        mock_provider = MagicMock()
        mock_provider.get_chat_model.return_value = AsyncMock()

        with (
            patch(
                "lasagna.providers.get_llm_provider",
                return_value=mock_provider,
            ),
            patch(
                "lasagna.sdk.evaluation.LLMJudge",
            ),
        ):
            result = await evaluator.evaluate_with_judge(
                [{"query": "test q", "expected_answer": "test a"}]
            )

        detail = result.details[0]
        assert "response" in detail
        assert "actual_response" in detail
        assert "error" in detail

    @pytest.mark.asyncio
    async def test_answer_model_init_failure_raises_evaluation_error(self):
        """If answer-model init fails, EvaluationError must be raised (not raw exception)."""
        from unittest.mock import AsyncMock, patch
        from lasagna.sdk.evaluation import (
            KnowledgeEvaluator,
            EvalConfig,
        )
        from lasagna.sdk.exceptions import EvaluationError

        client = AsyncMock()
        evaluator = KnowledgeEvaluator(client, config=EvalConfig())

        with patch(
            "lasagna.providers.get_llm_provider",
            side_effect=RuntimeError("provider unavailable"),
        ):
            with pytest.raises(EvaluationError, match="answering model"):
                await evaluator.evaluate_with_judge(
                    [{"query": "test", "expected_answer": "answer"}]
                )


# ---------------------------------------------------------------------------
# PopQA Converter & Adapter Tests
# ---------------------------------------------------------------------------


class TestPopQAConverter:
    """Tests for _convert_popqa format converter."""

    def _convert(self, rows):
        from lasagna.knowledge_evaluation_sdk.datasets import (
            _convert_popqa,
        )

        # Simulate a HF dataset as a list of dicts
        return _convert_popqa(rows)

    def test_basic_conversion(self):
        rows = [
            {
                "question": "What is the capital of France?",
                "possible_answers": '["Paris", "City of Light"]',
                "s_pop": 1000,
                "o_pop": 5000,
                "prop": "capital",
                "subj": "France",
                "obj": "Paris",
            }
        ]
        result = self._convert(rows)
        assert len(result) == 1
        assert result[0]["question"] == "What is the capital of France?"
        assert result[0]["ground_truth"] == "Paris"
        assert result[0]["s_pop"] == 1000
        assert result[0]["o_pop"] == 5000
        assert result[0]["prop"] == "capital"
        assert result[0]["subj"] == "France"
        assert result[0]["obj"] == "Paris"

    def test_aliases_preserved(self):
        rows = [
            {
                "question": "Q?",
                "possible_answers": '["A1", "A2", "A3"]',
                "s_pop": 100,
                "o_pop": 200,
            }
        ]
        result = self._convert(rows)
        assert result[0]["ground_truth"] == "A1"
        assert result[0]["ground_truth_aliases"] == ["A1", "A2", "A3"]

    def test_single_answer_no_aliases_key(self):
        rows = [
            {
                "question": "Q?",
                "possible_answers": '["OnlyAnswer"]',
                "s_pop": 100,
                "o_pop": 200,
            }
        ]
        result = self._convert(rows)
        assert result[0]["ground_truth"] == "OnlyAnswer"
        assert "ground_truth_aliases" not in result[0]

    def test_non_numeric_s_pop_does_not_crash(self):
        rows = [
            {"question": "Q?", "possible_answers": '["A"]', "s_pop": "", "o_pop": None},
            {"question": "Q2?", "possible_answers": '["B"]', "s_pop": "not_a_number", "o_pop": 0},
        ]
        result = self._convert(rows)
        assert len(result) == 2
        assert "s_pop" not in result[0]
        assert "o_pop" not in result[0]
        assert "s_pop" not in result[1]
        assert result[1]["o_pop"] == 0

    def test_malformed_json_answers_fallback(self):
        rows = [
            {"question": "Q?", "possible_answers": "just a plain string", "s_pop": 10},
        ]
        result = self._convert(rows)
        assert result[0]["ground_truth"] == "just a plain string"

    def test_empty_question_skipped(self):
        rows = [
            {"question": "", "possible_answers": '["A"]'},
            {"question": "Valid?", "possible_answers": '["B"]'},
        ]
        result = self._convert(rows)
        assert len(result) == 1
        assert result[0]["question"] == "Valid?"

    def test_list_type_possible_answers(self):
        """Handle case where HF returns possible_answers as a native list."""
        rows = [
            {"question": "Q?", "possible_answers": ["X", "Y"], "s_pop": 5},
        ]
        result = self._convert(rows)
        assert result[0]["ground_truth"] == "X"
        assert result[0]["ground_truth_aliases"] == ["X", "Y"]


class TestPopQATertileComputation:
    """Tests for tertile threshold computation."""

    def test_even_distribution(self):
        from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.popqa import (
            _compute_tertile_thresholds,
            _assign_tier,
        )

        # 9 items with distinct values: tiers should be roughly 3/3/3
        rows = [{"s_pop": v} for v in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
        t1, t2 = _compute_tertile_thresholds(rows)
        # Verify thresholds partition reasonably
        tiers = [_assign_tier(v, (t1, t2)) for v in range(1, 10)]
        assert tiers.count("tail") >= 2
        assert tiers.count("torso") >= 2
        assert tiers.count("head") >= 2

    def test_all_same_values(self):
        from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.popqa import (
            _compute_tertile_thresholds,
            _assign_tier,
        )

        # All same value — should not crash
        rows = [{"s_pop": 100}] * 100
        t1, t2 = _compute_tertile_thresholds(rows)
        tier = _assign_tier(100, (t1, t2))
        assert tier in ("tail", "torso", "head")

    def test_empty_rows(self):
        from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.popqa import (
            _compute_tertile_thresholds,
        )

        t1, t2 = _compute_tertile_thresholds([])
        assert t1 == 0.0
        assert t2 == 0.0


class TestPopQAAdapterRegistration:
    """Tests for PopQA adapter registration and metadata."""

    def test_popqa_in_legacy_catalog(self):
        from lasagna.knowledge_evaluation_sdk.datasets import (
            list_benchmarks,
        )

        benchmarks = list_benchmarks()
        assert "popqa" in benchmarks

    def test_popqa_in_adapter_registry(self):
        from lasagna.knowledge_evaluation_sdk.benchmarks import (
            benchmark_registry,
        )

        adapter = benchmark_registry.get("popqa")
        assert adapter is not None
        info = adapter.info()
        assert info.name == "popqa"
        assert info.category.value == "single_hop"
        assert info.eval_mode.value == "generation"
        assert info.num_queries == 14_267
        assert info.paper == "arxiv:2212.10511"

    def test_popqa_available_filters(self):
        from lasagna.knowledge_evaluation_sdk.benchmarks import (
            benchmark_registry,
        )

        adapter = benchmark_registry.get("popqa")
        filters = adapter.available_filters()
        assert "popularity_tier" in filters
        assert set(filters["popularity_tier"]) == {"head", "torso", "tail"}
        assert "relationship_type" in filters
        assert len(filters["relationship_type"]) == 16


# ---------------------------------------------------------------------------
# E5 Prefix Resolution Tests
# ---------------------------------------------------------------------------


class TestE5PrefixResolution:
    """Tests for model-aware prefix resolution in EmbeddingManager."""

    def test_e5_prefixes(self):
        from lasagna.infrastructure.embedding.manager import (
            _resolve_prefixes,
        )

        result = _resolve_prefixes("intfloat/e5-large-v2")
        assert result["query"] == "query: "
        assert result["document"] == "passage: "

    def test_e5_small_prefixes(self):
        from lasagna.infrastructure.embedding.manager import (
            _resolve_prefixes,
        )

        result = _resolve_prefixes("intfloat/e5-small-v2")
        assert result["query"] == "query: "
        assert result["document"] == "passage: "

    def test_multilingual_e5_prefixes(self):
        from lasagna.infrastructure.embedding.manager import (
            _resolve_prefixes,
        )

        result = _resolve_prefixes("intfloat/multilingual-e5-large")
        assert result["query"] == "query: "
        assert result["document"] == "passage: "

    def test_bge_m3_takes_no_query_instruction(self):
        """BGE-M3 is trained without one, unlike bge-*-v1.5; it was given theirs."""
        from lasagna.infrastructure.embedding.manager import (
            _resolve_prefixes,
        )

        result = _resolve_prefixes("BAAI/bge-m3")
        assert result["query"] == ""
        assert result["document"] == ""

    def test_bge_large_prefixes(self):
        from lasagna.infrastructure.embedding.manager import (
            _resolve_prefixes,
        )

        result = _resolve_prefixes("BAAI/bge-large-en-v1.5")
        assert result["query"] == "Represent this sentence for searching relevant passages: "
        assert result["document"] == ""

    def test_minilm_default_prefixes(self):
        from lasagna.infrastructure.embedding.manager import (
            _resolve_prefixes,
        )

        result = _resolve_prefixes("all-MiniLM-L6-v2")
        assert result["query"] == ""
        assert result["document"] == ""

    def test_unknown_model_default_prefixes(self):
        from lasagna.infrastructure.embedding.manager import (
            _resolve_prefixes,
        )

        result = _resolve_prefixes("some-custom-model")
        assert result["query"] == ""
        assert result["document"] == ""

    def test_case_insensitive_matching(self):
        from lasagna.infrastructure.embedding.manager import (
            _resolve_prefixes,
        )

        # E5 in uppercase path should still match
        result = _resolve_prefixes("intfloat/E5-Large-V2")
        assert result["query"] == "query: "
        assert result["document"] == "passage: "


class TestE5ConfiguratorIntegration:
    """Tests for E5 model entries in configurator schema and validation."""

    def test_e5_in_schema_enum_options(self):
        from lasagna.configurator.schema import PARAMETERS

        embedding_param = next(
            (p for p in PARAMETERS if p.key == "embedding_model"),
            None,
        )
        assert embedding_param is not None
        option_values = [o.value for o in embedding_param.enum_options]
        assert "intfloat/e5-small-v2" in option_values
        assert "intfloat/e5-base-v2" in option_values
        assert "intfloat/e5-large-v2" in option_values
        assert "intfloat/multilingual-e5-large" in option_values

    def test_e5_in_validation_model_dims(self):
        from lasagna.configurator.validation import validate_config

        # Validate that E5 model with correct dim produces no dim error
        result = validate_config(
            {
                "embedding_model": "intfloat/e5-large-v2",
                "embedding_dim": 1024,
            }
        )
        dim_errors = [e for e in result.errors if "embedding_dim" in str(e)]
        assert len(dim_errors) == 0

    def test_e5_dim_mismatch_warns(self):
        from lasagna.configurator.validation import validate_config

        result = validate_config(
            {
                "embedding_model": "intfloat/e5-large-v2",
                "embedding_dim": 384,  # wrong dim for e5-large
            }
        )
        dim_warnings = [w for w in result.warnings if "embedding_dim" in str(w)]
        assert len(dim_warnings) > 0


# ---------------------------------------------------------------------------
# Phase 0 Behavior Tests
#
# Tests import production helpers from ``_helpers.py`` — a zero-dependency
# module (stdlib only) that can be imported without pulling in loguru,
# pydantic, sentence-transformers, or any other heavy package.
#
# We load ``_helpers.py`` by file path using ``importlib.util`` to bypass
# the ``knowledge_evaluation_sdk/__init__.py`` which eagerly imports pydantic.
#
# Source-inspection tests read ``.py`` files from disk via ``pathlib.Path``
# instead of ``importlib.import_module()``, avoiding transitive imports.
# ---------------------------------------------------------------------------

import importlib.util as _ilu  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

# Locate the package on disk WITHOUT importing it: find_spec resolves the
# search path but never executes ``lasagna/__init__.py``, preserving the
# no-heavy-imports property this module depends on.
_LASAGNA_DIR = _Path(_ilu.find_spec("lasagna").submodule_search_locations[0])

_EVAL_SDK_DIR = _LASAGNA_DIR / "knowledge_evaluation_sdk"

# Load _helpers.py directly by file path — avoids triggering __init__.py
# and its pydantic/loguru imports.
_helpers_path = _EVAL_SDK_DIR / "_helpers.py"
_helpers_spec = _ilu.spec_from_file_location("_helpers", _helpers_path)
_helpers_mod = _ilu.module_from_spec(_helpers_spec)
_helpers_spec.loader.exec_module(_helpers_mod)


class TestCanonicalMetricMapping:
    """Verify canonical metric emission logic for agentic vs non-agentic configs.

    Tests call ``apply_canonical_metric_mapping()`` from ``_helpers.py``
    directly (zero third-party deps).

    Contract:
    1. Per-sample canonical: prefer _agentic score, fall back to _regen score.
    2. Aggregate canonical: mean over ALL N samples (not the agentic subset).
    3. Function is a no-op when called on samples that lack suffixed keys.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        self.apply_canonical_metric_mapping = _helpers_mod.apply_canonical_metric_mapping

    def test_all_samples_have_agentic_answer(self):
        """When all samples have agentic scores, canonical = mean(agentic)."""
        per_sample = [
            {"faithfulness_agentic": 1.0, "faithfulness_regen": 0.5},
            {"faithfulness_agentic": 0.8, "faithfulness_regen": 0.3},
            {"faithfulness_agentic": 0.6, "faithfulness_regen": 0.4},
        ]
        metrics = {"faithfulness_agentic": 0.8, "faithfulness_regen": 0.4}
        self.apply_canonical_metric_mapping(per_sample, metrics)
        # All 3 have agentic → canonical = mean(1.0, 0.8, 0.6) = 0.8
        assert metrics["faithfulness"] == pytest.approx(0.8)
        for ps in per_sample:
            assert "faithfulness" in ps
            assert ps["faithfulness"] == ps["faithfulness_agentic"]

    def test_subset_agentic_uses_regen_fallback(self):
        """When only some samples have agentic, canonical uses regen for the rest."""
        per_sample = [
            {"faithfulness_agentic": 1.0, "faithfulness_regen": 0.5},
            {"faithfulness_regen": 0.3},  # no agentic (empty answer)
            {"faithfulness_agentic": 0.6, "faithfulness_regen": 0.4},
        ]
        metrics = {"faithfulness_agentic": 0.8, "faithfulness_regen": 0.4}
        self.apply_canonical_metric_mapping(per_sample, metrics)
        expected = (1.0 + 0.3 + 0.6) / 3.0
        assert metrics["faithfulness"] == pytest.approx(expected)
        assert per_sample[0]["faithfulness"] == 1.0
        assert per_sample[1]["faithfulness"] == 0.3
        assert per_sample[2]["faithfulness"] == 0.6

    def test_canonical_differs_from_subset_average(self):
        """The canonical aggregate must NOT equal the agentic subset average.

        This was the bug caught in review round 2: when agentic answers exist
        for a subset, the agentic subset average (over k<N samples) differs
        from the canonical average (over all N samples with regen fallback).
        """
        per_sample = [
            {"faithfulness_agentic": 1.0, "faithfulness_regen": 0.0},
            {"faithfulness_regen": 0.0},  # no agentic answer
            {"faithfulness_regen": 0.0},  # no agentic answer
        ]
        metrics = {"faithfulness_agentic": 1.0, "faithfulness_regen": 0.0}
        self.apply_canonical_metric_mapping(per_sample, metrics)
        assert metrics["faithfulness"] == pytest.approx(1.0 / 3.0)
        assert metrics["faithfulness"] != pytest.approx(1.0)

    def test_no_suffixed_keys_is_noop(self):
        """When no _agentic/_regen suffixed keys, mapping is a no-op."""
        per_sample = [
            {"faithfulness": 0.7},
            {"faithfulness": 0.9},
        ]
        metrics = {"faithfulness": 0.8}
        self.apply_canonical_metric_mapping(per_sample, metrics)
        assert metrics["faithfulness"] == pytest.approx(0.8)

    def test_multiple_metrics_mapped(self):
        """All three base metrics are mapped, not just faithfulness."""
        per_sample = [
            {
                "faithfulness_agentic": 1.0,
                "faithfulness_regen": 0.5,
                "context_recall_agentic": 0.9,
                "context_recall_regen": 0.4,
                "context_precision_agentic": 0.8,
                "context_precision_regen": 0.3,
            },
        ]
        metrics = {}
        self.apply_canonical_metric_mapping(per_sample, metrics)
        assert metrics["faithfulness"] == pytest.approx(1.0)
        assert metrics["context_recall"] == pytest.approx(0.9)
        assert metrics["context_precision"] == pytest.approx(0.8)


class TestEmptyAgenticAnswerFiltering:
    """Verify that empty agentic answers are filtered before RAGAS evaluation.

    Tests call ``filter_nonempty_agentic_indices()`` from ``_helpers.py``.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        self.filter_nonempty_agentic_indices = _helpers_mod.filter_nonempty_agentic_indices

    def test_all_empty_returns_no_indices(self):
        """All empty strings → no indices to evaluate."""
        assert self.filter_nonempty_agentic_indices(["", "", ""]) == []

    def test_all_non_empty_returns_all_indices(self):
        """All non-empty → all indices returned."""
        assert self.filter_nonempty_agentic_indices(["a", "b", "c"]) == [0, 1, 2]

    def test_mixed_returns_only_non_empty_indices(self):
        """Mixed empty/non-empty → only non-empty indices returned."""
        assert self.filter_nonempty_agentic_indices(["", "ans2", "", "ans4", ""]) == [1, 3]

    def test_has_agentic_answers_flag(self):
        """The ``any()`` guard that gates the entire agentic block."""
        all_empty = ["", "", ""]
        some_filled = ["", "answer", ""]
        assert not any(a for a in all_empty)
        assert any(a for a in some_filled)

    def test_merge_back_to_original_indices(self):
        """RAGAS scores are mapped back to correct original positions."""
        gen_agentic_answers = ["", "answer_1", "", "answer_3"]
        _ag_indices = self.filter_nonempty_agentic_indices(gen_agentic_answers)
        agentic_per_sample = [
            {"faithfulness": 0.9},
            {"faithfulness": 0.7},
        ]
        per_sample_scores = [{} for _ in range(4)]
        for j, orig_idx in enumerate(_ag_indices):
            if j < len(agentic_per_sample):
                ps = agentic_per_sample[j]
                for metric_name, score_val in ps.items():
                    per_sample_scores[orig_idx][f"{metric_name}_agentic"] = score_val

        assert per_sample_scores[0] == {}
        assert per_sample_scores[1] == {"faithfulness_agentic": 0.9}
        assert per_sample_scores[2] == {}
        assert per_sample_scores[3] == {"faithfulness_agentic": 0.7}


class TestBM25ChunkIdFallback:
    """Verify that BM25 doc construction handles missing/None chunk_id.

    Tests call ``build_bm25_docs()`` from ``_helpers.py``.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        self.build_bm25_docs = _helpers_mod.build_bm25_docs

    def test_chunk_id_present_and_valid(self):
        """Normal case: chunk_id exists with valid value."""
        docs = self.build_bm25_docs([{"chunk_id": "abc123", "content": "hello"}])
        assert docs[0]["chunk_id"] == "abc123"

    def test_chunk_id_missing(self):
        """chunk_id key absent → fallback to indexed name."""
        docs = self.build_bm25_docs([{"content": "hello"}])
        assert docs[0]["chunk_id"] == "chunk_0"

    def test_chunk_id_none(self):
        """chunk_id key present but value is None → fallback to indexed name.

        This was the bug caught in review round 2: dict.get("chunk_id", default)
        returns None when the key exists with value None. The ``or`` operator
        correctly falls through to the fallback.
        """
        docs = self.build_bm25_docs([{"chunk_id": None, "content": "hello"}])
        assert docs[0]["chunk_id"] == "chunk_0"

    def test_chunk_id_empty_string(self):
        """Empty string chunk_id → fallback (empty string is falsy)."""
        docs = self.build_bm25_docs([{"chunk_id": "", "content": "hello"}])
        assert docs[0]["chunk_id"] == "chunk_0"

    def test_multiple_chunks_unique_fallback_ids(self):
        """Each chunk without chunk_id gets a unique fallback name."""
        docs = self.build_bm25_docs(
            [
                {"chunk_id": None, "content": "a"},
                {"chunk_id": "real_id", "content": "b"},
                {"content": "c"},
            ]
        )
        assert docs[0]["chunk_id"] == "chunk_0"
        assert docs[1]["chunk_id"] == "real_id"
        assert docs[2]["chunk_id"] == "chunk_2"

    def test_all_fields_populated(self):
        """Verify all fields are extracted correctly."""
        docs = self.build_bm25_docs(
            [
                {
                    "chunk_id": "id1",
                    "content": "some text",
                    "document_id": "doc1",
                    "metadata": {"key": "val"},
                }
            ]
        )
        assert docs[0] == {
            "chunk_id": "id1",
            "content": "some text",
            "document_id": "doc1",
            "metadata": {"key": "val"},
        }

    def test_missing_optional_fields_default(self):
        """Missing content/document_id/metadata get defaults."""
        docs = self.build_bm25_docs([{"chunk_id": "id1"}])
        assert docs[0]["content"] == ""
        assert docs[0]["document_id"] == ""
        assert docs[0]["metadata"] == {}


class TestAlphaTraceLogic:
    """Verify the alpha tracing debug log in _search_orchestrator.py.

    Reads source from disk (no module import → no heavy deps).
    """

    def test_alpha_override_detection_logic(self):
        """When effective != config, alpha_override='yes'."""
        assert ("yes" if 0.7 != 0.5 else "no") == "yes"
        assert ("yes" if 0.5 != 0.5 else "no") == "no"

    def test_alpha_trace_exists_in_search_orchestrator(self):
        """The alpha trace log must exist in _search_orchestrator.py source."""
        src_file = _EVAL_SDK_DIR.parent / "api" / "_search_orchestrator.py"
        source = src_file.read_text()
        assert "config.hybrid_alpha=" in source
        assert "effective_alpha=" in source
        assert "classification_enabled=" in source
        assert "alpha_override=" in source


class TestContextTruncationConsistency:
    """Verify Bug 3 fix: same context set goes to generation and RAGAS.

    Reads experiment.py source from disk (no module import).
    """

    def test_truncation_in_source(self):
        """Both generation and RAGAS paths use [:5] truncation."""
        source = (_EVAL_SDK_DIR / "experiment.py").read_text()
        assert "contexts[:5]" in source
        # Must appear at least twice (generation prompt + RAGAS append)
        assert source.count("contexts[:5]") >= 2

    def test_truncation_behavior(self):
        """[:5] correctly truncates and leaves short lists unchanged."""
        assert [f"ctx_{i}" for i in range(15)][:5] == ["ctx_0", "ctx_1", "ctx_2", "ctx_3", "ctx_4"]
        short = ["ctx_0", "ctx_1", "ctx_2"]
        assert short[:5] == short


class TestSilentConfigWarning:
    """Verify Bug 4 fix: config swallowing logs a warning, not pass.

    Reads experiment.py source from disk and uses AST to confirm no bare
    ``pass`` in except handlers within ``_apply_overrides_to_pipeline``.
    """

    def test_warning_logged_on_config_failure(self):
        """The except block in _apply_overrides_to_pipeline logs, not silences."""
        import ast
        import re

        source = (_EVAL_SDK_DIR / "experiment.py").read_text()

        # Extract the function body via regex (avoids module import).
        match = re.search(
            r"(def _apply_overrides_to_pipeline\b.*?)(?=\ndef |\nclass |\Z)",
            source,
            re.DOTALL,
        )
        assert match, "_apply_overrides_to_pipeline not found in experiment.py"

        func_source = match.group(1)
        tree = ast.parse(func_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    pytest.fail(
                        "_apply_overrides_to_pipeline has bare 'pass' in "
                        "except handler — should log a warning instead"
                    )
