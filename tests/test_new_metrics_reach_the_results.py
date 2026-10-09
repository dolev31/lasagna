"""The answer metrics beyond EM/F1 are scored per query and aggregated.

A metric that a function computes and the harness never records is invisible
to the significance tests and the bundle, which read ``per_sample_scores`` and
``metrics``. These run the offline path with a stub generator, judge and
embedder and check what lands in the result: values where the metric applies,
no key where it does not (never a 0.0 standing in for "not scored").
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest

from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

ragas_mod = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")
answer_mod = pytest.importorskip("lasagna.knowledge_evaluation_sdk.answer_metrics")

SAMPLES = [
    BenchmarkSample(
        query_id="cites",
        question="What is the capital of France?",
        ground_truth="Paris",
        context="Paris is the capital of France.",
    ),
    BenchmarkSample(
        query_id="plain",
        question="What colour is the sky?",
        ground_truth="blue",
        context="The sky is blue.",
    ),
    BenchmarkSample(
        query_id="none",
        question="Who was king in 2020?",
        context="France is a republic.",
        metadata={"unanswerable": True},
    ),
]
REPLIES = {
    "What is the capital of France?": "Paris is the capital of France [1].",
    "What colour is the sky?": "The sky is blue.",
    "Who was king in 2020?": "unanswerable",
}


class _Generator:
    def invoke(self, prompt):
        question = re.search(r"Question: (.*)\n", prompt).group(1)
        return SimpleNamespace(content=REPLIES[question])


class _RAGAS:
    requested = []

    def __init__(self, llm=None):
        pass

    def evaluate_detailed(self, questions, answers, contexts, ground_truths=None, metrics=None):
        _RAGAS.requested.append(list(metrics))
        rows = [{m: 1.0 for m in metrics} for _ in questions]
        return ragas_mod.DetailedEvalResult(aggregate={m: 1.0 for m in metrics}, per_sample=rows)


class _NLIJudge:
    """Entailed when the passage contains the statement word for word."""

    def generate(self, prompt, response_model):
        premise = prompt.split("Passage:\n", 1)[1].split("\n\nStatement:\n", 1)[0]
        hypothesis = prompt.split("\n\nStatement:\n", 1)[1].split("\n\n", 1)[0]
        return response_model(entailed=hypothesis.rstrip(".") in premise)


def _vectors(texts):
    # "Paris"-like texts point one way, sky texts the other.
    return [[1.0, 0.0] if "aris" in t else [0.0, 1.0] for t in texts]


def _run(monkeypatch, judge=_NLIJudge, **env):
    _RAGAS.requested = []
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ragas_mod, "RAGASEvaluator", _RAGAS)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda *a: judge() if judge else None)
    monkeypatch.setattr(answer_mod, "load_similarity_encoder", lambda *a: _vectors)
    import lasagna.providers as providers

    monkeypatch.setattr(
        providers, "get_llm_provider", lambda: SimpleNamespace(get_llm=lambda **k: _Generator())
    )
    task = BenchmarkTask(
        benchmark_name="b", config_name="c", run_index=0, mode="offline", config_overrides={}
    )
    result = asyncio.run(exp._eval_offline(task, SAMPLES))
    rows = {r["_sample_id"]: r for r in result.per_sample_scores}
    return result, rows


def test_each_answer_gets_rouge_containment_and_abstention(monkeypatch):
    result, rows = _run(monkeypatch)
    assert result.error is None
    # "Paris is the capital of France" vs "Paris": LCS 1, P = 1/6, R = 1.
    assert rows["cites"]["rouge_l"] == pytest.approx(2 / 7)
    assert rows["cites"]["exact_match"] == 0.0
    assert rows["cites"]["contains_answer"] == 1.0
    assert rows["plain"]["contains_answer"] == 1.0
    # Unanswerable: right because it abstained, as exact match scores it.
    assert rows["none"]["contains_answer"] == rows["none"]["exact_match"] == 1.0
    assert [rows[q]["abstention_rate"] for q in ("cites", "plain", "none")] == [0.0, 0.0, 1.0]
    assert result.metrics["abstention_rate"] == pytest.approx(1 / 3)
    assert result.metrics["abstention_rate_has_ans"] == 0.0
    assert result.metrics["abstention_rate_no_ans"] == 1.0


def test_correctness_is_scored_without_the_citation_markers(monkeypatch):
    _, rows = _run(monkeypatch)
    # "paris is capital of france" vs "paris": P = 1/5, R = 1. With the "[1]"
    # left in, "1" would be a sixth token and F1 2/7.
    assert rows["cites"]["token_f1"] == pytest.approx(1 / 3)


def test_similarity_is_scored_only_where_there_is_a_reference(monkeypatch):
    result, rows = _run(monkeypatch)
    assert rows["cites"]["answer_similarity"] == pytest.approx(1.0)
    assert rows["plain"]["answer_similarity"] == pytest.approx(1.0)
    assert "answer_similarity" not in rows["none"]
    assert result.metrics["answer_similarity"] == pytest.approx(1.0)


def test_similarity_can_be_turned_off(monkeypatch):
    result, rows = _run(monkeypatch, RAG_EVAL_SIMILARITY_MODEL="none")
    assert all("answer_similarity" not in r for r in rows.values())
    assert "answer_similarity" not in result.metrics


def test_citation_metrics_appear_only_on_answers_that_cite(monkeypatch):
    result, rows = _run(monkeypatch)
    assert rows["cites"]["citation_recall"] == 1.0
    assert rows["cites"]["citation_precision"] == 1.0
    assert "citation_recall" not in rows["plain"]
    assert result.metrics["citation_recall"] == 1.0


def test_without_a_judge_citations_are_absent_not_zero(monkeypatch):
    result, rows = _run(monkeypatch, judge=None)
    assert "citation_recall" not in rows["cites"]
    assert "citation_recall" not in result.metrics


def test_a_judge_that_cannot_decide_fails_the_metric_not_the_answer(monkeypatch):
    class _Down:
        def generate(self, prompt, response_model):
            raise ConnectionError("429")

    result, rows = _run(monkeypatch, judge=_Down)
    assert "citation_recall" not in rows["cites"]
    assert "judge produced no usable scores for citation_recall" in (result.error or "")


def test_cost_per_query_sums_a_querys_retrieval_and_generation_rows():
    rows = [
        {"_sample_id": "q1", "input_tokens": 0.0, "output_tokens": 0.0, "llm_calls": 0.0},
        {"_sample_id": "q2", "input_tokens": 10.0, "output_tokens": 2.0, "llm_calls": 1.0},
        {"_sample_id": "q1", "input_tokens": 100.0, "output_tokens": 20.0, "llm_calls": 3.0},
        {"_sample_id": "q2", "input_tokens": 50.0, "output_tokens": 4.0, "llm_calls": 1.0},
    ]
    # q1: 100 / 20 / 3, q2: 60 / 6 / 2.
    assert exp._per_query_costs(rows) == {
        "input_tokens_per_query": 80.0,
        "output_tokens_per_query": 13.0,
        "llm_calls_per_query": 2.5,
    }


def test_offline_results_carry_cost_per_query(monkeypatch):
    result, _ = _run(monkeypatch)
    assert result.metrics["llm_calls_per_query"] == 0.0  # the stub reports no usage


def test_aggregates_are_means_of_their_own_rows(monkeypatch):
    from lasagna.knowledge_evaluation_sdk.bundle import aggregate_mismatches

    result, _ = _run(monkeypatch)
    assert aggregate_mismatches([result.to_dict()]) == []


def test_a_lower_is_better_metric_ranks_the_smallest_first():
    from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults, lower_is_better
    from lasagna.knowledge_evaluation_sdk.runner import TaskResult

    def task(config, value):
        rows = [{"_sample_id": "q", "noise_sensitivity": value}]
        return TaskResult(
            task_key=config,
            benchmark_name="b",
            config_name=config,
            run_index=0,
            metrics={"noise_sensitivity": value},
            per_sample_scores=rows,
        )

    text = BenchmarkResults([task("noisy", 0.4), task("robust", 0.1)]).leaderboard(
        "noise_sensitivity"
    )
    assert "lower is better" in text
    assert text.index("1. robust") < text.index("2. noisy")
    assert lower_is_better("latency_ms") and lower_is_better("noise_sensitivity_regen")
    assert not lower_is_better("ndcg@10")
