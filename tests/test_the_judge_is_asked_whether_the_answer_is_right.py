"""The harness asks the judge whether the answer is right.

The live harness requested faithfulness, context recall and context precision:
groundedness and context quality, never whether the answer matched the
reference, except through exact match and token F1. ``factual_correctness``
(RAGAS claim-level F1 against the reference) is now requested too, and
``RAG_EVAL_JUDGE_METRICS`` chooses the judged set.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk._helpers import apply_canonical_metric_mapping
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

ragas_mod = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")


class _RAGAS:
    requested = []

    def __init__(self, llm=None):
        pass

    def evaluate_detailed(self, questions, answers, contexts, ground_truths=None, metrics=None):
        _RAGAS.requested.append(list(metrics))
        rows = [{m: 1.0 for m in metrics} for _ in questions]
        return ragas_mod.DetailedEvalResult(aggregate={m: 1.0 for m in metrics}, per_sample=rows)


def _run(monkeypatch, **env):
    _RAGAS.requested = []
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(ragas_mod, "RAGASEvaluator", _RAGAS)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda *a: None)
    import lasagna.providers as providers

    gen = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="Paris"))
    monkeypatch.setattr(
        providers, "get_llm_provider", lambda: SimpleNamespace(get_llm=lambda **k: gen)
    )
    sample = BenchmarkSample(
        query_id="cites", question="Capital of France?", ground_truth="Paris", context="Paris."
    )
    task = BenchmarkTask(
        benchmark_name="b", config_name="c", run_index=0, mode="offline", config_overrides={}
    )
    result = asyncio.run(exp._eval_offline(task, [sample]))
    rows = {r["_sample_id"]: r for r in result.per_sample_scores}
    return result, rows


def test_the_judge_is_asked_whether_the_answer_is_right(monkeypatch):
    result, rows = _run(monkeypatch)
    assert "factual_correctness" in _RAGAS.requested[0]
    assert rows["cites"]["factual_correctness"] == 1.0
    assert "factual_correctness" in result.metrics


def test_the_judged_metrics_can_be_chosen(monkeypatch):
    _run(monkeypatch, RAG_EVAL_JUDGE_METRICS="faithfulness, noise_sensitivity")
    assert _RAGAS.requested[0] == ["faithfulness", "noise_sensitivity"]


def test_a_misspelt_judged_metric_is_an_error(monkeypatch):
    monkeypatch.setenv("RAG_EVAL_JUDGE_METRICS", "faithfulnes")
    with pytest.raises(ValueError, match="faithfulnes"):
        ragas_mod.harness_metrics()


def test_without_context_only_metrics_that_need_none_are_judged(monkeypatch):
    monkeypatch.delenv("RAG_EVAL_JUDGE_METRICS", raising=False)
    assert ragas_mod.harness_metrics(has_context=False) == ["factual_correctness"]


def test_agentic_runs_get_a_canonical_score_for_every_judged_metric():
    rows = [
        {"factual_correctness_regen": 0.2, "factual_correctness_agentic": 0.6},
        {"factual_correctness_regen": 0.4},
    ]
    metrics = {}
    apply_canonical_metric_mapping(rows, metrics, base_keys=["factual_correctness"])
    assert [r["factual_correctness"] for r in rows] == [0.6, 0.4]
    assert metrics["factual_correctness"] == pytest.approx(0.5)
