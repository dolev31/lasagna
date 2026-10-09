"""SQuAD v2's unanswerable half is scored, by whether the system abstained.

Generation kept only questions with a gold answer, so the unanswerable half
of SQuAD v2 (the skill it exists to measure) was dropped, logged as "skipped
without gold context". A system that never abstains scored EM 1.000 where the
official rule gives about 0.5.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask


def _samples(n=4):
    out = []
    for i in range(n):
        answerable = i % 2 == 0
        out.append(
            BenchmarkSample(
                query_id=f"q{i}",
                question=f"question {i}",
                ground_truth="Paris" if answerable else "",
                context="Paris is the capital of France.",
                metadata={"unanswerable": not answerable},
            )
        )
    return out


class _Generator:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.reply(prompt))


class _Judge:
    def __init__(self, llm=None):
        pass

    def evaluate_detailed(
        self, questions, answers, contexts, ground_truths=None, metrics=None, **kw
    ):
        assert all(g for g in ground_truths), "the judge was given a question with no reference"
        rows = [{"faithfulness": 1.0} for _ in questions]
        return exp_ragas.DetailedEvalResult(aggregate={"faithfulness": 1.0}, per_sample=rows)


exp_ragas = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")


def _run(monkeypatch, reply):
    gen = _Generator(reply)
    monkeypatch.setattr(exp_ragas, "RAGASEvaluator", _Judge)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda: None)
    import lasagna.providers as providers

    monkeypatch.setattr(
        providers, "get_llm_provider", lambda: SimpleNamespace(get_llm=lambda **k: gen)
    )
    task = BenchmarkTask(
        benchmark_name="squad_v2", config_name="c", run_index=0, mode="offline", config_overrides={}
    )
    result = asyncio.run(exp._eval_offline(task, _samples()))
    return result, gen


def test_a_system_that_never_abstains_gets_half(monkeypatch):
    result, gen = _run(monkeypatch, lambda p: "Paris")
    assert result.n_samples == 4
    assert result.metrics["exact_match"] == pytest.approx(0.5)
    assert result.metrics["exact_match_has_ans"] == 1.0
    assert result.metrics["exact_match_no_ans"] == 0.0
    assert result.metrics["n_unanswerable"] == 2.0
    assert all("reply exactly: unanswerable" in p for p in gen.prompts)


def test_abstaining_where_there_is_no_answer_is_correct(monkeypatch):
    # Asked about a question it can answer, it answers; otherwise it abstains.
    replies = iter(["Paris", "unanswerable", "Paris", "Unanswerable."])
    result, _ = _run(monkeypatch, lambda p: next(replies))
    assert result.metrics["exact_match"] == 1.0


def test_an_empty_answer_is_a_failure_not_an_abstention(monkeypatch):
    replies = iter(["Paris", "", "Paris", ""])
    result, _ = _run(monkeypatch, lambda p: next(replies))
    assert result.metrics["exact_match_no_ans"] == 0.0


def test_other_benchmarks_keep_their_prompt(monkeypatch):
    gen = _Generator(lambda p: "blue")
    monkeypatch.setattr(exp_ragas, "RAGASEvaluator", _Judge)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda: None)
    import lasagna.providers as providers

    monkeypatch.setattr(
        providers, "get_llm_provider", lambda: SimpleNamespace(get_llm=lambda **k: gen)
    )
    sample = BenchmarkSample(
        query_id="q", question="sky?", ground_truth="blue", context="The sky is blue."
    )
    task = BenchmarkTask(
        benchmark_name="b", config_name="c", run_index=0, mode="offline", config_overrides={}
    )
    result = asyncio.run(exp._eval_offline(task, [sample]))
    assert "unanswerable" not in gen.prompts[0]
    assert "exact_match_no_ans" not in result.metrics


def test_the_squad_adapter_marks_unanswerables(monkeypatch):
    from lasagna.knowledge_evaluation_sdk import datasets
    from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.squad import SQuADv2Adapter

    rows = datasets._convert_squad(
        [
            {"question": "a?", "context": "c", "answers": {"text": ["x", "the x", "x"]}},
            {"question": "b?", "context": "c", "answers": {"text": []}},
        ]
    )
    monkeypatch.setattr(datasets, "load_benchmark", lambda **k: rows)
    samples = SQuADv2Adapter().load(split="validation")
    assert [s.metadata["unanswerable"] for s in samples] == [False, True]
    assert samples[0].ground_truth_answers == ["x", "the x"]
