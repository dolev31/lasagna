"""The harness's judge never runs on a thread with a running event loop.

RAGAS's ``evaluate`` is synchronous. Called inside a running loop, it applies
``nest_asyncio`` and runs its jobs on that same loop; the first time in a
process, with other callbacks queued behind the calling task, the outer loop
then pops from an empty queue (``IndexError: pop from an empty deque``), which
escapes ``asyncio.run`` and stops the whole run (#117). The live path and the check before answers already
judge in a thread; the offline (``generation_only``) path judged inline.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask
from lasagna.utils.token_usage import record_usage

ragas_mod = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class _RAGAS:
    """Stands in for RAGAS: notes whether it was called with a loop running, and
    spends 42 input tokens as a judge call would."""

    calls: list = []

    def __init__(self, llm=None):
        pass

    def evaluate_detailed(self, questions, answers, contexts, ground_truths=None, metrics=None):
        _RAGAS.calls.append(_loop_running())
        record_usage(SimpleNamespace(usage_metadata={"input_tokens": 42, "output_tokens": 7}))
        rows = [{m: 1.0 for m in metrics} for _ in questions]
        return ragas_mod.DetailedEvalResult(aggregate={m: 1.0 for m in metrics}, per_sample=rows)


def test_the_offline_path_judges_off_the_event_loop(monkeypatch):
    _RAGAS.calls = []
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")
    monkeypatch.setenv("RAG_EVAL_JUDGE_METRICS", "faithfulness")
    monkeypatch.setattr(ragas_mod, "RAGASEvaluator", _RAGAS)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda *a, **k: None)
    import lasagna.providers as providers

    def _answer(prompt):
        # Spends 5 input tokens, as the generator's callback would record them.
        record_usage(SimpleNamespace(usage_metadata={"input_tokens": 5, "output_tokens": 1}))
        return SimpleNamespace(content="Warsaw")

    gen = SimpleNamespace(invoke=_answer)
    monkeypatch.setattr(
        providers, "get_llm_provider", lambda: SimpleNamespace(get_llm=lambda **k: gen)
    )
    sample = BenchmarkSample(
        query_id="q1",
        question="Where was Marie Curie born?",
        ground_truth="Warsaw",
        context="Marie Curie was born in Warsaw, Poland, in 1867.",
    )
    task = BenchmarkTask(
        benchmark_name="b", config_name="c", run_index=0, mode="offline", config_overrides={}
    )

    result = asyncio.run(exp._eval_offline(task, [sample]))

    assert _RAGAS.calls == [False], "the judge ran on the event loop's thread"
    assert result.metrics["faithfulness"] == 1.0
    # Each side's spend is counted on its own side.
    assert result.eval_input_tokens == 42
    assert result.system_input_tokens == 5
