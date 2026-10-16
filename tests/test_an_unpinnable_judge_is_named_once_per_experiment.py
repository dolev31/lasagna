"""A judge that cannot be pinned is named once per experiment, not once per task.

A sweep scores every task with the same judge, so a note that the judge's
scores can vary between runs is one fact about the experiment. Said per task,
a six-task sweep would print it six times and bury the rest of the output.
The generator here is pinned, so the judge's note is the only one.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from loguru import logger

from lasagna.knowledge_evaluation_sdk import experiment as exp

ragas = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")

JUDGE = "claude-opus-4-7"


class _Answer:
    def __init__(self, **kwargs):
        pass

    def invoke(self, prompt):
        return SimpleNamespace(content="an answer")


class _Judge:
    def __init__(self, llm=None):
        pass

    def evaluate_detailed(self, questions, answers, contexts, ground_truths=None, **kw):
        rows = [{"faithfulness": 1.0} for _ in questions]
        return ragas.DetailedEvalResult(aggregate={"faithfulness": 1.0}, per_sample=rows)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    import lasagna.providers as providers
    from lasagna.providers.llm_provider import SimpleLLMProvider

    for var, _ in providers._API_KEY_TO_PLATFORM:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")  # pragma: allowlist secret
    monkeypatch.setenv("MODEL_NAME", "claude-haiku-4-5-20251001")  # a pinned generator
    monkeypatch.delenv("RAG_EVAL_GENERATOR_MODEL", raising=False)
    monkeypatch.delenv("RAG_EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")
    provider = SimpleLLMProvider("anthropic")
    monkeypatch.setattr(provider, "_build_chat_model", lambda *a, **k: _Answer())
    monkeypatch.setattr(providers, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(ragas, "RAGASEvaluator", _Judge)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda *a, **k: None)


def test_six_tasks_say_it_once_and_record_it(stubbed, tmp_path):
    logged = []
    sink = logger.add(lambda m: logged.append(m.record["message"]), level="WARNING")
    try:
        results = asyncio.run(
            exp.Experiment.from_dict(
                {
                    "name": "unpinnable judge",
                    "benchmarks": [{"name": "smoke", "n_samples": 2}],
                    "mode": "generation_only",
                    "n_runs": 3,
                    "judge_model": JUDGE,
                    "pipeline": {"generators": ["concise", "closed_book"]},
                    "output_dir": str(tmp_path / "out"),
                    "llm_cache": False,
                }
            ).run()
        )
    finally:
        logger.remove(sink)

    assert results.total_tasks == 6 and not results.failed_tasks
    temperature = [m for m in logged if "does not accept temperature" in m]
    assert temperature == [
        f"{JUDGE} does not accept temperature; judge scores can vary between runs; "
        "repeat runs (n_runs >= 3) to measure that variance"
    ], logged
    assert results.experiment_config["sampling"]["judge"] == {
        "model": JUDGE,
        "sampling": "provider-default",
    }
