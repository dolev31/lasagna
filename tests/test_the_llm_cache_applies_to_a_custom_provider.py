"""The LLM response cache serves a provider set with ``set_llm_provider``.

The cache was attached inside ``SimpleLLMProvider.get_chat_model`` only. A
custom ``LLMProvider`` that builds its own chat models never carried it, so a
warm run called the model for every answer again, while ``results.json``
still recorded ``llm_cache: {enabled: true, hits: 0, misses: 0}``. Every
``get_chat_model`` of an ``LLMProvider`` subclass now hands back a model with
the cache (and the token meter) attached; a model that cannot carry it is
recorded as such instead of reading as an empty cache.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from lasagna.knowledge_evaluation_sdk import experiment as exp  # noqa: E402
from lasagna.providers.llm_provider import LLMProvider  # noqa: E402

ragas = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")


class _ChatModel(BaseChatModel):
    """A LangChain chat model that records every prompt it is sent."""

    log: Any = None
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "custom-stub"

    @property
    def _identifying_params(self):
        return {"model": "in-house", "temperature": self.temperature}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = messages[-1].content
        self.log.append(prompt)
        usage = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="an answer", usage_metadata=usage))
            ]
        )


class _InHouseProvider(LLMProvider):
    """What a user writes for a model gateway Lasagna does not know."""

    def __init__(self, log, model_cls=_ChatModel):
        self.log = log
        self.model_cls = model_cls

    def get_chat_model(self, model_settings, temperature=0.0, max_tokens=None):
        if self.model_cls is _ChatModel:
            return _ChatModel(log=self.log, temperature=temperature)
        return self.model_cls(self.log)

    def validate_model_settings(self, model_settings):
        return True


class _Judge:
    def __init__(self, llm=None):
        pass

    def evaluate_detailed(self, questions, answers, contexts, ground_truths=None, **kw):
        rows = [{"faithfulness": 1.0} for _ in answers]
        return ragas.DetailedEvalResult(aggregate={"faithfulness": 1.0}, per_sample=rows)


def _use(monkeypatch, provider):
    import lasagna.providers as providers

    monkeypatch.setattr(providers, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(ragas, "RAGASEvaluator", _Judge)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda *a, **k: None)
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")
    monkeypatch.delenv("LASAGNA_LLM_CACHE", raising=False)


def _run(tmp_path, name):
    data = {
        "name": "custom",
        "benchmarks": [{"name": "smoke", "n_samples": 4}],
        "mode": "generation_only",
        "output_dir": str(tmp_path / name),
        "llm_cache": {"path": str(tmp_path / "llm.sqlite")},
    }
    return asyncio.run(exp.Experiment.from_dict(data).run())


def test_a_warm_run_with_a_custom_provider_calls_nothing(tmp_path, monkeypatch):
    log: list = []
    _use(monkeypatch, _InHouseProvider(log))

    cold = _run(tmp_path, "cold")
    n_cold = len(log)
    warm = _run(tmp_path, "warm")

    assert n_cold == 4
    assert len(log) == n_cold, "the warm run called the custom provider's model again"
    assert cold.experiment_config["llm_cache"]["misses"] == 4
    assert warm.experiment_config["llm_cache"]["hits"] == 4
    assert warm.experiment_config["llm_cache"]["hit_rate"] == 1.0
    # The token meter is attached too: a replay reads like the cold run.
    (task,) = warm._results
    assert task.system_llm_calls == 4 and task.system_input_tokens == 12


def test_a_sampling_custom_model_is_not_cached(tmp_path, monkeypatch):
    log: list = []

    class _Sampling(_InHouseProvider):
        def get_chat_model(self, model_settings, temperature=0.0, max_tokens=None):
            return _ChatModel(log=self.log, temperature=0.7)

    _use(monkeypatch, _Sampling(log))
    _run(tmp_path, "cold")
    _run(tmp_path, "warm")

    assert len(log) == 8, "a model that samples at 0.7 was replayed"


class _NotLangChain:
    """A provider may return any object with ``invoke``; it cannot carry a cache."""

    def __init__(self, log):
        self.log = log

    def invoke(self, prompt):
        self.log.append(prompt)
        return SimpleNamespace(content="an answer")


def test_a_model_that_cannot_carry_the_cache_is_recorded_as_such(tmp_path, monkeypatch):
    log: list = []
    _use(monkeypatch, _InHouseProvider(log, model_cls=_NotLangChain))

    results = _run(tmp_path, "cold")

    cache = results.experiment_config["llm_cache"]
    assert len(log) == 4
    assert cache["applied"] is False
    assert "_NotLangChain" in cache["not_applied"]
