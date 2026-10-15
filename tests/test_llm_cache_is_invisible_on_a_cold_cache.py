"""The LLM response cache changes what a run costs, never what it measures.

Every re-run of a sweep paid again for every answer it had already been given:
the same request at temperature 0, the same tokens, the same hours. Replaying
responses fixes the cost, and is only safe if it is invisible: an empty cache
must give the same numbers as no cache, a warm run the same numbers as the cold
run it replays, a sampled call must never be frozen into a replay, and a reader
must be able to tell from the results how much was replayed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from contextlib import closing
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from lasagna.knowledge_evaluation_sdk import experiment as exp  # noqa: E402
from lasagna.providers.llm_provider import SimpleLLMProvider  # noqa: E402
from lasagna.utils.llm_cache import (  # noqa: E402
    LLMCache,
    LLMCacheSettings,
    cache_openai_client,
    llm_cache_scope,
)
from lasagna.utils.token_usage import EVAL, UsageMeter, instrument_openai_client  # noqa: E402

ragas = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")


class _StubChatModel(BaseChatModel):
    """A real LangChain chat model, so the call goes through LangChain's cache
    path; it answers from the prompt and reports usage like a provider."""

    log: Any = None

    @property
    def _llm_type(self) -> str:
        return "stub-chat"

    @property
    def _identifying_params(self):
        return {"model": "stub"}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = messages[-1].content
        self.log.append(prompt)
        reply = "trehalose" if "tardigrade" in prompt.lower() else "a falling weight"
        usage = {
            "input_tokens": len(prompt.split()),
            "output_tokens": len(reply.split()),
            "total_tokens": len(prompt.split()) + len(reply.split()),
        }
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=reply, usage_metadata=usage))]
        )


class _Judge:
    """Scores without a model, deterministically from the answer."""

    def __init__(self, llm=None):
        pass

    def evaluate_detailed(self, questions, answers, contexts, ground_truths=None, **kw):
        rows = [{"faithfulness": 1.0 if a else 0.0} for a in answers]
        mean = sum(r["faithfulness"] for r in rows) / len(rows) if rows else 0.0
        return ragas.DetailedEvalResult(aggregate={"faithfulness": mean}, per_sample=rows)


@pytest.fixture
def calls(monkeypatch):
    """Every prompt the stub model actually received."""
    log: list = []
    provider = SimpleLLMProvider("openai")
    monkeypatch.setattr(provider, "_build_chat_model", lambda *a, **k: _StubChatModel(log=log))
    import lasagna.providers as providers

    monkeypatch.setattr(providers, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(ragas, "RAGASEvaluator", _Judge)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda *a, **k: None)
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")
    monkeypatch.delenv("LASAGNA_LLM_CACHE", raising=False)
    return log


def _run(tmp_path, name, llm_cache, **extra):
    data = {
        "name": "cache",
        "benchmarks": [{"name": "smoke", "n_samples": 4}],
        "mode": "generation_only",
        "output_dir": str(tmp_path / name),
        "llm_cache": llm_cache,
        **extra,
    }
    return asyncio.run(exp.Experiment.from_dict(data).run())


def _cache_at(tmp_path):
    return {"path": str(tmp_path / "llm.sqlite")}


_CACHE_KEYS = ("llm_cache_hits", "llm_cache_misses")


def _numbers(results):
    """Metrics and per-query scores of every task, minus timings and the
    cache's own counters."""
    out = {}
    for r in sorted(results._results, key=lambda r: (r.config_name, r.run_index)):
        metrics = {
            k: v for k, v in r.metrics.items() if "latency" not in k and k not in _CACHE_KEYS
        }
        rows = [
            {k: v for k, v in s.items() if "latency" not in k and not k.endswith("_ms")}
            for s in r.per_sample_scores
        ]
        tokens = (r.system_input_tokens, r.system_output_tokens, r.system_llm_calls)
        out[(r.config_name, r.run_index)] = (metrics, rows, tokens)
    return out


def test_an_empty_cache_gives_the_numbers_of_no_cache(tmp_path, calls):
    off = _run(tmp_path, "off", False)
    n_off = len(calls)
    cold = _run(tmp_path, "cold", _cache_at(tmp_path))

    assert n_off == 4 and len(calls) == 8, "an empty cache served something"
    assert _numbers(cold) == _numbers(off)
    (task_off,) = off._results
    (task_cold,) = cold._results
    assert not any(k in task_off.metrics for k in _CACHE_KEYS), "cache off, yet counted"
    assert task_cold.metrics["llm_cache_hits"] == 0.0
    assert task_cold.metrics["llm_cache_misses"] == 4.0


def test_a_warm_run_calls_nothing_and_reads_like_the_cold_one(tmp_path, calls):
    cold = _run(tmp_path, "cold", _cache_at(tmp_path))
    n_cold = len(calls)
    warm = _run(tmp_path, "warm", _cache_at(tmp_path))

    assert len(calls) == n_cold, "a warm run called the model again"
    assert warm.experiment_config["llm_cache"]["hit_rate"] == 1.0
    assert cold.experiment_config["llm_cache"]["hit_rate"] == 0.0
    # Token counts and LLM calls are the cold run's, replayed with the response.
    assert _numbers(warm) == _numbers(cold)
    (task,) = warm._results
    assert task.system_llm_calls == 4 and task.system_input_tokens > 0
    assert task.metrics["llm_cache_hits"] == 4.0


def test_repeated_runs_each_keep_their_own_responses(tmp_path, calls):
    """Run 1 replaying run 0 would make n_runs measure nothing."""
    _run(tmp_path, "cold", _cache_at(tmp_path), n_runs=2)
    assert len(calls) == 8, "run 1 was served run 0's responses"
    _run(tmp_path, "warm", _cache_at(tmp_path), n_runs=2)
    assert len(calls) == 8


def test_concurrent_tasks_share_one_cache_without_corrupting_it(tmp_path, calls):
    sweep = {
        "pipeline": {"generators": ["concise", "chain_of_thought"]},
        "n_runs": 2,
        "max_concurrency": 4,
    }
    cold = _run(tmp_path, "cold", _cache_at(tmp_path), **sweep)
    n_cold = len(calls)
    warm = _run(tmp_path, "warm", _cache_at(tmp_path), **sweep)

    assert n_cold == 16 and len(calls) == n_cold
    assert _numbers(warm) == _numbers(cold)
    with closing(sqlite3.connect(tmp_path / "llm.sqlite")) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 16


def test_threads_writing_at_once_lose_nothing(tmp_path):
    store = LLMCache(str(tmp_path / "threads.sqlite"))

    def write(t):
        for i in range(25):
            store.put(f"{t}-{i}", json.dumps({"t": t, "i": i}))

    threads = [threading.Thread(target=write, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(store) == 200
    assert json.loads(store.get("7-24")) == {"t": 7, "i": 24}


def test_a_sampled_model_is_never_cached(tmp_path, calls):
    import lasagna.providers as providers

    settings = LLMCacheSettings(path=str(tmp_path / "llm.sqlite"))
    sampled = providers.get_llm_provider().get_llm(temperature=0.7)
    greedy = providers.get_llm_provider().get_llm(temperature=0.0)
    with llm_cache_scope(settings, seed=42) as scope:
        for _ in range(2):
            sampled.invoke("How do tardigrades survive?")
    assert len(calls) == 2 and scope.lookups == 0
    with llm_cache_scope(settings, seed=42):
        for _ in range(2):
            greedy.invoke("How do tardigrades survive?")
    assert len(calls) == 3, "a temperature-0 call was not replayed"
    assert len(LLMCache(settings.path)) == 1


def test_outside_an_experiment_nothing_is_stored(tmp_path, calls, monkeypatch):
    """A server answering users' questions must not write them to disk."""
    import lasagna.providers as providers

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    model = providers.get_llm_provider().get_llm()
    model.invoke("How do tardigrades survive?")
    model.invoke("How do tardigrades survive?")
    assert len(calls) == 2
    assert LLMCache().path.startswith(str(tmp_path / "home"))
    assert not os.path.exists(LLMCache().path)


def test_outside_an_experiment_an_application_cache_still_works(calls):
    """LangChain's own global cache keeps working for library users."""
    import lasagna.providers as providers
    from langchain_core.caches import InMemoryCache
    from langchain_core.globals import set_llm_cache

    set_llm_cache(InMemoryCache())
    try:
        model = providers.get_llm_provider().get_llm()
        model.invoke("How do tardigrades survive?")
        model.invoke("How do tardigrades survive?")
    finally:
        set_llm_cache(None)
    assert len(calls) == 1


def test_no_cache_means_no_replay_even_of_an_application_cache(tmp_path, calls):
    import lasagna.providers as providers
    from langchain_core.caches import InMemoryCache
    from langchain_core.globals import set_llm_cache

    set_llm_cache(InMemoryCache())
    try:
        model = providers.get_llm_provider().get_llm()
        with llm_cache_scope(LLMCacheSettings(enabled=False), seed=1):
            model.invoke("How do tardigrades survive?")
            model.invoke("How do tardigrades survive?")
    finally:
        set_llm_cache(None)
    assert len(calls) == 2


def test_the_environment_switch_turns_it_off(tmp_path, calls, monkeypatch):
    monkeypatch.setenv("LASAGNA_LLM_CACHE", "0")
    _run(tmp_path, "a", _cache_at(tmp_path))
    second = _run(tmp_path, "b", _cache_at(tmp_path))
    assert len(calls) == 8
    assert second.experiment_config["llm_cache"]["enabled"] is False


# -- the judge ----------------------------------------------------------------


def _completion(content="1"):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "judge",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 31, "completion_tokens": 7, "total_tokens": 38},
        }
    )


def _judge_client():
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        return _completion()

    completions = SimpleNamespace(create=create)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions), base_url="http://j/")
    return client, sent


def test_the_judge_client_patched_twice_is_cached_once(tmp_path):
    pytest.importorskip("openai")
    client, sent = _judge_client()
    instrument_openai_client(cache_openai_client(cache_openai_client(client)), EVAL)
    settings = LLMCacheSettings(path=str(tmp_path / "llm.sqlite"))
    meter = UsageMeter()
    request = {"model": "judge", "messages": [{"role": "user", "content": "q"}]}

    with llm_cache_scope(settings, seed=42) as scope, meter, meter.scope(EVAL):
        first = client.chat.completions.create(**request, temperature=0)
        second = client.chat.completions.create(**request, temperature=0)
    assert len(sent) == 1 and (scope.hits, scope.misses) == (1, 1)
    assert second.choices[0].message.content == first.choices[0].message.content
    # A replayed judgement is counted at the usage it cost cold.
    assert (meter.usage(EVAL).input_tokens, meter.usage(EVAL).llm_calls) == (62, 2)

    with llm_cache_scope(settings, seed=42):
        client.chat.completions.create(**request, temperature=0.01)  # RAGAS's default
        client.chat.completions.create(**request, temperature=0.01)
    assert len(sent) == 3, "a sampled judgement was replayed"


def test_the_harness_judge_goes_through_the_cache(monkeypatch):
    pytest.importorskip("openai")
    import openai
    import ragas.llms

    client, _ = _judge_client()
    seen = {}
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: client)
    monkeypatch.setattr(
        ragas.llms, "llm_factory", lambda model, client: seen.setdefault("client", client)
    )
    monkeypatch.setattr(exp, "_active_judge_model", None)
    exp._build_judge_llm("judge-model")
    completions = seen["client"].chat.completions
    assert "_lasagna_llm_cached" in completions.__dict__
    assert "_lasagna_usage_counted" in completions.__dict__


# -- what the results say -------------------------------------------------------


def test_results_and_the_bundle_carry_the_hit_rate(tmp_path, calls):
    from lasagna.knowledge_evaluation_sdk.bundle import build_bundle

    cold = _run(tmp_path, "cold", _cache_at(tmp_path))
    warm = _run(tmp_path, "warm", _cache_at(tmp_path))

    saved = json.loads((tmp_path / "warm" / "results.json").read_text())
    record = saved["experiment_config"]["llm_cache"]
    assert record == {
        "enabled": True,
        "deterministic_only": True,
        "hits": 4,
        "misses": 0,
        "hit_rate": 1.0,
    }
    assert str(tmp_path) not in json.dumps(record), "a bundle must not carry a local path"
    bundle = build_bundle(warm)
    assert bundle["experiment"]["llm_cache"]["hit_rate"] == 1.0
    # Replayed or not, it is the same experiment.
    keys = {c["name"]: c["comparability_key"] for c in build_bundle(cold)["configs"]}
    assert keys == {c["name"]: c["comparability_key"] for c in bundle["configs"]}
    for row in warm._results[0].per_sample_scores:
        assert not any(k in row for k in _CACHE_KEYS), "counters leaked into per-query scores"


def test_a_typo_under_llm_cache_is_refused():
    with pytest.raises(ValueError, match="Unknown key"):
        LLMCacheSettings.parse({"pathh": "x"})
    with pytest.raises(ValueError, match="true or false"):
        LLMCacheSettings.parse({"deterministic_only": "yes"})
    assert LLMCacheSettings.parse(None).enabled is True
    assert LLMCacheSettings.parse(False).enabled is False


def test_no_cache_on_the_command_line(tmp_path, calls, monkeypatch):
    import yaml
    from typer.testing import CliRunner

    from lasagna import cli

    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        yaml.safe_dump(
            {
                "name": "cli",
                "benchmarks": [{"name": "smoke", "n_samples": 2}],
                "mode": "generation_only",
                "output_dir": str(tmp_path / "out"),
                "llm_cache": _cache_at(tmp_path),
            }
        )
    )
    for _ in range(2):
        result = CliRunner().invoke(cli.app, ["benchmark", "experiment", str(sweep), "--no-cache"])
        assert result.exit_code == 0, result.output
    assert len(calls) == 4
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert saved["experiment_config"]["llm_cache"]["enabled"] is False
