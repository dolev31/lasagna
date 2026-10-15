"""A sweep says what it will need before it runs, and what it cost if asked.

A sweep's first sign of a missing model, a misnamed arm or a bill was the run
itself: hours in, or a download in. `--dry-run` prints the plan and an
estimate from a three-sample probe, and must not cost more than those three
samples or leave anything in the results folder. A dollar figure is opt-in
(`prices:`): the run records tokens, and a price is not a measurement, so it is
named an estimate, appears only when every token it covers has a rate, and the
run metadata names the table it came from.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

pytest.importorskip("langchain_core")

import yaml  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from lasagna import cli  # noqa: E402
from lasagna.knowledge_evaluation_sdk import experiment as exp  # noqa: E402
from lasagna.knowledge_evaluation_sdk.cost import (  # noqa: E402
    BUILTIN_TABLE,
    PriceTable,
    estimate_cost,
)
from lasagna.knowledge_evaluation_sdk.results import lower_is_better  # noqa: E402
from lasagna.knowledge_evaluation_sdk.runner import TaskResult  # noqa: E402
from lasagna.providers.llm_provider import SimpleLLMProvider  # noqa: E402

ragas = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")


class _StubChatModel(BaseChatModel):
    log: Any = None

    @property
    def _llm_type(self) -> str:
        return "stub-chat"

    @property
    def _identifying_params(self):
        return {"model": "stub"}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.log.append(messages[-1].content)
        usage = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
        message = AIMessage(content="trehalose", usage_metadata=usage)
        return ChatResult(generations=[ChatGeneration(message=message)])


class _Judge:
    def __init__(self, llm=None):
        pass

    def evaluate_detailed(self, questions, answers, contexts, ground_truths=None, **kw):
        rows = [{"faithfulness": 1.0} for _ in questions]
        return ragas.DetailedEvalResult(aggregate={"faithfulness": 1.0}, per_sample=rows)


@pytest.fixture
def calls(monkeypatch, tmp_path):
    log: list = []
    provider = SimpleLLMProvider("openai")
    monkeypatch.setattr(provider, "_build_chat_model", lambda *a, **k: _StubChatModel(log=log))
    import lasagna.providers as providers

    monkeypatch.setattr(providers, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(ragas, "RAGASEvaluator", _Judge)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda *a, **k: None)
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")
    monkeypatch.setenv("RAG_EVAL_GENERATOR_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("RAG_EVAL_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # The static estimate counts with whatever is installed; pin it so the
    # test needs no tokenizer download.
    from lasagna.utils import tokenization

    monkeypatch.setattr(tokenization, "_get_tokenizer", lambda *a, **k: None)
    return log


def _sweep_file(tmp_path, **extra):
    data = {
        "name": "plan",
        "benchmarks": [{"name": "smoke"}],
        "mode": "generation_only",
        "n_runs": 2,
        "output_dir": str(tmp_path / "out"),
        "pipeline": {"generators": ["concise", "cited"]},
        **extra,
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def _cli(*args):
    return CliRunner().invoke(cli.app, ["benchmark", "experiment", *map(str, args)])


def test_a_dry_run_probes_three_samples_and_writes_no_results(tmp_path, calls):
    result = _cli(_sweep_file(tmp_path), "--dry-run")

    assert result.exit_code == 0, result.output
    assert len(calls) == 3, "the probe ran more than three samples"
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "home" / ".lasagna" / "research").exists()
    out = " ".join(result.output.split())
    assert "Arms (2): generator=concise, generator=cited" in out
    assert "Tasks (4): 1 benchmark x 2 arms x 2 runs" in out
    assert "smoke split=test 12 samples" in out
    assert "Tokens (static estimate, characters / 4)" in out
    # 12 samples x 4 tasks, from the probe's 100 input + 10 output per call.
    assert "Extrapolated to 48 queries" in out and "5,280 system tokens" in out
    # 48 x (100 x 0.15 + 10 x 0.60) / 1e6 dollars, from the built-in table.
    assert f"Cost (estimate, price table {BUILTIN_TABLE}): about $0.0010." in out
    assert "embedder" not in out, "generation_only never loads the retrieval models"


def test_no_probe_runs_nothing_at_all(tmp_path, calls):
    result = _cli(_sweep_file(tmp_path), "--dry-run", "--no-probe")
    assert result.exit_code == 0, result.output
    assert calls == []
    assert not (tmp_path / "home" / ".lasagna").exists()
    assert "Probe skipped" in result.output


def test_each_benchmark_gets_one_probe_of_three(tmp_path, calls, monkeypatch):
    from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
    from lasagna.knowledge_evaluation_sdk.benchmarks.adapters.smoke import SmokeBenchmarkAdapter

    benchmark_registry.get("smoke")  # load the built-ins first
    monkeypatch.setitem(benchmark_registry._adapters, "smoke_twin", SmokeBenchmarkAdapter())
    # The twin asks the same questions, which the cache would replay.
    path = _sweep_file(
        tmp_path, benchmarks=[{"name": "smoke"}, {"name": "smoke_twin"}], llm_cache=False
    )
    result = _cli(path, "--dry-run")
    assert result.exit_code == 0, result.output
    assert len(calls) == 6


def test_a_dry_run_refuses_what_the_run_would(tmp_path, calls):
    path = _sweep_file(tmp_path, pipeline={"generators": ["not_a_generator"]})
    result = _cli(path, "--dry-run")
    assert result.exit_code == 1 and "not_a_generator" in result.output
    assert calls == [] and not (tmp_path / "out").exists()


def test_no_probe_needs_a_dry_run(tmp_path, calls):
    result = _cli(_sweep_file(tmp_path), "--no-probe")
    assert result.exit_code == 1 and "--dry-run" in result.output
    assert calls == []


# -- cost ---------------------------------------------------------------------


def _task(**tokens):
    base = dict(task_key="t", benchmark_name="b", config_name="c", run_index=0)
    return TaskResult(**base, **tokens)


def test_cost_is_tokens_times_the_table():
    result = _task(
        system_input_tokens=1_000_000,
        system_output_tokens=500_000,
        system_llm_calls=10,
        eval_input_tokens=2_000_000,
        eval_llm_calls=4,
    )
    # gpt-4o-mini $0.15 in / $0.60 out; gpt-4.1 $2.00 in.
    cost = estimate_cost(result, PriceTable(), "gpt-4o-mini", "gpt-4.1")
    assert cost == pytest.approx(0.15 + 0.30 + 4.00)
    # A routed name is priced as the model it routes to.
    assert PriceTable().price("openai/gpt-4o-mini") == PriceTable().price("gpt-4o-mini")


def test_no_rate_no_number():
    priced = _task(system_input_tokens=1000, system_llm_calls=1, eval_input_tokens=1000)
    assert estimate_cost(priced, PriceTable(), "gpt-4o-mini", None) is None, (
        "the judge's tokens have no rate, so a sum would be partial"
    )
    assert estimate_cost(priced, PriceTable(), "my-local-model", "gpt-4.1") is None
    assert estimate_cost(_task(), PriceTable(), "gpt-4o-mini", "gpt-4.1") is None


def _run(tmp_path, **extra):
    data = {
        "name": "cost",
        "benchmarks": [{"name": "smoke", "n_samples": 4}],
        "mode": "generation_only",
        "output_dir": str(tmp_path / "out"),
        "llm_cache": False,
        **extra,
    }
    return asyncio.run(exp.Experiment.from_dict(data).run())


def test_a_sweep_that_asks_for_prices_gets_an_estimate_and_its_source(tmp_path, calls):
    results = _run(tmp_path, prices="builtin")
    (task,) = results._results
    expected = (task.system_input_tokens * 0.15 + task.system_output_tokens * 0.60) / 1e6
    assert task.system_input_tokens == 400
    assert task.metrics["cost_usd_estimate"] == pytest.approx(expected)
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    # The judge is recorded too: with none named it is the default, gpt-6-luna,
    # whose rate holds for prompts up to 272K tokens.
    assert saved["experiment_config"]["cost_estimate"] == {
        "price_table": BUILTIN_TABLE,
        "overrides": [],
        "is_estimate": True,
        "prices": {
            "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.6},
            "gpt-6-luna": {"input_per_1m": 0.1, "output_per_1m": 0.5, "max_prompt_tokens": 272_000},
        },
    }
    assert lower_is_better("cost_usd_estimate")


def test_without_prices_there_is_no_dollar_field(tmp_path, calls):
    results = _run(tmp_path)
    (task,) = results._results
    assert task.system_llm_calls == 4
    assert not [k for k in task.metrics if "cost" in k or "usd" in k]
    assert "cost_estimate" not in results.experiment_config


def test_a_model_the_table_does_not_price_gets_no_field(tmp_path, calls, monkeypatch):
    monkeypatch.setenv("RAG_EVAL_GENERATOR_MODEL", "my-local-model")
    (task,) = _run(tmp_path, prices="builtin")._results
    assert "cost_usd_estimate" not in task.metrics


def test_the_sweep_prices_win_over_the_table(tmp_path, calls):
    prices = {"gpt-4o-mini": {"input_per_1m": 1.0, "output_per_1m": 2.0}}
    results = _run(tmp_path, prices=prices)
    (task,) = results._results
    expected = (task.system_input_tokens * 1.0 + task.system_output_tokens * 2.0) / 1e6
    assert task.metrics["cost_usd_estimate"] == pytest.approx(expected)
    record = results.experiment_config["cost_estimate"]
    assert record["overrides"] == ["gpt-4o-mini"]
    assert record["prices"]["gpt-4o-mini"] == {"input_per_1m": 1.0, "output_per_1m": 2.0}


@pytest.mark.parametrize(
    "value",
    [
        "cheap",
        {"gpt-4o-mini": {"input_per_1m": 1.0}},
        {"m": {"input_per_1m": "x", "output_per_1m": 1}},
    ],
)
def test_a_malformed_price_is_refused(value):
    with pytest.raises(ValueError, match="prices"):
        PriceTable.parse(value)
