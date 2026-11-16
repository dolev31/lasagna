"""``benchmark run`` checks the judge beside its generator check, and only once (#96).

The CLI refuses a generation run with no generator LLM before it builds the
experiment. The judge is a second LLM client with its own variables
(``RAG_EVAL_JUDGE_BASE_URL``, ``RAG_EVAL_JUDGE_API_KEY``), so the same place
checks it. ``Experiment.preflight_judge`` is that check: ``run()`` skips its own
once, so one experiment probes the judge once whichever way it starts, and a
direct SDK run keeps its check.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import yaml
from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk import experiment as exp
from tests.test_the_judge_is_checked_before_answers_are_written import (
    _StubRAGAS,
    _clear_judge_env,
    _generation_sweep,
    _install_offline_openai,
    _ok_judge_handler,
    _stub_generator,
    ragas_mod,
)

_RUN = ["benchmark", "run", "smoke", "--mode", "generation_only", "--tracks", "generation"]


@pytest.fixture
def cli_run(tmp_path, monkeypatch):
    """A generation `benchmark run` with a working generator and spies on every step after the check."""
    _clear_judge_env(monkeypatch)
    prompts = _stub_generator(monkeypatch)
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")
    ingested: list = []

    async def _spy_ingest(*a, **k):
        ingested.append(a)
        raise AssertionError("corpus must not be ingested when the judge check fails")

    monkeypatch.setattr(exp, "_ensure_corpus_ingested", _spy_ingest)
    checks: list = []
    real_check = exp._check_judge_llm

    def _counting_check(*a, **k):
        checks.append(a)
        return real_check(*a, **k)

    monkeypatch.setattr(exp, "_check_judge_llm", _counting_check)

    def invoke(*extra):
        return CliRunner().invoke(cli.app, [*_RUN, "--output", str(tmp_path / "out"), *extra])

    invoke.prompts, invoke.ingested, invoke.checks = prompts, ingested, checks
    return invoke


def test_cli_stops_before_the_first_answer_without_judge_credentials(cli_run):
    result = cli_run()
    assert result.exit_code == 1
    for name in ("qwen3:0.6b", "RAG_EVAL_JUDGE_BASE_URL", "RAG_EVAL_JUDGE_API_KEY"):
        assert name in result.output
    assert cli_run.prompts == [] and cli_run.ingested == []
    assert len(cli_run.checks) == 1, "the failed CLI check is the only one; run() never started"
    assert "Running benchmark" not in result.output


def test_cli_stops_before_the_first_answer_when_the_judge_is_unreachable(cli_run, monkeypatch):
    import httpx

    monkeypatch.setenv("RAG_EVAL_JUDGE_API_KEY", "ollama")  # pragma: allowlist secret
    monkeypatch.setenv("RAG_EVAL_JUDGE_BASE_URL", "http://ollama:11434/v1")

    def _refused(request):
        raise httpx.ConnectError("Connection refused to ollama:11434")

    _install_offline_openai(monkeypatch, _refused)
    result = cli_run()
    assert result.exit_code == 1
    assert "APIConnectionError" in result.output
    assert "http://ollama:11434/v1" in result.output
    assert "RAG_EVAL_JUDGE_API_KEY" in result.output
    assert cli_run.prompts == [] and cli_run.ingested == []


def test_a_cli_run_with_a_working_judge_probes_it_exactly_once(cli_run, monkeypatch):
    monkeypatch.setenv("RAG_EVAL_JUDGE_API_KEY", "ollama")  # pragma: allowlist secret
    monkeypatch.setenv("RAG_EVAL_JUDGE_BASE_URL", "http://ollama:11434/v1")
    judge_requests: list = []
    _install_offline_openai(monkeypatch, _ok_judge_handler(judge_requests))
    monkeypatch.setattr(ragas_mod, "RAGASEvaluator", _StubRAGAS)

    result = cli_run("--samples", "2", "--runs", "2")

    assert result.exit_code == 0, result.output
    assert len(cli_run.prompts) == 4  # 2 runs x 2 samples: the generator ran
    # Once by the CLI; the experiment it started did not probe a second time.
    assert len(cli_run.checks) == 1
    assert len(judge_requests) == 1


def test_the_cli_check_does_not_excuse_a_later_run_of_the_same_experiment(monkeypatch, tmp_path):
    _clear_judge_env(monkeypatch)
    _stub_generator(monkeypatch)
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")
    monkeypatch.setenv("RAG_EVAL_JUDGE_API_KEY", "ollama")  # pragma: allowlist secret
    monkeypatch.setenv("RAG_EVAL_JUDGE_BASE_URL", "http://ollama:11434/v1")
    monkeypatch.setattr(ragas_mod, "RAGASEvaluator", _StubRAGAS)
    judge_requests: list = []
    _install_offline_openai(monkeypatch, _ok_judge_handler(judge_requests))

    experiment = exp.Experiment.from_dict(_generation_sweep(tmp_path))
    experiment.preflight_judge()
    experiment.preflight_judge()  # asking twice does not probe twice
    assert len(judge_requests) == 1
    asyncio.run(experiment.run())  # consumes the preflight
    assert len(judge_requests) == 1
    asyncio.run(experiment.run())  # a second run is a new experiment run
    assert len(judge_requests) == 2


def test_a_preflight_does_not_excuse_a_different_judge(monkeypatch, tmp_path):
    _clear_judge_env(monkeypatch)
    _stub_generator(monkeypatch)
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")
    monkeypatch.setenv("RAG_EVAL_JUDGE_API_KEY", "ollama")  # pragma: allowlist secret
    monkeypatch.setenv("RAG_EVAL_JUDGE_BASE_URL", "http://ollama:11434/v1")
    monkeypatch.setattr(ragas_mod, "RAGASEvaluator", _StubRAGAS)
    judge_requests: list = []
    _install_offline_openai(monkeypatch, _ok_judge_handler(judge_requests))

    experiment = exp.Experiment.from_dict(_generation_sweep(tmp_path))
    experiment.preflight_judge()
    monkeypatch.setenv("RAG_EVAL_JUDGE_BASE_URL", "http://other:11434/v1")
    asyncio.run(experiment.run())
    assert len(judge_requests) == 2, "the endpoint changed, so the earlier check no longer applies"


def test_a_direct_sdk_run_still_checks_the_judge_itself(monkeypatch, tmp_path):
    _clear_judge_env(monkeypatch)
    prompts = _stub_generator(monkeypatch)
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")
    with pytest.raises(RuntimeError, match="RAG_EVAL_JUDGE_BASE_URL"):
        asyncio.run(exp.Experiment.from_dict(_generation_sweep(tmp_path)).run())
    assert prompts == []


def test_a_retrieval_only_cli_run_never_probes_the_judge(monkeypatch, tmp_path):
    _clear_judge_env(monkeypatch)
    _stub_generator(monkeypatch)

    def _never(*a, **k):
        raise AssertionError("a retrieval-only run must not build or probe a judge")

    monkeypatch.setattr(exp, "_check_judge_llm", _never)
    monkeypatch.setattr(exp, "_build_judge_llm", _never)

    async def _stop(self, progress_callback=None):
        raise RuntimeError("reached run()")

    monkeypatch.setattr(exp.Experiment, "run", _stop)
    result = CliRunner().invoke(
        cli.app, ["benchmark", "run", "smoke", "--tracks", "retrieval", "-o", str(tmp_path)]
    )
    assert "reached run()" in result.output, result.output  # past the preflight, not stopped by it


def test_the_preflight_is_a_no_op_for_a_retrieval_fallback_and_a_custom_eval_fn(
    monkeypatch, tmp_path
):
    """No generator LLM and no judge named keeps the retrieval fallback; a custom eval_fn judges nothing."""
    _clear_judge_env(monkeypatch)
    import lasagna.providers as providers

    def _no_provider():
        raise providers.LLMProviderNotConfigured("no key")

    monkeypatch.setattr(providers, "get_llm_provider", _no_provider)

    def _never(*a, **k):
        raise AssertionError("no judge probe expected")

    monkeypatch.setattr(exp, "_check_judge_llm", _never)
    fallback = exp.Experiment.from_dict(
        {
            "name": "both",
            "benchmarks": [{"name": "smoke", "n_samples": 2}],
            "tracks": ["retrieval", "generation"],
            "mode": "live",
        }
    )
    fallback.preflight_judge()

    async def _eval(task, samples):
        raise AssertionError("not run")

    custom = exp.Experiment.from_dict(_generation_sweep(tmp_path), eval_fn=_eval)
    custom.preflight_judge()


def test_dry_run_no_probe_makes_no_judge_request(monkeypatch, tmp_path):
    _clear_judge_env(monkeypatch)
    _stub_generator(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")
    from lasagna.utils import tokenization

    monkeypatch.setattr(tokenization, "_get_tokenizer", lambda *a, **k: None)

    def _never(*a, **k):
        raise AssertionError("--dry-run --no-probe must not touch the judge")

    monkeypatch.setattr(exp, "_check_judge_llm", _never)
    monkeypatch.setattr(exp, "_build_judge_llm", _never)
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump(_generation_sweep(tmp_path)))
    result = CliRunner().invoke(
        cli.app, ["benchmark", "experiment", str(sweep_path), "--dry-run", "--no-probe"]
    )
    assert result.exit_code == 0, result.output


def test_benchmark_experiment_checks_judge_before_clearing_checkpoints_and_only_once(
    monkeypatch, tmp_path
):
    _clear_judge_env(monkeypatch)
    prompts = _stub_generator(monkeypatch)
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    sweep = _generation_sweep(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel_ckpt = out_dir / "checkpoint_existing.json"
    sentinel_ckpt.write_text('{"task_key": "existing"}')

    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump(sweep))

    # Missing judge credentials must fail before cp.clear() deletes existing checkpoints.
    failed = CliRunner().invoke(cli.app, ["benchmark", "experiment", str(sweep_path)])
    assert failed.exit_code == 1
    assert "RAG_EVAL_JUDGE_BASE_URL" in failed.output
    assert "Starting experiment" not in failed.output
    assert sentinel_ckpt.exists(), "failed judge preflight must not clear existing checkpoints"
    assert prompts == []

    # With a working judge, benchmark experiment probes the judge exactly once.
    monkeypatch.setenv("RAG_EVAL_JUDGE_API_KEY", "ollama")  # pragma: allowlist secret
    monkeypatch.setenv("RAG_EVAL_JUDGE_BASE_URL", "http://ollama:11434/v1")
    judge_requests: list = []
    _install_offline_openai(monkeypatch, _ok_judge_handler(judge_requests))
    monkeypatch.setattr(ragas_mod, "RAGASEvaluator", _StubRAGAS)

    ok = CliRunner().invoke(cli.app, ["benchmark", "experiment", str(sweep_path)])
    assert ok.exit_code == 0, ok.output
    assert len(judge_requests) == 1
    assert len(prompts) == 2


def test_preflight_judge_skips_local_file_with_corpus_on_retrieval_track(monkeypatch, tmp_path):
    """Regression test: a local data path with contexts runs in live mode on retrieval-only."""
    from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    local_file = tmp_path / "my_data.jsonl"
    local_file.write_text(
        json.dumps(
            {
                "question": "What is Lasagna?",
                "contexts": ["Lasagna is a RAG evaluation framework."],
                "answer": "A RAG evaluation framework.",
            }
        )
        + "\n"
    )

    def _never(*a, **k):
        raise AssertionError("retrieval-only local data with a corpus must not probe the judge")

    monkeypatch.setattr(exp, "_check_judge_llm", _never)
    monkeypatch.setattr(exp, "_build_judge_llm", _never)
    benchmark_registry.names()  # populate built-ins before taking the snapshot
    before = dict(benchmark_registry._adapters)
    try:
        retrieval_exp = exp.Experiment.from_dict(
            {
                "name": "local_retrieval",
                "benchmarks": [{"path": str(local_file)}],
                "tracks": ["retrieval"],
                "pipeline": {"base_profile": "speed"},
            }
        )
        retrieval_exp.preflight_judge()
    finally:
        benchmark_registry._adapters.clear()
        benchmark_registry._adapters.update(before)
