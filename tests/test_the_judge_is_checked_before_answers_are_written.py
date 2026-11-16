"""The judge is checked once per experiment before the first answer is written.

A generation run checks that the generator has an LLM, ingests the corpus,
retrieves for every question, writes an answer for every question, and only
then builds the RAGAS judge. Because the judge uses its own OpenAI-compatible
client (``RAG_EVAL_JUDGE_BASE_URL`` and ``RAG_EVAL_JUDGE_API_KEY``, falling
back to ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY``) rather than the generator's
provider or ``OLLAMA_BASE_URL``, a missing key or base URL surfaced only after
every answer had been generated (#96).
"""

from __future__ import annotations

import asyncio
import json
import traceback
from types import SimpleNamespace

import httpx
import openai
import pytest
import yaml
from loguru import logger
from typer.testing import CliRunner

from lasagna import cli
from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.providers.llm_provider import SimpleLLMProvider

ragas_mod = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")


class _StubRAGAS:
    """Scores without calling the judge client during task-end evaluation."""

    def __init__(self, llm=None):
        self.llm = llm

    def evaluate_detailed(self, questions, answers, contexts, ground_truths=None, **kw):
        rows = [{"faithfulness": 1.0} for _ in questions]
        return ragas_mod.DetailedEvalResult(aggregate={"faithfulness": 1.0}, per_sample=rows)


def _clear_judge_env(monkeypatch):
    for var in (
        "RAG_EVAL_JUDGE_MODEL",
        "RAG_EVAL_JUDGE_BASE_URL",
        "RAG_EVAL_JUDGE_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def _stub_generator(monkeypatch):
    """Configure a working generator that records every prompt it is given."""
    import lasagna.providers as providers

    prompts: list[str] = []

    class _Gen:
        def invoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content="Paris")

        async def ainvoke(self, prompt):
            prompts.append(prompt)
            return SimpleNamespace(content="Paris")

    gen = _Gen()
    provider = SimpleNamespace(
        get_llm=lambda *a, **k: gen,
        get_chat_model=lambda *a, **k: gen,
        validate_model_settings=lambda *a, **k: True,
    )
    monkeypatch.setattr(providers, "get_llm_provider", lambda: provider)
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")
    return prompts


def _install_offline_openai(monkeypatch, handler):
    """Replace ``openai.OpenAI`` with a real client on a mock HTTP transport."""
    created_kwargs: list[dict] = []

    class _OfflineOpenAI(openai.OpenAI):
        def __init__(self, **kw):
            created_kwargs.append(dict(kw))
            super().__init__(
                **{"api_key": kw.get("api_key") or "sk-mock", **kw},  # pragma: allowlist secret
                max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            )

    monkeypatch.setattr(openai, "OpenAI", _OfflineOpenAI)
    return created_kwargs


def _ok_judge_handler(requests_seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests_seen.append((str(request.url), body))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-probe",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "qwen3:0.6b"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"ok": true}'},
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
            },
        )

    return handler


def _generation_sweep(tmp_path, **extra):
    return {
        "name": "judge-preflight",
        "benchmarks": [{"name": "smoke", "n_samples": 2}],
        "mode": "generation_only",
        "n_runs": 1,
        "output_dir": str(tmp_path / "out"),
        "llm_cache": False,
        **extra,
    }


def test_missing_judge_credentials_stop_before_first_answer(tmp_path, monkeypatch):
    """1. Generator on Ollama + RAG_EVAL_JUDGE_MODEL without judge key/URL fails upfront."""
    _clear_judge_env(monkeypatch)
    prompts = _stub_generator(monkeypatch)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("MODEL_NAME", "qwen3:0.6b")
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(exp.Experiment.from_dict(_generation_sweep(tmp_path)).run())

    msg = str(exc_info.value)
    assert "qwen3:0.6b" in msg
    assert "RAG_EVAL_JUDGE_BASE_URL" in msg
    assert "RAG_EVAL_JUDGE_API_KEY" in msg
    assert "OPENAI_API_KEY" in msg
    assert "OLLAMA_BASE_URL" in msg
    assert "OpenAIError" in msg
    assert prompts == [], "no answer should be generated when the judge is unconfigured"
    assert exp._active_judge_model is None, "_active_judge_model must be reset on failure"


@pytest.mark.parametrize(
    "case,base_url,api_key,handler_factory,expected_fragment",
    [
        (
            "auth_rejected",
            None,
            "super-secret-ollama-key",  # pragma: allowlist secret
            lambda: (
                lambda req: httpx.Response(
                    401,
                    json={
                        "error": {
                            "message": "Incorrect API key provided: super-secret-ollama-key",
                            "type": "invalid_request_error",
                        }
                    },
                )
            ),
            "AuthenticationError",
        ),
        (
            "unreachable_endpoint",
            "http://ollama:11434/v1",
            "ollama",  # pragma: allowlist secret
            lambda: (
                lambda req: (_ for _ in ()).throw(
                    httpx.ConnectError("Connection refused to ollama:11434")
                )
            ),
            "APIConnectionError",
        ),
        (
            "unknown_model",
            "http://ollama:11434/v1",
            "ollama",  # pragma: allowlist secret
            lambda: (
                lambda req: httpx.Response(
                    404,
                    json={
                        "error": {
                            "message": "model 'qwen3:0.6b' not found, try pulling it first",
                            "type": "invalid_request_error",
                        }
                    },
                )
            ),
            "NotFoundError",
        ),
        (
            "invalid_endpoint",
            "http://",
            "ollama",  # pragma: allowlist secret
            lambda: (lambda req: (_ for _ in ()).throw(httpx.InvalidURL("Invalid URL 'http://'"))),
            "APIConnectionError",
        ),
    ],
)
def test_unreachable_or_rejected_judge_stops_before_first_answer(
    tmp_path,
    monkeypatch,
    case,
    base_url,
    api_key,
    handler_factory,
    expected_fragment,
):
    """2. Unreachable, invalid, 401-rejected, or 404-unknown judge fails before generation."""
    _clear_judge_env(monkeypatch)
    prompts = _stub_generator(monkeypatch)
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")
    monkeypatch.setenv("RAG_EVAL_JUDGE_API_KEY", api_key)
    if base_url is not None:
        monkeypatch.setenv("RAG_EVAL_JUDGE_BASE_URL", base_url)
    _install_offline_openai(monkeypatch, handler_factory())

    ingested = []

    async def _spy_ingest(*a, **k):
        ingested.append(a)
        raise AssertionError("corpus must not be ingested when judge validation fails")

    monkeypatch.setattr(exp, "_ensure_corpus_ingested", _spy_ingest)

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(exp.Experiment.from_dict(_generation_sweep(tmp_path)).run())

    msg = str(exc_info.value)
    formatted_tb = "".join(traceback.format_exception(exc_info.value))
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True
    assert "qwen3:0.6b" in msg, case
    assert expected_fragment in msg, (case, msg)
    assert "RAG_EVAL_JUDGE_BASE_URL" in msg
    assert "RAG_EVAL_JUDGE_API_KEY" in msg
    if base_url:
        assert base_url in msg
    if len(api_key) >= 8:
        assert api_key not in msg, "API key value must never be exposed in the error message"
        assert api_key not in formatted_tb, (
            "API key value must never be exposed via chained traceback causes"
        )
        assert "<redacted>" in msg
        assert "Incorrect API key provided: <redacted>" in msg
    assert prompts == [] and ingested == []
    assert exp._active_judge_model is None


def test_judge_client_construction_failure_stops_before_first_answer(tmp_path, monkeypatch):
    """2b. If judge client construction fails or returns None, the run stops upfront."""
    _clear_judge_env(monkeypatch)
    prompts = _stub_generator(monkeypatch)
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")
    monkeypatch.setenv("RAG_EVAL_JUDGE_API_KEY", "ollama")  # pragma: allowlist secret

    def _broken_openai(**kw):
        raise ValueError("Malformed client configuration")

    monkeypatch.setattr(openai, "OpenAI", _broken_openai)

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(exp.Experiment.from_dict(_generation_sweep(tmp_path)).run())

    msg = str(exc_info.value)
    assert "qwen3:0.6b" in msg
    assert "[ValueError] Malformed client configuration" in msg
    assert "RAG_EVAL_JUDGE_BASE_URL" in msg
    assert prompts == []


def test_default_judge_model_fails_early_without_credentials(tmp_path, monkeypatch):
    """3. With no judge named and no OPENAI_API_KEY, the default judge fails upfront."""
    _clear_judge_env(monkeypatch)
    prompts = _stub_generator(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")  # pragma: allowlist secret

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(exp.Experiment.from_dict(_generation_sweep(tmp_path)).run())

    msg = str(exc_info.value)
    assert exp.DEFAULT_JUDGE_MODEL in msg
    assert "OPENAI_API_KEY" in msg
    assert "RAG_EVAL_JUDGE_BASE_URL" in msg
    assert "RAG_EVAL_JUDGE_API_KEY" in msg
    assert "OpenAIError" in msg
    assert prompts == []
    assert exp._active_judge_model is None


def test_valid_judge_is_checked_once_per_experiment_and_excluded_from_task_usage(
    tmp_path, monkeypatch
):
    """4 & 5. Multi-task experiment validates the judge once and does not count it in task usage."""
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")
    monkeypatch.setenv("RAG_EVAL_JUDGE_BASE_URL", "http://ollama:11434/v1")
    monkeypatch.setenv("RAG_EVAL_JUDGE_API_KEY", "ollama")  # pragma: allowlist secret
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")
    monkeypatch.delenv("LASAGNA_LLM_CACHE", raising=False)

    judge_requests: list = []
    created_kwargs = _install_offline_openai(monkeypatch, _ok_judge_handler(judge_requests))
    monkeypatch.setattr(ragas_mod, "RAGASEvaluator", _StubRAGAS)

    # Use a real instrumented SimpleLLMProvider stub so system tokens and cache
    # misses are tracked accurately across 4 tasks (2 generators x 2 runs).
    from tests.test_llm_cache_is_invisible_on_a_cold_cache import _StubChatModel
    import lasagna.providers as providers

    gen_log: list[str] = []
    provider = SimpleLLMProvider("openai")
    monkeypatch.setattr(provider, "_build_chat_model", lambda *a, **k: _StubChatModel(log=gen_log))
    monkeypatch.setattr(providers, "get_llm_provider", lambda: provider)

    sweep = _generation_sweep(
        tmp_path,
        n_runs=2,
        pipeline={"generators": ["concise", "cited"]},
        llm_cache={"path": str(tmp_path / "llm.sqlite"), "deterministic_only": False},
    )
    results = asyncio.run(exp.Experiment.from_dict(sweep).run())

    # 4 tasks x 2 samples = 8 answers generated; none of them have citations so
    # citation scoring makes 0 judge calls, and _StubRAGAS makes 0 judge calls.
    # Therefore the ONLY request on the judge wire is the 1 pre-flight check.
    assert len(results._results) == 4
    assert len(gen_log) == 8
    assert len(judge_requests) == 1, (
        f"Expected exactly 1 pre-flight judge request across 4 tasks, got {len(judge_requests)}"
    )
    url, body = judge_requests[0]
    assert url == "http://ollama:11434/v1/chat/completions"
    assert body["model"] == "qwen3:0.6b"
    assert created_kwargs[0]["base_url"] == "http://ollama:11434/v1"
    assert created_kwargs[0]["api_key"] == "ollama"  # pragma: allowlist secret

    # Task-level eval usage and LLM cache must not include the pre-flight probe.
    for task_res in results._results:
        assert task_res.error is None, task_res.error
        assert task_res.eval_llm_calls == 0
        assert task_res.eval_input_tokens == 0
        assert task_res.eval_output_tokens == 0
        assert task_res.system_llm_calls == 2
        assert task_res.system_input_tokens > 0
    assert results.experiment_config["llm_cache"]["misses"] == 8
    assert results.experiment_config["llm_cache"]["hits"] == 0
    assert exp._active_judge_model is None


def test_retrieval_only_experiment_never_checks_the_judge(tmp_path, monkeypatch):
    """6. Retrieval-only experiments do not build or validate a judge."""
    _clear_judge_env(monkeypatch)

    def _never(*a, **k):
        raise AssertionError("Retrieval-only run must not check or build a judge")

    monkeypatch.setattr(exp, "_check_judge_llm", _never)
    monkeypatch.setattr(exp, "_build_judge_llm", _never)

    import lasagna.api.rag_backend as rag_backend_mod
    from lasagna.core.pipeline import KnowledgePipeline
    from lasagna.knowledge_evaluation_sdk.benchmarks.adapters import smoke

    async def _ingest(benchmark_name, samples, **kw):
        ids = {c: c for c in smoke._CORPUS}
        return {
            "session_id": "stub-session",
            "corpus_to_valid": ids,
            "valid_to_corpus": ids,
            "corpus_fingerprint": "stub",
            "corpus_provenance": "native",
        }

    class _RetrievalPipeline:
        def __init__(self):
            self._config, self._component_names, self._instances = {}, {}, {}

        def _apply_session_config(self, session_id):
            pass

        def query(self, question, k, session_id, config):
            doc_id = next(iter(smoke._CORPUS))
            chunk = SimpleNamespace(document_id=doc_id, content=smoke._CORPUS[doc_id], metadata={})
            return [SimpleNamespace(chunk=chunk, score=1.0)]

    monkeypatch.setattr(exp, "_ensure_corpus_ingested", _ingest)
    monkeypatch.setattr(
        KnowledgePipeline, "from_profile", classmethod(lambda c, p=None: _RetrievalPipeline())
    )
    backend = SimpleNamespace(
        embedding_manager=SimpleNamespace(model_name="stub-embedder", dimension=8),
        hybrid_searcher=SimpleNamespace(
            _legacy_native_serves=lambda s, c=None: True,
            _backend_supports_native_hybrid=lambda s, c=None: True,
        ),
    )
    monkeypatch.setattr(rag_backend_mod, "get_rag_backend", lambda *a, **k: backend)

    sweep = {
        "name": "retrieval-only",
        "benchmarks": [{"name": "smoke", "n_samples": 2}],
        "tracks": ["retrieval"],
        "mode": "live",
        "n_runs": 1,
        "output_dir": str(tmp_path / "out"),
        "pipeline": {"base_profile": "speed"},
    }
    results = asyncio.run(exp.Experiment.from_dict(sweep).run())
    assert len(results._results) == 1
    assert results._results[0].error is None
    assert results.experiment_config["judge_model"] is None


def test_retrieval_fallback_survives_when_generator_is_unconfigured(tmp_path, monkeypatch):
    """7. Both tracks with no generator LLM configured skips judge check so retrieval runs."""
    _clear_judge_env(monkeypatch)
    import lasagna.providers as providers

    def _no_provider():
        raise providers.LLMProviderNotConfigured("no key")

    monkeypatch.setattr(providers, "get_llm_provider", _no_provider)

    cfg = exp._parse_config(
        {
            "name": "both-tracks-no-llm",
            "benchmarks": [{"name": "smoke", "n_samples": 2}],
            "tracks": ["retrieval", "generation"],
            "mode": "live",
        }
    )
    assert exp._should_check_judge(cfg, None) is False


def test_offline_dataset_validation_precedes_judge_validation(tmp_path, monkeypatch):
    """8. Offline dataset problems raise before judge credential validation."""
    _clear_judge_env(monkeypatch)
    _stub_generator(monkeypatch)

    monkeypatch.setattr(
        exp,
        "_offline_dataset_problem",
        lambda names: ValueError("dataset cache is missing for smoke"),
    )

    def _never_judge(*a, **k):
        raise AssertionError("_check_judge_llm must not run before _offline_dataset_problem")

    monkeypatch.setattr(exp, "_check_judge_llm", _never_judge)

    with pytest.raises(ValueError, match="dataset cache is missing"):
        asyncio.run(exp.Experiment.from_dict(_generation_sweep(tmp_path)).run())
    assert exp._active_judge_model is None


def test_dry_run_no_probe_does_not_check_judge_while_probing_dry_run_does(tmp_path, monkeypatch):
    """9. --dry-run --no-probe makes no judge call; --dry-run (probing) validates the judge."""
    _clear_judge_env(monkeypatch)
    _stub_generator(monkeypatch)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    from lasagna.utils import tokenization

    monkeypatch.setattr(tokenization, "_get_tokenizer", lambda *a, **k: None)

    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump(_generation_sweep(tmp_path)))

    # --dry-run --no-probe must succeed without touching the judge.
    no_probe = CliRunner().invoke(
        cli.app, ["benchmark", "experiment", str(sweep_path), "--dry-run", "--no-probe"]
    )
    assert no_probe.exit_code == 0, no_probe.output
    assert "Probe skipped" in no_probe.output

    # --dry-run (with probe enabled) runs Experiment(probe).run(), which must
    # fail early on the missing judge credentials.
    probed = CliRunner().invoke(cli.app, ["benchmark", "experiment", str(sweep_path), "--dry-run"])
    assert probed.exit_code == 1
    probed_msg = probed.output or str(probed.exception)
    assert "RAG_EVAL_JUDGE_BASE_URL" in probed_msg
    assert "RAG_EVAL_JUDGE_API_KEY" in probed_msg


def test_judge_construction_failure_log_is_truthful(monkeypatch):
    """10. _build_judge_llm logs that judge metrics will fail, never 'RAGAS will use its default'."""
    _clear_judge_env(monkeypatch)
    monkeypatch.setattr(exp, "_active_judge_model", None)
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")

    logged: list[str] = []
    logger.enable("lasagna")
    sink = logger.add(lambda m: logged.append(m.record["message"]), level="WARNING")
    try:
        assert exp._build_judge_llm() is None
    finally:
        logger.remove(sink)

    combined = "\n".join(logged)
    assert "Judge LLM construction failed; judge metrics will fail" in combined
    assert "RAGAS will use its default" not in combined


def test_check_judge_llm_preserves_type_error_from_client_construction(monkeypatch):
    """A genuine TypeError during judge construction is surfaced (and redacted), not swallowed."""
    _clear_judge_env(monkeypatch)
    secret = "sk-typeerror-secret-987654"  # pragma: allowlist secret
    monkeypatch.setenv("RAG_EVAL_JUDGE_API_KEY", secret)
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:0.6b")

    def _bad_build(model_name=None, *, _require=False):
        assert _require is True
        raise TypeError(f"unexpected argument 'timeout' for client with key {secret}")

    monkeypatch.setattr(exp, "_build_judge_llm", _bad_build)
    with pytest.raises(RuntimeError) as exc_info:
        exp._check_judge_llm("qwen3:0.6b")

    msg = str(exc_info.value)
    assert "[TypeError]" in msg
    assert "unexpected argument 'timeout'" in msg
    assert "returned None" not in msg
    assert secret not in msg
    assert exc_info.value.__cause__ is None
