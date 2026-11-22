"""A reasoning model as generator gets room to finish its answer, and an answer cut off is named.

Every loop and the harness's generator built their LLM with ``get_llm``, which
asked for at most 1000 output tokens, and no setting changed it. A reasoning
model (Qwen3, the DeepSeek-R1 distills, gpt-oss) reasons inside that budget,
so an answer that needed more came back empty, and the task failed with
"could not be generated" and no reason (#114). In ``generation_only`` mode an
empty answer was not even counted as failed. And to an OpenAI-compatible
server the limit went as ``max_completion_tokens``, which Ollama's ``/v1``
ignores, so an answer there had no limit at all.

These run the real provider (``SimpleLLMProvider``, ``ChatOpenAI``) against an
OpenAI client on a mock HTTP transport, so the request the generator sends and
the reply it gets back are the ones a server would see and send.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

ragas_mod = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")

SAMPLES = [
    BenchmarkSample(
        query_id="q1",
        question="Where was Marie Curie born?",
        ground_truth="Warsaw",
        context="Marie Curie was born in Warsaw, Poland, in 1867.",
    ),
    BenchmarkSample(
        query_id="q2",
        question="Who wrote Hamlet?",
        ground_truth="William Shakespeare",
        context="Hamlet is a tragedy written by William Shakespeare.",
    ),
]


class _RAGAS:
    """Stands in for the judge, which is not what these tests are about."""

    def __init__(self, llm=None):
        pass

    def evaluate_detailed(self, questions, answers, contexts, ground_truths=None, metrics=None):
        rows = [{m: 1.0 for m in metrics} for _ in questions]
        return ragas_mod.DetailedEvalResult(aggregate={m: 1.0 for m in metrics}, per_sample=rows)


def _reset():
    from lasagna.core.config import reset_rag_config
    from lasagna.providers import reset_providers

    reset_rag_config()
    reset_providers()


@pytest.fixture
def generator_server(monkeypatch):
    """A generator endpoint that ends every reply at the limit with no text, and what it got."""
    requests: list[dict] = []
    reply = {"finish_reason": "length", "content": ""}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        limit = body.get("max_tokens") or body.get("max_completion_tokens") or 0
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-gen",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "qwen3:8b"),
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": reply["finish_reason"],
                        "message": {"role": "assistant", "content": reply["content"]},
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": limit,
                    "total_tokens": 50 + limit,
                },
            },
        )

    transport = httpx.MockTransport(handler)

    class _OfflineOpenAI(openai.OpenAI):
        def __init__(self, **kw):
            kw.pop("http_client", None)
            super().__init__(**kw, max_retries=0, http_client=httpx.Client(transport=transport))

    class _OfflineAsyncOpenAI(openai.AsyncOpenAI):
        def __init__(self, **kw):
            kw.pop("http_client", None)
            super().__init__(
                **kw, max_retries=0, http_client=httpx.AsyncClient(transport=transport)
            )

    monkeypatch.setattr(openai, "OpenAI", _OfflineOpenAI)
    monkeypatch.setattr(openai, "AsyncOpenAI", _OfflineAsyncOpenAI)
    for var in ("RAG_LLM_MAX_TOKENS", "LASAGNA_LLM_MAX_TOKENS", "RAG_EVAL_GENERATOR_MODEL"):
        monkeypatch.delenv(var, raising=False)
    for var in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "OLLAMA_BASE_URL", "VLLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock")  # pragma: allowlist secret
    monkeypatch.setenv("OPENAI_BASE_URL", "http://generator.invalid/v1")
    monkeypatch.setenv("MODEL_NAME", "qwen3:8b")
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")
    monkeypatch.setenv("RAG_EVAL_JUDGE_METRICS", "faithfulness")
    monkeypatch.setenv("LASAGNA_LLM_CACHE", "off")
    monkeypatch.setattr(ragas_mod, "RAGASEvaluator", _RAGAS)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda *a, **k: None)
    _reset()
    yield SimpleNamespace(requests=requests, reply=reply)
    _reset()


def _limits(requests: list[dict]) -> set:
    return {r.get("max_tokens", r.get("max_completion_tokens")) for r in requests}


def _ask_the_generator() -> None:
    from lasagna.providers import get_llm_provider

    get_llm_provider().get_llm().invoke("Where was Marie Curie born?")


def _answer_offline():
    task = BenchmarkTask(
        benchmark_name="b", config_name="c", run_index=0, mode="offline", config_overrides={}
    )
    return asyncio.run(exp._eval_offline(task, SAMPLES))


def test_the_generator_asks_for_the_limit_the_setting_sets(generator_server, monkeypatch):
    monkeypatch.setenv("RAG_LLM_MAX_TOKENS", "8192")
    _reset()
    _ask_the_generator()
    assert generator_server.requests, "the generator sent no request"
    assert _limits(generator_server.requests) == {8192}


def test_without_the_setting_the_generator_sends_what_it_sent(generator_server):
    """Nothing changes for a generator that never needed more: same requests, same cache keys."""
    _ask_the_generator()
    assert _limits(generator_server.requests) == {1000}


def test_a_caller_that_sets_its_own_limit_keeps_it(generator_server, monkeypatch):
    """The LLM chunkers pass their own budgets."""
    from lasagna.providers import get_llm_provider

    monkeypatch.setenv("RAG_LLM_MAX_TOKENS", "8192")
    _reset()
    get_llm_provider().get_llm(max_tokens=128).invoke("Expand this query.")
    assert _limits(generator_server.requests) == {128}


@pytest.mark.parametrize("platform", ["vllm", "ollama"])
def test_an_openai_compatible_server_gets_the_limit_as_max_tokens(
    generator_server, monkeypatch, platform
):
    """Ollama's ``/v1`` reads ``max_tokens`` and ignores ``max_completion_tokens``;
    exactly one is sent, as OpenAI's own servers refuse the two together."""
    import lasagna.providers.llm_provider as llm_provider
    from lasagna.providers.llm_provider import SimpleLLMProvider

    monkeypatch.setenv("RAG_LLM_MAX_TOKENS", "8192")
    monkeypatch.setenv(f"{platform.upper()}_BASE_URL", "http://generator.invalid/v1")
    _reset()
    if platform == "ollama":
        # Without the `local` extra (as in the eval image), Ollama goes through
        # the OpenAI-compatible path.
        monkeypatch.setattr(
            llm_provider.SimpleLLMProvider,
            "_create_ollama",
            staticmethod(
                lambda name, settings, kw: SimpleLLMProvider._create_openai_compatible(
                    "ollama", name, settings, kw
                )
            ),
        )
    SimpleLLMProvider(default_provider=platform).get_llm(model_name="qwen3:8b").invoke("hi")
    (body,) = generator_server.requests
    assert body["max_tokens"] == 8192
    assert "max_completion_tokens" not in body


@pytest.mark.parametrize(
    "base_url", [None, "https://api.openai.com/v1", "http://generator.invalid/v1"]
)
def test_the_openai_platform_gets_the_request_it_got(generator_server, monkeypatch, base_url):
    """OpenAI, Azure and their gateways refuse ``max_tokens`` beside ``max_completion_tokens``."""
    if base_url is None:
        monkeypatch.delenv("OPENAI_BASE_URL")
    else:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    _reset()
    _ask_the_generator()
    (body,) = generator_server.requests
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 1000


def test_an_answer_cut_off_by_the_limit_is_named_in_the_tasks_error(generator_server):
    error = _answer_offline().error or ""
    assert "2 of 2 answers could not be generated" in error
    assert "2 LLM replies came back empty at their output-token limit" in error
    assert "the generator's and the loops' limit is 1000: raise RAG_LLM_MAX_TOKENS" in error


def test_an_empty_answer_that_was_not_cut_off_fails_without_the_cut_off_note(generator_server):
    """A reply that ended on its own and said nothing still fails; it is not said to be cut off."""
    generator_server.reply["finish_reason"] = "stop"
    error = _answer_offline().error or ""
    assert "2 of 2 answers could not be generated" in error
    assert "cut off" not in error


def test_the_limit_is_recorded_and_in_the_resume_fingerprint(generator_server, monkeypatch):
    from lasagna.knowledge_evaluation_sdk.run_provenance import rag_overrides, resume_fingerprint

    assert exp._sampling_record()["generator"]["max_tokens"] == 1000
    before = resume_fingerprint("judge")

    monkeypatch.setenv("RAG_LLM_MAX_TOKENS", "8192")
    _reset()
    assert exp._sampling_record()["generator"]["max_tokens"] == 8192
    assert rag_overrides()["RAG_LLM_MAX_TOKENS"] == "8192"
    assert resume_fingerprint("judge") != before


def test_an_arm_cannot_set_the_limit_every_arm_shares():
    """``get_llm`` reads the process's setting, so an arm's would be ignored silently."""
    from lasagna.knowledge_evaluation_sdk.experiment import Experiment

    sweep = Experiment.from_dict(
        {
            "name": "limits",
            "benchmarks": [{"name": "smoke"}],
            "pipeline": {
                "base_profile": "speed",
                "config_overrides": [{"name": "long", "llm_max_tokens": 8192}],
            },
        }
    )
    with pytest.raises(ValueError, match="set RAG_LLM_MAX_TOKENS in the environment"):
        sweep.generate_tasks()


@pytest.mark.parametrize("value", ["0", "lots"])
def test_a_limit_that_is_not_a_positive_number_is_refused(value, monkeypatch):
    from lasagna.core.config import RAGConfig

    monkeypatch.setenv("RAG_LLM_MAX_TOKENS", value)
    with pytest.raises(ValueError, match="LLM_MAX_TOKENS"):
        RAGConfig()


@pytest.mark.parametrize(
    ("generation_info", "response_metadata", "text", "counted"),
    [
        ({"finish_reason": "length"}, {}, "", 1),  # OpenAI, vLLM
        ({}, {"done_reason": "length"}, "", 1),  # Ollama's own API
        ({}, {"stop_reason": "max_tokens"}, "", 1),  # Anthropic
        ({}, {"finish_reason": "MAX_TOKENS"}, "", 1),  # Google, a thinking model
        ({"finish_reason": "stop"}, {"done_reason": "stop"}, "", 0),
        # Cut off mid-text, as a query2doc passage at its 128: not an empty answer.
        ({"finish_reason": "length"}, {}, "The Kamchatka Peninsula lies in", 0),
    ],
)
def test_a_reply_that_came_back_empty_at_its_limit_is_counted(
    generation_info, response_metadata, text, counted
):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    from lasagna.utils.token_usage import SYSTEM, UsageMeter, usage_callback_handler

    message = AIMessage(content=text, response_metadata=response_metadata)
    result = LLMResult(
        generations=[[ChatGeneration(message=message, generation_info=generation_info)]]
    )
    meter = UsageMeter()
    with meter, meter.scope(SYSTEM):
        usage_callback_handler().on_llm_end(result)
    assert meter.usage(SYSTEM).cut_off == counted
