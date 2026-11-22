"""A reasoning model as judge gets room to finish its judgement, and a judgement cut off is named.

Qwen3, the DeepSeek-R1 distills and gpt-oss reason before they answer, and the
reasoning counts against the reply's token limit. RAGAS builds its judge with
``max_tokens=1024`` and nothing raised it, so a judgement that needed more was
cut off (``finish_reason: length``), instructor raised
``IncompleteOutputException``, RAGAS recorded NaN, and the task failed with
"judge produced no usable scores", which did not say why. On vLLM with
Qwen3-8B as judge, 17 judgements of one small run were lost this way.

These run the real judge (``_build_judge_llm``, RAGAS, instructor) against an
OpenAI client on a mock HTTP transport, so the request the judge sends and the
reply it gets back are the ones a server would see and send.
"""

from __future__ import annotations

import asyncio
import json
import re
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


class _Generator:
    def invoke(self, prompt):
        question = re.search(r"Question: (.*)\n", prompt).group(1)
        return SimpleNamespace(content=f"The answer to '{question}' is in the passage.")


def _cut_off(body: dict, finish_reason: str = "length") -> httpx.Response:
    """What a reasoning model sends when its reasoning used up ``max_tokens``.

    With ``finish_reason="stop"``, a reply that ended on its own and still
    holds no usable JSON: a judge that failed for another reason.
    """
    limit = body.get("max_tokens") or body.get("max_completion_tokens")
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-cut",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "qwen3"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": '{"statements": ["Marie'},
                }
            ],
            "usage": {"prompt_tokens": 200, "completion_tokens": limit or 0},
        },
    )


@pytest.fixture
def judge_server(monkeypatch):
    """A judge endpoint that cuts every reply off at the limit, and the requests it got."""
    requests: list[dict] = []
    finish = {"reason": "length"}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return _cut_off(body, finish["reason"])

    class _OfflineOpenAI(openai.OpenAI):
        def __init__(self, **kw):
            super().__init__(
                **{"api_key": "sk-mock", **kw},  # pragma: allowlist secret
                max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            )

    monkeypatch.setattr(openai, "OpenAI", _OfflineOpenAI)
    for var in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "RAG_EVAL_JUDGE_MAX_TOKENS",
        "RAG_EVAL_JUDGE_EXTRA_BODY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:8b")
    monkeypatch.setenv("RAG_EVAL_JUDGE_BASE_URL", "http://judge.invalid/v1")
    monkeypatch.setenv("RAG_EVAL_JUDGE_API_KEY", "none")  # pragma: allowlist secret
    monkeypatch.setenv("RAG_EVAL_JUDGE_METRICS", "faithfulness")
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")
    monkeypatch.setenv("LASAGNA_LLM_CACHE", "off")
    import lasagna.providers as providers

    monkeypatch.setattr(
        providers, "get_llm_provider", lambda: SimpleNamespace(get_llm=lambda **k: _Generator())
    )
    return SimpleNamespace(requests=requests, finish=finish)


def _judge_offline_answers() -> object:
    task = BenchmarkTask(
        benchmark_name="b", config_name="c", run_index=0, mode="offline", config_overrides={}
    )
    return asyncio.run(exp._eval_offline(task, SAMPLES))


def _limits(requests: list[dict]) -> set:
    return {r.get("max_tokens", r.get("max_completion_tokens")) for r in requests}


def test_the_judge_asks_for_the_token_limit_the_environment_sets(judge_server, monkeypatch):
    monkeypatch.setenv("RAG_EVAL_JUDGE_MAX_TOKENS", "8192")
    _judge_offline_answers()
    assert judge_server.requests, "the judge sent no request"
    assert _limits(judge_server.requests) == {8192}


def test_without_the_setting_the_judge_sends_what_ragas_sends(judge_server):
    """Nothing changes for a judge that never needed more: same requests, same cache keys."""
    from ragas.llms.base import InstructorModelArgs

    _judge_offline_answers()
    assert _limits(judge_server.requests) == {InstructorModelArgs().max_tokens}
    assert not any("chat_template_kwargs" in r for r in judge_server.requests)
    # The limit the run record and the error state is the one RAGAS applies.
    assert ragas_mod.RAGAS_JUDGE_MAX_TOKENS == InstructorModelArgs().max_tokens


def test_a_judgement_cut_off_by_the_token_limit_is_named_in_the_tasks_error(judge_server):
    result = _judge_offline_answers()
    error = result.error or ""
    assert "judge produced no usable scores for faithfulness" in error
    assert "cut off" in error
    assert "1024" in error
    assert "RAG_EVAL_JUDGE_MAX_TOKENS" in error


def test_a_judge_that_fails_for_another_reason_is_not_said_to_be_cut_off(judge_server):
    judge_server.finish["reason"] = "stop"
    error = _judge_offline_answers().error or ""
    assert "judge produced no usable scores for faithfulness" in error
    assert "cut off" not in error


def test_the_check_before_answers_names_a_judge_cut_off_not_its_credentials(judge_server):
    """The probe a run sends its judge first: a reasoning judge can spend the limit on it too."""
    with pytest.raises(RuntimeError) as raised:
        exp._check_judge_llm("qwen3:8b")
    message = str(raised.value)
    assert "1 judge reply was cut off at the judge's limit of 1024" in message
    assert "RAG_EVAL_JUDGE_MAX_TOKENS" in message
    assert "RAG_EVAL_JUDGE_BASE_URL" not in message


def test_the_judge_sends_the_extra_body_the_environment_sets(judge_server, monkeypatch):
    """How a server turns a model's reasoning off: vLLM and SGLang read Qwen3's
    ``enable_thinking`` from ``chat_template_kwargs``. With reasoning on, Qwen3-8B
    on vLLM 0.24 answered the judge's JSON-object request with ``{}`` (finish
    ``stop``); with it off, with the object asked for."""
    monkeypatch.setenv(
        "RAG_EVAL_JUDGE_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}'
    )
    _judge_offline_answers()
    assert judge_server.requests, "the judge sent no request"
    assert all(
        r.get("chat_template_kwargs") == {"enable_thinking": False} for r in judge_server.requests
    )


@pytest.mark.parametrize("value", ["enable_thinking=false", "[1]", '"off"'])
def test_an_extra_body_that_is_not_a_json_object_is_refused(judge_server, monkeypatch, value):
    monkeypatch.setenv("RAG_EVAL_JUDGE_EXTRA_BODY", value)
    with pytest.raises(ValueError, match="RAG_EVAL_JUDGE_EXTRA_BODY"):
        exp._build_judge_llm(_require=True)


@pytest.mark.parametrize(
    "field", ["max_tokens", "max_completion_tokens", "temperature", "response_format", "model"]
)
def test_an_extra_body_cannot_replace_what_the_judge_sets(judge_server, monkeypatch, field):
    """Its fields would replace the judge's own, and the run record would be false."""
    monkeypatch.setenv("RAG_EVAL_JUDGE_EXTRA_BODY", json.dumps({field: 1}))
    with pytest.raises(ValueError, match=field):
        exp._build_judge_llm(_require=True)


def test_the_check_before_answers_names_a_judge_that_answered_without_the_json(judge_server):
    """A reply that ended on its own and holds no usable JSON is no credentials problem."""
    judge_server.finish["reason"] = "stop"
    with pytest.raises(RuntimeError) as raised:
        exp._check_judge_llm("qwen3:8b")
    message = str(raised.value)
    assert "answered, but not with the JSON it was asked for" in message
    assert "RAG_EVAL_JUDGE_EXTRA_BODY" in message
    assert "RAG_EVAL_JUDGE_BASE_URL" not in message


def test_a_limit_that_is_not_a_positive_number_is_refused(judge_server, monkeypatch):
    monkeypatch.setenv("RAG_EVAL_JUDGE_MAX_TOKENS", "lots")
    with pytest.raises(ValueError, match="RAG_EVAL_JUDGE_MAX_TOKENS"):
        exp._build_judge_llm(_require=True)


def test_the_judge_settings_are_recorded_and_in_the_resume_fingerprint(monkeypatch):
    from lasagna.knowledge_evaluation_sdk.run_provenance import rag_overrides, resume_fingerprint

    monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "qwen3:8b")
    monkeypatch.delenv("RAG_EVAL_JUDGE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("RAG_EVAL_JUDGE_EXTRA_BODY", raising=False)
    assert exp._sampling_record()["judge"]["max_tokens"] == 1024
    before = resume_fingerprint("qwen3:8b")

    monkeypatch.setenv("RAG_EVAL_JUDGE_MAX_TOKENS", "8192")
    assert exp._sampling_record()["judge"]["max_tokens"] == 8192
    assert rag_overrides()["RAG_EVAL_JUDGE_MAX_TOKENS"] == "8192"
    assert resume_fingerprint("qwen3:8b") != before

    with_limit = resume_fingerprint("qwen3:8b")
    body = '{"chat_template_kwargs": {"enable_thinking": false}}'
    monkeypatch.setenv("RAG_EVAL_JUDGE_EXTRA_BODY", body)
    assert rag_overrides()["RAG_EVAL_JUDGE_EXTRA_BODY"] == json.loads(body)
    assert resume_fingerprint("qwen3:8b") != with_limit


def test_a_stub_judge_has_no_cut_offs():
    """A ``Mock`` judge makes up any attribute it is asked for; it counts none."""
    from unittest.mock import MagicMock

    for judge in (MagicMock(), None, object()):
        assert ragas_mod.cut_off_replies(judge) == 0
        assert ragas_mod.cut_off_note(judge) == ""
