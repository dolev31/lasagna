"""Requests are built for the model they go to.

Every chat model was built with ``temperature=0``. Claude 4.7 and later answer
that with HTTP 400 (the Anthropic SDK 1.0 raises ``TypeError`` before the
request leaves), and OpenAI's gpt-6-luna, the replacement OpenAI names for
gpt-4o and gpt-4.1, rejects temperature at its default reasoning effort. So
with a current model every answer, HyDE call and judge call failed. The cache
also counted any temperature-0 call as deterministic, including ones whose
temperature the client had silently dropped.

No request leaves the machine: Claude models are built against a stand-in
``langchain_anthropic``, OpenAI models are real ``ChatOpenAI`` objects whose
request payload is read, and the judge talks to an ``httpx.MockTransport``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from types import SimpleNamespace

import pytest

from lasagna.providers.llm_provider import SimpleLLMProvider

#: Every Claude id at 4.7 or later in the status table on
#: https://platform.claude.com/docs/en/about-claude/model-deprecations
#: (2026-10-15), all of which reject a non-default temperature.
CLAUDE_REJECTING = [
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5-5",
    "claude-sonnet-5",
    "claude-haiku-5-5",
]

#: The active ids before 4.7 in the same table, plus their aliases: these
#: still take temperature 0.
CLAUDE_ACCEPTING = [
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5",
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
]

SAMPLING = {"temperature", "top_p", "top_k"}


class _ChatAnthropic:
    """Stands in for langchain_anthropic.ChatAnthropic, which CI does not install."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def anthropic(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "langchain_anthropic", types.SimpleNamespace(ChatAnthropic=_ChatAnthropic)
    )
    return SimpleLLMProvider("anthropic")


@pytest.fixture
def openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-no-request-is-made")  # pragma: allowlist secret
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)


def _payload(model) -> dict:
    """The body ChatOpenAI would send for one user message."""
    return model._get_request_payload("hello")


# -- Anthropic ----------------------------------------------------------------


@pytest.mark.parametrize("model_id", CLAUDE_REJECTING)
def test_claude_4_7_and_later_get_no_sampling_parameters(anthropic, model_id):
    model = anthropic.get_chat_model({"model_name": model_id}, temperature=0.0, max_tokens=50)
    assert not SAMPLING & set(model.kwargs), (
        f"{model_id} was built with {sorted(SAMPLING & set(model.kwargs))}: a 400 from the API"
    )
    assert model.kwargs["max_tokens"] == 50
    assert model.cache.deterministic is False, "an unpinned model's answers were cached as fixed"


@pytest.mark.parametrize("model_id", CLAUDE_ACCEPTING)
def test_older_claude_models_still_decode_at_temperature_0(anthropic, model_id):
    model = anthropic.get_chat_model({"model_name": model_id}, temperature=0.0)
    assert model.kwargs.get("temperature") == 0.0
    assert model.cache.deterministic is True


def test_the_anthropic_default_is_a_current_model(anthropic, monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    model = anthropic.get_llm()
    assert model.kwargs["model"] == "claude-sonnet-5-5"
    assert "temperature" not in model.kwargs


@pytest.mark.parametrize("model_id", ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"])
def test_claude_through_openrouter_gets_no_temperature(monkeypatch, model_id):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")  # pragma: allowlist secret
    model = SimpleLLMProvider("openrouter").get_chat_model({"model_name": model_id})
    assert "temperature" not in _payload(model)


# -- OpenAI -------------------------------------------------------------------


#: OpenAI reasoning models whose page lists reasoning effort "none"
#: (https://developers.openai.com/api/docs/models/<id>, 2026-10-15), and dated
#: snapshots of two of them.
OPENAI_EFFORT_NONE = [
    "gpt-6-luna",
    "gpt-6-sol",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.5",
    "gpt-5.5-2026-04-23",
    "gpt-5.4-nano",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5.1-2025-11-13",
]

#: Reasoning models whose page lists no "none" effort.
OPENAI_NO_EFFORT_NONE = ["gpt-6-astra", "gpt-6.1-sol", "gpt-5", "gpt-5-2025-08-07"]


@pytest.mark.parametrize("model_id", OPENAI_EFFORT_NONE)
def test_openai_reasoning_models_are_pinned_with_effort_none(openai_key, model_id):
    model = SimpleLLMProvider("openai").get_chat_model({"model_name": model_id}, temperature=0.0)
    body = _payload(model)
    assert body.get("temperature") == 0.0
    effort = body.get("reasoning_effort") or (body.get("reasoning") or {}).get("effort")
    assert effort == "none", f"{model_id} at temperature 0 without effort none is a 400"
    assert model.cache.deterministic is True


def test_the_openai_default_is_gpt_6_luna_and_pinned(openai_key):
    body = _payload(SimpleLLMProvider("openai").get_llm())
    assert body["model"] == "gpt-6-luna"
    assert (body["temperature"], body["reasoning_effort"]) == (0.0, "none")


def test_an_effort_the_user_set_wins_and_unpins(openai_key):
    model = SimpleLLMProvider("openai").get_chat_model(
        {"model_name": "gpt-6-luna", "reasoning_effort": "high"}, temperature=0.0
    )
    body = _payload(model)
    assert body["reasoning_effort"] == "high"
    assert "temperature" not in body
    assert model.cache.deterministic is False


@pytest.mark.parametrize("model_id", OPENAI_NO_EFFORT_NONE)
def test_models_with_no_effort_none_get_no_temperature(openai_key, model_id):
    model = SimpleLLMProvider("openai").get_chat_model({"model_name": model_id})
    assert "temperature" not in _payload(model)
    assert model.cache.deterministic is False


@pytest.mark.parametrize(
    "platform,model_id",
    [("openai", "gpt-4.1-nano"), ("openai", "my-gateway-model"), ("vllm", "default")],
)
def test_unknown_and_older_models_are_built_as_before(openai_key, platform, model_id):
    model = SimpleLLMProvider(platform).get_chat_model({"model_name": model_id}, temperature=0.0)
    body = _payload(model)
    assert body.get("temperature") == 0.0
    assert "reasoning_effort" not in body
    assert model.cache.deterministic is True


def test_a_temperature_the_client_drops_is_not_counted_as_deterministic(openai_key):
    """langchain-openai drops temperature for gpt-5-mini without a word."""
    model = SimpleLLMProvider("openai").get_chat_model({"model_name": "gpt-5-mini"})
    assert "temperature" not in _payload(model)
    assert model.cache.deterministic is False, "a sampled answer would be replayed as fixed"


# -- the judge ----------------------------------------------------------------


def _judge_wire(monkeypatch, model_id):
    """What `_build_judge_llm(model_id)` sends, read off the HTTP request."""
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    pytest.importorskip("ragas")
    from pydantic import BaseModel

    from lasagna.knowledge_evaluation_sdk import experiment as exp

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": '{"ok": true}'},
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    class _OfflineOpenAI(openai.OpenAI):
        """A real client (instructor checks isinstance) on a mock transport."""

        def __init__(self, **kw):
            super().__init__(
                **{"api_key": "sk-test", **kw},  # pragma: allowlist secret
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            )

    monkeypatch.setattr(openai, "OpenAI", _OfflineOpenAI)
    monkeypatch.setattr(exp, "_active_judge_model", None)
    judge = exp._build_judge_llm(model_id)
    assert judge is not None

    class Verdict(BaseModel):
        ok: bool

    assert judge.generate("Is the answer right?", Verdict).ok is True
    (body,) = bodies
    return body


def test_the_judge_asks_gpt_6_luna_for_effort_none(monkeypatch):
    body = _judge_wire(monkeypatch, "gpt-6-luna")
    assert body["reasoning_effort"] == "none"
    assert body["temperature"] == 0.01, "RAGAS's judge temperature, not the 1.0 it forces"


def test_a_claude_judge_behind_a_proxy_gets_no_sampling_parameters(monkeypatch):
    body = _judge_wire(monkeypatch, "claude-sonnet-5-5")
    assert not {"temperature", "top_p", "top_k"} & set(body)


def test_an_older_judge_is_asked_exactly_as_before(monkeypatch):
    body = _judge_wire(monkeypatch, "gpt-4.1-nano")
    assert (body["temperature"], body["top_p"]) == (0.01, 0.1)
    assert "reasoning_effort" not in body


def test_with_nothing_configured_the_judge_is_a_current_model(monkeypatch):
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    monkeypatch.delenv("RAG_EVAL_JUDGE_MODEL", raising=False)
    seen = {}
    import ragas.llms

    monkeypatch.setattr(
        ragas.llms, "llm_factory", lambda model, client: seen.setdefault("m", model)
    )
    monkeypatch.setattr(exp, "_active_judge_model", None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")  # pragma: allowlist secret
    exp._build_judge_llm()
    assert seen["m"] == "gpt-6-luna", "RAGAS would fall back to gpt-4o-mini, shut down 2027-01-14"


# -- what the user sees -------------------------------------------------------

NOTE = (
    "claude-sonnet-5-5 does not accept temperature; answers can vary between runs; "
    "repeat runs (n_runs >= 3) to measure that variance"
)


@pytest.fixture
def anthropic_only(monkeypatch, tmp_path):
    import lasagna.providers as providers

    for var, _ in providers._API_KEY_TO_PLATFORM:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")  # pragma: allowlist secret
    monkeypatch.setenv("MODEL_NAME", "claude-sonnet-5-5")
    monkeypatch.delenv("RAG_EVAL_GENERATOR_MODEL", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def test_doctor_says_once_that_the_model_cannot_be_pinned(anthropic_only):
    from lasagna.core.diagnostics import diagnose

    d = diagnose(check_gh=False)
    lines = [f"{c.detail} {c.advice}" for c in d.checks]
    assert sum("does not accept temperature" in line for line in lines) == 1, lines
    (check,) = [c for c in d.checks if c.name == "llm_sampling"]
    assert (check.status, check.detail) == ("warning", NOTE)


def test_doctor_is_quiet_about_a_pinned_model(anthropic_only, monkeypatch):
    from lasagna.core.diagnostics import diagnose

    monkeypatch.setenv("MODEL_NAME", "claude-haiku-4-5-20251001")
    (check,) = [c for c in diagnose(check_gh=False).checks if c.name == "llm_sampling"]
    assert check.status == "ok"


def test_an_experiment_says_it_once_and_records_it(anthropic_only, monkeypatch, tmp_path):
    from loguru import logger

    from lasagna.knowledge_evaluation_sdk import experiment as exp
    from lasagna.knowledge_evaluation_sdk import ragas_evaluator

    class _Answer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, prompt):
            return SimpleNamespace(content="an answer")

    class _Judge:
        def __init__(self, llm=None):
            pass

        def evaluate_detailed(self, questions, answers, contexts, ground_truths=None, **kw):
            rows = [{"faithfulness": 1.0} for _ in questions]
            return ragas_evaluator.DetailedEvalResult(
                aggregate={"faithfulness": 1.0}, per_sample=rows
            )

    import lasagna.providers as providers

    provider = SimpleLLMProvider("anthropic")
    monkeypatch.setattr(provider, "_build_chat_model", lambda *a, **k: _Answer())
    monkeypatch.setattr(providers, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(ragas_evaluator, "RAGASEvaluator", _Judge)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda *a, **k: None)
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")

    logged = []
    logger.enable("lasagna")
    sink = logger.add(lambda m: logged.append(m.record["message"]), level="WARNING")
    try:
        results = asyncio.run(
            exp.Experiment.from_dict(
                {
                    "name": "sampling",
                    "benchmarks": [{"name": "smoke", "n_samples": 2}],
                    "mode": "generation_only",
                    "n_runs": 2,
                    "output_dir": str(tmp_path / "out"),
                    "llm_cache": False,
                }
            ).run()
        )
    finally:
        logger.remove(sink)
        logger.disable("lasagna")

    assert [m for m in logged if "does not accept temperature" in m] == [NOTE]
    sampling = results.experiment_config["sampling"]
    assert sampling["generator"] == {"model": "claude-sonnet-5-5", "sampling": "provider-default"}
    assert sampling["judge"] == {"model": "gpt-6-luna", "sampling": "pinned"}


# -- comparability ------------------------------------------------------------


def test_dateless_current_ids_are_snapshots_and_old_aliases_still_float():
    from lasagna.knowledge_evaluation_sdk.bundle import _is_floating_alias

    for pinned in ("claude-sonnet-5-5", "claude-opus-4-6", "gpt-6-luna", "gpt-5.6-luna"):
        assert _is_floating_alias(pinned) is False, pinned
    for alias in ("claude-sonnet-4-5", "claude-haiku-4-5", "gpt-5.5", "gpt-4o-mini"):
        assert _is_floating_alias(alias) is True, alias
    assert _is_floating_alias("gpt-5.5-2026-04-23") is False


# -- prices -------------------------------------------------------------------


def test_the_price_table_has_the_current_models_and_not_the_retired_ones():
    from lasagna.knowledge_evaluation_sdk.cost import PriceTable

    table = PriceTable()
    # https://platform.claude.com/docs/en/about-claude/pricing and
    # https://developers.openai.com/api/docs/models/<id>, read 2026-10-15.
    expected = {
        "claude-fable-5-1": (10.00, 50.00),
        "claude-opus-5-5": (4.00, 20.00),
        "claude-sonnet-5-5": (2.00, 10.00),
        "claude-haiku-5-5": (0.10, 0.50),
        "gpt-6-luna": (0.10, 0.50),
        "gpt-6-sol": (2.00, 10.00),
    }
    for model, rates in expected.items():
        price = table.price(model)
        assert price is not None, f"no rate for {model}"
        assert (price.input_per_1m, price.output_per_1m) == rates, model
    assert table.price("openai/gpt-6-luna") == table.price("gpt-6-luna")
    assert table.price("claude-sonnet-4-20250514") is None, "a retired model is still priced"


def test_a_rate_that_only_holds_for_short_prompts_is_not_stretched():
    from lasagna.knowledge_evaluation_sdk.cost import PriceTable, estimate_cost
    from lasagna.knowledge_evaluation_sdk.runner import TaskResult

    def task(tokens, calls):
        return TaskResult(
            task_key="k",
            benchmark_name="b",
            config_name="c",
            run_index=0,
            system_input_tokens=tokens,
            system_output_tokens=10,
            system_llm_calls=calls,
        )

    short = estimate_cost(task(2_000_000, 100), PriceTable(), "claude-haiku-5-5", None)
    assert short == pytest.approx((2_000_000 * 0.10 + 10 * 0.50) / 1e6)
    # Two calls averaging 150,000 tokens: Haiku 5.5 bills those at $0.50 / $2.50.
    assert estimate_cost(task(300_000, 2), PriceTable(), "claude-haiku-5-5", None) is None


def test_a_retrieval_only_run_still_records_no_judge(monkeypatch):
    """judge_model is in the comparability key; a run that judges nothing keeps None."""
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    monkeypatch.delenv("RAG_EVAL_JUDGE_MODEL", raising=False)

    def cfg(tracks, judge=""):
        return exp.ExperimentConfig(name="j", tracks=tracks, judge_model=judge)

    assert exp._resolve_judge_model(cfg(["retrieval"])) is None
    assert exp._resolve_judge_model(cfg(["retrieval", "generation"])) == "gpt-6-luna"
    assert exp._resolve_judge_model(cfg(["generation"], "my-judge")) == "my-judge"


def test_a_judge_on_openrouter_sends_gpt_6_luna_no_temperature():
    """OpenRouter's catalog lists no temperature for OpenAI's reasoning models."""
    from lasagna.providers.model_capabilities import pin_openai_client

    sent = []
    completions = SimpleNamespace(create=lambda **kw: sent.append(kw))
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions), base_url="https://openrouter.ai/api/v1/"
    )
    pin_openai_client(client, 0.01).chat.completions.create(
        model="openai/gpt-6-luna", messages=[], temperature=1.0
    )
    assert "temperature" not in sent[0] and "reasoning_effort" not in sent[0]
