"""Tokens must be counted, and dollars must not be invented.

Every cost figure this package produced was zero. `input_tokens` and
`output_tokens` were summed out of `chunk.metadata`, a key nothing in the
codebase ever wrote, and the price table downstream faithfully multiplied those
zeros into `$0.00000`. `quality_per_dollar` divided by the result and reported
`inf`, which sorts to the top — so a config with no usage data was recommended
as the most efficient one. The `AIMessage` carrying the real numbers was
reduced to `.content` on the following line.

The counting is at the provider rather than at call sites, because a call site
that nobody remembers to instrument produces an undercount, and an undercount
flatters whichever technique spends the most. That is the comparison this
package exists to get right.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from lasagna.utils.token_usage import (
    EVAL,
    SYSTEM,
    TokenUsage,
    UsageMeter,
    extract_usage,
    instrument_openai_client,
    record_usage,
    usage_callback_handler,
)


class TestExtractionAcrossProviderShapes:
    """Providers disagree about where usage lives and what to call it.

    A shape this does not recognise costs a silent zero, not an error, so the
    breadth of these cases is the safety margin.
    """

    def test_langchain_usage_metadata(self):
        msg = SimpleNamespace(usage_metadata={"input_tokens": 12, "output_tokens": 3})
        assert extract_usage(msg) == TokenUsage(12, 3, 1)

    def test_openai_style_prompt_completion_names(self):
        msg = SimpleNamespace(usage_metadata={"prompt_tokens": 12, "completion_tokens": 3})
        assert extract_usage(msg) == TokenUsage(12, 3, 1)

    def test_response_metadata_token_usage(self):
        msg = SimpleNamespace(
            usage_metadata=None,
            response_metadata={"token_usage": {"prompt_tokens": 5, "completion_tokens": 7}},
        )
        assert extract_usage(msg) == TokenUsage(5, 7, 1)

    def test_additional_kwargs_usage(self):
        msg = SimpleNamespace(
            usage_metadata=None,
            response_metadata={},
            additional_kwargs={"usage": {"input_tokens": 4, "output_tokens": 6}},
        )
        assert extract_usage(msg) == TokenUsage(4, 6, 1)

    def test_llm_output_token_usage(self):
        result = SimpleNamespace(
            usage_metadata=None,
            llm_output={"token_usage": {"prompt_tokens": 9, "completion_tokens": 1}},
        )
        assert extract_usage(result) == TokenUsage(9, 1, 1)

    def test_openai_sdk_usage_object(self):
        """The OpenAI SDK returns an object, not a dict."""
        response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=20, completion_tokens=8))
        assert extract_usage(response) == TokenUsage(20, 8, 1)

    def test_a_response_with_no_usage_reads_as_unmeasured_not_free(self):
        usage = extract_usage(SimpleNamespace(content="hi"))
        assert usage.llm_calls == 0
        assert not usage.measured

    def test_a_partial_report_still_counts_as_a_call(self):
        """Some providers report input only. Half a number beats no number."""
        usage = extract_usage(SimpleNamespace(usage_metadata={"input_tokens": 11}))
        assert usage == TokenUsage(11, 0, 1)

    def test_string_counts_are_coerced(self):
        usage = extract_usage(SimpleNamespace(usage_metadata={"input_tokens": "11"}))
        assert usage.input_tokens == 11

    def test_garbage_does_not_raise(self):
        assert extract_usage(None).llm_calls == 0
        assert extract_usage("a string").llm_calls == 0
        assert extract_usage(SimpleNamespace(usage_metadata="not a dict")).llm_calls == 0


class TestTheMeter:
    def test_nothing_is_recorded_without_an_active_meter(self):
        """The normal case for a library user must cost nothing."""
        usage = record_usage(SimpleNamespace(usage_metadata={"input_tokens": 5}))
        assert usage.input_tokens == 5  # returned, but nowhere to accumulate

    def test_a_meter_collects_what_happens_inside_it(self):
        meter = UsageMeter()
        with meter, meter.scope(SYSTEM):
            record_usage(SimpleNamespace(usage_metadata={"input_tokens": 5, "output_tokens": 2}))
            record_usage(SimpleNamespace(usage_metadata={"input_tokens": 3, "output_tokens": 1}))
        assert meter.usage(SYSTEM) == TokenUsage(8, 3, 2)

    def test_scopes_do_not_bleed(self):
        meter = UsageMeter()
        with meter:
            with meter.scope(SYSTEM):
                record_usage(SimpleNamespace(usage_metadata={"input_tokens": 5}))
            with meter.scope(EVAL):
                record_usage(SimpleNamespace(usage_metadata={"input_tokens": 900}))
        assert meter.usage(SYSTEM).input_tokens == 5
        assert meter.usage(EVAL).input_tokens == 900

    def test_a_meter_stops_collecting_once_it_exits(self):
        meter = UsageMeter()
        with meter, meter.scope(SYSTEM):
            record_usage(SimpleNamespace(usage_metadata={"input_tokens": 5}))
        record_usage(SimpleNamespace(usage_metadata={"input_tokens": 1000}))
        assert meter.usage(SYSTEM).input_tokens == 5

    def test_an_unmeasured_response_does_not_inflate_the_call_count(self):
        meter = UsageMeter()
        with meter, meter.scope(SYSTEM):
            record_usage(SimpleNamespace(content="no usage here"))
        assert meter.usage(SYSTEM).llm_calls == 0

    def test_it_reaches_across_asyncio_to_thread(self):
        """The pipeline runs in a worker thread; contextvars must follow it.

        `pipeline.query` is dispatched with `asyncio.to_thread`, so a meter that
        did not propagate would count nothing for every live-mode run — the
        exact silent zero being fixed.
        """
        meter = UsageMeter()

        def _in_worker():
            record_usage(SimpleNamespace(usage_metadata={"input_tokens": 42}))

        async def _run():
            with meter, meter.scope(SYSTEM):
                await asyncio.to_thread(_in_worker)

        asyncio.run(_run())
        assert meter.usage(SYSTEM).input_tokens == 42

    def test_concurrent_recording_does_not_lose_counts(self):
        """Two RAGAS worker threads report into one meter."""
        meter = UsageMeter()
        barrier = threading.Barrier(8)

        def _worker():
            barrier.wait()
            for _ in range(50):
                meter.record(TokenUsage(1, 1, 1), SYSTEM)

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert meter.usage(SYSTEM).llm_calls == 400

    def test_total_spans_scopes(self):
        meter = UsageMeter()
        meter.record(TokenUsage(1, 1, 1), SYSTEM)
        meter.record(TokenUsage(2, 2, 1), EVAL)
        assert meter.total() == TokenUsage(3, 3, 2)

    def test_a_snapshot_does_not_alias_the_meter(self):
        meter = UsageMeter()
        meter.record(TokenUsage(1, 1, 1), SYSTEM)
        snap = meter.snapshot()
        meter.record(TokenUsage(9, 9, 1), SYSTEM)
        assert snap[SYSTEM] == TokenUsage(1, 1, 1)


class TestTheProviderIsTheChokePoint:
    """One attachment point, or the count is wrong the next time somebody
    adds an LLM call somewhere."""

    def test_every_chat_model_carries_the_handler(self, monkeypatch):
        from lasagna.providers.llm_provider import SimpleLLMProvider

        # Constructing a client validates that a key is present; it makes no
        # network call. Nothing here reaches an API.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key-for-construction-only")
        model = SimpleLLMProvider().get_chat_model(
            {"platform": "openai", "model_name": "gpt-4o-mini"}
        )
        names = [type(h).__name__ for h in (getattr(model, "callbacks", None) or [])]
        assert "_UsageCallbackHandler" in names

    def test_attaching_twice_does_not_double_count(self):
        from lasagna.utils.token_usage import attach_usage_meter

        model = SimpleNamespace(callbacks=None)
        attach_usage_meter(model)
        attach_usage_meter(model)
        assert len(model.callbacks) == 1

    def test_existing_callbacks_are_preserved(self):
        from lasagna.utils.token_usage import attach_usage_meter

        sentinel = object()
        model = SimpleNamespace(callbacks=[sentinel])
        attach_usage_meter(model)
        assert sentinel in model.callbacks

    def test_a_model_that_cannot_take_callbacks_is_not_fatal(self):
        """Less accounting is a worse run; a crash at construction is no run."""
        from lasagna.utils.token_usage import attach_usage_meter

        class _Frozen:
            __slots__ = ()

        frozen = _Frozen()
        assert attach_usage_meter(frozen) is frozen


class TestTheCallbackHandler:
    def test_it_records_an_llm_result(self):
        handler = usage_callback_handler()
        meter = UsageMeter()
        result = SimpleNamespace(
            usage_metadata=None,
            llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 2}},
        )
        with meter, meter.scope(SYSTEM):
            handler.on_llm_end(result)
        assert meter.usage(SYSTEM) == TokenUsage(7, 2, 1)

    def test_it_falls_back_to_the_generated_message(self):
        """Several providers leave `llm_output` empty and fill the message."""
        handler = usage_callback_handler()
        meter = UsageMeter()
        message = SimpleNamespace(usage_metadata={"input_tokens": 4, "output_tokens": 1})
        result = SimpleNamespace(
            usage_metadata=None,
            llm_output=None,
            generations=[[SimpleNamespace(message=message)]],
        )
        with meter, meter.scope(SYSTEM):
            handler.on_llm_end(result)
        assert meter.usage(SYSTEM) == TokenUsage(4, 1, 1)

    def test_it_is_silent_with_no_meter(self):
        usage_callback_handler().on_llm_end(SimpleNamespace(usage_metadata={"input_tokens": 1}))


class TestTheJudgeIsCountedToo:
    """RAGAS builds its judge on a raw OpenAI client, not through the provider.

    That is why `eval_cost_usd` had no producer at all: the entire cost of
    *measuring*, which frequently exceeds the cost of the pipeline, went past
    every instrumentation point in the package.
    """

    def _client(self, usage):
        created = {}

        class _Completions:
            def create(self, **kwargs):
                created.update(kwargs)
                return SimpleNamespace(usage=usage, choices=[])

        return SimpleNamespace(chat=SimpleNamespace(completions=_Completions())), created

    def test_completions_are_counted_under_eval(self):
        client, _ = self._client(SimpleNamespace(prompt_tokens=100, completion_tokens=20))
        wrapped = instrument_openai_client(client, EVAL)
        meter = UsageMeter()
        with meter, meter.scope(SYSTEM):  # system scope active...
            wrapped.chat.completions.create(model="m")
        # ...but the judge's spend is attributed to eval regardless.
        assert meter.usage(EVAL) == TokenUsage(100, 20, 1)
        assert meter.usage(SYSTEM).llm_calls == 0

    def test_the_wrapper_is_transparent(self):
        client, created = self._client(SimpleNamespace(prompt_tokens=1, completion_tokens=1))
        client.api_key = "sk-not-a-real-key"  # pragma: allowlist secret
        wrapped = instrument_openai_client(client, EVAL)
        wrapped.chat.completions.create(model="gpt-4o", temperature=0)
        assert created == {"model": "gpt-4o", "temperature": 0}
        assert wrapped.api_key == "sk-not-a-real-key"  # pragma: allowlist secret

    def test_the_response_is_returned_unchanged(self):
        client, _ = self._client(SimpleNamespace(prompt_tokens=1, completion_tokens=1))
        wrapped = instrument_openai_client(client, EVAL)
        assert wrapped.chat.completions.create(model="m").usage.prompt_tokens == 1

    def test_something_that_is_not_a_client_is_left_alone(self):
        """An uncounted judge beats a broken evaluation."""
        thing = SimpleNamespace()
        assert instrument_openai_client(thing, EVAL) is thing

    def test_instrumenting_twice_does_not_double_count(self):
        client, _ = self._client(SimpleNamespace(prompt_tokens=10, completion_tokens=2))
        instrument_openai_client(client, EVAL)
        instrument_openai_client(client, EVAL)
        meter = UsageMeter()
        with meter:
            client.chat.completions.create(model="m")
        assert meter.usage(EVAL) == TokenUsage(10, 2, 1)


class TestTheInstrumentedJudgeStillWorksAsAJudge:
    """Instrumentation that disables what it measures is worse than none.

    The first version of `instrument_openai_client` returned a transparent
    proxy with `__getattr__` delegation. RAGAS hands the client to
    `instructor.from_openai`, which dispatches on `isinstance(client, OpenAI)`
    — a proxy is not an `OpenAI`, so the dispatch fell through and returned
    `None`, `InstructorLLM.client` became `None`, and every judge call died
    with `AttributeError: 'NoneType' object has no attribute 'chat'` before
    reaching the network. That happened inside an `except Exception` that
    logged and continued, so every RAGAS metric silently vanished from the
    results while the run reported success.

    The tests above could not catch it: they exercise a `SimpleNamespace`
    duck-type, which is exactly the shape the real dispatch rejects. These go
    through the real libraries.

    Nothing here makes a network call — the failure was an `AttributeError`
    raised during construction, well before any HTTP.
    """

    def _real_client(self):
        # pragma: allowlist nextline secret
        return pytest.importorskip("openai").OpenAI(api_key="sk-not-a-real-key-no-request-is-made")

    def test_the_instrumented_client_is_still_an_openai_instance(self):
        openai = pytest.importorskip("openai")

        client = self._real_client()
        assert isinstance(instrument_openai_client(client, EVAL), openai.OpenAI)

    def test_instructor_still_accepts_it(self):
        """The exact dispatch that returned None."""
        instructor = pytest.importorskip("instructor")

        client = instrument_openai_client(self._real_client(), EVAL)
        assert instructor.from_openai(client) is not None

    def test_the_ragas_judge_builds_with_a_usable_client(self):
        """`InstructorLLM.client` was None, so `.client.chat` raised."""
        pytest.importorskip("instructor")
        llm_factory = pytest.importorskip("ragas.llms").llm_factory

        judge = llm_factory("gpt-4o", client=instrument_openai_client(self._real_client(), EVAL))
        assert getattr(judge, "client", "missing") is not None

    def test_the_same_object_is_returned_not_a_wrapper(self):
        """The patch is in place; anything that identity-compares still works."""
        client = self._real_client()
        assert instrument_openai_client(client, EVAL) is client

    def test_counting_survives_the_patch(self, monkeypatch):
        """Guards the guard: fixing the dispatch must not lose the counting.

        The first version of this test replaced the *instrumented* `create`
        with a stub that called `record_usage` itself, then asserted the meter
        saw the tokens — which proved only that `record_usage` works. It passed
        identically with `instrument_openai_client` deleted. That is precisely
        the false assurance this file criticises elsewhere, written into the
        test meant to prevent it.

        The stub now replaces the *transport underneath* the patch, so the
        assertion can only hold if the wrapper is really in the call path. No
        request leaves the machine.
        """
        client = self._real_client()

        recorded = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=31, completion_tokens=7))
        completions = type(client.chat.completions)

        def _fake_create(self, *args, **kwargs):
            return recorded

        # Patched on the class, i.e. below the instance attribute the wrapper
        # is installed on, so `instrument_openai_client` still has something
        # real to wrap.
        monkeypatch.setattr(completions, "create", _fake_create, raising=False)
        instrument_openai_client(client, EVAL)

        meter = UsageMeter()
        with meter:
            assert client.chat.completions.create(model="m") is recorded
        assert meter.usage(EVAL) == TokenUsage(31, 7, 1), (
            "the instrumented wrapper is not in the call path"
        )


class TestDollarsAreGone:
    def test_the_price_table_is_not_importable(self):
        with pytest.raises(ImportError):
            from lasagna.knowledge_evaluation_sdk.experiment import (  # noqa: F401
                _get_cost_rates,
            )

    def test_the_tradeoff_analyzer_has_no_price_table(self):
        import lasagna.knowledge_evaluation_sdk.private_eval.tradeoff_analyzer as ta

        assert not hasattr(ta, "DEFAULT_COST_TABLE")

    def test_efficiency_reports_tokens_not_dollars(self):
        from lasagna.knowledge_evaluation_sdk.private_eval.tradeoff_analyzer import (
            ConfigEfficiency,
        )

        fields = ConfigEfficiency.__dataclass_fields__
        assert "quality_per_1k_tokens" in fields
        assert not [f for f in fields if "cost" in f or "dollar" in f]

    def test_a_config_with_no_token_data_is_not_ranked_infinitely_efficient(self):
        """`inf` sorted a config with no measurements to the top of the table."""
        from lasagna.knowledge_evaluation_sdk.private_eval.tradeoff_analyzer import (
            TradeoffAnalyzer,
        )

        table = TradeoffAnalyzer().efficiency_table(
            {
                "no_data": {
                    "quality": 0.9,
                    "per_sample": [{"latency_ms": 10.0}],
                }
            }
        )
        assert table[0].quality_per_1k_tokens == 0.0
