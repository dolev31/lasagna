"""With no LLM configured, agentic search must say so, not answer an opaque 500."""

from __future__ import annotations

from lasagna.api.rest.routes.search import _missing_llm
from lasagna.providers import LLMProviderNotConfigured


def test_the_message_is_found_through_wrapping():
    try:
        try:
            raise LLMProviderNotConfigured("No LLMProvider configured. Set OPENAI_API_KEY")
        except LLMProviderNotConfigured as inner:
            raise RuntimeError("agentic graph failed") from inner
    except RuntimeError as outer:
        assert _missing_llm(outer).startswith("No LLMProvider configured")


def test_other_failures_stay_500():
    assert _missing_llm(RuntimeError("boom")) is None


def test_it_is_still_a_runtime_error_for_existing_handlers():
    assert issubclass(LLMProviderNotConfigured, RuntimeError)
