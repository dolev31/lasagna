"""Ingest-time LLM calls use the configured provider, and failed enrichment is reported (#49).

Contextual enrichment, graph extraction and RAPTOR each built a fresh
``SimpleLLMProvider().get_llm()``: an OpenAI client, whatever provider was
configured. With only ``OLLAMA_BASE_URL`` set it asked for ``OPENAI_API_KEY``
and failed, so a ``max_quality`` corpus was ingested with no context and no
graph. Given an OpenAI client, ``ContextualEnricher`` called the chat model's
``generate`` with a string (``generate`` takes lists of messages), caught the
error and returned every chunk unchanged. Neither reached the run record: the
arm scored plain chunks under the name of contextual retrieval.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from lasagna import providers  # noqa: E402
from lasagna.core.strict import strict  # noqa: E402
from lasagna.knowledge_evaluation_sdk import bundle as bundle_mod  # noqa: E402
from lasagna.knowledge_evaluation_sdk import experiment as exp  # noqa: E402
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample  # noqa: E402
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask  # noqa: E402
from lasagna.processing.contextual import ContextualEnricher  # noqa: E402
from lasagna.providers.llm_provider import LLMProvider  # noqa: E402
from tests.test_event_loop_safety import executable_source  # noqa: E402
from tests.test_fast_ingest import MockConfig  # noqa: E402

_DOCUMENT = "Tardigrades survive extreme cold, heat and radiation. " * 12


class _DownModel(BaseChatModel):
    """A chat model whose every call fails, as an unreachable endpoint does."""

    @property
    def _llm_type(self) -> str:
        return "down"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("model is down")


class _Provider(LLMProvider):
    """The configured provider: hands out one model and records what was asked of it."""

    def __init__(self, model):
        self.model = model
        self.asked = []

    def get_chat_model(self, model_settings, temperature=0.0, max_tokens=None):
        self.asked.append(dict(model_settings))
        return self.model

    def validate_model_settings(self, model_settings):
        return True


@pytest.fixture
def configured(monkeypatch):
    """Configure ``model`` as the only LLM: no OpenAI credentials anywhere."""
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "AZURE_OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    def use(model):
        provider = _Provider(model)
        monkeypatch.setattr(providers, "_llm_provider", provider)
        monkeypatch.setattr(providers, "_auto_config_done", True)
        return provider

    return use


# --- The enricher reads a chat model's answer -------------------------------


def test_contextual_enrichment_reads_a_chat_models_answer():
    """``invoke`` and ``.content``: a LangChain chat model has ``generate`` too, for message lists."""
    enricher = ContextualEnricher(
        llm_provider=FakeListChatModel(responses=["About tardigrades."]), strict=True
    )
    assert enricher.enrich(_DOCUMENT, "They survive.") == "About tardigrades.\n\nThey survive."


def test_async_enrichment_reads_a_chat_models_answer():
    enricher = ContextualEnricher(
        llm_provider=FakeListChatModel(responses=["About tardigrades."]), strict=True
    )
    enriched = asyncio.run(enricher.aenrich(_DOCUMENT, "They survive."))
    assert enriched == "About tardigrades.\n\nThey survive."


def test_a_failed_enrichment_is_counted_with_its_reason():
    """Not strict, the chunk is kept as it was, and the enricher says how many and why."""
    enricher = ContextualEnricher(llm_provider=_DownModel())
    assert enricher.enrich_batch(_DOCUMENT, ["One.", "Two."]) == ["One.", "Two."]
    assert enricher.failed == 2
    assert "model is down" in enricher.first_error


# --- Every ingest-time call goes through the configured provider ------------


def test_benchmark_ingest_enriches_with_the_configured_provider(configured):
    from lasagna.knowledge_evaluation_sdk._fast_ingest import _chunk_one_doc

    configured(FakeListChatModel(responses=["About tardigrades."]))
    _, chunks, _ = _chunk_one_doc(
        corpus_id="d1",
        text=_DOCUMENT,
        valid_id="v1",
        session_id="s",
        filename="v1.txt",
        config=MockConfig(enable_contextual_retrieval=True),
    )
    assert chunks
    assert all(c["content"].startswith("About tardigrades.\n\n") for c in chunks)


def test_failed_enrichment_at_benchmark_ingest_is_tallied_for_the_corpus(configured):
    """Counted, not raised, even under evaluation: the run record names it (below)."""
    from lasagna.knowledge_evaluation_sdk._fast_ingest import _chunk_one_doc

    configured(_DownModel())
    tally: dict = {}
    with strict():
        _, chunks, _ = _chunk_one_doc(
            corpus_id="d1",
            text=_DOCUMENT,
            valid_id="v1",
            session_id="s",
            filename="v1.txt",
            config=MockConfig(enable_contextual_retrieval=True),
            enrichment=tally,
        )
    assert tally["chunks"] == len(chunks) > 0
    assert tally["failed"] == len(chunks)
    assert "model is down" in tally["error"]
    assert not any(c["content"].startswith("About") for c in chunks)


def test_graph_extraction_uses_the_configured_provider(configured):
    from lasagna.retrieval.graph.entity_extractor import extract_graph

    reply = '{"entities": [{"name": "Tardigrade", "type": "CONCEPT"}], "relationships": []}'
    provider = configured(FakeListChatModel(responses=[reply]))
    config = SimpleNamespace(graph_extraction_model="extractor-model", graph_entity_types=None)
    found = extract_graph([_DOCUMENT], config)
    assert found.extractions[0]["entities"] == [{"name": "Tardigrade", "type": "CONCEPT"}]
    assert provider.asked[-1]["model_name"] == "extractor-model"


def test_no_ingest_feature_builds_its_own_llm_client():
    """The whole class: enrichment, graph extraction and RAPTOR, at upload and benchmark ingest."""
    from lasagna.api._document_pipeline import DocumentPipeline
    from lasagna.knowledge_evaluation_sdk._fast_ingest import _chunk_one_doc
    from lasagna.retrieval.graph.entity_extractor import extract_graph

    for func in (DocumentPipeline.process_document, _chunk_one_doc, extract_graph):
        assert "SimpleLLMProvider" not in executable_source(func), func.__qualname__


# --- A corpus whose enrichment failed is named on every task's row ----------


class _Pipeline:
    """Stands in for KnowledgePipeline: one document per query, nothing falls back."""

    def __init__(self):
        self._config = {"enable_hybrid_search": False}
        self._component_names = {}
        self._instances = {}

    def query(self, question, top_k, session_id, config):
        return [SimpleNamespace(chunk=SimpleNamespace(document_id="v1", content="x"), score=1.0)]


@pytest.fixture
def run_on_corpus(monkeypatch):
    """Run one retrieval task of four queries on a corpus described by ``contextual``."""
    from lasagna.api import rag_backend
    from lasagna.core import pipeline as pipeline_mod

    def run(contextual):
        async def _ingested(*a, **k):
            return {
                "session_id": "bench_corpus_test",
                "corpus_to_valid": {"d1": "v1"},
                "valid_to_corpus": {"v1": "d1"},
                "corpus_fingerprint": "fp",
                "contextual": contextual,
            }

        async def _released(*a, **k):
            return None

        monkeypatch.setattr(exp, "_ensure_corpus_ingested", _ingested)
        monkeypatch.setattr(exp, "_release_corpus", _released)
        monkeypatch.setattr(exp, "_apply_overrides_to_pipeline", lambda *a, **k: None)
        monkeypatch.setattr(
            rag_backend,
            "get_rag_backend",
            lambda: SimpleNamespace(embedding_manager=SimpleNamespace(model_name="m", dimension=4)),
        )
        pipeline = _Pipeline()
        monkeypatch.setattr(
            pipeline_mod.KnowledgePipeline, "from_profile", classmethod(lambda cls, p: pipeline)
        )
        task = BenchmarkTask(
            benchmark_name="smoke",
            config_name="max_quality",
            run_index=0,
            mode="live",
            config_overrides={},
            tracks=("retrieval",),
        )
        samples = [
            BenchmarkSample(query_id=f"q{i}", question=f"question {i}?", expected_docs={"d1": 1})
            for i in range(4)
        ]
        return asyncio.run(exp._eval_live(task, samples))

    return run


def test_a_corpus_whose_enrichment_failed_is_named_on_the_tasks_row(run_on_corpus):
    result = run_on_corpus({"chunks": 12, "failed": 5, "error": "model is down"})
    assert result.error is None
    assert result.system["degraded_stages"]["retrieval"]["contextual_retrieval"] == {
        "queries": 4,
        "of": 4,
        "reasons": [
            {
                "reason": "contextual enrichment failed for 5 of 12 chunks at ingest; "
                "they were stored without context (model is down)",
                "queries": 4,
            }
        ],
    }
    row = {"benchmark_name": "smoke", "config_name": "max_quality", "system": result.system}
    (line,) = bundle_mod.degraded_findings([row])
    assert line.startswith(
        "contextual_retrieval fell back during the run: smoke, arm 'max_quality'"
    )


def test_a_corpus_enriched_in_full_records_nothing(run_on_corpus):
    result = run_on_corpus({"chunks": 12, "failed": 0, "error": ""})
    assert "degraded_stages" not in result.system
