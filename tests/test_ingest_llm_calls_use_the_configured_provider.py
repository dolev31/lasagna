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

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402 - after importorskip
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402 - after importorskip
from langchain_core.messages import AIMessage  # noqa: E402 - after importorskip
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402 - after importorskip

from lasagna import providers  # noqa: E402 - after importorskip
from lasagna.core.strict import strict  # noqa: E402 - after importorskip
from lasagna.knowledge_evaluation_sdk import bundle as bundle_mod  # noqa: E402 - after importorskip
from lasagna.knowledge_evaluation_sdk import experiment as exp  # noqa: E402 - after importorskip
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample  # noqa: E402 - after importorskip
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask  # noqa: E402 - after importorskip
from lasagna.processing.contextual import ContextualEnricher  # noqa: E402 - after importorskip
from lasagna.providers.llm_provider import LLMProvider  # noqa: E402 - after importorskip
from tests.test_event_loop_safety import executable_source  # noqa: E402 - after importorskip
from tests.test_fast_ingest import MockBackend, MockConfig  # noqa: E402 - after importorskip

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


# --- The corpus's tally is the sum of its documents' ------------------------


class _FailsOnMarkedChunks(BaseChatModel):
    """Situates every chunk except those that say FAIL: that call fails."""

    @property
    def _llm_type(self) -> str:
        return "fails-on-marked"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        chunk = messages[-1].content.split("<chunk>\n", 1)[1].split("\n</chunk>", 1)[0]
        if "FAIL" in chunk:
            raise RuntimeError("model is down")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="About it."))])


def test_the_corpus_tally_is_the_sum_of_every_documents_chunks(configured):
    """Through the real pool, documents failing none, some or all of their chunks.

    With contextual retrieval on, the pool has one worker today, so this does
    not reproduce a lost update; it pins the totals to the chunks stored.
    """
    from lasagna.knowledge_evaluation_sdk._fast_ingest import fast_ingest_corpus

    configured(_FailsOnMarkedChunks())
    failing, passing = "FAIL a tardigrade dries out. ", "Moss and lichen hold water. "
    corpus = {"all": failing * 20, "none": passing * 20, "mixed": failing * 10 + passing * 10}
    backend = MockBackend()
    tally: dict = {}
    asyncio.run(
        fast_ingest_corpus(
            corpus=corpus,
            corpus_to_valid={name: f"v_{name}" for name in corpus},
            ready_ids=set(),
            session_id="s",
            config=MockConfig(enable_contextual_retrieval=True),
            backend=backend,
            enrichment=tally,
        )
    )
    stored = backend._vector_backend.inserted
    without_context = [r for r in stored if not r["metadata"]["contextual_enriched"]]
    assert tally["chunks"] == len(stored)
    assert tally["failed"] == len(without_context) == sum("FAIL" in r["content"] for r in stored)
    assert 0 < tally["failed"] < tally["chunks"]
    assert tally["error"] == "model is down"
    assert all(
        r["content"].startswith("About it.\n\n")
        for r in stored
        if r["metadata"]["contextual_enriched"]
    )


# --- A corpus stored without its context is not reused ----------------------


@pytest.fixture
def corpus_runs(monkeypatch, tmp_path):
    """`_ensure_corpus_ingested` on the smoke corpus with contextual retrieval on.

    The stored chunks live in ``rows``; ingest marks them enriched unless
    ``model_down``. ``ingests`` counts the ingests and ``deleted`` the deletes.
    ``rows_as_stored`` is what the store's ``query`` returns for a row.
    """
    import lasagna.api.rag_backend as rag_backend_mod
    from lasagna.core.config import rag_config
    from lasagna.knowledge_evaluation_sdk import _fast_ingest as fast_ingest_mod

    h = SimpleNamespace(
        rows=[],
        model_down=False,
        ingests=0,
        deleted=0,
        graphs_dropped=0,
        rows_as_stored=lambda row: {"metadata": row["metadata"]},
    )

    def delete(**kw):
        h.deleted += 1
        h.rows.clear()

    store = SimpleNamespace(
        session_filter=lambda c, s: None,
        delete=delete,
        query=lambda **kw: [h.rows_as_stored(r) for r in h.rows],
    )
    backend = SimpleNamespace(
        _collection_manager=SimpleNamespace(resolve_backend=lambda s: store),
        hybrid_searcher=SimpleNamespace(clear_session=lambda s: None),
    )
    monkeypatch.setattr(rag_backend_mod, "get_rag_backend", lambda *a, **k: backend)
    monkeypatch.setattr(exp, "_present_document_ids", lambda *a: {r["document_id"] for r in h.rows})

    def drop_graph(session_id, collection_name):
        h.graphs_dropped += 1

    monkeypatch.setattr(exp, "_drop_corpus_graph", drop_graph)
    monkeypatch.setattr(exp, "_results_root", lambda: str(tmp_path))
    monkeypatch.setattr(
        rag_config,
        "_env_overrides",
        {**(rag_config._env_overrides or {}), "enable_contextual_retrieval": True},
    )

    async def fast_ingest(corpus, corpus_to_valid, ready_ids, enrichment=None, **kw):
        h.ingests += 1
        for cid in corpus:
            mark = {"contextual_enriched": not h.model_down}
            h.rows.append({"document_id": corpus_to_valid[cid], "metadata": mark})
        enrichment.update(
            chunks=len(corpus),
            failed=len(corpus) if h.model_down else 0,
            error="model is down" if h.model_down else "",
        )
        return len(corpus), set(corpus_to_valid.values()), set()

    monkeypatch.setattr(fast_ingest_mod, "fast_ingest_corpus", fast_ingest)

    def run():
        exp._corpus_sessions.clear()
        return asyncio.run(exp._ensure_corpus_ingested("smoke", []))

    h.run = run
    yield h
    exp._corpus_sessions.clear()


def test_a_reused_corpus_stored_without_context_says_so(corpus_runs):
    """Reused, it recorded nothing; deleted, it could vanish under another run."""
    corpus_runs.model_down = True
    first = corpus_runs.run()
    assert first["contextual"]["failed"] == first["contextual"]["chunks"] == 24

    corpus_runs.model_down = False  # back, but the corpus is what it is
    second = corpus_runs.run()
    assert (corpus_runs.ingests, corpus_runs.deleted) == (1, 0)
    assert second["contextual"]["failed"] == second["contextual"]["chunks"] == 24
    assert "reused" in second["contextual"]["error"]


def test_a_reused_corpus_enriched_in_full_reports_no_failure(corpus_runs):
    corpus_runs.run()
    reused = corpus_runs.run()
    assert (corpus_runs.ingests, corpus_runs.deleted) == (1, 0)
    assert reused["contextual"]["failed"] == 0


def test_a_mark_stored_as_a_property_of_the_row_is_read(corpus_runs):
    """Weaviate keeps metadata keys as the row's own properties."""
    corpus_runs.model_down = True
    corpus_runs.run()
    corpus_runs.rows_as_stored = lambda row: dict(row["metadata"])
    assert corpus_runs.run()["contextual"]["failed"] == 24
    assert corpus_runs.ingests == 1


def test_a_contextual_corpus_from_before_the_mark_is_ingested_again(corpus_runs):
    """Stored before #49 marked its chunks, it was ingested when enrichment failed."""
    corpus_runs.run()
    for row in corpus_runs.rows:
        row["metadata"] = {}
    corpus_runs.run()
    assert (corpus_runs.ingests, corpus_runs.deleted, corpus_runs.graphs_dropped) == (2, 1, 1)


def test_a_corpus_whose_marks_cannot_be_read_is_ingested_again(corpus_runs):
    corpus_runs.run()

    def unreadable(row):
        raise RuntimeError("store is busy")

    corpus_runs.rows_as_stored = unreadable
    again = corpus_runs.run()
    assert (corpus_runs.ingests, corpus_runs.deleted) == (2, 1)
    assert again["contextual"]["failed"] == 0


def test_with_no_llm_every_chunk_is_marked_and_counted(configured, monkeypatch):
    """Not strict, and no LLM could be built: no chunk is taken for enriched."""
    from lasagna.knowledge_evaluation_sdk._fast_ingest import _chunk_one_doc

    def no_llm(**kw):
        raise RuntimeError("no LLM is configured")

    monkeypatch.setattr(providers, "get_ingest_llm", no_llm)
    tally: dict = {}
    _, chunks, _ = _chunk_one_doc(
        corpus_id="d1",
        text=_DOCUMENT,
        valid_id="v1",
        session_id="s",
        filename="v1.txt",
        config=MockConfig(enable_contextual_retrieval=True),
        enrichment=tally,
    )
    assert chunks and all(c["metadata"]["contextual_enriched"] is False for c in chunks)
    assert tally == {"chunks": len(chunks), "failed": len(chunks), "error": "no LLM is configured"}
