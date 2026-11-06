"""The shared graph follows its documents, and a reused benchmark corpus finds or rebuilds it (#55).

A deleted document's part of the graph stays out of every later search; a
session that is reset forgets its graphs; and a benchmark that reuses an
ingested corpus says in its run record whether the corpus brought its graph,
or had none and got it rebuilt from the chunks already stored. A document is
done once every chunk of it was extracted, entities or none; one that could
not be (no LLM, a failed call) leaves the graph incomplete, and the bundle
reports that as a fallback.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from tests._document_harness import SESSION, Harness

pytest.importorskip("networkx")

EXTRACTION = {
    "entities": [{"name": "Warsaw", "type": "LOCATION"}, {"name": "Vistula", "type": "LOCATION"}],
    "relationships": [{"source": "Warsaw", "target": "Vistula", "type": "LIES_ON"}],
}


@pytest.fixture
def stub_extraction(monkeypatch):
    from lasagna.providers.llm_provider import SimpleLLMProvider
    from lasagna.retrieval.graph.entity_extractor import EntityExtractor

    calls = []

    def extract(self, texts, llm, max_workers=4):
        calls.append(list(texts))
        return [EXTRACTION for _ in texts]

    monkeypatch.setattr(SimpleLLMProvider, "get_llm", lambda self, **kw: object())
    monkeypatch.setattr(EntityExtractor, "extract_batch_sync", extract)
    return calls


@pytest.fixture(autouse=True)
def _fresh_loaded_graphs():
    import importlib

    gs = importlib.import_module("lasagna.retrieval.graph.graph_store")
    gs._loaded.clear()
    yield
    gs._loaded.clear()


def _graph_harness(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    h.settings["enable_graph_retrieval"] = True
    return h


def test_deleting_a_document_removes_its_part_of_the_graph(tmp_path, monkeypatch, stub_extraction):
    h = _graph_harness(tmp_path, monkeypatch)
    one = h.upload(h.write(tmp_path, "warsaw.txt", "Warsaw lies on the Vistula."))
    two = h.upload(h.write(tmp_path, "vistula.txt", "The Vistula flows through Warsaw."))
    collection = h.config().collection_name
    h.pipeline.delete_document(one.document_id, SESSION)
    _, mentions, edges = h.db.load_graph(SESSION, collection)
    assert {doc for _, _, doc in mentions} == {two.document_id}
    assert {doc for *_, doc in edges} == {two.document_id}
    h.pipeline.delete_document(two.document_id, SESSION)
    assert h.db.load_graph(SESSION, collection) == ([], [], [])


def test_a_reset_session_forgets_the_graphs_this_process_loaded(
    tmp_path, monkeypatch, stub_extraction
):
    import importlib

    from lasagna.api.rag_backend import RAGBackend
    from lasagna.retrieval.graph.graph_store import DatabaseGraphStore

    gs = importlib.import_module("lasagna.retrieval.graph.graph_store")
    h = _graph_harness(tmp_path, monkeypatch)
    h.upload(h.write(tmp_path, "warsaw.txt", "Warsaw lies on the Vistula."))
    DatabaseGraphStore(SESSION, h.config().collection_name, db=h.db).snapshot()
    assert any(key[0] == SESSION for key in gs._loaded)
    RAGBackend.clear_session_state(h.backend, SESSION)
    assert not any(key[0] == SESSION for key in gs._loaded)


class _StoredChunks:
    """A vector store holding a reused corpus's chunks, as a benchmark finds them."""

    def __init__(self, rows):
        self.rows = rows

    def session_filter(self, collection, session):
        return None

    def iter_all(self, **kw):
        return list(self.rows)


def _corpus_db(tmp_path, monkeypatch):
    import lasagna.core.database as database
    from lasagna.core.database import RAGDatabase

    db = RAGDatabase(str(tmp_path / "corpus.db"))
    monkeypatch.setattr(database, "get_rag_database", lambda: db)
    return db


ROWS = [
    {
        "chunk_id": "d1_chunk_1",
        "content": "The Vistula.",
        "document_id": "d1",
        "metadata": {"chunk_index": 1},
    },
    {
        "chunk_id": "d1_chunk_0",
        "content": "Warsaw.",
        "document_id": "d1",
        "metadata": {"chunk_index": 0},
    },
    {
        "chunk_id": "d2_chunk_0",
        "content": "Kraków.",
        "document_id": "d2",
        "metadata": {"chunk_index": 0},
    },
]


def test_a_reused_corpus_with_no_graph_gets_it_rebuilt_then_reused(
    tmp_path, monkeypatch, stub_extraction
):
    from lasagna.core.config import get_rag_config
    from lasagna.retrieval.graph.graph_store import ensure_corpus_graph

    db = _corpus_db(tmp_path, monkeypatch)
    config = get_rag_config().with_query_overrides({"enable_graph_retrieval": True})
    first = ensure_corpus_graph(
        "s", "c", _StoredChunks(ROWS), {"d1", "d2"}, set(), config, lambda n: np.ones(3)
    )
    assert first["state"] == "rebuilt" and first["rebuilt"] == 2 and first["documents"] == 2
    # Extracted from the stored chunks, in chunk order.
    assert stub_extraction[0] == ["Warsaw.", "The Vistula."]
    _, mentions, _ = db.load_graph("s", "c")
    assert {chunk for _, chunk, _ in mentions} == {"d1_chunk_0", "d1_chunk_1", "d2_chunk_0"}
    again = ensure_corpus_graph(
        "s", "c", _StoredChunks(ROWS), {"d1", "d2"}, set(), config, lambda n: np.ones(3)
    )
    assert again["state"] == "reused" and again["rebuilt"] == 0
    assert len(stub_extraction) == 2  # nothing extracted again


def test_a_corpus_ingested_now_reports_its_graph_as_built(tmp_path, monkeypatch, stub_extraction):
    from lasagna.core.config import get_rag_config
    from lasagna.retrieval.graph.graph_store import ensure_corpus_graph, index_document_graph

    _corpus_db(tmp_path, monkeypatch)
    config = get_rag_config().with_query_overrides({"enable_graph_retrieval": True})
    index_document_graph("s", "c", "d1", [EXTRACTION], ["d1_chunk_0"], lambda n: np.ones(3))
    info = ensure_corpus_graph(
        "s", "c", _StoredChunks(ROWS), {"d1"}, {"d1"}, config, lambda n: np.ones(3)
    )
    assert info["state"] == "built" and info["documents"] == 1


def test_the_benchmark_ingest_stores_graphs_only_for_documents_that_made_it(tmp_path, monkeypatch):
    from lasagna.knowledge_evaluation_sdk._fast_ingest import _store_pending_graphs
    from lasagna.retrieval.graph.entity_extractor import GraphExtraction

    db = _corpus_db(tmp_path, monkeypatch)
    pending = {
        "d1": (GraphExtraction([EXTRACTION], 0, "m"), ["d1_chunk_0"]),
        "d2": (GraphExtraction([EXTRACTION], 0, "m"), ["d2_chunk_0"]),
    }
    config = SimpleNamespace(collection_name="c")
    stored = _store_pending_graphs(
        pending, {"d1"}, "s", config, SimpleNamespace(encode_document=lambda n: np.ones(3))
    )
    assert stored == 1
    _, mentions, _ = db.load_graph("s", "c")
    assert {doc for _, _, doc in mentions} == {"d1"}
    assert db.extracted_graph_documents("s", "c") == {"d1": "m"}


def _run_with_graph(monkeypatch, graph):
    """One live retrieval task, over a corpus whose graph check said ``graph``."""
    from lasagna.api import rag_backend
    from lasagna.core import pipeline as pipeline_mod
    from lasagna.knowledge_evaluation_sdk import experiment as exp
    from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
    from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

    async def _ingested(*a, **k):
        return {
            "session_id": "s",
            "corpus_to_valid": {"d1": "v1"},
            "valid_to_corpus": {"v1": "d1"},
            "corpus_fingerprint": "fp",
            "graph": graph,
        }

    async def _released(*a, **k):
        return None

    class _Pipeline:
        _component_names: dict = {}
        _instances: dict = {}

        def __init__(self):
            self._config = {"enable_hybrid_search": False}

        def query(self, *a, **k):
            return [
                SimpleNamespace(chunk=SimpleNamespace(document_id="v1", content="x"), score=1.0)
            ]

    monkeypatch.setattr(exp, "_ensure_corpus_ingested", _ingested)
    monkeypatch.setattr(exp, "_release_corpus", _released)
    monkeypatch.setattr(exp, "_apply_overrides_to_pipeline", lambda *a, **k: None)
    monkeypatch.setattr(
        rag_backend,
        "get_rag_backend",
        lambda: SimpleNamespace(embedding_manager=SimpleNamespace(model_name="m", dimension=4)),
    )
    monkeypatch.setattr(
        pipeline_mod.KnowledgePipeline, "from_profile", classmethod(lambda c, p: _Pipeline())
    )
    task = BenchmarkTask(
        benchmark_name="smoke",
        config_name="graph",
        run_index=0,
        mode="live",
        config_overrides={},
        tracks=("retrieval",),
    )
    sample = BenchmarkSample(query_id="q", question="q?", expected_docs={"d1": 1})
    return asyncio.run(exp._eval_live(task, [sample]))


def test_the_run_record_says_how_the_corpus_got_its_graph(monkeypatch):
    graph = {"state": "rebuilt", "documents": 2, "missing": 0, "entities": 3, "rebuilt": 2}
    result = _run_with_graph(monkeypatch, graph)
    assert result.system["graph"] == graph
    assert "degraded_stages" not in result.system


def test_an_incomplete_graph_is_reported_as_a_fallback(monkeypatch):
    from lasagna.knowledge_evaluation_sdk import bundle as bundle_mod

    graph = {"state": "incomplete", "documents": 1, "missing": 1, "entities": 2, "rebuilt": 0}
    result = _run_with_graph(monkeypatch, graph)
    stage = result.system["degraded_stages"]["retrieval"]["graph"]
    assert stage["queries"] == stage["of"] == 1
    row = result.to_dict()
    assert "ACTIVE_FALLBACKS" in bundle_mod._integrity([row], {})["warnings"]
    (line,) = bundle_mod.degraded_findings([row])
    assert "graph fell back" in line and "documents with no graph: 1" in line


class _LLM:
    """An extraction model that answers, or fails on chunks naming a word."""

    def __init__(self, fail_on=()):
        self.fail_on = fail_on
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if any(word in prompt for word in self.fail_on):
            raise RuntimeError("the endpoint refused the call")
        return SimpleNamespace(content=json.dumps(EXTRACTION))


def _use_llm(monkeypatch, llm):
    from lasagna.providers.llm_provider import SimpleLLMProvider

    monkeypatch.setattr(SimpleLLMProvider, "get_llm", lambda self, **kw: llm)


def _graph_config(**extra):
    from lasagna.core.config import get_rag_config

    return get_rag_config().with_query_overrides({"enable_graph_retrieval": True, **extra})


def _ones(text):
    return np.ones(3)


# Long enough to be sent to the model (EntityExtractor skips under 50 characters).
LONG_ROWS = [
    {
        "chunk_id": "d1_chunk_0",
        "content": "Warsaw lies on the Vistula, the longest river of Poland.",
        "document_id": "d1",
        "metadata": {"chunk_index": 0},
    },
    {
        "chunk_id": "d2_chunk_0",
        "content": "Krakow lies on the Vistula too, upstream of the capital city.",
        "document_id": "d2",
        "metadata": {"chunk_index": 0},
    },
]


def test_a_document_that_names_no_entity_is_reused_not_extracted_again(tmp_path, monkeypatch):
    from lasagna.retrieval.graph.graph_store import ensure_corpus_graph

    db = _corpus_db(tmp_path, monkeypatch)
    llm = _LLM()
    _use_llm(monkeypatch, llm)
    # Every chunk is too short to extract: done, with no entity and no call.
    first = ensure_corpus_graph(
        "s", "c", _StoredChunks(ROWS), {"d1", "d2"}, set(), _graph_config(), _ones
    )
    assert first["state"] == "rebuilt" and first["missing"] == 0 and first["entities"] == 0
    assert set(db.extracted_graph_documents("s", "c")) == {"d1", "d2"}
    again = ensure_corpus_graph(
        "s", "c", _StoredChunks(ROWS), {"d1", "d2"}, set(), _graph_config(), _ones
    )
    assert again["state"] == "reused" and again["rebuilt"] == 0 and again["documents"] == 2
    assert llm.calls == 0


def test_with_no_llm_a_reused_corpus_graph_is_incomplete(tmp_path, monkeypatch):
    from lasagna.providers.llm_provider import SimpleLLMProvider
    from lasagna.retrieval.graph.graph_store import ensure_corpus_graph

    _corpus_db(tmp_path, monkeypatch)
    where = os.path.join(os.path.expanduser("~"), "keys", "openai.env")

    def no_llm(self, **kw):
        raise RuntimeError(f"no API key in {where}")

    monkeypatch.setattr(SimpleLLMProvider, "get_llm", no_llm)
    info = ensure_corpus_graph(
        "s", "c", _StoredChunks(LONG_ROWS), {"d1", "d2"}, set(), _graph_config(), _ones
    )
    assert info["state"] == "incomplete" and info["missing"] == 2 and info["rebuilt"] == 0
    # Bundles are published: the error says what failed, not where.
    assert "no API key" in info["error"] and os.path.expanduser("~") not in info["error"]


def test_a_corpus_ingested_now_without_a_graph_is_incomplete_not_built(tmp_path, monkeypatch):
    from lasagna.retrieval.graph.graph_store import ensure_corpus_graph

    _corpus_db(tmp_path, monkeypatch)
    # The ingest ran and its extraction failed: nothing stored for d1.
    info = ensure_corpus_graph(
        "s", "c", _StoredChunks(LONG_ROWS), {"d1"}, {"d1"}, _graph_config(), _ones
    )
    assert info["state"] == "incomplete" and info["missing"] == 1


def test_a_failed_chunk_leaves_its_document_to_be_extracted_again(tmp_path, monkeypatch):
    from lasagna.retrieval.graph.graph_store import ensure_corpus_graph

    db = _corpus_db(tmp_path, monkeypatch)
    config = _graph_config(graph_extraction_model="graph-model")
    _use_llm(monkeypatch, _LLM(fail_on=("Krakow",)))
    info = ensure_corpus_graph(
        "s", "c", _StoredChunks(LONG_ROWS), {"d1", "d2"}, set(), config, _ones
    )
    assert info["state"] == "incomplete" and info["missing"] == 1 and info["documents"] == 1
    assert db.extracted_graph_documents("s", "c") == {"d1": "graph-model"}
    assert info["extraction_model"] == "graph-model"

    healed = _LLM()
    _use_llm(monkeypatch, healed)
    again = ensure_corpus_graph(
        "s", "c", _StoredChunks(LONG_ROWS), {"d1", "d2"}, set(), config, _ones
    )
    assert again["state"] == "rebuilt" and again["rebuilt"] == 1 and again["missing"] == 0
    assert healed.calls == 1  # d2's chunk only


def test_a_rebuild_reads_only_the_documents_it_needs(tmp_path, monkeypatch, stub_extraction):
    from lasagna.retrieval.graph.graph_store import ensure_corpus_graph, index_document_graph

    _corpus_db(tmp_path, monkeypatch)
    index_document_graph("s", "c", "d1", [EXTRACTION], ["d1_chunk_0"], _ones)
    store = _StoredChunks(ROWS)
    seen = []
    store.iter_all = lambda **kw: seen.append(kw["filter_expr"]) or list(ROWS)
    ensure_corpus_graph("s", "c", store, {"d1", "d2"}, set(), _graph_config(), _ones)
    (expr,) = seen

    def leaves(e):
        if e.children:
            return [leaf for child in e.children for leaf in leaves(child)]
        return [(e.field, e.value)]

    # d1 has its graph; only d2's chunks are read, not the whole corpus.
    assert ("document_id", ["d2"]) in leaves(expr)
    assert stub_extraction == [["Kraków."]]


def test_a_stale_corpus_drops_its_graph_with_its_vectors(tmp_path, monkeypatch):
    import inspect

    from lasagna.knowledge_evaluation_sdk import experiment as exp
    from lasagna.retrieval.graph.graph_store import index_document_graph

    db = _corpus_db(tmp_path, monkeypatch)
    index_document_graph("s", "c", "d1", [EXTRACTION], ["d1_chunk_0"], _ones)
    index_document_graph("s", "other", "d1", [EXTRACTION], ["d1_chunk_0"], _ones)
    exp._drop_corpus_graph("s", "c")
    assert db.load_graph("s", "c") == ([], [], [])
    assert db.extracted_graph_documents("s", "c") == {}
    # Another collection's graph is not this corpus's.
    assert db.extracted_graph_documents("s", "other") == {"d1": None}
    # Both stale-corpus paths, where the vectors are deleted, drop it.
    assert inspect.getsource(exp._ensure_corpus_ingested).count("_drop_corpus_graph(") == 2


def test_two_graph_arms_with_different_extraction_settings_get_corpora_of_their_own():
    from lasagna.knowledge_evaluation_sdk import experiment as exp
    from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

    def key(**overrides):
        return exp._corpus_cache_key(
            BenchmarkTask(
                benchmark_name="smoke",
                config_name="graph",
                run_index=0,
                mode="live",
                config_overrides={"enable_graph_retrieval": True, **overrides},
                tracks=("retrieval",),
            )
        )

    assert key(graph_extraction_model="a") != key(graph_extraction_model="b")
    assert key(graph_entity_types=["PERSON"]) != key(graph_entity_types=["ORG"])
    assert key(graph_extraction_model="a") == key(graph_extraction_model="a")
