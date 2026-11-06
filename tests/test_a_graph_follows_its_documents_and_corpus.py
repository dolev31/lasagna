"""The shared graph follows its documents, and a reused benchmark corpus finds or rebuilds it (#55).

A deleted document's part of the graph stays out of every later search; a
session that is reset forgets its graphs; and a benchmark that reuses an
ingested corpus says in its run record whether the corpus brought its graph,
or had none and got it rebuilt from the chunks already stored.
"""

from __future__ import annotations

import asyncio
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

    db = _corpus_db(tmp_path, monkeypatch)
    pending = {"d1": ([EXTRACTION], ["d1_chunk_0"]), "d2": ([EXTRACTION], ["d2_chunk_0"])}
    config = SimpleNamespace(collection_name="c")
    stored = _store_pending_graphs(
        pending, {"d1"}, "s", config, SimpleNamespace(encode_document=lambda n: np.ones(3))
    )
    assert stored == 1
    _, mentions, _ = db.load_graph("s", "c")
    assert {doc for _, _, doc in mentions} == {"d1"}


def test_the_run_record_says_how_the_corpus_got_its_graph(monkeypatch):
    from lasagna.api import rag_backend
    from lasagna.core import pipeline as pipeline_mod
    from lasagna.knowledge_evaluation_sdk import experiment as exp
    from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
    from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

    graph = {"state": "rebuilt", "documents": 2, "entities": 3, "rebuilt": 2}

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
    result = asyncio.run(exp._eval_live(task, [sample]))
    assert result.system["graph"] == graph
