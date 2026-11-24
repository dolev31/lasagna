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
    from lasagna import providers
    from lasagna.retrieval.graph.entity_extractor import EntityExtractor

    calls = []

    def extract(self, texts, llm, max_workers=4):
        calls.append(list(texts))
        return [EXTRACTION for _ in texts]

    monkeypatch.setattr(providers, "get_ingest_llm", lambda **kw: object())
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


def _lifecycle(h):
    from lasagna.lifecycle.manager import DocumentLifecycleManager
    from lasagna.lifecycle.policy import LifecyclePolicy

    return DocumentLifecycleManager(
        db=h.db,
        pipeline=h.pipeline,
        backend=h.backend,
        policy=LifecyclePolicy(),
        collection_name=h.config().collection_name,
    )


def test_deleting_a_session_drops_its_graph(tmp_path, monkeypatch, stub_extraction):
    h = _graph_harness(tmp_path, monkeypatch)
    h.upload(h.write(tmp_path, "warsaw.txt", "Warsaw lies on the Vistula."))
    h.upload(h.write(tmp_path, "vistula.txt", "The Vistula flows through Warsaw."))
    collection = h.config().collection_name
    # A part left by a document whose own removal failed earlier.
    h.db.save_graph_document(
        SESSION,
        collection,
        "gone",
        [("Kraków", "LOCATION", None)],
        [("Kraków", "gone_chunk_0")],
        [],
    )
    assert h.db.load_graph(SESSION, collection)[1]
    result = _lifecycle(h).bulk_delete_session(SESSION, confirm=True)
    assert result.deleted == 2 and result.errors == 0
    assert h.db.load_graph(SESSION, collection) == ([], [], [])
    assert h.db.extracted_graph_documents(SESSION, collection) == {}


def test_deleting_a_session_with_no_documents_drops_what_is_left_of_its_graph(
    tmp_path, monkeypatch
):
    h = _graph_harness(tmp_path, monkeypatch)
    collection = h.config().collection_name
    h.db.save_graph_document(
        SESSION,
        collection,
        "gone",
        [("Kraków", "LOCATION", None)],
        [("Kraków", "gone_chunk_0")],
        [],
    )
    result = _lifecycle(h).bulk_delete_session(SESSION, confirm=True)
    assert result.requested == 0
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
    assert db.extracted_graph_documents("s", "c") == {"d1": {"model": "m", "unparsed_chunks": 0}}


def _run_with_graph(monkeypatch, graph, expansion_fails=False):
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
            if expansion_fails:
                from lasagna.utils import degraded

                degraded.record("graphrag", "the graph expansion failed")
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

    graph = {
        "state": "incomplete",
        "documents": 1,
        "missing": 1,
        "entities": 2,
        "rebuilt": 0,
        "error": "no LLM configured",
    }
    result = _run_with_graph(monkeypatch, graph)
    # The stage name the search's own graph expansion records.
    stage = result.system["degraded_stages"]["retrieval"]["graphrag"]
    assert stage["queries"] == stage["of"] == 1
    row = result.to_dict()
    assert "ACTIVE_FALLBACKS" in bundle_mod._integrity([row], {})["warnings"]
    (line,) = bundle_mod.degraded_findings([row])
    assert "graphrag fell back" in line
    assert "the corpus graph is missing 1 of 2 documents (no LLM configured)" in line


def test_an_incomplete_graph_keeps_what_the_expansion_recorded(monkeypatch):
    graph = {"state": "incomplete", "documents": 1, "missing": 1, "entities": 2, "rebuilt": 0}
    result = _run_with_graph(monkeypatch, graph, expansion_fails=True)
    stage = result.system["degraded_stages"]["retrieval"]["graphrag"]
    assert stage["queries"] == stage["of"] == 1
    assert [r["reason"] for r in stage["reasons"]] == [
        "the corpus graph is missing 1 of 2 documents",
        "the graph expansion failed",
    ]


def test_a_graph_that_could_not_be_checked_says_why(monkeypatch):
    graph = {"state": "unknown", "error": "the metadata database is locked"}
    result = _run_with_graph(monkeypatch, graph)
    (reason,) = result.system["degraded_stages"]["retrieval"]["graphrag"]["reasons"]
    assert reason["reason"] == (
        "the corpus graph could not be checked (the metadata database is locked)"
    )


def test_a_graph_left_empty_by_unreadable_replies_is_a_fallback(monkeypatch):
    graph = {
        "state": "rebuilt",
        "documents": 2,
        "missing": 0,
        "entities": 0,
        "rebuilt": 2,
        "unparsed_chunks": 2,
    }
    result = _run_with_graph(monkeypatch, graph)
    (reason,) = result.system["degraded_stages"]["retrieval"]["graphrag"]["reasons"]
    assert reason["reason"] == (
        "the corpus graph is empty: 2 replies of the extraction model held no readable JSON"
    )
    # A graph with entities and a few unreadable replies is not a fallback.
    result = _run_with_graph(monkeypatch, {**graph, "entities": 5})
    assert "degraded_stages" not in result.system


class _LLM:
    """An extraction model that answers, fails, or answers in prose on chunks naming a word."""

    def __init__(self, fail_on=(), prose_on=()):
        self.fail_on = fail_on
        self.prose_on = prose_on
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if any(word in prompt for word in self.fail_on):
            raise RuntimeError("the endpoint refused the call")
        if any(word in prompt for word in self.prose_on):
            return SimpleNamespace(content="There are no named entities in this passage.")
        return SimpleNamespace(content=json.dumps(EXTRACTION))


def _use_llm(monkeypatch, llm):
    from lasagna import providers

    monkeypatch.setattr(providers, "get_ingest_llm", lambda **kw: llm)


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
    from lasagna import providers
    from lasagna.retrieval.graph.graph_store import ensure_corpus_graph

    _corpus_db(tmp_path, monkeypatch)
    where = os.path.join(os.path.expanduser("~"), "keys", "openai.env")

    def no_llm(**kw):
        raise RuntimeError(f"no API key in {where}")

    monkeypatch.setattr(providers, "get_ingest_llm", no_llm)
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
    # d2 kept what its other chunks named, and is not counted as rebuilt.
    assert info["rebuilt"] == 1
    assert db.extracted_graph_documents("s", "c") == {
        "d1": {"model": "graph-model", "unparsed_chunks": 0}
    }
    assert info["extraction_model"] == "graph-model"

    healed = _LLM()
    _use_llm(monkeypatch, healed)
    again = ensure_corpus_graph(
        "s", "c", _StoredChunks(LONG_ROWS), {"d1", "d2"}, set(), config, _ones
    )
    assert again["state"] == "rebuilt" and again["rebuilt"] == 1 and again["missing"] == 0
    assert healed.calls == 1  # d2's chunk only


def test_a_reply_in_prose_is_the_models_answer_not_a_failure(tmp_path, monkeypatch):
    from lasagna.retrieval.graph.graph_store import ensure_corpus_graph

    db = _corpus_db(tmp_path, monkeypatch)
    llm = _LLM(prose_on=("Krakow",))
    _use_llm(monkeypatch, llm)
    first = ensure_corpus_graph(
        "s", "c", _StoredChunks(LONG_ROWS), {"d1", "d2"}, set(), _graph_config(), _ones
    )
    # At temperature 0 the model answers the same way next time: done.
    assert first["state"] == "rebuilt" and first["missing"] == 0
    assert first["unparsed_chunks"] == 1
    assert db.extracted_graph_documents("s", "c")["d2"]["unparsed_chunks"] == 1
    again = ensure_corpus_graph(
        "s", "c", _StoredChunks(LONG_ROWS), {"d1", "d2"}, set(), _graph_config(), _ones
    )
    assert again["state"] == "reused" and again["unparsed_chunks"] == 1
    assert llm.calls == 2


def test_a_document_another_model_extracted_is_extracted_again(tmp_path, monkeypatch):
    from lasagna.retrieval.graph.graph_store import ensure_corpus_graph

    _corpus_db(tmp_path, monkeypatch)
    llm = _LLM()
    _use_llm(monkeypatch, llm)

    def check(model):
        return ensure_corpus_graph(
            "s",
            "c",
            _StoredChunks(LONG_ROWS),
            {"d1", "d2"},
            set(),
            _graph_config(graph_extraction_model=model),
            _ones,
        )

    assert check("model-a")["extraction_model"] == "model-a"
    assert check("model-a")["state"] == "reused" and llm.calls == 2
    # The default model changed (an arm that names none keys no model).
    switched = check("model-b")
    assert switched["state"] == "rebuilt" and switched["rebuilt"] == 2
    assert switched["extraction_model"] == "model-b" and llm.calls == 4


def test_a_document_with_no_stored_chunks_is_not_missing(tmp_path, monkeypatch):
    from lasagna.retrieval.graph.graph_store import ensure_corpus_graph

    _corpus_db(tmp_path, monkeypatch)
    _use_llm(monkeypatch, _LLM())
    # d3 chunked to nothing: there is nothing to extract.
    info = ensure_corpus_graph(
        "s", "c", _StoredChunks(LONG_ROWS), {"d1", "d2", "d3"}, set(), _graph_config(), _ones
    )
    assert info["state"] == "rebuilt" and info["missing"] == 0 and info["no_chunks"] == 1


def test_a_stale_model_with_no_llm_to_replace_it_is_incomplete(tmp_path, monkeypatch):
    from lasagna import providers
    from lasagna.retrieval.graph.graph_store import ensure_corpus_graph

    _corpus_db(tmp_path, monkeypatch)
    _use_llm(monkeypatch, _LLM())
    ensure_corpus_graph(
        "s",
        "c",
        _StoredChunks(LONG_ROWS),
        {"d1", "d2"},
        set(),
        _graph_config(graph_extraction_model="old"),
        _ones,
    )

    def no_llm(**kw):
        raise RuntimeError("no LLM configured")

    monkeypatch.setattr(providers, "get_ingest_llm", no_llm)
    info = ensure_corpus_graph(
        "s",
        "c",
        _StoredChunks(LONG_ROWS),
        {"d1", "d2"},
        set(),
        _graph_config(graph_extraction_model="new"),
        _ones,
    )
    # The graph is the old model's: not the one this arm asked for.
    assert info["state"] == "incomplete" and info["missing"] == 2 and info["documents"] == 0
    assert info["error"] == "no LLM configured"


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
    assert db.extracted_graph_documents("s", "other") == {
        "d1": {"model": None, "unparsed_chunks": 0}
    }
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


def _env_overrides(monkeypatch, **values):
    """RAG_* settings as the config read them at start, without RAG_GRAPH_EXTRACTION_MODEL."""
    from lasagna.core.config import rag_config

    overrides = dict(rag_config._env_overrides or {})
    overrides.pop("graph_extraction_model", None)
    monkeypatch.setattr(rag_config, "_env_overrides", {**overrides, **values})


def test_a_graph_arm_that_names_no_model_is_keyed_by_the_default_model(monkeypatch):
    from lasagna.knowledge_evaluation_sdk import experiment as exp
    from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

    _env_overrides(monkeypatch, enable_graph_retrieval=False)

    def key(**overrides):
        return exp._corpus_cache_key(
            BenchmarkTask(
                benchmark_name="smoke",
                config_name="arm",
                run_index=0,
                mode="live",
                config_overrides=overrides,
                tracks=("retrieval",),
            )
        )

    monkeypatch.setenv("MODEL_NAME", "model-a")
    graph_a, plain_a = key(enable_graph_retrieval=True), key(chunk_size=300)
    monkeypatch.setenv("MODEL_NAME", "model-b")
    # Two runs with different default models never share one graph.
    assert key(enable_graph_retrieval=True) != graph_a
    # An arm without the graph does not care which model is the default.
    assert key(chunk_size=300) == plain_a


def test_a_graph_on_for_every_arm_is_keyed_by_the_default_model_too(monkeypatch):
    from lasagna.knowledge_evaluation_sdk import experiment as exp
    from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

    _env_overrides(monkeypatch, enable_graph_retrieval=True)

    def key(**overrides):
        return exp._corpus_cache_key(
            BenchmarkTask(
                benchmark_name="smoke",
                config_name="arm",
                run_index=0,
                mode="live",
                config_overrides=overrides,
                tracks=("retrieval",),
            )
        )

    monkeypatch.setenv("MODEL_NAME", "model-a")
    plain_a, off_a = key(), key(enable_graph_retrieval=False)
    monkeypatch.setenv("MODEL_NAME", "model-b")
    # The Compose stack turns the graph on for every arm.
    assert key() != plain_a
    # An arm that turns it off keys no model.
    assert key(enable_graph_retrieval=False) == off_a
