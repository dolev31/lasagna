"""A knowledge graph one process builds is the graph every other process searches (#55).

The graph lived in a dict in the memory of the process that ingested. In the
Compose stack the Celery worker ingests and the API workers search, so every
search expanded through an empty graph, and nothing said so; a restart lost
every graph. It is now kept in the metadata database the processes share.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from tests._document_harness import SESSION, Harness
from tests._paths import PACKAGE_DIR

pytest.importorskip("networkx")

TEXT = "Marie Curie was born in Warsaw. Warsaw lies on the Vistula river."
EXTRACTION = {
    "entities": [
        {"name": "Marie Curie", "type": "PERSON"},
        {"name": "Warsaw", "type": "LOCATION"},
        {"name": "Vistula", "type": "LOCATION"},
    ],
    "relationships": [
        {"source": "Marie Curie", "target": "Warsaw", "type": "BORN_IN"},
        {"source": "Warsaw", "target": "Vistula", "type": "LIES_ON"},
    ],
}


def _graph_upload(tmp_path, monkeypatch):
    """An upload through the real pipeline, graph on, with a stub extractor and LLM."""
    from lasagna.providers.llm_provider import SimpleLLMProvider
    from lasagna.retrieval.graph.entity_extractor import EntityExtractor

    monkeypatch.setattr(SimpleLLMProvider, "get_llm", lambda self, **kw: object())
    monkeypatch.setattr(
        EntityExtractor,
        "extract_batch_sync",
        lambda self, texts, llm, max_workers=4: [EXTRACTION for _ in texts],
    )
    h = Harness(tmp_path, monkeypatch)
    h.settings["enable_graph_retrieval"] = True
    doc = h.upload(h.write(tmp_path, "curie.txt", TEXT))
    return h, doc


#: What a second process runs: it opens the same metadata database, as an API
#: worker opens the stack's PostgreSQL, and expands a search through the graph.
_CHILD = """
import inspect, json, sys
import lasagna.core.database as database
from lasagna.core.database import RAGDatabase
from lasagna.retrieval.graph import GraphRetriever

db = RAGDatabase(sys.argv[1])
database.get_rag_database = lambda: db
retriever = GraphRetriever(embedding_manager=None)
kwargs = {}
if "collection_name" in inspect.signature(retriever.expand).parameters:
    kwargs["collection_name"] = sys.argv[2]
print(json.dumps(sorted(retriever.expand("Where was Marie Curie born?", set(), sys.argv[3], **kwargs))))
"""


def test_a_graph_an_upload_built_is_expanded_by_a_process_that_built_none(tmp_path, monkeypatch):
    h, doc = _graph_upload(tmp_path, monkeypatch)
    collection = h.config().collection_name
    run = subprocess.run(
        [sys.executable, "-c", _CHILD, str(tmp_path / "rag.db"), collection, SESSION],
        env={"PYTHONPATH": str(PACKAGE_DIR.parent), "HF_HUB_OFFLINE": "1", "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    found = json.loads(run.stdout.strip().splitlines()[-1])
    chunk_ids = {c["chunk_id"] for c in h.chunks(doc.document_id)}
    assert found and set(found) <= chunk_ids


# -- the store itself --------------------------------------------------------


def _store(tmp_path, collection="coll_a"):
    from lasagna.core.database import RAGDatabase
    from lasagna.retrieval.graph.graph_store import DatabaseGraphStore

    return DatabaseGraphStore(SESSION, collection, db=RAGDatabase(str(tmp_path / "graph.db")))


def _entities(chunk, names, vector=None):
    from lasagna.retrieval.graph.graph_store import GraphEntity

    return [GraphEntity(n, "LOCATION", vector, chunk) for n in names]


def _relationships(*pairs):
    from lasagna.retrieval.graph.graph_store import GraphRelationship

    return [GraphRelationship(a, b, "RELATED") for a, b in pairs]


@pytest.fixture(autouse=True)
def _fresh_loaded_graphs():
    import importlib

    gs = importlib.import_module("lasagna.retrieval.graph.graph_store")

    loaded = getattr(gs, "_loaded", {})  # absent before the shared store
    loaded.clear()
    yield
    loaded.clear()


def test_the_store_answers_what_the_retriever_asks(tmp_path, monkeypatch):
    import lasagna.core.database as database

    store = _store(tmp_path)
    monkeypatch.setattr(database, "get_rag_database", lambda: store.db)
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    store.add_document(
        "d1",
        _entities("d1_chunk_0", ["Warsaw", "Vistula"], vec),
        _relationships(("Warsaw", "Vistula")),
    )
    assert store.num_entities == 2 and store.num_relationships == 1
    assert store.find_entities_by_name("which river is Warsaw on?") == ["Warsaw"]
    assert set(store.find_entities_by_embedding(vec, top_k=5)) == {"Warsaw", "Vistula"}
    assert store.get_neighbors("Warsaw", hops=1) == {"Vistula"}
    assert store.get_chunk_ids_for_entities({"Vistula"}) == {"d1_chunk_0"}


def test_storing_a_document_again_replaces_its_part(tmp_path, monkeypatch):
    """A reindex gives the chunks new ids; the old ones must not stay in the graph."""
    import lasagna.core.database as database

    store = _store(tmp_path)
    monkeypatch.setattr(database, "get_rag_database", lambda: store.db)
    store.add_document("d1", _entities("d1_chunk_0", ["Warsaw"]), [])
    store.add_document("d1", _entities("d1_r2_chunk_0", ["Warsaw"]), [])
    assert store.get_chunk_ids_for_entities({"Warsaw"}) == {"d1_r2_chunk_0"}


def test_removing_a_document_keeps_what_another_still_states(tmp_path, monkeypatch):
    import lasagna.core.database as database

    store = _store(tmp_path)
    monkeypatch.setattr(database, "get_rag_database", lambda: store.db)
    store.add_document(
        "d1", _entities("d1_chunk_0", ["Warsaw", "Vistula"]), _relationships(("Warsaw", "Vistula"))
    )
    store.add_document(
        "d2", _entities("d2_chunk_0", ["Warsaw", "Poland"]), _relationships(("Warsaw", "Poland"))
    )
    store.remove_document("d1")
    assert store.find_entities_by_name("Vistula") == []
    assert store.get_chunk_ids_for_entities({"Warsaw"}) == {"d2_chunk_0"}
    assert store.get_neighbors("Warsaw") == {"Poland"}
    assert store.num_entities == 2


def test_each_collection_has_a_graph_of_its_own(tmp_path, monkeypatch):
    import lasagna.core.database as database
    from lasagna.retrieval.graph.graph_store import DatabaseGraphStore

    a = _store(tmp_path, "coll_a")
    monkeypatch.setattr(database, "get_rag_database", lambda: a.db)
    b = DatabaseGraphStore(SESSION, "coll_b", db=a.db)
    a.add_document("d1", _entities("d1_chunk_0", ["Warsaw"]), [])
    assert b.num_entities == 0
    assert a.num_entities == 1


def test_a_process_sees_what_another_wrote_once_the_session_changes(tmp_path, monkeypatch):
    """The loaded graph is kept until the content generation moves, as BM25's is."""
    import lasagna.core.database as database
    from lasagna.retrieval.graph.graph_store import DatabaseGraphStore

    reader = _store(tmp_path)
    monkeypatch.setattr(database, "get_rag_database", lambda: reader.db)
    assert reader.num_entities == 0
    # Another process writes: its own store object, and it bumps the session.
    other = DatabaseGraphStore(SESSION, "coll_a", db=reader.db)
    other.db.save_graph_document(
        SESSION, "coll_a", "d9", [("Warsaw", "LOCATION", None)], [("Warsaw", "d9_chunk_0")], []
    )
    assert reader.num_entities == 0  # still the loaded graph
    reader.db.bump_content_generation(SESSION)
    assert reader.num_entities == 1


def test_an_in_memory_graph_removes_a_document_as_the_database_does():
    from lasagna.retrieval.graph.graph_store import InMemoryGraphStore

    g = InMemoryGraphStore()
    g.add_document("d1", _entities("d1_chunk_0", ["A", "B"]), _relationships(("A", "B")))
    g.add_document("d2", _entities("d2_chunk_0", ["B", "C"]), _relationships(("B", "C")))
    g.remove_document("d1")
    assert sorted(g._graph.nodes()) == ["B", "C"]
    assert g.get_chunk_ids_for_entities({"B"}) == {"d2_chunk_0"}
