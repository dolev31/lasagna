"""Reindexing or rebuilding over MCP leaves the server's shared components alone.

``reindex_document`` and ``rebuild_collection`` called
``RAGBackend.reload_embedding_manager``, which assigned the session's embedder
to the backend's shared ``embedding_manager`` (and the chunker's) and built a
new shared reranker. One reindex for a session on another model changed what
the health check reports as the default model and what every later reader of
those attributes embedded with, while concurrent requests were using them.
The upload and search paths stopped doing this earlier; these two tools now
resolve the session's embedder for the request alone.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lasagna.api.rag_backend import RAGBackend
from lasagna.mcp import server

DEFAULT = SimpleNamespace(model_name="BAAI/bge-base-en-v1.5", dimension=768)
SESSION = SimpleNamespace(model_name="sentence-transformers/all-MiniLM-L6-v2", dimension=384)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    """A RAGBackend whose collaborators are stubs, recording what happens in order."""
    b = object.__new__(RAGBackend)  # not the process singleton
    events = []
    b.embedding_manager = DEFAULT
    b.chunker = SimpleNamespace(embedding_manager=DEFAULT)
    b.reranker = shared_reranker = object()
    b._create_reranker = lambda: object()

    def embedder_for(config):
        events.append("collection checked")
        return SESSION

    manager = SimpleNamespace(embedding_manager=DEFAULT, embedder_for=embedder_for)

    def maybe_reload_embedding_manager(config):
        manager.embedding_manager = embedder_for(config)

    manager.maybe_reload_embedding_manager = maybe_reload_embedding_manager
    b._collection_manager = manager

    source = tmp_path / "doc.txt"
    source.write_text("a document")
    doc = SimpleNamespace(
        document_id="d1",
        session_id="s1",
        filename="doc.txt",
        file_type="txt",
        file_size=10,
        upload_date=datetime.now(timezone.utc),
        chunks_count=1,
        status=SimpleNamespace(value="ready"),
    )
    b.get_documents = lambda session_id: [doc]
    b.get_document_file_path = lambda document_id, session_id: str(source)
    b.get_session_config = lambda session_id: SimpleNamespace(
        collection_name="lasagna_minilm", embedding_dim=384, embedding_model=SESSION.model_name
    )
    b.get_collection_embedding_dim = lambda name: 384
    b.ensure_collection_for_config = lambda config: None
    b.recreate_collection_for_config = lambda config: None
    b.get_vector_backend = lambda session_id: SimpleNamespace(
        delete=lambda **kw: events.append("chunks deleted")
    )
    b.process_document = lambda document, path, config=None: events.append("reprocessed")

    monkeypatch.setattr(server, "_get_backend", lambda: b)
    b.events, b.shared_reranker = events, shared_reranker
    return b


def _untouched(b):
    assert b.embedding_manager is DEFAULT
    assert b.chunker.embedding_manager is DEFAULT
    assert b._collection_manager.embedding_manager is DEFAULT
    assert b.reranker is b.shared_reranker


def test_reindex_document_leaves_the_shared_embedder_and_reranker_alone(backend):
    reply = server.reindex_document("d1", session_id="s1")
    assert "reindexed successfully" in reply, reply
    _untouched(backend)
    # The collection is still checked for this session's model before its chunks go.
    assert backend.events == ["collection checked", "chunks deleted", "reprocessed"]


def test_rebuild_collection_leaves_the_shared_embedder_and_reranker_alone(backend):
    reply = server.rebuild_collection(session_id="s1", confirm=True)
    assert reply.startswith("Collection rebuilt successfully"), reply
    _untouched(backend)
    assert backend.events == ["collection checked"]
