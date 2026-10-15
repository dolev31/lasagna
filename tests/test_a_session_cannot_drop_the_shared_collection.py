"""A session-scoped operation never drops the collection every session shares.

Reproduced with auth on and only a tenant's own session token: one
``patch_rag_config`` of ``embedding_dim`` followed by ``reindex_document(...,
force_rebuild=True)`` dropped ``rag_documents_v2``, and every other tenant's
search came back empty. Sessions are a metadata filter inside that one
collection, so "rebuild it for my new size" means "delete everyone's vectors".
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import lasagna.api._profile_guard as guard
from lasagna.mcp import server


class _Store:
    def __init__(self, rows):
        self.rows = rows

    def has_collection(self, name):
        return True

    def get_collection_stats(self, name):
        if isinstance(self.rows, Exception):
            raise self.rows
        return {"row_count": self.rows}

    def delete(self, **kwargs):
        pass


class _Backend:
    """What `reindex_document` touches, recording any drop."""

    def __init__(self, tmp_path, *, rows, stored_dim=384, configured_dim=768):
        source = tmp_path / "doc.txt"
        source.write_text("tenant B's own document")
        self.source = str(source)
        self.store = _Store(rows)
        self.stored_dim = stored_dim
        self.configured_dim = configured_dim
        self.dropped = []

    def get_documents(self, session_id):
        return [
            SimpleNamespace(
                document_id="b-doc",
                session_id=session_id,
                filename="doc.txt",
                file_type="txt",
                file_size=10,
                upload_date=datetime.now(timezone.utc),
                chunks_count=1,
                status=SimpleNamespace(value="ready"),
            )
        ]

    def get_document_file_path(self, document_id, session_id):
        return self.source

    def get_session_config(self, session_id):
        return SimpleNamespace(
            collection_name="rag_documents_v2", embedding_dim=self.configured_dim
        )

    def get_collection_embedding_dim(self, name):
        return self.stored_dim

    def get_vector_backend(self, session_id):
        return self.store

    def recreate_collection_for_config(self, config):
        self.dropped.append(config.collection_name)

    def ensure_collection_for_config(self, config):
        pass

    def reload_embedding_manager(self, config):
        pass

    def process_document(self, doc, path, config=None):
        pass


@pytest.fixture
def backend(tmp_path, monkeypatch):
    def make(**kwargs):
        fake = _Backend(tmp_path, **kwargs)
        monkeypatch.setattr(server, "_get_backend", lambda: fake)
        monkeypatch.setattr(server, "_get_rebuild_signature", lambda: None)
        monkeypatch.setattr(server, "_set_rebuild_signature", lambda sig: None)
        return fake

    return make


def test_reindex_with_force_rebuild_never_drops_a_collection_that_holds_documents(backend):
    fake = backend(rows=12)
    reply = server.reindex_document("b-doc", session_id="tenant-b", force_rebuild=True)
    assert fake.dropped == []
    assert reply.startswith("Failed to reindex document")
    assert "rebuild_collection" in reply


def test_an_unreadable_row_count_is_not_an_empty_collection(backend):
    fake = backend(rows=RuntimeError("stats unavailable"))
    server.reindex_document("b-doc", session_id="tenant-b", force_rebuild=True)
    assert fake.dropped == []


def test_reindex_with_force_rebuild_still_rebuilds_an_empty_collection(backend):
    fake = backend(rows=0)
    reply = server.reindex_document("b-doc", session_id="tenant-b", force_rebuild=True)
    assert fake.dropped == ["rag_documents_v2"]
    assert "reindexed successfully" in reply


def test_a_patch_of_embedding_dim_is_refused_while_documents_are_stored(monkeypatch):
    """The first half of the attack: the tenant's own size change."""
    import json

    monkeypatch.setattr(guard, "stored_embedding_dim", lambda collection, vector_backend=None: 384)
    reply = json.loads(
        server.patch_rag_config(
            "tenant-b-patch",
            json.dumps({"embedding_model": "BAAI/bge-base-en-v1.5", "embedding_dim": 768}),
        )
    )
    assert "384" in reply["error"]


def test_a_mismatch_on_an_unreadable_collection_is_refused_not_rebuilt():
    """The upload path decided "empty" from a stats call that had failed."""
    from lasagna.api._collection_manager import CollectionManager

    manager = CollectionManager.__new__(CollectionManager)
    manager._default_backend = _Store(RuntimeError("stats unavailable"))
    manager.get_collection_embedding_dim = lambda name, backend=None: 384
    manager.recreate_collection_for_config = lambda config, backend=None: pytest.fail(
        "dropped a collection whose row count could not be read"
    )
    config = SimpleNamespace(collection_name="rag_documents_v2", embedding_dim=768)
    # The refusal's wording changed (0.1.0: no "MISMATCH"/"DELETED" banner);
    # what it pins is that the collection is refused, not dropped.
    with pytest.raises(ValueError, match="embedded at 384 dimensions"):
        manager.maybe_reload_embedding_manager(config)
