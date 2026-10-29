"""Two services starting together must not fail when the other wins the race.

Both services check ``has_collection`` before either creates, so the loser
sees no collection and then its ``create_collection`` raises because the
winner already created it. The manager must treat that as "already exists":
return, and still ensure the index. A plain ``ValueError`` cannot carry that
distinction, so the backends raise ``CollectionExistsError`` (a ``ValueError``
subclass, so existing ``except ValueError`` handlers keep working).
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import lasagna.api._collection_manager as cm
from lasagna.core.config import get_rag_config
from lasagna.vector_backends.base import (
    CollectionExistsError,
    CollectionSchema,
    FieldSchema,
)


def _schema():
    return CollectionSchema(
        fields=[
            FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True),
            FieldSchema(name="embedding", dtype="FLOAT_VECTOR", dimension=4),
        ]
    )


class _RacingBackend:
    """A backend whose collection appears between the check and the create."""

    backend_name = "fake"
    is_lite = True

    def __init__(self):
        self.index_created = False

    def has_collection(self, name):
        return False

    def create_collection(self, name, schema, index_config=None):
        raise CollectionExistsError(f"Collection {name} already exists")

    def describe_fields(self, name):
        return [{"name": "embedding", "dtype": "FLOAT_VECTOR"}]

    def list_indexes(self, name):
        return []

    def prepare_index_params(self):
        return SimpleNamespace(add_index=lambda **kw: None)

    def create_index(self, *args, **kwargs):
        self.index_created = True


def _manager(monkeypatch, backend):
    monkeypatch.setattr(cm, "_stamp_store", lambda: None)
    config = get_rag_config()
    monkeypatch.setattr(cm, "rag_config", config)
    manager = cm.CollectionManager(
        backend,
        "fake",
        {},
        SimpleNamespace(dimension=getattr(config, "effective_embedding_dim", config.embedding_dim)),
    )
    return manager, config


def test_initialize_collection_survives_a_lost_creation_race(monkeypatch):
    backend = _RacingBackend()
    manager, _config = _manager(monkeypatch, backend)
    manager.initialize_collection()
    assert backend.index_created


def test_ensure_collection_for_config_survives_a_lost_creation_race(monkeypatch):
    backend = _RacingBackend()
    manager, config = _manager(monkeypatch, backend)
    manager.ensure_collection_for_config(config)
    assert backend.index_created


def test_ensure_backend_collection_survives_a_lost_creation_race(monkeypatch):
    """The third check-then-create: `ensure_backend_collection` checks
    `has_collection` and then creates, so a concurrent creator (the API
    process racing a Celery worker) wins in between. The loser must not
    fail: it treats the winner's collection as "already exists" and still
    ensures the index instead of dropping anything.
    """
    backend = _RacingBackend()
    manager, config = _manager(monkeypatch, backend)
    manager.ensure_backend_collection(backend, config)
    assert backend.index_created


def test_qdrant_create_on_existing_name_raises_collection_exists_error():
    from lasagna.vector_backends.qdrant_backend import QdrantBackend

    backend = QdrantBackend.__new__(QdrantBackend)
    backend._client = SimpleNamespace(
        get_collections=lambda: SimpleNamespace(collections=[SimpleNamespace(name="c1")])
    )
    with pytest.raises(CollectionExistsError) as excinfo:
        backend.create_collection("c1", _schema())
    assert isinstance(excinfo.value, ValueError)


def _qdrant_backend_with_client(monkeypatch, client):
    """QdrantBackend with a fake client, stubbing client-only names.

    When qdrant-client is not installed, the module never binds Distance
    or VectorParams, and reaching the create call NameErrors before the
    mapping under test. The mapping itself does not use those names, so
    stub them only when they are missing; a real client keeps its own.
    The stubs go through monkeypatch, so the module is left as it was
    found once the test ends instead of leaking into later tests.
    """
    import lasagna.vector_backends.qdrant_backend as qmod

    if not hasattr(qmod, "Distance"):
        monkeypatch.setattr(
            qmod,
            "Distance",
            SimpleNamespace(COSINE="cosine", EUCLID="euclid", DOT="dot"),
            raising=False,
        )
    if not hasattr(qmod, "VectorParams"):
        monkeypatch.setattr(
            qmod, "VectorParams", lambda **kwargs: SimpleNamespace(**kwargs), raising=False
        )
    backend = qmod.QdrantBackend.__new__(qmod.QdrantBackend)
    backend._client = client
    return backend


def test_qdrant_create_maps_a_lost_race_to_collection_exists_error(monkeypatch):
    """The check passes (empty list) but another process wins the create.

    This is the narrower Qdrant window: the client raises its own error,
    so the backend must map it to CollectionExistsError itself.
    """

    def _create_collection(**kwargs):
        raise Exception("Collection c1 already exists")

    backend = _qdrant_backend_with_client(
        monkeypatch,
        SimpleNamespace(
            get_collections=lambda: SimpleNamespace(collections=[]),
            create_collection=_create_collection,
        ),
    )
    with pytest.raises(CollectionExistsError) as excinfo:
        backend.create_collection("c1", _schema())
    assert isinstance(excinfo.value, ValueError)


def test_qdrant_create_maps_a_409_conflict_to_collection_exists_error(monkeypatch):
    class _Conflict(Exception):
        status_code = 409

    def _create_collection(**kwargs):
        raise _Conflict("Unexpected Response: 409")

    backend = _qdrant_backend_with_client(
        monkeypatch,
        SimpleNamespace(
            get_collections=lambda: SimpleNamespace(collections=[]),
            create_collection=_create_collection,
        ),
    )
    with pytest.raises(CollectionExistsError) as excinfo:
        backend.create_collection("c1", _schema())
    assert isinstance(excinfo.value, ValueError)


def test_pgvector_create_on_existing_name_raises_collection_exists_error():
    from lasagna.vector_backends.pgvector_backend import PgVectorBackend

    class _Cursor:
        def fetchone(self):
            return (True,)

    class _Conn:
        def execute(self, *args, **kwargs):
            return _Cursor()

    @contextmanager
    def _pool_connection():
        yield _Conn()

    backend = PgVectorBackend.__new__(PgVectorBackend)
    backend._schema = "public"
    backend._pool = SimpleNamespace(connection=lambda: _pool_connection())
    with pytest.raises(CollectionExistsError) as excinfo:
        backend.create_collection("c1", _schema())
    assert isinstance(excinfo.value, ValueError)


def test_pinecone_create_on_existing_name_raises_collection_exists_error():
    from lasagna.vector_backends.pinecone_backend import PineconeBackend

    backend = PineconeBackend.__new__(PineconeBackend)
    backend._client = SimpleNamespace(list_indexes=lambda: [SimpleNamespace(name="c1")])
    with pytest.raises(CollectionExistsError) as excinfo:
        backend.create_collection("c1", _schema())
    assert isinstance(excinfo.value, ValueError)


def test_weaviate_create_on_existing_name_raises_collection_exists_error():
    from lasagna.vector_backends.weaviate_backend import WeaviateBackend

    backend = WeaviateBackend.__new__(WeaviateBackend)
    backend._client = SimpleNamespace(
        collections=SimpleNamespace(exists=lambda name: True),
        schema=SimpleNamespace(get=lambda: {"classes": [{"class": "c1"}]}),
    )
    with pytest.raises(CollectionExistsError) as excinfo:
        backend.create_collection("c1", _schema())
    assert isinstance(excinfo.value, ValueError)


def test_milvus_create_maps_an_existing_name_to_collection_exists_error():
    from lasagna.vector_backends.milvus_backend import MilvusBackend

    def _create_collection(**kwargs):
        raise Exception("Collection c1 already exists")

    backend = MilvusBackend.__new__(MilvusBackend)
    backend._client = SimpleNamespace(create_collection=_create_collection)
    backend._session_column = {}
    backend._native_hybrid = {}
    backend._to_milvus_schema = lambda schema: SimpleNamespace()
    with pytest.raises(CollectionExistsError) as excinfo:
        backend.create_collection("c1", _schema())
    assert isinstance(excinfo.value, ValueError)
