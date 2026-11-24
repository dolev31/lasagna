"""A benchmark task on a non-Milvus stack must never construct a MilvusBackend.

dolev31/lasagna#32: ``lasagna benchmark run smoke`` inside a stack built with
``VECTOR_BACKEND=qdrant`` talked to Milvus and failed with::

    MilvusException: (code=100, message=collection 'rag_bench_emb_...' does not exist)

Root cause chain:

1. ``KnowledgePipeline.from_profile("speed")`` stores the built-in profile's
   *full* configuration. That config's ``vector_backend`` is the packaged
   ``settings.lasagna.toml`` value (``"milvus"``): the profile builder merges
   the TOML over the environment-aware defaults, so the file beats
   ``RAG_VECTOR_BACKEND``.
2. The evaluation harness applies the pipeline config as session overrides
   (``trusted=True``, so the ``_SERVER_KEYS`` check in ``_session_keys`` is
   skipped).
3. ``RAGConfig.vector_backend`` resolved session overrides first, so every
   ``with_session(...).vector_backend`` returned ``"milvus"`` and
   ``CollectionManager.resolve_backend()`` built a ``MilvusBackend``
   (Milvus Lite, since a Qdrant stack sets no Milvus URI).

The rule, already stated in ``_session_keys``: ``vector_backend`` and
``vector_backend_config`` are the server's infrastructure
(``_SERVER_KEYS``). A session must never redirect the vector store.

Each test below fails on ``main`` and passes with the fix.
"""

from __future__ import annotations

import uuid

import pytest

from lasagna.core.config import RAGConfig, reset_rag_config
from lasagna.core.pipeline import KnowledgePipeline


@pytest.fixture
def qdrant_env(monkeypatch, tmp_path):
    """Fresh providers with the environment a Qdrant deployment sets."""
    import lasagna.providers as providers

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LASAGNA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_SESSION_BACKEND", "memory")
    monkeypatch.setenv("RAG_VECTOR_BACKEND", "qdrant")
    monkeypatch.setenv("RAG_VECTOR_BACKEND_CONFIG", '{"url": "http://qdrant:6333"}')
    for name in ("_config_provider", "_llm_provider", "_session_provider"):
        monkeypatch.setattr(providers, name, None)
    monkeypatch.setattr(providers, "_auto_config_done", False)
    reset_rag_config()
    yield
    monkeypatch.undo()
    reset_rag_config()


def _new_id() -> str:
    return f"bench-{uuid.uuid4().hex[:8]}"


def _poison_session_like_the_harness(sid: str) -> None:
    """Write what the benchmark harness wrote: the profile's full config,
    whose vector_backend is the packaged settings file's "milvus"."""
    from lasagna.providers import get_session_provider

    get_session_provider().set_config_overrides(
        sid,
        {
            "vector_backend": "milvus",
            "vector_backend_config": {},
            "collection_name": "rag_bench_emb_c4c84ef1",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "enable_hybrid_search": False,
        },
    )


def test_session_overrides_cannot_redirect_vector_backend(qdrant_env):
    sid = _new_id()
    _poison_session_like_the_harness(sid)

    session = RAGConfig().with_session(sid)

    assert session.vector_backend == "qdrant", (
        "a session override redirected the vector store to Milvus (dolev31/lasagna#32)"
    )
    assert session.vector_backend_config == {"url": "http://qdrant:6333"}


def test_pipeline_apply_session_config_strips_vector_backend(monkeypatch, qdrant_env):
    import lasagna.core.session_config as session_config_pkg

    written = {}

    class _StubManager:
        def patch_config(self, config, session_id=None, trusted=False):
            written["config"] = dict(config)
            written["session_id"] = session_id
            written["trusted"] = trusted
            return {}

    monkeypatch.setattr(
        session_config_pkg, "get_knowledge_config_manager", lambda **kw: _StubManager()
    )

    # The built-in profile's full config, as from_profile() stores it: the
    # packaged settings file's vector_backend rides along.
    pipeline = KnowledgePipeline(
        config={
            "vector_backend": "milvus",
            "vector_backend_config": {},
            "collection_name": "rag_bench_emb_c4c84ef1",
            "enable_reranking": False,
        }
    )
    pipeline._apply_session_config("bench-session")

    assert "vector_backend" not in written["config"]
    assert "vector_backend_config" not in written["config"]
    # The harness names one collection per arm on purpose; that stays.
    assert written["config"]["collection_name"] == "rag_bench_emb_c4c84ef1"
    assert written["config"]["enable_reranking"] is False
    assert written["trusted"] is True


def test_resolve_backend_never_builds_milvus_on_a_qdrant_stack(qdrant_env, monkeypatch):
    try:
        from lasagna.api import _collection_manager as cm
    except Exception as exc:  # noqa: BLE001 — intentionally broad: any missing
        # server dependency (e.g. sentence_transformers) means this box cannot
        # import the collection manager; CI has the full set and runs this test.
        pytest.skip(f"needs the full server dependency set: {exc}")
    from lasagna.core.config import RAGConfig as _FreshRAGConfig  # noqa: F401

    sid = _new_id()
    _poison_session_like_the_harness(sid)

    fresh = _FreshRAGConfig()
    assert fresh.vector_backend == "qdrant"
    monkeypatch.setattr(cm, "rag_config", fresh)

    built = []

    def _spy_get_backend(name, **kwargs):
        built.append(name)
        if name == "milvus":
            raise AssertionError("resolve_backend built a MilvusBackend on a Qdrant stack")
        raise AssertionError(f"unexpected backend build: {name}")

    monkeypatch.setattr(cm, "get_backend", _spy_get_backend)

    class _StubBackend:
        backend_name = "qdrant"

        def is_connected(self):
            return True

    manager = cm.CollectionManager(
        default_backend=_StubBackend(),
        default_backend_name="qdrant",
        default_backend_config={"url": "http://qdrant:6333"},
        embedding_manager=object(),
    )

    resolved = manager.resolve_backend(sid)

    assert isinstance(resolved, _StubBackend)
    assert "milvus" not in built
