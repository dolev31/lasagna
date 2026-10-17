"""Uploading a file the session already holds returns that document, once.

The README's quickstart says "same session on every run", and every rerun
uploaded paper.pdf again: `lasagna status` listed two copies and search
returned the same passage once per copy. A retriever that hands back one
chunk twice also reached the caller twice.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.api import _search_orchestrator as orch_mod
from lasagna.api._document_pipeline import DocumentPipeline
from lasagna.core.base_components import BaseRetriever, ScoredDocument
from lasagna.core.config import rag_config
from tests._paths import PACKAGE_DIR

QUICKSTART = """
import asyncio, sys
from lasagna.sdk import KnowledgeClient, config_for_speed

async def main():
    client = KnowledgeClient(session_id="quickstart", config=config_for_speed())
    doc = await client.upload("paper.md", allow_duplicates="--force" in sys.argv)
    await client.documents.wait_for_processing(doc.document_id)
    print("duplicate_of:", doc.duplicate_of)
    print("message:", doc.message)
    results = await client.search_documents("What did the ablation show?")
    print("chunks:", [r.chunk.chunk_id for r in results.results])
    await client.close()

asyncio.run(main())
"""


def _minilm_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
    return isinstance(hit, str)


@pytest.mark.skipif(not _minilm_cached(), reason="all-MiniLM-L6-v2 is not cached")
def test_rerunning_the_quickstart_keeps_one_copy(tmp_path):
    (tmp_path / "paper.md").write_text(
        "# Ablation\n\nThe ablation showed that removing the reranker lowered NDCG@10.\n"
    )
    (tmp_path / "quickstart.py").write_text(QUICKSTART)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        HOME=str(tmp_path / "home"),
        PYTHONPATH=str(PACKAGE_DIR.parent),
        HF_HUB_OFFLINE="1",
        COLUMNS="300",
    )

    def run(*argv):
        done = subprocess.run(
            [sys.executable, *argv], cwd=tmp_path, env=env, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stdout + done.stderr[-3000:]
        return done.stdout

    first = run("quickstart.py")
    assert "duplicate_of: None" in first
    second = run("quickstart.py")
    original = first.split("chunks: ['")[1].split("_chunk_")[0]
    assert f"duplicate_of: {original}" in second
    assert "(same content), so it was not uploaded again" in second
    assert second.count("_chunk_") == 1, second

    cli = run("-m", "lasagna", "upload", "paper.md", "-s", "quickstart")
    assert f"Already uploaded: paper.md has the same content as {original}" in cli
    assert "--force" in cli
    status = run("-m", "lasagna", "status", "-s", "quickstart")
    assert "Documents: 1" in status, status

    forced = run("quickstart.py", "--force")
    assert "duplicate_of: None" in forced
    assert run("-m", "lasagna", "status", "-s", "quickstart").count("paper.md") == 2


def _pipeline():
    pipeline = DocumentPipeline.__new__(DocumentPipeline)
    pipeline._storage = SimpleNamespace(retrieve=lambda uri: open(uri, "rb"))
    pipeline._get_document_metadata = lambda doc_id: SimpleNamespace(document_id=doc_id)
    return pipeline


@pytest.fixture
def stored(tmp_path, monkeypatch):
    """A session holding one ready and one failed copy of a.txt, and b.txt."""
    files = {}
    for name, text in (("a.txt", "same words"), ("b.txt", "other text")):
        files[name] = tmp_path / name
        files[name].write_text(text)
    rows = [
        {
            "document_id": "failed",
            "status": "error",
            "file_size": 10,
            "file_path": str(files["a.txt"]),
        },
        {
            "document_id": "ready",
            "status": "ready",
            "file_size": 10,
            "file_path": str(files["a.txt"]),
        },
        {
            "document_id": "other",
            "status": "ready",
            "file_size": 10,
            "file_path": str(files["b.txt"]),
        },
    ]
    db = SimpleNamespace(get_documents=lambda session_id: rows)
    monkeypatch.setattr("lasagna.core.database.get_rag_database", lambda: db)
    return files


def test_the_same_bytes_match_the_ready_copy_and_not_the_failed_one(tmp_path, stored):
    upload = tmp_path / "again.txt"
    upload.write_text("same words")
    found = _pipeline()._find_same_content(str(upload), "s")
    assert found.document_id == "ready"


def test_other_bytes_of_the_same_size_do_not_match(tmp_path, stored):
    upload = tmp_path / "new.txt"
    upload.write_text("new words!")
    assert _pipeline()._find_same_content(str(upload), "s") is None


def test_an_explicit_document_id_is_never_deduplicated(tmp_path, monkeypatch):
    """A reindex and a benchmark corpus name their ids: the caller decides."""
    pipeline = DocumentPipeline.__new__(DocumentPipeline)

    def never(*a):
        raise AssertionError("looked for a duplicate")

    pipeline._find_same_content = never
    pipeline._get_session_config = lambda sid: (_ for _ in ()).throw(RuntimeError("reached"))
    with pytest.raises(RuntimeError, match="reached"):
        pipeline.upload_document(str(tmp_path / "x"), "s", document_id="doc_x")
    with pytest.raises(RuntimeError, match="reached"):
        pipeline.upload_document(str(tmp_path / "x"), "s", _chunker_override=object())


class _Retriever(BaseRetriever):
    _forwards_session_config = True

    def retrieve(self, query, collection, top_k=5, **kwargs):
        return [
            ScoredDocument(content=c, score=s, document_id="d", chunk_id=c)
            for c, s in (("c0", 0.9), ("c1", 0.8), ("c0", 0.7))
        ]


class _Backend:
    backend_name = "fake"


def test_search_returns_a_chunk_once(monkeypatch):
    from lasagna.vector_backends.base import VectorStoreAdapter

    _Backend.session_filter = VectorStoreAdapter.session_filter
    monkeypatch.setattr(orch_mod, "SearchResult", lambda **kw: SimpleNamespace(**kw))
    backend = _Backend()
    o = orch_mod.SearchOrchestrator.__new__(orch_mod.SearchOrchestrator)
    o.embedding_manager = SimpleNamespace(
        get_cache_stats=lambda: {}, encode_query=lambda q: np.ones(3)
    )
    o.hybrid_searcher = None
    o.reranker = o.query_enhancer = o._auto_merger = None
    o._collection_manager = SimpleNamespace(resolve_backend=lambda sid: backend)
    o._document_pipeline = SimpleNamespace(
        get_documents_metadata_batch=lambda ids: {i: SimpleNamespace(document_id=i) for i in ids}
    )
    o._default_backend = backend
    o._get_session_config = lambda sid: rag_config
    o._session_embedder = lambda config: o.embedding_manager
    o.backend = backend
    speed = dict(
        enable_hybrid_search=False,
        enable_reranking=False,
        enable_query_routing=False,
        enable_query_classification=False,
        enable_search_cache=False,
        enable_hyde=False,
        enable_mmr=False,
        enable_context_reordering=False,
        enable_hierarchical=False,
        enable_rag_fusion=False,
        enable_query_expansion=False,
    )
    out = o.search(
        "q",
        "s",
        top_k=5,
        min_score=None,
        _retriever_override=_Retriever(),
        query_config_overrides=speed,
    )
    assert [r.chunk.chunk_id for r in out] == ["c0", "c1"]
