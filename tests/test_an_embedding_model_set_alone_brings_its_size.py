"""Setting only ``embedding_model`` gives the collection that model's size.

``KnowledgeConfig(embedding_model="BAAI/bge-small-en-v1.5")`` kept the settings
file's ``embedding_dim`` of 1024: the collection was made at 1024, Milvus
refused the 384-dimension vectors ("expected dim 1024, got 384") with a
pymilvus traceback, the document failed, and search returned ``[]`` without a
word. The same held for ``RAG_EMBEDDING_MODEL`` alone.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from lasagna.core import embedding_dims
from lasagna.core.config import rag_config
from tests._paths import PACKAGE_DIR


def _config(env=None, query=None):
    config = rag_config.with_session(None)
    config._env_overrides = dict(env or {})
    return config.with_query_overrides(query) if query else config


def test_a_known_model_has_its_size():
    assert embedding_dims.known_embedding_dim("BAAI/bge-small-en-v1.5") == 384
    assert embedding_dims.known_embedding_dim("text-embedding-3-large") == 3072
    assert embedding_dims.known_embedding_dim("acme/never-heard-of-it") is None


def test_a_model_folder_gives_its_size(tmp_path):
    (tmp_path / "1_Pooling").mkdir()
    (tmp_path / "1_Pooling" / "config.json").write_text(
        json.dumps({"word_embedding_dimension": 512})
    )
    modules = [{"idx": 1, "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}]
    (tmp_path / "modules.json").write_text(json.dumps(modules))
    assert embedding_dims.known_embedding_dim(str(tmp_path)) == 512
    # A Dense layer after pooling decides the output size.
    (tmp_path / "2_Dense").mkdir()
    (tmp_path / "2_Dense" / "config.json").write_text(json.dumps({"out_features": 256}))
    modules.append({"idx": 2, "path": "2_Dense", "type": "sentence_transformers.models.Dense"})
    (tmp_path / "modules.json").write_text(json.dumps(modules))
    embedding_dims._read.clear()
    assert embedding_dims.known_embedding_dim(str(tmp_path)) == 256


def test_a_model_from_the_environment_brings_its_size():
    assert _config().embedding_dim == 1024  # the settings file's model and size
    config = _config(env={"embedding_model": "BAAI/bge-small-en-v1.5"})
    assert config.embedding_dim == 384
    assert config.collection_name.endswith("__384")
    assert config.embedding_size_unknown() is None


def test_a_size_set_with_the_model_is_kept():
    config = _config(env={"embedding_model": "BAAI/bge-small-en-v1.5", "embedding_dim": 192})
    assert config.embedding_dim == 192


def test_a_session_or_query_model_brings_its_size_over_an_environment_size():
    config = _config(env={"embedding_dim": 768}, query={"embedding_model": "all-MiniLM-L6-v2"})
    assert config.embedding_dim == 384


def test_an_unreadable_size_is_an_error_before_anything_is_stored(tmp_path):
    from lasagna.api._document_pipeline import DocumentPipeline

    config = _config(env={"embedding_model": "acme/never-heard-of-it"})
    problem = config.embedding_size_unknown()
    assert "embedding_dim was not set with it" in problem
    assert "RAG_EMBEDDING_DIM" in problem and "hf download acme/never-heard-of-it" in problem

    pipeline = DocumentPipeline.__new__(DocumentPipeline)
    pipeline._find_same_content = lambda *a: None
    pipeline._get_session_config = lambda session_id: config

    def stored(*a, **k):
        raise AssertionError("the file was stored")

    pipeline._store_file = stored
    (tmp_path / "doc.txt").write_text("text")
    with pytest.raises(ValueError, match="embedding_dim was not set with it"):
        pipeline.upload_document(str(tmp_path / "doc.txt"), "s")


UPLOAD = """
import asyncio
from lasagna.sdk import KnowledgeClient, KnowledgeConfig

async def main():
    config = KnowledgeConfig(embedding_model="BAAI/bge-small-en-v1.5", enable_reranking=False)
    client = KnowledgeClient(session_id="e10", config=config)
    doc = await client.upload("paper.md")
    done = await client.documents.wait_for_processing(doc.document_id)
    print("status:", done.status.value)
    results = await client.search_documents("What did the ablation show?")
    print("results:", len(results.results))
    await client.close()

asyncio.run(main())
"""


def _cached(model: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model, "config.json"), str)


@pytest.mark.skipif(not _cached("BAAI/bge-small-en-v1.5"), reason="bge-small is not cached")
def test_the_sdk_indexes_and_finds_with_only_the_model_set(tmp_path):
    (tmp_path / "paper.md").write_text(
        "# Ablation\n\nThe ablation showed that removing the reranker lowered NDCG@10.\n"
    )
    (tmp_path / "upload.py").write_text(UPLOAD)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(PACKAGE_DIR.parent), HF_HUB_OFFLINE="1")
    run = subprocess.run(
        [sys.executable, "upload.py"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr[-3000:]
    assert "status: ready" in run.stdout and "results: 1" in run.stdout, run.stdout
    assert "expected dim" not in run.stderr
