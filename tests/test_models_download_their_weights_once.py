"""A first run downloads each model's weights once.

BAAI/bge-m3 ships only ``pytorch_model.bin``. transformers loaded it and then
started a background thread that downloaded ``model.safetensors`` from the
repository's conversion pull request: 2.3 GB that no later load reads, since
loads resolve the main branch and read the ``.bin`` again. The release gate's
cold start measured 4.6 GB for the default configuration, half of it that copy,
and the first ``lasagna search`` did not exit until the copy had finished.
transformers skips the copy when ``DISABLE_SAFETENSORS_CONVERSION`` is set
while it loads, so every model Lasagna loads now loads that way.
"""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.core import model_downloads
from lasagna.core.config import rag_config

VAR = model_downloads.NO_CONVERSION_VAR


@pytest.fixture
def seen(monkeypatch):
    """What each fake load saw of the switch; the switch is unset to start."""
    monkeypatch.delenv(VAR, raising=False)
    return []


def _record(seen):
    seen.append(os.environ.get(VAR))


def test_transformers_still_reads_the_switch():
    """If transformers renames it, the copy comes back without any error."""
    from transformers import modeling_utils

    assert f'is_env_variable_true("{VAR}")' in inspect.getsource(modeling_utils)


def test_the_embedding_model_loads_without_the_copy(monkeypatch, seen):
    from lasagna.infrastructure.embedding import manager as manager_mod

    class FakeST:
        def __init__(self, path, device=None):
            _record(seen)

        def encode(self, texts, **kwargs):
            return np.ones(4, dtype=np.float32)

    monkeypatch.setattr(manager_mod, "SentenceTransformer", FakeST)
    monkeypatch.setattr(manager_mod, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(manager_mod.model_downloads, "is_cached", lambda model: True)
    config = rag_config.with_query_overrides(
        {
            "embedding_model": "BAAI/bge-m3",
            "embedding_backend": "sentence-transformers",
            "embedding_dim": 4,
            "enable_embedding_cache": False,
            "use_gpu": False,
        }
    )
    manager_mod.EmbeddingManager(config).encode_query("hello")
    assert seen == ["1"]
    assert VAR not in os.environ


def test_the_fallback_embedder_loads_without_the_copy(monkeypatch, seen):
    from lasagna.infrastructure.embedding import manager as manager_mod

    class FakeST:
        def __init__(self, path, device=None):
            if path == "t-primary":
                raise OSError("t-primary is not on the Hub")
            _record(seen)

        def encode(self, texts, **kwargs):
            return np.ones(4, dtype=np.float32)

    monkeypatch.setattr(manager_mod, "SentenceTransformer", FakeST)
    monkeypatch.setattr(manager_mod, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(manager_mod.model_downloads, "is_cached", lambda model: True)
    config = rag_config.with_query_overrides(
        {
            "embedding_model": "t-primary",
            "embedding_model_fallback": "t-fallback",
            "embedding_backend": "sentence-transformers",
            "embedding_dim": 4,
            "enable_embedding_cache": False,
            "use_gpu": False,
        }
    )
    manager = manager_mod.EmbeddingManager(config)
    manager.encode_query("hello")
    assert manager.model_name == "t-fallback" and seen == ["1"]


def test_flag_embedding_loads_without_the_copy(monkeypatch, seen):
    from lasagna.infrastructure.embedding import manager as manager_mod

    class FakeFlag:
        def __init__(self, name, **kwargs):
            _record(seen)

        def encode(self, texts, **kwargs):
            return {"dense_vecs": np.ones((len(texts), 4), dtype=np.float32)}

    monkeypatch.setattr(manager_mod, "BGEM3FlagModel", FakeFlag, raising=False)
    monkeypatch.setattr(manager_mod, "FLAG_EMBEDDING_AVAILABLE", True)
    monkeypatch.setattr(manager_mod.model_downloads, "is_cached", lambda model: True)
    config = rag_config.with_query_overrides(
        {
            "embedding_model": "BAAI/bge-m3",
            "embedding_backend": "flag-embedding",
            "embedding_dim": 4,
            "enable_embedding_cache": False,
            "use_gpu": False,
        }
    )
    manager_mod.EmbeddingManager(config).encode_query("hello")
    assert seen == ["1"]


def test_the_cross_encoder_loads_without_the_copy(monkeypatch, seen):
    from lasagna.retrieval.search import ranker

    class FakeCrossEncoder:
        def __init__(self, name, device=None, **kwargs):
            _record(seen)

        def predict(self, pairs, **kwargs):
            return np.zeros(len(pairs))

    monkeypatch.setattr(ranker, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(ranker, "CROSS_ENCODER_AVAILABLE", True)
    session = SimpleNamespace(
        enable_reranking=True,
        reranker_model="mixedbread-ai/mxbai-rerank-large-v1",
        reranker_batch_size=8,
        rerank_top_k=5,
        enable_mmr=False,
    )
    hit = [{"id": "a", "content": "a", "score": 1.0, "metadata": {}}]
    ranker.CrossEncoderReranker().rerank("q", hit, top_k=1, config=session)
    assert seen == ["1"]
    assert VAR not in os.environ


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self


def _fake_from_pretrained(seen, result):
    def from_pretrained(name, **kwargs):
        _record(seen)
        return result

    return staticmethod(from_pretrained)


def test_splade_loads_without_the_copy(monkeypatch, seen):
    import transformers

    from lasagna.retrieval.search.hybrid import SPLADESparseEncoder

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda name, **kw: object())
    )
    monkeypatch.setattr(
        transformers.AutoModelForMaskedLM,
        "from_pretrained",
        _fake_from_pretrained(seen, (_FakeModel(), {"missing_keys": []})),
    )
    SPLADESparseEncoder("naver/splade-cocondenser-ensembledistil")._ensure_loaded()
    assert seen == ["1"]


def test_monot5_loads_without_the_copy(monkeypatch, seen):
    import transformers

    from lasagna.retrieval.search import rerankers

    tokenizer = SimpleNamespace(unk_token_id=2, convert_tokens_to_ids=lambda tokens: [0, 1])
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda name, **kw: tokenizer)
    )
    monkeypatch.setattr(
        transformers.AutoModelForSeq2SeqLM,
        "from_pretrained",
        _fake_from_pretrained(seen, _FakeModel()),
    )
    rerankers._MonoT5("castorini/monot5-base-msmarco-10k")
    assert seen == ["1"]


def test_colbert_loads_without_the_copy(monkeypatch, seen):
    import transformers

    from lasagna.retrieval.search import ranker

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda name, **kw: object())
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", _fake_from_pretrained(seen, _FakeModel())
    )
    ranker.ColBERTReranker()._ensure_model()
    assert seen == ["1"]


def test_the_answer_similarity_model_loads_without_the_copy(monkeypatch, seen):
    import sentence_transformers

    from lasagna.knowledge_evaluation_sdk import answer_metrics

    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", lambda name, device=None: _record(seen)
    )
    answer_metrics.load_similarity_encoder.__wrapped__("sentence-transformers/all-MiniLM-L6-v2")
    assert seen == ["1"]


def test_a_value_the_environment_already_has_is_left_alone(monkeypatch):
    monkeypatch.setenv(VAR, "0")
    with model_downloads.one_weights_copy():
        assert os.environ[VAR] == "0"
    assert os.environ[VAR] == "0"


def test_overlapping_loads_keep_the_switch_until_the_last_one_ends(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    first = model_downloads.one_weights_copy()
    second = model_downloads.one_weights_copy()
    first.__enter__()
    second.__enter__()
    first.__exit__(None, None, None)
    assert os.environ.get(VAR) == "1"
    second.__exit__(None, None, None)
    assert VAR not in os.environ


def test_a_load_that_fails_still_removes_the_switch(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)
    with pytest.raises(OSError):
        with model_downloads.one_weights_copy():
            raise OSError("no such model")
    assert VAR not in os.environ
