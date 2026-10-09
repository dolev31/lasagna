"""The embedder components: contracts, prefixes and input types, API plumbing.

API clients are stubbed: each stub embeds a text as a fixed function of it and
records what it was sent, so the tests can check the request a real client
would have received. No network, no key.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.core.config import rag_config
from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness
from lasagna.infrastructure.embedding import manager as manager_module
from lasagna.infrastructure.embedding.embedders import (
    BGEM3Embedder,
    CohereEmbedder,
    OpenAIEmbedder,
    SentenceTransformersEmbedder,
    VoyageEmbedder,
)
from lasagna.infrastructure.embedding.manager import _resolve_prefixes


def _vector(text, dim=3):
    return [float(len(text)), float(text.count("a")) + 1.0, 2.0] + [0.0] * (dim - 3)


class _Stub:
    """Embeds at the size the model would return: native, or as requested."""

    def __init__(self, native=3):
        self.calls = []
        self.native = native

    def _dim(self, kwargs):
        return kwargs.get("dimensions") or kwargs.get("output_dimension") or self.native


class OpenAIStub(_Stub):
    @property
    def embeddings(self):
        return self

    def create(self, model, input, **kwargs):
        self.calls.append({"model": model, "input": list(input), **kwargs})
        dim = self._dim(kwargs)
        items = [SimpleNamespace(index=i, embedding=_vector(t, dim)) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(items)))  # the API does not promise order


class CohereStub(_Stub):
    def embed(self, texts, model, input_type, embedding_types, **kwargs):
        self.calls.append(
            {"texts": list(texts), "model": model, "input_type": input_type, **kwargs}
        )
        dim = self._dim(kwargs)
        return SimpleNamespace(embeddings=SimpleNamespace(float_=[_vector(t, dim) for t in texts]))


class VoyageStub(_Stub):
    def embed(self, texts, model, input_type, truncation, **kwargs):
        self.calls.append(
            {"texts": list(texts), "model": model, "input_type": input_type, **kwargs}
        )
        return SimpleNamespace(embeddings=[_vector(t, self._dim(kwargs)) for t in texts])


API = [(OpenAIEmbedder, OpenAIStub), (CohereEmbedder, CohereStub), (VoyageEmbedder, VoyageStub)]


def _arm(**overrides):
    return rag_config.with_query_overrides(overrides)


class TestAPIEmbedders:
    @pytest.mark.parametrize("cls,stub", API)
    def test_the_embedder_passes_the_stage_contract(self, cls, stub):
        native = cls.native_dims[cls.default_model]
        report = ComponentTestHarness().validate(cls(client=stub(native)))
        assert report.passed, report.summary()

    def test_a_size_other_than_the_listed_one_is_believed_and_reported(self):
        embedder = VoyageEmbedder(client=VoyageStub(native=7))
        assert embedder.encode(["x"]).shape == (1, 7)
        assert embedder.dimension == 7

    @pytest.mark.parametrize("cls,stub", API)
    def test_vectors_come_back_in_input_order_and_unit_length(self, cls, stub):
        out = cls(client=stub()).encode(["a", "bbbb", "aa"])
        expected = np.array([_vector(t) for t in ["a", "bbbb", "aa"]], dtype=np.float32)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        assert np.allclose(out, expected)

    def test_cohere_and_voyage_embed_queries_and_documents_differently(self):
        cohere, voyage = CohereStub(), VoyageStub()
        for embedder in (CohereEmbedder(client=cohere), VoyageEmbedder(client=voyage)):
            embedder.encode(["a passage"])
            embedder.encode_query("a question")
        assert [c["input_type"] for c in cohere.calls] == ["search_document", "search_query"]
        assert [c["input_type"] for c in voyage.calls] == ["document", "query"]

    def test_requests_are_batched_at_the_providers_limit(self):
        stub = CohereStub()
        out = CohereEmbedder(client=stub).encode([f"t{i}" for i in range(200)])
        assert [len(c["texts"]) for c in stub.calls] == [96, 96, 8]
        assert out.shape == (200, 3)

    def test_the_truncation_setting_becomes_the_apis_dimension(self):
        stub = OpenAIStub()
        embedder = OpenAIEmbedder(config=_arm(embedding_truncate_dim=256), client=stub)
        embedder.encode(["x"])
        assert stub.calls[0]["dimensions"] == 256
        assert embedder.dimension == 256

    @pytest.mark.parametrize(
        "cls,model",
        [
            (OpenAIEmbedder, "text-embedding-ada-002"),
            (CohereEmbedder, "embed-english-v3.0"),
            (VoyageEmbedder, "voyage-3"),
        ],
    )
    def test_a_dimension_is_refused_for_a_fixed_size_model(self, cls, model):
        with pytest.raises(ValueError, match="fixed output size"):
            cls(model=model, dimensions=256, client=object())

    def test_a_provider_model_in_the_settings_is_used_and_a_hub_id_is_not(self):
        large = OpenAIEmbedder(config=_arm(embedding_model="text-embedding-3-large"), client=1)
        hub = OpenAIEmbedder(config=_arm(embedding_model="BAAI/bge-m3"), client=1)
        assert large.model_name == "text-embedding-3-large" and large.dimension == 3072
        assert hub.model_name == "text-embedding-3-small"

    def test_an_empty_text_is_sent_as_a_space(self):
        stub = OpenAIStub()
        OpenAIEmbedder(client=stub).encode(["", "x"])
        assert stub.calls[0]["input"] == [" ", "x"]

    @pytest.mark.parametrize(
        "cls,var", [(OpenAIEmbedder, "OPENAI_API_KEY"), (VoyageEmbedder, "VOYAGE_API_KEY")]
    )
    def test_a_missing_key_is_named(self, cls, var, monkeypatch):
        for v in cls.key_vars:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setitem(sys.modules, cls.package, SimpleNamespace(OpenAI=None, Client=None))
        with pytest.raises(RuntimeError, match=var):
            cls()

    @pytest.mark.parametrize("cls", [OpenAIEmbedder, CohereEmbedder, VoyageEmbedder])
    def test_a_missing_package_names_the_install(self, cls, monkeypatch):
        monkeypatch.setitem(sys.modules, cls.package, None)
        with pytest.raises(ImportError, match=f"pip install {cls.pip_name}"):
            cls()


class _Manager:
    """Stands in for EmbeddingManager: prefixes applied, as it applies them."""

    model_name = "stub-model"
    is_loaded = True
    model = object()
    dimension = 3
    backend_name = "sentence-transformers"
    query_prefix = "query: "

    def encode_documents_batch(self, texts, show_progress=True):
        return [np.asarray(_vector(t)) for t in texts]

    def encode_query(self, query):
        return np.asarray(_vector(self.query_prefix + query))

    def encode_queries_batch(self, queries):
        return [self.encode_query(q) for q in queries]


class TestLocalEmbedders:
    def test_the_sentence_transformers_embedder_passes_the_stage_contract(self):
        report = ComponentTestHarness().validate(
            SentenceTransformersEmbedder(embedding_manager=_Manager())
        )
        assert report.passed, report.summary()

    def test_the_model_named_in_the_settings_is_the_one_reported(self):
        embedder = SentenceTransformersEmbedder(config=_arm(embedding_model="intfloat/e5-small-v2"))
        assert embedder.model_name == "intfloat/e5-small-v2"

    def test_the_pipelines_loaded_model_is_reused_not_replaced(self, monkeypatch):
        config = _arm(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        singleton = SimpleNamespace(signature=manager_module._config_signature(config))
        monkeypatch.setattr(manager_module, "_embedding_manager", singleton)
        assert SentenceTransformersEmbedder(config=config).manager is singleton

        other = SentenceTransformersEmbedder(config=_arm(embedding_model="intfloat/e5-small-v2"))
        assert other.manager is not singleton
        assert manager_module._embedding_manager is singleton

    def test_bge_m3_sparse_output_names_its_dependency(self):
        with pytest.raises(ImportError, match="FlagEmbedding"):
            BGEM3Embedder(embedding_manager=_Manager()).encode_sparse(["text"])

    def test_bge_m3_always_names_bge_m3(self):
        assert BGEM3Embedder(config=_arm(embedding_model="intfloat/e5-small-v2")).model_name == (
            "BAAI/bge-m3"
        )

    @pytest.mark.parametrize(
        "name", ["sentence_transformers", "bge_m3", "openai", "cohere", "voyage"]
    )
    def test_every_embedder_is_registered_with_a_citation(self, name):
        entry = next(e for e in registry.list_embedders() if e["name"] == name)
        assert entry["implements_contract"] and entry.get("paper")


@pytest.mark.parametrize(
    "model,query,document",
    [
        ("intfloat/e5-large-v2", "query: ", "passage: "),
        ("intfloat/multilingual-e5-large-instruct", "Instruct: ", ""),
        ("intfloat/e5-mistral-7b-instruct", "Instruct: ", ""),
        ("Alibaba-NLP/gte-Qwen2-7B-instruct", "Instruct: ", ""),
        ("Alibaba-NLP/gte-large-en-v1.5", "", ""),
        ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
        ("BAAI/bge-large-zh-v1.5", "为这个句子生成表示以用于检索相关文章：", ""),
        ("BAAI/bge-m3", "", ""),
        ("nomic-ai/nomic-embed-text-v1.5", "search_query: ", "search_document: "),
        ("mixedbread-ai/mxbai-embed-large-v1", "Represent this sentence", ""),
        ("Snowflake/snowflake-arctic-embed-m-v1.5", "Represent this sentence", ""),
        ("Snowflake/snowflake-arctic-embed-l-v2.0", "query: ", ""),
        ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    ],
)
def test_each_family_gets_its_model_cards_prefixes(model, query, document):
    prefixes = _resolve_prefixes(model)
    assert prefixes["query"].startswith(query) and (query or prefixes["query"] == "")
    assert prefixes["document"] == document


def test_qwen3_embedding_uses_its_own_instruction_template():
    assert _resolve_prefixes("Qwen/Qwen3-Embedding-0.6B")["query"] == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    )


def _cached(repo):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


BGE_SMALL = "BAAI/bge-small-en-v1.5"


@pytest.mark.skipif(not _cached(BGE_SMALL), reason=f"{BGE_SMALL} is not cached")
def test_a_real_bge_model_embeds_queries_with_its_instruction():
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformersEmbedder(
        config=_arm(embedding_model=BGE_SMALL, use_gpu=False, enable_embedding_cache=False)
    )
    model = SentenceTransformer(BGE_SMALL, device="cpu")
    instruction = "Represent this sentence for searching relevant passages: "
    expected_q = model.encode(instruction + "what is rag", normalize_embeddings=True)
    expected_d = model.encode("RAG retrieves passages.", normalize_embeddings=True)
    assert np.allclose(embedder.encode_query("what is rag"), expected_q, atol=1e-5)
    assert np.allclose(embedder.encode(["RAG retrieves passages."])[0], expected_d, atol=1e-5)
    assert embedder.dimension == 384 and embedder.model_name == BGE_SMALL
    assert ComponentTestHarness().validate(embedder).passed
