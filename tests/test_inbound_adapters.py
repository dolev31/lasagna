"""A LangChain or LlamaIndex component becomes a pipeline stage in one call.

Before ``lasagna.from_langchain`` and ``lasagna.from_llamaindex``, a reranker or
embedder someone already had in LangChain had to be rewritten against
``BaseReranker`` or ``BaseEmbedder`` before a sweep could name it, and the
rewrite was where ids, scores and the query-versus-passage split got lost.
These tests hold the adapters to what the stage contracts require, with fakes
that follow each library's interface (no model, no network).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever as LCBaseRetriever

import lasagna
from lasagna.core.base_components import (
    BaseEmbedder,
    BaseReranker,
    BaseRetriever,
    ScoredDocument,
)
from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness


def _docs() -> List[ScoredDocument]:
    return [
        ScoredDocument(content="alpha beta gamma", score=0.1, document_id="d1", chunk_id="c1"),
        ScoredDocument(content="delta", score=0.9, document_id="d2", chunk_id="c2"),
        ScoredDocument(content="alpha beta", score=0.5, document_id="d3", chunk_id="c3"),
    ]


class OverlapCompressor(BaseDocumentCompressor):
    """Orders by word overlap with the query and reports it as relevance_score,
    the way CohereRerank and CrossEncoderReranker do."""

    top_n: int = 10

    def compress_documents(self, documents, query, callbacks=None):
        words = set(query.split())
        scored = []
        for doc in documents:
            copy = Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            copy.metadata["relevance_score"] = float(len(words & set(doc.page_content.split())))
            scored.append(copy)
        scored.sort(key=lambda d: d.metadata["relevance_score"], reverse=True)
        return scored[: self.top_n]


class TestALangChainCompressorIsAReranker:
    def test_it_passes_the_reranker_contract(self):
        name = lasagna.from_langchain(OverlapCompressor(), name="t_lc_overlap")
        component = registry.get("reranker", name)
        assert isinstance(component, BaseReranker)
        report = ComponentTestHarness().validate(component)
        assert report.passed, report.failures

    def test_its_order_scores_and_ids_come_through(self):
        name = lasagna.from_langchain(OverlapCompressor(), name="t_lc_overlap_ids")
        out = registry.get("reranker", name).rerank("alpha beta", _docs(), top_k=2)
        assert [(d.chunk_id, d.document_id, d.score) for d in out] == [
            ("c1", "d1", 2.0),
            ("c3", "d3", 2.0),
        ]
        assert all("_lasagna_position" not in d.metadata for d in out)

    def test_documents_rebuilt_without_metadata_are_matched_by_text(self):
        class Rebuilds:
            def compress_documents(self, documents, query, callbacks=None):
                return [Document(page_content=d.page_content) for d in reversed(documents)]

        name = lasagna.from_langchain(Rebuilds(), name="t_lc_rebuilds")
        out = registry.get("reranker", name).rerank("q", _docs(), top_k=3)
        assert [d.chunk_id for d in out] == ["c3", "c2", "c1"]
        # No score was reported, so the order returned is the ranking.
        assert [d.score for d in out] == [1.0, 0.5, pytest.approx(1 / 3)]

    def test_a_score_the_passage_already_carried_is_not_taken_for_the_compressors(self):
        class Reverses:
            def compress_documents(self, documents, query, callbacks=None):
                return list(reversed(documents))

        docs = _docs()
        for d in docs:
            d.metadata["score"] = 99.0
        name = lasagna.from_langchain(Reverses(), name="t_lc_reverses")
        out = registry.get("reranker", name).rerank("q", docs, top_k=3)
        assert [d.chunk_id for d in out] == ["c3", "c2", "c1"]
        assert [d.score for d in out] == [1.0, 0.5, pytest.approx(1 / 3)]

    def test_a_cap_below_the_requested_depth_is_reported(self):
        from loguru import logger

        seen: list = []
        sink = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
        try:
            name = lasagna.from_langchain(OverlapCompressor(top_n=1), name="t_lc_capped")
            out = registry.get("reranker", name).rerank("alpha", _docs(), top_k=3)
        finally:
            logger.remove(sink)
        assert len(out) == 1
        assert any("top_n=1" in m and "at least 3" in m for m in seen), seen


class CountingEmbeddings(Embeddings):
    """Queries and passages embed differently, as asymmetric models (E5, BGE) do."""

    def __init__(self, model: str = "fake-small"):
        self.model = model
        self.calls: list = []

    def embed_documents(self, texts):
        self.calls.append(("documents", list(texts)))
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def embed_query(self, text):
        self.calls.append(("query", text))
        return [float(len(text)), 0.0, 1.0]


class TestALangChainEmbedderKeepsQueriesAndPassagesApart:
    def test_queries_use_embed_query_and_passages_embed_documents(self):
        fake = CountingEmbeddings()
        name = lasagna.from_langchain(fake, name="t_lc_embed")
        component = registry.get("embedder", name)
        assert isinstance(component, BaseEmbedder)

        fake.calls.clear()
        query = component.encode_query("cats")
        passages = component.encode(["ab", "abc"])
        assert fake.calls == [("query", "cats"), ("documents", ["ab", "abc"])]
        assert query.tolist() == [4.0, 0.0, 1.0]
        assert passages.shape == (2, 3) and passages.dtype == np.float32
        assert component.dimension == 3

    def test_it_passes_the_embedder_contract(self):
        name = lasagna.from_langchain(CountingEmbeddings(), name="t_lc_embed_contract")
        report = ComponentTestHarness().validate(registry.get("embedder", name))
        assert report.passed, report.failures


class TestALangChainRetrieverBringsItsOwnIndex:
    class Fixed(LCBaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager=None):
            return [
                Document(page_content="first", metadata={"corpus_id": "C9", "score": 0.7}),
                Document(page_content="second", metadata={"doc_id": "D2"}),
                Document(page_content="third", metadata={"_id": "U3"}),
                Document(page_content="no id at all"),
            ]

    def test_ids_come_from_the_metadata_in_order_then_a_text_hash(self):
        name = lasagna.from_langchain(self.Fixed(), name="t_lc_retriever", id_key="corpus_id")
        component = registry.get("retriever", name)
        assert isinstance(component, BaseRetriever)
        out = component.retrieve("q", collection="ignored", top_k=10)
        ids = [d.document_id for d in out]
        assert ids[:3] == ["C9", "D2", "U3"]
        assert ids[3].startswith("sha256:") and len(ids[3]) == len("sha256:") + 16
        # Stable: the same text hashes to the same id on the next call.
        assert component.retrieve("q", collection="x", top_k=10)[3].document_id == ids[3]
        assert out[0].score == 0.7, "a reported score is kept"
        assert out[1].score == 0.5, "an unscored document gets 1/rank"
        assert len({d.chunk_id for d in out}) == 4
        assert len(component.retrieve("q", collection="x", top_k=2)) == 2


class TestLlamaIndexComponents:
    """Duck-typed, as LlamaIndex is an optional extra."""

    class ReversePostprocessor:
        """Reverses the nodes and scores them by position, keeping each node."""

        def postprocess_nodes(self, nodes, query_bundle=None, query_str=None):
            out = list(reversed(nodes))
            for rank, item in enumerate(out):
                item.score = 10.0 - rank
            return out

    def test_a_postprocessor_is_a_conforming_reranker(self):
        name = lasagna.from_llamaindex(self.ReversePostprocessor(), name="t_li_reverse")
        component = registry.get("reranker", name)
        report = ComponentTestHarness().validate(component)
        assert report.passed, report.failures
        out = component.rerank("q", _docs(), top_k=2)
        assert [(d.chunk_id, d.document_id, d.score) for d in out] == [
            ("c3", "d3", 10.0),
            ("c2", "d2", 9.0),
        ]

    def test_a_retriever_falls_back_to_the_source_document_id(self):
        class Node:
            def __init__(self, text, metadata, node_id, ref_doc_id):
                self.text, self.metadata = text, metadata
                self.node_id, self.ref_doc_id = node_id, ref_doc_id

            def get_content(self):
                return self.text

        class Scored:
            def __init__(self, node, score):
                self.node, self.score = node, score

        class Retriever:
            def retrieve(self, query):
                return [
                    Scored(Node("from metadata", {"doc_id": "M1"}, "n1", "R1"), 0.9),
                    Scored(Node("from the source", {}, "n2", "R2"), None),
                ]

        name = lasagna.from_llamaindex(Retriever(), name="t_li_retriever")
        out = registry.get("retriever", name).retrieve("q", collection="x", top_k=5)
        assert [(d.document_id, d.chunk_id, d.score) for d in out] == [
            ("M1", "n1", 0.9),
            ("R2", "n2", 0.5),
        ]


def test_the_listing_says_what_was_wrapped():
    class ProvenanceCompressor(OverlapCompressor):
        pass

    name = lasagna.from_langchain(ProvenanceCompressor())
    assert name == "langchain.ProvenanceCompressor"
    entry = next(e for e in registry.list_rerankers() if e["name"] == name)
    assert entry["description"] == "wrapped: langchain.ProvenanceCompressor"
    assert entry["implements_contract"] is True
    assert "wrapped: langchain.ProvenanceCompressor" in registry.get("reranker", name).model_name


def test_two_configurations_of_one_class_never_share_an_identity():
    # The identity keys the search cache and an embedder's corpus. Wrapping a
    # second model under the same name must not be served the first one's.
    lasagna.from_langchain(OverlapCompressor(top_n=5), name="t_lc_identity")
    first = registry.get("reranker", "t_lc_identity").model_name
    lasagna.from_langchain(OverlapCompressor(top_n=7), name="t_lc_identity")
    assert registry.get("reranker", "t_lc_identity").model_name != first

    from lasagna.knowledge_evaluation_sdk.experiment import _ingest_component_fingerprint

    lasagna.from_langchain(CountingEmbeddings("fake-small"), name="t_lc_identity_emb")
    fingerprint = _ingest_component_fingerprint("embedder", "t_lc_identity_emb")
    lasagna.from_langchain(CountingEmbeddings("fake-large"), name="t_lc_identity_emb")
    assert _ingest_component_fingerprint("embedder", "t_lc_identity_emb") != fingerprint


def test_a_sweep_arm_names_the_wrapped_reranker_and_its_manifest_shows_it():
    from lasagna.knowledge_evaluation_sdk.experiment import Experiment
    from lasagna.knowledge_evaluation_sdk.stages import stage_manifest

    class SweepCompressor(OverlapCompressor):
        pass

    name = lasagna.from_langchain(SweepCompressor())
    tasks = Experiment.from_dict(
        {
            "name": "wrapped reranker",
            "benchmarks": [{"name": "smoke"}],
            "tracks": ["retrieval"],
            "pipeline": {"base_profile": "speed", "rerankers": ["cross_encoder", name]},
        }
    ).generate_tasks()
    assert {t.config_overrides.get("reranker") for t in tasks} == {"cross_encoder", name}

    from lasagna.core.config import rag_config

    arm = next(t for t in tasks if t.config_overrides.get("reranker") == name)
    manifest = stage_manifest(rag_config, {"reranker": arm.config_overrides["reranker"]})
    assert manifest["reranker"]["component"] == "langchain.SweepCompressor"


class TestWhatCannotBeWrapped:
    def test_an_unsupported_object_gets_one_line_naming_what_it_has(self):
        class ChatModel:
            def invoke(self, prompt):
                return "hi"

        with pytest.raises(TypeError) as err:
            lasagna.from_langchain(ChatModel())
        message = str(err.value)
        assert "\n" not in message
        assert "Cannot wrap" in message and "ChatModel" in message
        assert "ChatModel has invoke." in message
        assert "compress_documents" in message and "embed_query" in message
        assert "@lasagna.<stage>" in message

    def test_the_other_librarys_component_is_pointed_at_its_adapter(self):
        with pytest.raises(TypeError, match="use lasagna.from_llamaindex"):
            lasagna.from_langchain(TestLlamaIndexComponents.ReversePostprocessor())

    def test_a_class_instead_of_an_object_is_refused_with_the_fix(self):
        with pytest.raises(TypeError, match=r"pass OverlapCompressor\(\.\.\.\)"):
            lasagna.from_langchain(OverlapCompressor)

    def test_a_built_in_name_is_refused(self):
        with pytest.raises(ValueError, match="built-in reranker"):
            lasagna.from_langchain(OverlapCompressor(), name="cross_encoder")


def test_a_sweep_file_can_name_a_langchain_object_where_it_is_defined(tmp_path):
    # `lasagna benchmark experiment sweep.yaml` runs no Python of the user's
    # before reading the sweep, so from_langchain cannot have been called. A
    # `module:attr` naming the object is wrapped the same way.
    import textwrap

    from lasagna.knowledge_evaluation_sdk.experiment import Experiment

    (tmp_path / "t_lc_sweep_ideas.py").write_text(
        textwrap.dedent(
            """
            from langchain_core.documents.compressor import BaseDocumentCompressor
            from langchain_core.embeddings import DeterministicFakeEmbedding


            class KeywordCompressor(BaseDocumentCompressor):
                def compress_documents(self, documents, query, callbacks=None):
                    words = set(query.split())
                    return sorted(
                        documents,
                        key=lambda d: len(words & set(d.page_content.split())),
                        reverse=True,
                    )


            compressor = KeywordCompressor()
            embeddings = DeterministicFakeEmbedding(size=8)
            """
        )
    )
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        textwrap.dedent(
            """
            name: wrapped by reference
            benchmarks: [{name: smoke}]
            tracks: [retrieval]
            pipeline:
              base_profile: speed
              rerankers: [cross_encoder, "t_lc_sweep_ideas:compressor"]
            """
        )
    )
    tasks = Experiment.from_file(str(sweep)).generate_tasks()
    assert {t.config_overrides.get("reranker") for t in tasks} == {
        "cross_encoder",
        "t_lc_sweep_ideas:compressor",
    }
    entry = next(e for e in registry.list_rerankers() if e["name"] == "t_lc_sweep_ideas:compressor")
    assert entry["description"] == "wrapped: langchain.KeywordCompressor"

    # Named under the wrong stage, it says which stage it fills.
    sweep.write_text(sweep.read_text().replace(":compressor", ":embeddings"))
    with pytest.raises(ValueError, match="wraps as an embedder, not a reranker"):
        Experiment.from_file(str(sweep)).generate_tasks()


def test_a_plain_function_is_still_a_function_component():
    # The library check reads the object's class hierarchy; a function's is
    # `function`, so the one-line form is unaffected.
    @lasagna.reranker("t_lc_plain_function")
    def length(query, text):
        return float(len(text))

    from lasagna.components import FunctionReranker

    assert issubclass(registry.get_class("reranker", "t_lc_plain_function"), FunctionReranker)
