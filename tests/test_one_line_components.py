"""Adding a technique is one line: a decorator on a plain function.

The paper says "adding a new technique is a single line of code; the framework
tests it inside a complete modern pipeline". These tests hold the library to
that: a function becomes a conforming component of its stage with one
decorator, a sweep can name it as `module:function` with no registration or
packaging, and the framework checks its contract before any corpus is ingested.
"""

from __future__ import annotations

import textwrap

import pytest

import lasagna
from lasagna.components import load
from lasagna.core.base_components import (
    BaseChunker,
    BaseEmbedder,
    BaseQueryEnhancer,
    BaseReranker,
    ScoredDocument,
)
from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness


def _docs():
    return [
        ScoredDocument(content="alpha beta gamma", score=0.1, chunk_id="c1"),
        ScoredDocument(content="delta", score=0.9, chunk_id="c2"),
        ScoredDocument(content="alpha beta", score=0.5, chunk_id="c3"),
    ]


def test_a_scoring_function_is_a_conforming_reranker():
    @lasagna.reranker("t_overlap")
    def overlap(query, text):
        return len(set(query.split()) & set(text.split()))

    cls = registry.get_class("reranker", "t_overlap")
    assert issubclass(cls, BaseReranker)
    entry = next(e for e in registry.list_rerankers() if e["name"] == "t_overlap")
    assert entry["implements_contract"] is True
    out = registry.get("reranker", "t_overlap").rerank("alpha beta", _docs(), top_k=2)
    assert [d.chunk_id for d in out] == ["c1", "c3"]
    assert overlap("a", "a") == 1, "the decorator returns the function unchanged"


def test_the_function_never_sees_the_callers_objects():
    @lasagna.reranker("t_mutator")
    def mutator(query, text):
        return 1.0

    docs = _docs()
    registry.get("reranker", "t_mutator").rerank("q", docs, top_k=3)
    assert [d.score for d in docs] == [0.1, 0.9, 0.5], "scores were written onto the input"


def test_a_listwise_function_orders_the_list_itself():
    @lasagna.reranker("t_listwise")
    def by_length(query, documents, top_k):
        return sorted(documents, key=lambda d: len(d.content))

    out = registry.get("reranker", "t_listwise").rerank("q", _docs(), top_k=2)
    assert [d.chunk_id for d in out] == ["c2", "c3"]


def test_bare_decorator_and_one_call_forms():
    @lasagna.query_enhancer
    def t_bare_enhancer(query):
        return f"{query} synonyms"

    def t_split(text):
        return text.split("\n\n")

    lasagna.chunker("t_paragraphs", t_split)

    enh = registry.get("query_enhancer", "t_bare_enhancer")
    assert isinstance(enh, BaseQueryEnhancer)
    assert enh.enhance("cats") == ["cats", "cats synonyms"]

    ch = registry.get("chunker", "t_paragraphs")
    assert isinstance(ch, BaseChunker)
    chunks = ch.chunk("one two\n\nthree")
    assert [c.content for c in chunks] == ["one two", "three"]
    assert chunks[1].start_char == 9 and chunks[1].chunk_index == 1


def test_an_embedder_reports_its_dimension():
    @lasagna.embedder("t_len")
    def lengths(texts):
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    emb = registry.get("embedder", "t_len")
    assert isinstance(emb, BaseEmbedder)
    assert emb.dimension == 3
    assert emb.encode(["ab", "abc"]).shape == (2, 3)


def test_every_function_component_passes_its_stage_contract():
    @lasagna.reranker("t_contract_rr")
    def score(query, text):
        return float(len(text))

    @lasagna.query_enhancer("t_contract_qe")
    def variants(query):
        return [query.upper()]

    for stage, name in (("reranker", "t_contract_rr"), ("query_enhancer", "t_contract_qe")):
        report = ComponentTestHarness().validate(registry.get(stage, name))
        assert report.passed, report.failures


def test_two_functions_never_share_an_identity():
    # The identity keys the search cache: two lambdas, or two functions one
    # factory made, share module:qualname, and the second was served the
    # first's results.
    lasagna.reranker("t_id_a", lambda q, t: 1.0)
    lasagna.reranker("t_id_b", lambda q, t: 2.0)
    a = registry.get("reranker", "t_id_a").model_name
    b = registry.get("reranker", "t_id_b").model_name
    assert a != b

    def make(sign):
        def embed(texts):
            return [[sign * len(t), 1.0] for t in texts]

        return embed

    lasagna.embedder("t_id_pos", make(1.0))
    lasagna.embedder("t_id_neg", make(-1.0))
    pos = registry.get("embedder", "t_id_pos").model_name
    neg = registry.get("embedder", "t_id_neg").model_name
    assert pos != neg

    # Same name, different closure value (a notebook cell run again with an
    # edit): the identity changes too, so a shared cache cannot serve the old.
    lasagna.embedder("t_id_rerun", make(1.0))
    first = registry.get("embedder", "t_id_rerun").model_name
    lasagna.embedder("t_id_rerun", make(2.0))
    assert registry.get("embedder", "t_id_rerun").model_name != first


def test_a_built_in_name_is_refused():
    # Registering `cross_encoder` replaced the control arm of every sweep in
    # the process, under the built-in's name and its model's provenance.
    with pytest.raises(ValueError, match="built-in reranker"):

        @lasagna.reranker("cross_encoder")
        def longer_first(query, text):
            return len(text.split())

    assert registry.is_builtin("reranker", "cross_encoder")
    assert registry.get_class("reranker", "cross_encoder").__name__ != "cross_encoder"


def test_overriding_a_built_in_through_the_registry_drops_its_built_in_status(monkeypatch):
    from lasagna.core.registry import ComponentRegistry

    monkeypatch.setattr(ComponentRegistry, "_instance", None)
    fresh = ComponentRegistry()

    class Mine(BaseReranker):
        def rerank(self, query, documents, top_k=5, **kwargs):
            return list(documents)[:top_k]

    assert fresh.is_builtin("reranker", "cross_encoder")
    fresh.register_class("reranker", "cross_encoder", Mine, override=True)
    assert not fresh.is_builtin("reranker", "cross_encoder")


def test_a_listwise_function_with_the_interfaces_own_signature_is_listwise():
    # top_k has a default, as in BaseReranker.rerank; counting only required
    # parameters called it with one passage's text and blamed correct code.
    @lasagna.reranker("t_listwise_default")
    def by_length(query, documents, top_k=5):
        return sorted(documents, key=lambda d: len(d.content))[:top_k]

    out = registry.get("reranker", "t_listwise_default").rerank("q", _docs(), top_k=2)
    assert [d.chunk_id for d in out] == ["c2", "c3"]


def test_a_pointwise_function_with_an_option_stays_pointwise():
    @lasagna.reranker("t_pointwise_option")
    def overlap(query, text, lowercase=True):
        return len(set(query.split()) & set(text.split()))

    out = registry.get("reranker", "t_pointwise_option").rerank("alpha beta", _docs(), top_k=2)
    assert [d.chunk_id for d in out] == ["c1", "c3"]


def test_an_ambiguous_reranker_shape_is_refused_with_the_fix():
    with pytest.raises(TypeError, match=r"\(query, documents, top_k\)"):
        lasagna.reranker("t_ambiguous", lambda q, x, y=1: 0.0)


def test_an_agentic_pipeline_class_can_be_named_by_reference(tmp_path):
    from lasagna.core.base_components import BaseAgenticPipeline

    (tmp_path / "t_pipes.py").write_text(
        "from lasagna.core.base_components import BaseAgenticPipeline\n"
        "class Mine(BaseAgenticPipeline):\n"
        "    async def run(self, input):\n"
        "        return None\n"
    )
    name = load("agentic_pipeline", "t_pipes:Mine", [tmp_path])
    assert issubclass(registry.get_class("agentic_pipeline", name), BaseAgenticPipeline)

    with pytest.raises(TypeError, match="is a class, not a function"):
        lasagna.components.register("agentic_pipeline", lambda q: q, "t_fn_pipeline")


def test_a_reference_means_one_file(tmp_path):
    # Two sweep folders each holding my_ideas.py: the second sweep silently
    # ran the first folder's function.
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "t_same_name.py").write_text("def f(query, text):\n    return 1.0\n")
    (two / "t_same_name.py").write_text("def f(query, text):\n    return 2.0\n")
    load("reranker", "t_same_name:f", [one])
    assert load("reranker", "t_same_name:f", [one]) == "t_same_name:f"
    with pytest.raises(ImportError, match="already"):
        load("reranker", "t_same_name:f", [two])


def test_a_file_reference_does_not_take_over_a_module_name(tmp_path):
    import sys

    (tmp_path / "t_shadow_json.py").write_text("def v(query):\n    return [query]\n")
    load("query_enhancer", f"{tmp_path / 't_shadow_json.py'}:v")
    assert "t_shadow_json" not in sys.modules


def test_a_class_that_skips_the_interface_is_refused_with_the_fix():
    class NotAReranker:
        def rerank(self, query, documents, top_k=5):
            return documents

    with pytest.raises(TypeError, match="@lasagna.reranker"):
        lasagna.reranker("t_bad_class", NotAReranker)


def test_a_module_reference_loads_and_registers_under_its_own_name(tmp_path):
    (tmp_path / "t_ideas.py").write_text(
        textwrap.dedent(
            """
            def keyword_overlap(query, text):
                return len(set(query.split()) & set(text.split()))
            """
        )
    )
    name = load("reranker", "t_ideas:keyword_overlap", [tmp_path])
    assert registry.has("reranker", name)
    assert issubclass(registry.get_class("reranker", name), BaseReranker)


def test_a_file_reference_loads(tmp_path):
    (tmp_path / "ideas_file.py").write_text("def v(query):\n    return [query + ' x']\n")
    ref = f"{tmp_path / 'ideas_file.py'}:v"
    load("query_enhancer", ref)
    assert registry.get("query_enhancer", ref).enhance("q") == ["q", "q x"]


def test_a_sweep_names_a_function_in_one_line_and_it_is_checked_first(tmp_path):
    from lasagna.knowledge_evaluation_sdk.experiment import Experiment

    (tmp_path / "t_sweep_ideas.py").write_text(
        "def good(query, text):\n    return float(len(text))\n"
        "def broken(query, text):\n    raise RuntimeError('boom')\n"
    )
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        textwrap.dedent(
            """
            name: one line
            benchmarks: [{name: smoke}]
            tracks: [retrieval]
            pipeline:
              base_profile: speed
              rerankers: [cross_encoder, "t_sweep_ideas:good"]
            """
        )
    )
    tasks = Experiment.from_file(str(sweep)).generate_tasks()
    assert {t.config_overrides.get("reranker") for t in tasks} == {
        "cross_encoder",
        "t_sweep_ideas:good",
    }

    sweep.write_text(sweep.read_text().replace(":good", ":broken"))
    with pytest.raises(ValueError, match="fails its contract"):
        Experiment.from_file(str(sweep)).generate_tasks()


def test_a_sweep_refuses_a_component_that_skips_the_interface():
    from lasagna.knowledge_evaluation_sdk.experiment import _check_components_conform

    class DuckTyped:
        def rerank(self, query, documents, top_k=5):
            return documents

    registry.register_class("reranker", "t_duck_sweep", DuckTyped, override=True)
    try:
        with pytest.raises(ValueError, match="does not implement BaseReranker"):
            _check_components_conform([{"name": "x", "overrides": {"reranker": "t_duck_sweep"}}])
    finally:
        registry.unregister_reranker("t_duck_sweep")


def test_an_embedder_or_chunker_arm_gets_a_corpus_of_its_own(tmp_path):
    # The corpus used to be ingested with the built-in embedder, so an embedder
    # arm embedded only its queries and measured two models against each
    # other. Each component arm now has its own corpus, keyed by what the
    # component is, so an edited function is never served the old corpus.
    from lasagna.knowledge_evaluation_sdk.experiment import Experiment, _corpus_cache_key

    (tmp_path / "t_ingest_ideas.py").write_text(
        "def e(texts):\n    return [[1.0, float(len(t))] for t in texts]\n"
        "def c(text):\n    return text.split('. ')\n"
    )
    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        textwrap.dedent(
            """
            name: ingest arms
            benchmarks: [{name: smoke}]
            tracks: [retrieval]
            pipeline:
              base_profile: speed
              config_overrides:
                - {name: control}
                - {name: emb, embedder: "t_ingest_ideas:e"}
                - {name: chk, chunker: "t_ingest_ideas:c"}
            """
        )
    )
    tasks = {t.config_name: t for t in Experiment.from_file(str(sweep)).generate_tasks()}
    keys = {name: _corpus_cache_key(t) for name, t in tasks.items()}
    assert len(set(keys.values())) == 3, keys


def test_an_edited_function_gets_a_new_corpus():
    from lasagna.knowledge_evaluation_sdk.experiment import _ingest_component_fingerprint

    lasagna.embedder("t_fp_emb", lambda texts: [[1.0, 0.0] for _ in texts])
    first = _ingest_component_fingerprint("embedder", "t_fp_emb")
    lasagna.embedder("t_fp_emb", lambda texts: [[2.0, 0.0] for _ in texts])
    assert _ingest_component_fingerprint("embedder", "t_fp_emb") != first


def test_a_document_processor_sweep_is_still_refused(tmp_path):
    from lasagna.knowledge_evaluation_sdk.experiment import Experiment

    sweep = tmp_path / "sweep.yaml"
    sweep.write_text(
        "name: dp\nbenchmarks: [{name: smoke}]\ntracks: [retrieval]\n"
        "pipeline: {base_profile: speed, config_overrides: [{name: a, document_processor: default}]}\n"
    )
    with pytest.raises(ValueError, match="loaded as text"):
        Experiment.from_file(str(sweep)).generate_tasks()
