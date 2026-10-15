"""Baselines a RAG claim is measured against: no retrieval, and more context.

There was no way to ask the two questions every RAG result has to answer:
does retrieval help at all (against the same model answering from the question
alone), and would more passages help (against the same prompt over all of them)?
Generation needed a context in both harness paths, so a closed-book arm could
not be scored, and the harness always handed the generator exactly five
passages. A closed-book arm must never see a passage, must not be judged on
faithfulness to passages it never saw, and adding these arms must leave the
default path exactly as it was.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lasagna.core.base_components import BaseGenerator
from lasagna.core.registry import registry
from lasagna.core.testing import ComponentTestHarness
from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
from lasagna.knowledge_evaluation_sdk.benchmarks.adapters import smoke
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask
from lasagna.knowledge_evaluation_sdk.stages import declared_stages, isolation_report, stage_of

ragas = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")

CORPUS = sorted(smoke._CORPUS)


class _LLM:
    def __init__(self):
        self.prompts = []

    def _reply(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content="trehalose" if "tardigrade" in prompt.lower() else "lime")

    def invoke(self, prompt):
        return self._reply(prompt)

    async def ainvoke(self, prompt):
        return self._reply(prompt)


class _Judge:
    calls: list = []

    def __init__(self, llm=None):
        pass

    def evaluate_detailed(
        self, questions, answers, contexts, ground_truths=None, metrics=None, **kw
    ):
        _Judge.calls.append({"metrics": list(metrics or []), "contexts": contexts})
        rows = [{m: 0.5 for m in metrics or []} for _ in questions]
        return ragas.DetailedEvalResult(aggregate={m: 0.5 for m in metrics or []}, per_sample=rows)


@pytest.fixture
def llm(monkeypatch):
    model = _LLM()
    import lasagna.providers as providers

    monkeypatch.setattr(
        providers, "get_llm_provider", lambda: SimpleNamespace(get_llm=lambda **k: model)
    )
    monkeypatch.setattr(ragas, "RAGASEvaluator", _Judge)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda *a, **k: None)
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "none")
    _Judge.calls = []
    return model


# -- a live path without a vector store ----------------------------------------


class _Result:
    def __init__(self, doc_id, rank):
        self.chunk = SimpleNamespace(
            document_id=doc_id, content=f"PASSAGE{rank}: {smoke._CORPUS[doc_id]}", metadata={}
        )
        self.score = 1.0 / (rank + 1)


class _Pipeline:
    """Stands in for KnowledgePipeline: every query returns the k passages it
    asked for, each tagged with its rank."""

    built: list = []

    def __init__(self):
        self._config, self._component_names, self._instances = {}, {}, {}
        self.queries = []
        _Pipeline.built.append(self)

    def replace(self, slot, name):
        self._component_names[slot] = name
        self._instances.pop(slot, None)
        return self

    def _apply_session_config(self, session_id):
        pass

    def get_component(self, slot):
        if slot not in self._instances and slot in self._component_names:
            self._instances[slot] = registry.get(slot, self._component_names[slot])
        return self._instances.get(slot)

    def query(self, question, k, session_id, config):
        self.queries.append(question)
        start = sum(map(ord, question)) % len(CORPUS)
        return [_Result(CORPUS[(start + j) % len(CORPUS)], j) for j in range(k)]


@pytest.fixture
def live(monkeypatch, llm):
    import lasagna.api.rag_backend as rag_backend
    from lasagna.core.pipeline import KnowledgePipeline

    async def ingest(benchmark_name, samples, **kw):
        ids = {c: c for c in CORPUS}
        return {
            "session_id": "stub-session",
            "corpus_to_valid": ids,
            "valid_to_corpus": ids,
            "corpus_fingerprint": "stub",
            "corpus_provenance": "native",
        }

    _Pipeline.built = []
    monkeypatch.setattr(exp, "_ensure_corpus_ingested", ingest)
    monkeypatch.setattr(
        KnowledgePipeline, "from_profile", classmethod(lambda c, p=None: _Pipeline())
    )
    backend = SimpleNamespace(
        embedding_manager=SimpleNamespace(model_name="stub-embedder", dimension=8),
        hybrid_searcher=SimpleNamespace(
            is_native_mode_available=lambda: True, _backend_supports_native_hybrid=lambda s: True
        ),
    )
    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda *a, **k: backend)
    return llm


def _sweep(tmp_path, arms, mode="live", **extra):
    data = {
        "name": "baselines",
        "benchmarks": [{"name": "smoke"}],
        "mode": mode,
        "output_dir": str(tmp_path / "out"),
        "max_concurrency": 1,
        "llm_cache": False,
        "pipeline": {"base_profile": "speed", "config_overrides": arms},
        **extra,
    }
    return asyncio.run(exp.Experiment.from_dict(data).run())


def _by_arm(results):
    return {r.config_name: r for r in results._results}


# -- the generators -------------------------------------------------------------


@pytest.mark.parametrize("name", ["closed_book", "long_context"])
def test_the_baseline_generators_pass_their_contract(name):
    gen = registry.get("generator", name)
    assert isinstance(gen, BaseGenerator)
    assert ComponentTestHarness().validate(gen).passed
    with pytest.raises(RuntimeError, match="needs an LLM"):
        gen.generate("q?", ["Paris is in France."], llm=None)


def test_closed_book_never_puts_a_passage_in_its_prompt():
    model = _LLM()
    gen = registry.get("generator", "closed_book")
    gen.generate("Capital of France?", ["SECRET PASSAGE about Paris."], llm=model)
    assert "SECRET PASSAGE" not in model.prompts[0]
    assert "Capital of France?" in model.prompts[0]
    assert gen.needs_context is False and BaseGenerator.needs_context is True


def test_long_context_is_the_concise_prompt_over_more_passages():
    a, b = _LLM(), _LLM()
    passages = [f"p{i}" for i in range(10)]
    registry.get("generator", "concise").generate("q?", passages, llm=a)
    registry.get("generator", "long_context").generate("q?", passages, llm=b)
    assert a.prompts == b.prompts
    assert registry.get("generator", "long_context").context_k == 10
    assert registry.get("generator", "concise").context_k is None


# -- generation_only (gold context) ---------------------------------------------


def test_closed_book_is_scored_without_the_gold_context(llm):
    task = BenchmarkTask(
        benchmark_name="smoke",
        config_name="cb",
        run_index=0,
        mode="generation_only",
        config_overrides={"generator": "closed_book"},
    )
    samples = smoke.SmokeBenchmarkAdapter().load()
    result = asyncio.run(exp._eval_offline(task, samples))

    assert result.error is None, result.error
    assert result.n_samples == 12
    for prompt, sample in zip(llm.prompts, samples):
        assert sample.context[:40] not in prompt, "the gold context reached a closed-book prompt"
    assert "exact_match" in result.metrics and "token_f1" in result.metrics
    assert not any(k.startswith(("faithfulness", "context_")) for k in result.metrics)
    (call,) = _Judge.calls
    assert call["metrics"] == ["factual_correctness"]
    assert all(c == [] for c in call["contexts"])
    assert result.system == {"generator_context": "none"}


def test_closed_book_scores_questions_that_have_no_gold_context(llm):
    sample = BenchmarkSample(query_id="q", question="Capital of France?", ground_truth="Paris")
    task = BenchmarkTask(
        benchmark_name="b",
        config_name="cb",
        run_index=0,
        mode="offline",
        config_overrides={"generator": "closed_book"},
    )
    result = asyncio.run(exp._eval_offline(task, [sample]))
    assert result.n_samples == 1 and result.metrics["generations_failed"] == 0.0
    assert llm.prompts and "Capital of France?" in llm.prompts[0]
    plain = BenchmarkTask(
        benchmark_name="b", config_name="c", run_index=0, mode="offline", config_overrides={}
    )
    # The default still needs the context it is told to answer from.
    assert asyncio.run(exp._eval_offline(plain, [sample])).metrics["generations_failed"] == 1.0


# -- live -----------------------------------------------------------------------


def test_long_context_reads_ten_passages_and_is_judged_on_the_same_ten(tmp_path, live):
    _sweep(tmp_path, [{"name": "long", "generator": "long_context"}])
    assert all(p.count("PASSAGE") == 10 for p in live.prompts) and len(live.prompts) == 12
    assert "PASSAGE9:" in live.prompts[0]
    (call,) = _Judge.calls
    assert {len(c) for c in call["contexts"]} == {10}
    assert "faithfulness" in call["metrics"]


def test_the_default_path_still_reads_the_first_five(tmp_path, live):
    results = _sweep(tmp_path, [{"name": "default"}, {"name": "concise", "generator": "concise"}])
    default, named = live.prompts[:12], live.prompts[12:]
    assert default == named, "naming concise must reproduce the default prompt"
    assert all(p.count("PASSAGE") == 5 and "PASSAGE4:" in p for p in default)
    assert [{len(c) for c in call["contexts"]} for call in _Judge.calls] == [{5}, {5}]
    rows = {
        name: [{k: v for k, v in s.items() if "_ms" not in k} for s in r.per_sample_scores]
        for name, r in _by_arm(results).items()
    }
    assert rows["default"] == rows["concise"]


def test_closed_book_live_never_sees_or_fetches_a_passage(tmp_path, live):
    results = _sweep(tmp_path, [{"name": "cb", "generator": "closed_book"}])
    assert live.prompts and not any("PASSAGE" in p for p in live.prompts)
    (pipeline,) = _Pipeline.built
    assert len(pipeline.queries) == 12, "only the retrieval track should have queried"
    task = _by_arm(results)["cb"]
    assert "ndcg@10" in task.metrics, "the retrieval track still measures this arm"
    assert "faithfulness" not in task.metrics and "exact_match" in task.metrics
    assert task.system["generator_context"] == "none"


def test_a_no_rag_arm_retrieves_nothing_and_is_filed_under_the_retriever(tmp_path, live):
    results = _sweep(tmp_path, [{"name": "control"}, {"name": "none", "no_rag": True}])
    arms = _by_arm(results)
    assert arms["none"].error is None, arms["none"].error
    assert sorted(len(p.queries) for p in _Pipeline.built) == [0, 24]
    assert not any("@10" in k for k in arms["none"].metrics), "a no_rag arm has no retrieval"
    assert "ndcg@10" in arms["control"].metrics
    assert "exact_match" in arms["none"].metrics and "faithfulness" not in arms["none"].metrics
    assert not any("PASSAGE" in p for p in live.prompts[-12:])

    assert stage_of("no_rag") == "retriever"
    assert declared_stages({"no_rag": True}) == {"retriever"}
    control, none = arms["control"].system["stages"], arms["none"].system["stages"]
    assert [s for s in control if control[s]["digest"] != none[s]["digest"]] == ["retriever"]
    report = isolation_report([r.to_dict() for r in results._results])
    assert report["undeclared"] == [] and report["inert"] == []


def test_the_reports_say_why_a_context_free_arm_has_no_faithfulness(tmp_path, live):
    results = _sweep(
        tmp_path,
        [{"name": "rag"}, {"name": "cb", "generator": "closed_book"}],
        tracks=["generation"],
    )
    note = (
        "cb: no context: faithfulness and context metrics not reported; "
        "compare exact_match, token_f1, factual_correctness"
    )
    assert note in results.leaderboard("exact_match")
    assert note in results.to_markdown("exact_match")
    assert "rag: no context" not in results.leaderboard("exact_match")


def test_a_no_rag_arm_that_cannot_be_scored_is_refused_before_it_runs():
    def tasks(**data):
        base = {"name": "x", "benchmarks": [{"name": "smoke"}], "mode": "live"}
        return exp.Experiment.from_dict({**base, **data}).generate_tasks()

    with pytest.raises(ValueError, match="no_rag: true gives the generator no passages"):
        tasks(pipeline={"config_overrides": [{"name": "n", "no_rag": True, "generator": "cited"}]})
    with pytest.raises(ValueError, match="scored on generation only"):
        tasks(pipeline={"config_overrides": [{"name": "n", "no_rag": True}]}, tracks=["retrieval"])
    assert tasks(pipeline={"config_overrides": [{"name": "n", "no_rag": True}]})
