"""An agentic arm's exact match and F1 score the answer its loop wrote.

The generation track asks the harness's own generator to answer from the
passages a search returned, so every arm is scored the same way. For an
agentic arm the loop writes an answer too. The judged metrics already scored
it: ``faithfulness`` was the loop's answer, ``faithfulness_regen`` the
harness's. Exact match, token F1, ROUGE-L, containment, abstention and answer
similarity scored only the harness's answer. So one row judged two answers,
and ``lasagna results significance -m exact_match``, the check #20 asks for,
compared the harness's answers from each loop's passages, not the loops.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lasagna.knowledge_evaluation_sdk import answer_metrics
from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.benchmarks._base import BenchmarkSample
from lasagna.knowledge_evaluation_sdk.runner import BenchmarkTask

exp_ragas = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")

_FREE = ("exact_match", "token_f1", "rouge_l", "contains_answer", "answer_similarity")


class _Pipeline:
    """Stands in for KnowledgePipeline. An agentic arm's search returns the
    loop's passages with its answer on the first, as ``_convert_agentic_results``
    stashes it; a question missing from ``answers`` got no answer from the loop."""

    def __init__(self, answers):
        self._config = {"enable_hybrid_search": False}
        self._component_names = {}
        self._instances = {}
        self._answers = answers

    def query(self, question, top_k, session_id, config):
        meta = {}
        if config.get("use_agentic_rag"):
            meta["_routing_tier_used"] = "comprehensive"
            if self._answers.get(question):
                meta["_agentic_answer"] = self._answers[question]
        chunk = SimpleNamespace(
            document_id="v1", content="Tardigrades survive in space.", metadata=meta
        )
        return [SimpleNamespace(chunk=chunk, score=1.0)]


class _Generator:
    """The harness's generator: it reads the passages and gets it wrong."""

    reply = "Lisbon"

    async def ainvoke(self, prompt):
        return SimpleNamespace(content=self.reply)

    def invoke(self, prompt):
        return SimpleNamespace(content=self.reply)


class _Judge:
    """Records which answers it was asked to judge; every one is faithful."""

    judged: list = []

    def __init__(self, llm=None):
        pass

    def evaluate_detailed(self, questions, answers, contexts, ground_truths=None, **kw):
        _Judge.judged.append(list(answers))
        rows = [{"faithfulness": 1.0} for _ in answers]
        return exp_ragas.DetailedEvalResult(aggregate={"faithfulness": 1.0}, per_sample=rows)


def _encoder(texts):
    """Answer similarity without a model: 1.0 when both name the tardigrade."""
    return [[1.0, 0.0] if "tardigrade" in t.lower() else [0.0, 1.0] for t in texts]


@pytest.fixture
def run(monkeypatch):
    from lasagna.api import rag_backend
    from lasagna.core import pipeline as pipeline_mod
    import lasagna.providers as providers

    async def _ingested(*a, **k):
        return {
            "session_id": "bench_corpus_test",
            "corpus_to_valid": {"d1": "v1"},
            "valid_to_corpus": {"v1": "d1"},
            "corpus_fingerprint": "fp",
        }

    async def _released(*a, **k):
        return None

    monkeypatch.setattr(exp, "_ensure_corpus_ingested", _ingested)
    monkeypatch.setattr(exp, "_release_corpus", _released)
    monkeypatch.setattr(
        exp,
        "_apply_overrides_to_pipeline",
        lambda pipeline, overrides, session_id: pipeline._config.update(overrides or {}),
    )
    monkeypatch.setattr(
        rag_backend,
        "get_rag_backend",
        lambda: SimpleNamespace(embedding_manager=SimpleNamespace(model_name="m", dimension=4)),
    )
    model = _Generator()
    monkeypatch.setattr(
        providers,
        "get_llm_provider",
        lambda: SimpleNamespace(
            get_llm=lambda *a, **k: model, get_chat_model=lambda *a, **k: model
        ),
    )
    monkeypatch.setattr(exp_ragas, "RAGASEvaluator", _Judge)
    monkeypatch.setattr(exp, "_build_judge_llm", lambda: None)
    monkeypatch.setenv("RAG_EVAL_SIMILARITY_MODEL", "stub")
    monkeypatch.setattr(answer_metrics, "load_similarity_encoder", lambda name=None: _encoder)
    _Judge.judged = []

    def _run(answers, n=3, overrides=None, harness="Lisbon", fails=False):
        model.reply = harness
        pipeline = _Pipeline(answers)
        monkeypatch.setattr(
            pipeline_mod.KnowledgePipeline, "from_profile", classmethod(lambda cls, p: pipeline)
        )
        if overrides is None:
            overrides = {"agentic_pipeline": "self_ask", "use_agentic_rag": True}
        task = BenchmarkTask(
            benchmark_name="smoke",
            config_name="arm",
            run_index=0,
            mode="live",
            config_overrides=overrides,
            tracks=("generation",),
        )
        samples = [
            BenchmarkSample(
                query_id=f"q{i}",
                question=f"question {i}?",
                expected_docs={"d1": 1},
                ground_truth="tardigrade",
            )
            for i in range(n)
        ]
        result = asyncio.run(exp._eval_live(task, samples))
        assert (result.error is not None) == fails, result.error
        return result

    return _run


def _rows(result):
    return [r for r in result.per_sample_scores if "exact_match" in r]


def test_the_loops_answer_is_the_arms_score(run):
    loop = {f"question {i}?": "tardigrade [1]" for i in range(3)}
    result = run(loop)
    for key in _FREE:
        assert result.metrics[key] == 1.0, key
        assert result.metrics[f"{key}_agentic"] == 1.0, key
        assert result.metrics[f"{key}_regen"] == 0.0, key
    for row in _rows(result):
        assert row["exact_match"] == row["exact_match_agentic"] == 1.0
        assert row["exact_match_regen"] == 0.0
        assert row["_agentic_answer"] == "tardigrade [1]"


def test_the_judge_and_exact_match_read_the_same_answer(run):
    loop = {"question 0?": "tardigrade", "question 1?": "Lisbon"}
    result = run(loop, n=2)
    assert _Judge.judged == [["Lisbon", "Lisbon"], ["tardigrade", "Lisbon"]]
    first, second = _rows(result)
    assert (first["exact_match"], second["exact_match"]) == (1.0, 0.0)
    assert first["faithfulness"] == first["faithfulness_agentic"]


def test_a_question_the_loop_did_not_answer_keeps_the_harness_answer(run):
    """As the judged metrics do: the canonical key falls back to the regenerated
    answer, and the ``_agentic`` mean covers only the questions the loop answered."""
    # A blank reply is no answer.
    result = run({"question 0?": "tardigrade", "question 1?": "\n "}, n=2)
    answered, unanswered = _rows(result)
    assert answered["exact_match"] == 1.0
    assert "exact_match_agentic" not in unanswered
    assert unanswered["exact_match"] == unanswered["exact_match_regen"] == 0.0
    assert result.metrics["exact_match"] == 0.5
    assert result.metrics["exact_match_agentic"] == 1.0
    assert result.metrics["exact_match_regen"] == 0.0


def test_an_arm_without_a_loop_keeps_its_keys(run):
    result = run({}, n=2, overrides={})
    assert result.metrics["exact_match"] == 0.0
    assert "generations_failed_regen" not in result.metrics
    assert not [k for k in result.metrics if k.endswith(("_agentic", "_regen"))]
    for row in _rows(result):
        assert not [k for k in row if k.endswith(("_agentic", "_regen"))]
        assert "_agentic_answer" not in row


def test_a_question_the_loop_answered_is_not_a_failed_generation(run):
    """The harness's generator wrote nothing (in the Search-o1 smoke run the loop
    answered with no passage left for it to read), but the row is scored on the
    loop's answer, so the arm keeps its generation metrics."""
    loop = {f"question {i}?": "tardigrade" for i in range(3)}
    result = run(loop, harness="")
    assert result.metrics["generations_failed"] == 0.0
    assert result.metrics["exact_match"] == 1.0
    # The harness's misses stay visible: they are averaged into _regen.
    assert result.metrics["generations_failed_regen"] == 3.0


def test_with_no_answer_from_either_the_question_failed(run):
    result = run({}, n=2, harness="", fails=True)
    assert "2 of 2 answers could not be generated" in result.error


def test_the_viewer_shows_the_answer_each_score_is_for():
    from lasagna.knowledge_evaluation_sdk.viewer import _evidence

    html = _evidence({"_agentic_answer": "tardigrade", "_generated_answer": "Lisbon"})
    assert "Loop's answer:</b> tardigrade" in html
    assert "Harness's answer (_regen):</b> Lisbon" in html
    assert "<b>Answer:</b> Lisbon" in _evidence({"_generated_answer": "Lisbon"})
