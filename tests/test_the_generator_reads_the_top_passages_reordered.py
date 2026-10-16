"""The generator reads the top passages in lost-in-the-middle order, not every other rank.

The generation track asks the search for ten passages and hands the generator
the first five. With context reordering on (standard, balanced, max_quality),
the search had already reordered all ten: ranks 1, 3, 5, 7, 9 first, then the
rest reversed. The first five were ranks 1, 3, 5, 7 and 9, so the generator
never read the second-best passage, and the judge's context precision was
scored on that order. Now the generator reads ranks 1 to 5 reordered
(1, 3, 5, 4, 2), and the judge sees the same five in ranked order. Serving is
unchanged: a search outside the harness still returns its list reordered.
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest

from lasagna.knowledge_evaluation_sdk import experiment as exp
from tests.test_retrieval_metrics_score_the_ranked_order import (  # noqa: F401 (fixtures)
    QUESTION,
    _profile,
    _search,
    live,
    orchestrator,
    ranked,
)

ragas = pytest.importorskip("lasagna.knowledge_evaluation_sdk.ragas_evaluator")


class _LLM:
    def __init__(self):
        self.prompts = []

    def _reply(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content="trehalose")

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
def llm(monkeypatch, live):  # noqa: F811
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


def _run(tmp_path, profile, arms=None):
    data = {
        "name": "presentation",
        "benchmarks": [{"name": "smoke"}],
        "mode": "live",
        "tracks": ["generation"],
        "output_dir": str(tmp_path / "out"),
        "max_concurrency": 1,
        "llm_cache": False,
        "pipeline": {"base_profile": profile, **({"config_overrides": arms} if arms else {})},
    }
    return asyncio.run(exp.Experiment.from_dict(data).run())._results


def _ranks(text):
    return [int(n) for n in re.findall(r"PASSAGE(\d+):", text)]


def test_the_generator_reads_ranks_one_to_five_reordered(tmp_path, llm):
    (result,) = _run(tmp_path, "standard")
    assert result.error is None, result.error
    assert llm.prompts and {tuple(_ranks(p)) for p in llm.prompts} == {(0, 2, 4, 3, 1)}
    (call,) = _Judge.calls
    assert {tuple(_ranks("".join(c))) for c in call["contexts"]} == {(0, 1, 2, 3, 4)}
    order = result.system["stages"]["reranker"]["settings"]["context_order"]
    assert order == {"retrieval": "ranked", "generation": "lost_in_the_middle"}


def test_long_context_reads_all_ten_reordered(tmp_path, llm):
    _run(tmp_path, "standard", [{"name": "long", "generator": "long_context"}])
    assert {tuple(_ranks(p)) for p in llm.prompts} == {(0, 2, 4, 6, 8, 9, 7, 5, 3, 1)}


def test_without_reordering_the_generator_reads_the_ranked_five(tmp_path, llm):
    (result,) = _run(tmp_path, "speed")
    assert {tuple(_ranks(p)) for p in llm.prompts} == {(0, 1, 2, 3, 4)}
    order = result.system["stages"]["reranker"]["settings"]["context_order"]
    assert order == {"retrieval": "ranked", "generation": "ranked"}


def test_a_search_outside_the_harness_still_returns_the_reordered_list(orchestrator):  # noqa: F811
    out = _search(orchestrator, QUESTION, 10, {**_profile("standard"), "use_agentic_rag": False})
    docs = ranked(QUESTION)
    assert out == [docs[i] for i in (0, 2, 4, 6, 8, 9, 7, 5, 3, 1)]
