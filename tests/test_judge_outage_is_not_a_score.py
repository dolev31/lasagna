"""A judge that could not run has not judged.

`LLMJudge.evaluate` used to return score 1 / FAIL when the judge model raised
(quota, network, expired key), so an outage lowered the mean and the pass rate
like a wrong answer would, and an outage that hit one configuration more than
another became a measured quality difference.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lasagna.knowledge_evaluation_sdk.judge import JudgeUnavailableError, LLMJudge
from lasagna.knowledge_evaluation_sdk.models import (
    EvalCase,
    EvalMetrics,
    EvalResult,
    JudgeScore,
    ScoreLevel,
)


def _judge_that_fails():
    model = MagicMock()
    structured = MagicMock()
    structured.ainvoke = AsyncMock(side_effect=ConnectionError("judge endpoint down"))
    model.with_structured_output.return_value = structured
    return LLMJudge(model=model)


def test_an_outage_raises_instead_of_scoring_the_answer():
    with pytest.raises(JudgeUnavailableError):
        asyncio.run(_judge_that_fails().evaluate("Capital of France?", "Paris", "Paris", None))


def test_batch_marks_outages_as_unjudged():
    scores = asyncio.run(_judge_that_fails().batch_evaluate([("q", "a", "a", None)] * 3))
    assert scores == [None, None, None]


def test_unjudged_cases_do_not_count_as_failures():
    case = EvalCase(utterance="q", expected_response="a")
    good = JudgeScore(
        score=5,
        level=ScoreLevel.EXCELLENT,
        reasoning="ok",
        relevance=1.0,
        completeness=1.0,
        factual_accuracy=1.0,
    )
    results = [
        EvalResult(
            case=case,
            agent_response="a",
            rag_context=None,
            rag_used=True,
            judge_score=good,
            latency_ms=1.0,
            rag_latency_ms=1.0,
            error=None,
        ),
        EvalResult(
            case=case,
            agent_response="a",
            rag_context=None,
            rag_used=True,
            judge_score=None,
            latency_ms=1.0,
            rag_latency_ms=1.0,
            error=None,
        ),
    ]
    metrics = EvalMetrics.from_results(results)
    assert metrics.unjudged_cases == 1
    assert metrics.pass_rate == 100.0
    assert metrics.avg_score == 5.0
