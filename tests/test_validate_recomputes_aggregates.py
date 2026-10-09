"""`results validate` must catch a headline that disagrees with its own data.

A bundle built from a results.json whose NDCG@10 was edited to 0.665, over
per-query scores averaging 0.609, validated: the run id covers the bundle as
built, not the numbers it was built from.
"""

from __future__ import annotations

from lasagna.knowledge_evaluation_sdk.bundle import aggregate_mismatches


def _row(headline, scores, metric="ndcg@10"):
    return {
        "config_name": "dense",
        "run_index": 0,
        "metrics": {metric: headline, "avg_latency_ms": 12.0},
        "per_sample_scores": [{metric: s} for s in scores],
    }


def test_an_edited_headline_is_caught():
    problems = aggregate_mismatches([_row(0.665, [0.6, 0.618])])
    assert len(problems) == 1 and "ndcg@10 is 0.665" in problems[0]


def test_a_consistent_row_passes():
    assert aggregate_mismatches([_row(0.609, [0.6, 0.618])]) == []


def test_metrics_with_unscored_queries_are_not_second_guessed():
    assert aggregate_mismatches([_row(0.5, [0.5, float("nan")])]) == []
    assert aggregate_mismatches([_row(0.5, [0.5])] + [{"metrics": {"x": 1}}]) == []
