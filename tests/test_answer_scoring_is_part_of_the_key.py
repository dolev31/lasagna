"""What scores an answer is part of what a number means.

The answer-similarity embedder is set by an environment variable; two runs
scored by different embedders produce different numbers and must not share a
comparability key. And a published leaderboard ranks latency and tokens
smallest first.
"""

from __future__ import annotations

from lasagna.knowledge_evaluation_sdk.bundle import comparability_key
from lasagna.knowledge_evaluation_sdk.leaderboard import group


def _held(**extra):
    return {"metrics_version": "1.5", "benchmark": "b", "fixed_stages": {}, **extra}


def test_a_different_similarity_embedder_is_a_different_key():
    a = comparability_key(_held(similarity_model="all-MiniLM-L6-v2"))
    b = comparability_key(_held(similarity_model="BAAI/bge-small-en-v1.5"))
    assert a != b


def test_older_bundles_keep_the_key_they_carry():
    # Before 1.5 the field was not hashed, so adding it cannot change their key.
    old = {"metrics_version": "1.4", "benchmark": "b", "fixed_stages": {}}
    assert comparability_key(old) == comparability_key({**old, "similarity_model": "x"})


def test_latency_ranks_smallest_first():
    rows = [
        {
            "benchmark": "b",
            "comparability_key": "k",
            "metric": "p95_latency_ms",
            "mean": 900.0,
            "fixed_stages": {"x": "y"},
        },
        {
            "benchmark": "b",
            "comparability_key": "k",
            "metric": "p95_latency_ms",
            "mean": 120.0,
            "fixed_stages": {"x": "y"},
        },
    ]
    grouped = group(rows)
    (only,) = grouped["b"].values()
    assert [r["mean"] for r in only] == [120.0, 900.0]


def test_quality_still_ranks_largest_first():
    rows = [
        {
            "benchmark": "b",
            "comparability_key": "k",
            "metric": "ndcg@10",
            "mean": 0.4,
            "fixed_stages": {"x": "y"},
        },
        {
            "benchmark": "b",
            "comparability_key": "k",
            "metric": "ndcg@10",
            "mean": 0.7,
            "fixed_stages": {"x": "y"},
        },
    ]
    (only,) = group(rows)["b"].values()
    assert [r["mean"] for r in only] == [0.7, 0.4]
