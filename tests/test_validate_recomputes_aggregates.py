"""`results validate` must catch a headline that disagrees with its own data.

A bundle built from a results.json whose NDCG@10 was edited to 0.665, over
per-query scores averaging 0.609, validated: the run id covers the bundle as
built, not the numbers it was built from.
"""

from __future__ import annotations

import pytest

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


def test_an_end_to_end_row_is_checked_metric_by_metric():
    """An e2e task's list mixes retrieval rows and generation rows, so no
    metric appeared on every row and the check skipped them all: a headline
    ndcg@10 of 0.9 over queries averaging 0.5 validated."""
    row = {
        "config_name": "e2e",
        "run_index": 0,
        "metrics": {"ndcg@10": 0.9, "faithfulness": 0.1},
        "per_sample_scores": [
            {"_sample_id": "q1", "ndcg@10": 0.4},
            {"_sample_id": "q2", "ndcg@10": 0.6},
            {"_sample_id": "q1", "faithfulness": 0.7},
            {"_sample_id": "q2", "faithfulness": 0.9},
        ],
    }
    problems = aggregate_mismatches([row])
    assert len(problems) == 2
    assert any("ndcg@10 is 0.9" in p for p in problems)
    assert any("faithfulness is 0.1" in p for p in problems)


def test_a_consistent_end_to_end_row_passes():
    row = {
        "config_name": "e2e",
        "run_index": 0,
        "metrics": {"ndcg@10": 0.5, "faithfulness": 0.8, "avg_latency_ms": 40.0},
        "per_sample_scores": [
            {"_sample_id": "q1", "ndcg@10": 0.4, "latency_ms": 30.0},
            {"_sample_id": "q2", "ndcg@10": 0.6, "latency_ms": 50.0},
            {"_sample_id": "q1", "faithfulness": 0.7},
            {"_sample_id": "q2", "faithfulness": 0.9},
        ],
    }
    assert aggregate_mismatches([row]) == []


class TestRankedListsAreRechecked:
    """The run id is not a signature: edit a score and recompute the id, and the
    bundle validates. The ranked list and the expected documents are exported
    per query, so the binary metrics can be recomputed from them, which catches
    a list that was reversed, or a score raised, without the other."""

    def _row(self, retrieved, expected, **scores):
        return {
            "config_name": "dense",
            "run_index": 0,
            "metrics": {},
            "per_sample_scores": [
                {"_sample_id": "q1", "_retrieved_ids": retrieved, "_expected_docs": expected}
                | scores
            ],
        }

    def test_scores_that_match_their_list_pass(self):
        from lasagna.knowledge_evaluation_sdk.bundle import ranked_list_mismatches

        row = self._row(
            ["d3", "d1", "d9"],
            ["d1"],
            **{"mrr@3": 0.5, "hit_rate@3": 1.0, "precision@3": 1 / 3, "recall@3": 1.0},
        )
        assert ranked_list_mismatches([row]) == []

    def test_a_reversed_list_is_caught(self):
        from lasagna.knowledge_evaluation_sdk.bundle import ranked_list_mismatches

        row = self._row(["d9", "d3", "d1"], ["d1"], **{"mrr@3": 1.0})
        problems = ranked_list_mismatches([row])
        assert len(problems) == 1 and "mrr@3" in problems[0] and "'q1'" in problems[0]

    def test_a_raised_hit_rate_is_caught(self):
        from lasagna.knowledge_evaluation_sdk.bundle import ranked_list_mismatches

        row = self._row(["d9", "d3"], ["d1"], **{"hit_rate@2": 1.0, "precision@2": 0.5})
        assert len(ranked_list_mismatches([row])) == 2

    def test_graded_metrics_are_not_second_guessed(self):
        """Grades are not exported, so ndcg cannot be recomputed."""
        from lasagna.knowledge_evaluation_sdk.bundle import ranked_list_mismatches

        row = self._row(["d9"], ["d1"], **{"ndcg@1": 0.9})
        assert ranked_list_mismatches([row]) == []

    def test_validate_reports_it(self, tmp_path):
        import json

        typer_testing = pytest.importorskip("typer.testing")
        from lasagna.cli import app
        from lasagna.knowledge_evaluation_sdk.bundle import _content_hash

        bundle = {
            "schema_version": "1.0",
            "experiment": {"environment": {"lasagna_version": "x", "python": "3", "platform": "p"}},
            "configs": [],
            "results": [self._row(["d9", "d3", "d1"], ["d1"], **{"mrr@3": 1.0})],
            "integrity": {"warnings": [], "blocking": []},
        }
        bundle["run_id"] = _content_hash(bundle)
        path = tmp_path / "run.json"
        path.write_text(json.dumps(bundle))
        out = typer_testing.CliRunner().invoke(app, ["results", "validate", str(path)])
        assert out.exit_code == 1
        assert "ranked list" in out.output
