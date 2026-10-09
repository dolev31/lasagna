"""`lasagna benchmark leaderboard` must rank configs only on what they all ran.

`BenchmarkResults.leaderboard` pooled every task of a config across benchmarks
and silently dropped failed ones. A config that ran only the easy benchmark
outranked one that ran the easy and the hard one (`1. A 0.9000`, `2. B
0.5500`), and the `± std (n=2)` it printed was the spread between benchmarks,
labelled as if it were runs. An arm that scored 30 of 100 queries (0.8831)
ranked above one that scored all 100 (0.5131).
"""

from __future__ import annotations

from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults
from lasagna.knowledge_evaluation_sdk.runner import TaskResult


def _task(config, benchmark, run, value, n_queries=10, error=None, first_query=0):
    scores = [{"_sample_id": f"q{i}", "ndcg@10": value} for i in range(first_query, n_queries)]
    return TaskResult(
        task_key=f"{benchmark}|{config}|{run}",
        benchmark_name=benchmark,
        config_name=config,
        run_index=run,
        metrics={} if error else {"ndcg@10": value},
        per_sample_scores=[] if error else scores,
        error=error,
    )


def _section(text, benchmark):
    """The lines of one benchmark's section."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == benchmark)
    out = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(" ") and not line.startswith("-"):
            break
        out.append(line)
    return out


class TestConfigsAreRankedPerBenchmark:
    def test_a_config_that_skipped_the_hard_benchmark_does_not_win_overall(self):
        res = BenchmarkResults(
            [
                _task("A", "easy", 0, 0.9),
                _task("A", "easy", 1, 0.9),
                _task("B", "easy", 0, 0.8),
                _task("B", "hard", 0, 0.3),
            ]
        )
        text = res.leaderboard("ndcg@10")
        easy = _section(text, "easy")
        assert any("1." in line and "A" in line and "0.9000" in line for line in easy)
        assert any("2." in line and "B" in line and "0.8000" in line for line in easy)
        hard = _section(text, "hard")
        assert any("A" in line and "not run" in line for line in hard)
        assert "0.5500" not in text, "a cross-benchmark average was printed"

    def test_the_spread_is_labelled_as_runs(self):
        res = BenchmarkResults([_task("A", "easy", 0, 0.9), _task("A", "easy", 1, 0.8)])
        assert "2 runs" in res.leaderboard("ndcg@10")

    def test_a_failed_task_is_shown(self):
        res = BenchmarkResults(
            [
                _task("A", "easy", 0, 0.9),
                _task("B", "easy", 0, 0.8),
                _task("B", "easy", 1, 0.0, error="judge timed out"),
            ]
        )
        easy = _section(res.leaderboard("ndcg@10"), "easy")
        assert any("B" in line and "1 of 2 runs failed" in line for line in easy)

    def test_a_config_whose_every_run_failed_is_listed_unranked(self):
        res = BenchmarkResults(
            [_task("A", "easy", 0, 0.9), _task("B", "easy", 0, 0.0, error="boom")]
        )
        easy = _section(res.leaderboard("ndcg@10"), "easy")
        assert any("B" in line and "failed" in line and "2." not in line for line in easy)


class TestPartialQuerySetsAreNotRanked:
    def test_an_arm_that_scored_fewer_queries_is_set_apart(self):
        res = BenchmarkResults(
            [
                _task("partial", "easy", 0, 0.8831, n_queries=30),
                _task("full", "easy", 0, 0.5131, n_queries=100),
            ]
        )
        easy = _section(res.leaderboard("ndcg@10"), "easy")
        assert any("1." in line and "full" in line for line in easy)
        partial = [line for line in easy if "partial" in line]
        assert partial and "1." not in partial[0] and "2." not in partial[0]
        assert "30 of 100 queries" in partial[0]

    def test_configs_on_the_same_queries_are_all_ranked(self):
        res = BenchmarkResults(
            [_task("a", "easy", 0, 0.5), _task("b", "easy", 0, 0.6), _task("c", "easy", 0, 0.4)]
        )
        easy = _section(res.leaderboard("ndcg@10"), "easy")
        assert [line.split()[1] for line in easy if line.strip()[:2] in ("1.", "2.", "3.")] == [
            "b",
            "a",
            "c",
        ]
