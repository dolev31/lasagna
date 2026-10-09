"""A losing config must be debuggable, not just detectable.

The two eval paths write two different row shapes into the same
`per_sample_scores` list — retrieval rows carrying `ndcg@10`, generation rows
carrying `faithfulness` — and both stamp `_sample_id`. Nothing ever joined
them. So "did this config lose at retrieval or at generation?" was answerable
from the data in the file and not answerable through any API the package
offered, and the two failures need opposite fixes.

The trajectory fields were worse than unused: `_generated_answer`,
`_retrieved_contexts`, `_ground_truth` and `_question` were all write-only.
`save_trajectories: full` cost storage in every results file and bought
nothing, because nothing read them back.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults
from lasagna.knowledge_evaluation_sdk.runner import HARNESS_KEYS, BenchmarkTask, TaskResult


def _mixed_run():
    """One config, two eval paths, same three queries.

    This is the shape a live run produces: IR rows first, generation rows
    appended after them, both keyed by `_sample_id`.
    """
    return BenchmarkResults(
        [
            TaskResult(
                task_key="t",
                benchmark_name="hotpotqa",
                config_name="reranked",
                run_index=0,
                per_sample_scores=[
                    {"_sample_id": "q1", "ndcg@10": 0.9, "recall@10": 1.0, "latency_ms": 40.0},
                    {"_sample_id": "q2", "ndcg@10": 0.1, "recall@10": 0.0, "latency_ms": 45.0},
                    {"_sample_id": "q1", "faithfulness": 0.2, "exact_match": 0.0},
                    {"_sample_id": "q2", "faithfulness": 0.95, "exact_match": 1.0},
                ],
            )
        ]
    )


class TestTheTwoRowShapesAreJoined:
    def test_one_row_per_query_not_two(self):
        rows = _mixed_run().per_query()
        assert len(rows) == 2

    def test_retrieval_and_generation_land_on_the_same_row(self):
        """The whole point. Separately they cannot be compared per query."""
        row = next(r for r in _mixed_run().per_query() if r["sample_id"] == "q1")
        assert row["ndcg@10"] == 0.9
        assert row["faithfulness"] == 0.2

    def test_it_distinguishes_the_two_failure_modes(self):
        """q1 retrieved well and generated badly; q2 the reverse.

        An aggregate score cannot tell these apart, and they need opposite
        fixes.
        """
        rows = {r["sample_id"]: r for r in _mixed_run().per_query()}
        assert rows["q1"]["ndcg@10"] > rows["q2"]["ndcg@10"]
        assert rows["q1"]["faithfulness"] < rows["q2"]["faithfulness"]

    def test_identity_columns_are_present(self):
        row = _mixed_run().per_query()[0]
        assert {"benchmark", "config", "sample_id", "n_runs"} <= set(row)

    def test_it_can_be_restricted_to_one_benchmark(self):
        results = BenchmarkResults(
            [
                TaskResult(
                    task_key=f"t{i}",
                    benchmark_name=name,
                    config_name="c",
                    run_index=0,
                    per_sample_scores=[{"_sample_id": "q1", "ndcg@10": 0.5}],
                )
                for i, name in enumerate(["hotpotqa", "scifact"])
            ]
        )
        assert {r["benchmark"] for r in results.per_query("scifact")} == {"scifact"}

    def test_a_failed_task_contributes_nothing(self):
        results = BenchmarkResults(
            [
                TaskResult(
                    task_key="t",
                    benchmark_name="b",
                    config_name="c",
                    run_index=0,
                    error="exploded",
                    per_sample_scores=[{"_sample_id": "q1", "ndcg@10": 0.0}],
                )
            ]
        )
        assert results.per_query() == []


class TestRepeatedRuns:
    def _repeated(self, values):
        return BenchmarkResults(
            [
                TaskResult(
                    task_key=f"t{i}",
                    benchmark_name="b",
                    config_name="c",
                    run_index=i,
                    per_sample_scores=[{"_sample_id": "q1", "ndcg@10": v}],
                )
                for i, v in enumerate(values)
            ]
        )

    def test_scores_are_averaged_across_runs(self):
        """Consistent with the significance tests, which also use every run."""
        row = self._repeated([0.2, 0.4, 0.9]).per_query()[0]
        assert row["ndcg@10"] == pytest.approx(0.5)
        assert row["n_runs"] == 3

    def test_n_runs_counts_runs_not_rows(self):
        """An e2e sample contributes two rows per run — a retrieval row and a
        generation row — and counting rows reported n_runs=2 for a single run
        on exactly the benchmarks per_query() exists to serve."""
        assert _mixed_run().per_query()[0]["n_runs"] == 1

    def test_latency_is_summed_within_a_run_not_averaged_across_row_shapes(self):
        """Retrieval and generation are two stages of answering one query.

        Averaging 40ms against 900ms reports 470ms for a query that took 940ms:
        neither stage, and not the total.
        """
        results = BenchmarkResults(
            [
                TaskResult(
                    task_key="t",
                    benchmark_name="b",
                    config_name="c",
                    run_index=0,
                    per_sample_scores=[
                        {"_sample_id": "q1", "ndcg@10": 0.9, "latency_ms": 40.0},
                        {"_sample_id": "q1", "faithfulness": 0.5, "latency_ms": 900.0},
                    ],
                )
            ]
        )
        assert results.per_query()[0]["latency_ms"] == pytest.approx(940.0)

    def test_latency_is_averaged_across_runs_not_summed(self):
        """Summing across runs would report 3x the per-query latency."""
        results = BenchmarkResults(
            [
                TaskResult(
                    task_key=f"t{i}",
                    benchmark_name="b",
                    config_name="c",
                    run_index=i,
                    per_sample_scores=[{"_sample_id": "q1", "latency_ms": 100.0}],
                )
                for i in range(3)
            ]
        )
        row = results.per_query()[0]
        assert row["latency_ms"] == pytest.approx(100.0)
        assert row["n_runs"] == 3

    def test_tokens_follow_the_same_rule_as_latency(self):
        results = BenchmarkResults(
            [
                TaskResult(
                    task_key="t",
                    benchmark_name="b",
                    config_name="c",
                    run_index=0,
                    per_sample_scores=[
                        {"_sample_id": "q1", "input_tokens": 300.0, "llm_calls": 1.0},
                        {"_sample_id": "q1", "input_tokens": 700.0, "llm_calls": 2.0},
                    ],
                )
            ]
        )
        row = results.per_query()[0]
        assert row["input_tokens"] == pytest.approx(1000.0)
        assert row["llm_calls"] == pytest.approx(3.0)

    def test_a_metric_present_in_only_some_runs_is_not_diluted(self):
        """Averaged over the runs that scored it, not over all runs.

        Dividing by the run count would report 0.3 for a metric that measured
        0.9 in the one run that computed it.
        """
        results = BenchmarkResults(
            [
                TaskResult(
                    task_key="t0",
                    benchmark_name="b",
                    config_name="c",
                    run_index=0,
                    per_sample_scores=[{"_sample_id": "q1", "ndcg@10": 0.5, "faithfulness": 0.9}],
                ),
                TaskResult(
                    task_key="t1",
                    benchmark_name="b",
                    config_name="c",
                    run_index=1,
                    per_sample_scores=[{"_sample_id": "q1", "ndcg@10": 0.5}],
                ),
            ]
        )
        assert results.per_query()[0]["faithfulness"] == pytest.approx(0.9)


class TestTrajectoriesBecomeReadable:
    """These fields were write-only until this view existed."""

    def test_the_generated_answer_survives_to_the_row(self):
        results = BenchmarkResults(
            [
                TaskResult(
                    task_key="t",
                    benchmark_name="b",
                    config_name="c",
                    run_index=0,
                    per_sample_scores=[
                        {
                            "_sample_id": "q1",
                            "faithfulness": 0.0,
                            "_generated_answer": "Paris is in Germany.",
                            "_question": "Where is Paris?",
                        }
                    ],
                )
            ]
        )
        row = results.per_query()[0]
        assert row["_generated_answer"] == "Paris is in Germany."
        assert row["_question"] == "Where is Paris?"

    def test_non_numeric_fields_are_not_averaged(self):
        results = BenchmarkResults(
            [
                TaskResult(
                    task_key="t",
                    benchmark_name="b",
                    config_name="c",
                    run_index=0,
                    per_sample_scores=[
                        {"_sample_id": "q1", "_retrieved_ids": ["d1", "d2"], "ndcg@10": 1.0}
                    ],
                )
            ]
        )
        assert results.per_query()[0]["_retrieved_ids"] == ["d1", "d2"]

    def test_a_boolean_is_not_treated_as_a_number(self):
        results = BenchmarkResults(
            [
                TaskResult(
                    task_key="t",
                    benchmark_name="b",
                    config_name="c",
                    run_index=0,
                    per_sample_scores=[{"_sample_id": "q1", "abstained": True}],
                )
            ]
        )
        assert results.per_query()[0]["abstained"] is True


class TestReportingOptionsAreNotPartOfTheSystem:
    """`save_trajectories` decides how much is *recorded*, not what runs.

    It was absent from the harness-key set, so it was merged into the pipeline
    config and hashed into `config_hash` -- which is the session id and the
    checkpoint key. Turning on trajectory logging produced a different session,
    a cold corpus cache, a checkpoint that could not resume, and a run that
    compared as a different system to the identical run without it.
    """

    def _task(self, **overrides):
        return BenchmarkTask(
            benchmark_name="b",
            config_name="c",
            config_overrides=overrides,
            run_index=0,
        )

    def test_it_is_declared_a_harness_key(self):
        assert "save_trajectories" in HARNESS_KEYS

    def test_it_does_not_change_the_config_hash(self):
        plain = self._task(enable_hyde=True)
        logged = self._task(enable_hyde=True, save_trajectories="full")
        assert plain.config_hash == logged.config_hash

    def test_a_real_setting_still_does(self):
        """Guards the guard: an exclusion that swallowed everything would also
        pass the test above."""
        assert self._task(enable_hyde=True).config_hash != self._task(enable_hyde=False).config_hash

    def test_the_task_key_is_unchanged_too(self):
        """It is the checkpoint key; a change silently discards a resume."""
        plain = self._task(enable_hyde=True)
        logged = self._task(enable_hyde=True, save_trajectories="full")
        assert plain.task_key == logged.task_key

    def test_it_never_reaches_the_pipeline_config(self):
        from lasagna.knowledge_evaluation_sdk.experiment import _apply_overrides_to_pipeline

        class _Pipeline:
            _config: dict = {}
            _component_names: dict = {}

            def replace(self, *a, **k):
                raise AssertionError("no component should be replaced here")

        pipeline = _Pipeline()
        try:
            _apply_overrides_to_pipeline(pipeline, {"save_trajectories": "full"}, "session")
        except Exception:
            # Session publishing needs a live store; the config merge above it
            # is what this asserts.
            pass
        assert "save_trajectories" not in pipeline._config


class TestTheCliExports:
    def _write(self, tmp_path):
        path = tmp_path / "results.json"
        _mixed_run().save(str(path))
        return path

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "lasagna", "results", "per-query", *args],
            capture_output=True,
            text=True,
        )

    def test_jsonl_export_keeps_every_row(self, tmp_path):
        out = tmp_path / "rows.jsonl"
        result = self._run(str(self._write(tmp_path)), "-o", str(out))
        assert result.returncode == 0, result.stderr
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert len(rows) == 2
        assert {"ndcg@10", "faithfulness"} <= set(rows[0])

    def test_csv_export_drops_only_the_underscore_fields(self, tmp_path):
        import csv

        out = tmp_path / "rows.csv"
        assert self._run(str(self._write(tmp_path)), "-o", str(out)).returncode == 0
        rows = list(csv.DictReader(out.open()))
        assert len(rows) == 2
        assert "ndcg@10" in rows[0] and "faithfulness" in rows[0]
        assert not [c for c in rows[0] if c.startswith("_")]

    def test_sorting_puts_the_worst_query_first(self, tmp_path):
        result = self._run(str(self._write(tmp_path)), "--sort-by", "ndcg@10")
        assert result.returncode == 0, result.stderr
        assert result.stdout.index("q2") < result.stdout.index("q1")

    def test_an_unknown_sort_metric_lists_the_real_ones(self, tmp_path):
        result = self._run(str(self._write(tmp_path)), "--sort-by", "nonsense")
        assert result.returncode != 0
        assert "ndcg@10" in result.stdout + result.stderr


class TestThePrintedTable:
    """The table a researcher reads before exporting anything."""

    def _file(self, tmp_path, configs, n=6):
        rows = []
        for c, name in enumerate(configs):
            rows.append(
                {
                    "task_key": name,
                    "run_index": 0,
                    "benchmark_name": "bench",
                    "config_name": name,
                    "metrics": {},
                    "per_sample_scores": [
                        {
                            "_sample_id": f"q{i}",
                            "ndcg@10": 0.1 * ((i + c) % 9),
                            "f1@10": 0.5,
                            "hit_rate@10": 1.0,
                            "input_tokens": 3.0,
                            "latency_ms": 40.0,
                            "map@10": 0.4,
                            "mrr@10": 0.4,
                            "precision@10": 0.1,
                        }
                        for i in range(n)
                    ],
                }
            )
        path = tmp_path / "results.json"
        path.write_text(json.dumps({"results": rows}))
        return str(path)

    def _invoke(self, *args):
        typer_testing = pytest.importorskip("typer.testing")
        from lasagna.cli import app

        return typer_testing.CliRunner().invoke(app, ["results", "per-query", *args])

    def test_the_sort_column_is_shown(self, tmp_path):
        """`--sort-by ndcg@10` showed the first six metrics alphabetically,
        which left out ndcg@10 itself."""
        out = self._invoke(self._file(tmp_path, ["a", "b"]), "--sort-by", "ndcg@10")
        assert out.exit_code == 0, out.output
        # Only ndcg@10 takes these values in the fixture.
        assert "0.600" in out.output and "0.300" in out.output

    def test_two_long_config_names_can_be_told_apart(self, tmp_path):
        """Both arms printed as `reranke…`."""
        names = ["reranker=cross_encoder", "reranker=my_tech:soft_overlap"]
        out = self._invoke(self._file(tmp_path, names))
        assert out.exit_code == 0, out.output
        for name in names:
            assert name in out.output

    def test_the_header_counts_queries_and_configs_not_rows(self, tmp_path):
        """50 queries x 2 arms was headed "100 queries"."""
        out = self._invoke(self._file(tmp_path, ["a", "b"], n=5))
        assert out.exit_code == 0, out.output
        assert "10 rows" in out.output and "5 queries" in out.output
        assert "10 queries" not in out.output
