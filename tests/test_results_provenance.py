"""A results file records what its numbers were computed under, and a bundle keeps it.

`results.json` recorded neither the metric definitions nor the package version,
the environment's setting overrides, nor which commit of each model ran; and
`results bundle` stamped this build's METRICS_VERSION on whatever it was given,
so a file computed under 1.4 was published as 1.5. `--resume` restored a
checkpoint computed under another judge, version or setting. Every run said
`determinism: seeded`, though concurrent tasks share the process-wide random
generators. A checkpoint was renamed into place before its data was flushed,
so a crash could leave an empty file that later read as a finished task.
"""

from __future__ import annotations

import json
import os

import pytest

from lasagna.knowledge_evaluation_sdk import bundle as bundle_module
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION, build_bundle
from lasagna.knowledge_evaluation_sdk.experiment import (
    BenchmarkSpec,
    Experiment,
    ExperimentConfig,
    _run_metadata,
)
from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults
from lasagna.knowledge_evaluation_sdk.runner import TaskResult
from lasagna.sdk._version import __version__


def _cfg(**kwargs):
    return ExperimentConfig(name="t", benchmarks=[BenchmarkSpec(name="smoke")], **kwargs)


def _row(**extra):
    row = {
        "benchmark_name": "smoke",
        "config_name": "c",
        "run_index": 0,
        "metrics": {"ndcg@10": 0.5},
        "split": "test",
        "resolved_mode": "live",
    }
    row.update(extra)
    return row


class _Results:
    def __init__(self, meta):
        self._results = [_row()]
        self.experiment_config = meta


class TestTheRunRecordsWhatItRanUnder:
    def test_metrics_version_package_version_and_overrides(self, monkeypatch):
        monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
        monkeypatch.setenv("RAG_VECTOR_BACKEND_CONFIG", '{"uri": "http://db:19530", "token": "s"}')
        monkeypatch.setenv("RAG_EVAL_JUDGE_MODEL", "gpt-4o-2024-08-06")
        monkeypatch.setenv("RAG_JWT_SECRET", "never-recorded")
        meta = _run_metadata(_cfg())
        assert meta["metrics_version"] == METRICS_VERSION
        assert meta["package_version"] == __version__
        overrides = meta["rag_overrides"]
        assert overrides["RAG_CHUNK_SIZE"] == "500"
        assert overrides["RAG_EVAL_JUDGE_MODEL"] == "gpt-4o-2024-08-06"
        assert overrides["RAG_VECTOR_BACKEND_CONFIG"] == {
            "uri": "<redacted>",
            "token": "<redacted>",
        }
        assert "never-recorded" not in json.dumps(meta)

    def test_save_puts_the_versions_beside_the_results(self, tmp_path):
        path = tmp_path / "results.json"
        meta = {"metrics_version": METRICS_VERSION, "package_version": __version__}
        BenchmarkResults([TaskResult("k", "smoke", "c", 0)], "e", meta).save(str(path))
        data = json.loads(path.read_text())
        assert (data["metrics_version"], data["package_version"]) == (METRICS_VERSION, __version__)
        assert BenchmarkResults.load(str(path)).experiment_config["metrics_version"] == (
            METRICS_VERSION
        )

    def test_saving_an_old_file_does_not_stamp_it_with_this_builds_version(self, tmp_path):
        old = tmp_path / "old.json"
        old.write_text(json.dumps({"experiment_config": {"n_runs": 1}, "results": []}))
        BenchmarkResults.load(str(old)).save(str(tmp_path / "again.json"))
        again = json.loads((tmp_path / "again.json").read_text())
        assert again["metrics_version"] is None
        assert "metrics_version" not in again["experiment_config"]


class TestModelRevisions:
    @pytest.fixture
    def cache(self, tmp_path):
        repo = tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2"
        snapshot = repo / "snapshots" / "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
        return str(tmp_path)

    def test_a_cached_model_resolves_to_its_commit_offline(self, cache):
        from lasagna.knowledge_evaluation_sdk.run_provenance import model_revisions

        assert model_revisions(
            ["all-MiniLM-L6-v2", "BAAI/never-downloaded", "my_ideas:keyword_overlap"],
            cache_dir=cache,
        ) == {
            "BAAI/never-downloaded": None,
            "all-MiniLM-L6-v2": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            "my_ideas:keyword_overlap": None,
        }

    def test_the_models_a_run_used_are_its_tasks_embedder_reranker_and_similarity_model(self):
        from lasagna.knowledge_evaluation_sdk.run_provenance import models_used

        rows = [
            {"system": {"embedder_model": "all-MiniLM-L6-v2", "reranker_model": "my:fn"}},
            {"system": {"reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"}},
        ]
        assert models_used(rows, {"similarity_model": "BAAI/bge-small-en-v1.5"}) == [
            "BAAI/bge-small-en-v1.5",
            "all-MiniLM-L6-v2",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ]


class TestTheBundleUsesTheRunsVersion:
    def test_held_fixed_carries_the_runs_metrics_version_not_this_builds(self):
        held = bundle_module.held_fixed_from(_row(), {"metrics_version": "1.4"})
        assert held["metrics_version"] == "1.4"

    def test_a_result_without_a_metrics_version_is_refused_with_how_to_rerun(self):
        error = getattr(bundle_module, "MetricsVersionError", ValueError)
        with pytest.raises(error) as raised:
            build_bundle(_Results({"n_runs": 3}))
        message = str(raised.value)
        assert "no metrics_version" in message
        assert "lasagna benchmark experiment" in message

    def test_a_result_from_other_metric_definitions_is_refused(self):
        error = getattr(bundle_module, "MetricsVersionError", ValueError)
        with pytest.raises(error) as raised:
            build_bundle(_Results({"metrics_version": "1.4", "package_version": "0.1.0a1"}))
        message = str(raised.value)
        assert "1.4" in message and METRICS_VERSION in message and "0.1.0a1" in message

    def test_a_current_result_bundles_with_its_version_throughout(self):
        out = build_bundle(_Results({"metrics_version": METRICS_VERSION, "n_runs": 3}))
        assert out["metrics_version"] == METRICS_VERSION
        assert {c["held_fixed"]["metrics_version"] for c in out["configs"]} == {METRICS_VERSION}

    def test_validate_lists_a_missing_or_different_version(self):
        good = build_bundle(_Results({"metrics_version": METRICS_VERSION, "n_runs": 3}))
        assert bundle_module.metrics_version_problems(good) == []
        older = dict(good, metrics_version="1.4")
        assert any("1.4" in p for p in bundle_module.metrics_version_problems(older))
        missing = {k: v for k, v in good.items() if k != "metrics_version"}
        assert any(
            "no metrics_version" in p for p in bundle_module.metrics_version_problems(missing)
        )


class TestResume:
    def _keys(self, **cfg):
        return {t.task_key for t in Experiment(_cfg(**cfg)).generate_tasks()}

    def test_a_checkpoint_is_not_reused_under_another_judge_or_setting(self, monkeypatch):
        monkeypatch.delenv("RAG_EVAL_JUDGE_MODEL", raising=False)
        base = self._keys()
        assert base == self._keys(), "the key must be stable for an unchanged run"
        assert base.isdisjoint(self._keys(judge_model="gpt-4o-2024-08-06"))
        monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
        assert base.isdisjoint(self._keys())

    def test_a_checkpoint_is_not_reused_under_other_metric_definitions(self, monkeypatch):
        base = self._keys()
        monkeypatch.setattr(bundle_module, "METRICS_VERSION", "9.9")
        assert base.isdisjoint(self._keys())

    def test_the_runner_does_not_restore_a_checkpoint_from_another_environment(
        self, tmp_path, monkeypatch
    ):
        import asyncio

        from lasagna.knowledge_evaluation_sdk.checkpoint import CheckpointManager
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkRunner

        cp = CheckpointManager(str(tmp_path))
        task = Experiment(_cfg()).generate_tasks()[0]
        cp.save(task.task_key, TaskResult(task.task_key, "smoke", task.config_name, 0).to_dict())

        monkeypatch.setenv("RAG_CHUNK_SIZE", "500")
        rerun = Experiment(_cfg()).generate_tasks()[0]
        ran = []

        async def _eval(t, samples):
            ran.append(t.task_key)
            return TaskResult(t.task_key, t.benchmark_name, t.config_name, t.run_index)

        asyncio.run(BenchmarkRunner(max_concurrency=1).run_tasks([rerun], _eval, cp))
        assert ran == [rerun.task_key]


class TestDeterminism:
    def test_the_label_says_what_seeding_guarantees(self):
        assert _run_metadata(_cfg(max_concurrency=1))["determinism"] == "seeded_per_task"
        assert _run_metadata(_cfg(max_concurrency=4))["determinism"] == ("seeded_global_concurrent")
        # A sweep the harness runs one task at a time records that.
        assert _run_metadata(_cfg(max_concurrency=4), concurrency=1)["determinism"] == (
            "seeded_per_task"
        )

    def test_both_labels_count_as_seeded_in_a_bundle(self):
        from lasagna.knowledge_evaluation_sdk.bundle import _integrity, json_schema

        allowed = json_schema()["properties"]["experiment"]["properties"]["determinism"]["enum"]
        for label in ("seeded_per_task", "seeded_global_concurrent"):
            assert label in allowed
            assert (
                "UNSEEDED"
                not in _integrity([_row()], {"n_runs": 3, "determinism": label})["warnings"]
            )


def test_a_checkpoint_reaches_the_disk_before_it_is_renamed_into_place(tmp_path, monkeypatch):
    from pathlib import Path

    from lasagna.knowledge_evaluation_sdk import checkpoint

    order = []
    real_rename = Path.rename
    monkeypatch.setattr(checkpoint.os, "fsync", lambda fd: order.append("fsync"), raising=False)
    monkeypatch.setattr(
        Path, "rename", lambda self, target: (order.append("rename"), real_rename(self, target))[1]
    )
    checkpoint.CheckpointManager(str(tmp_path)).save("k", {"metrics": {}})
    assert order == ["fsync", "rename"]
    assert os.listdir(tmp_path / ".checkpoints")
