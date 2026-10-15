"""A number is only comparable if you know what produced it.

`results.json` used to record four things: mode, n_runs, seed, and the benchmark
names. Two researchers reporting `ndcg@10 = 0.71` on "the same benchmark" could
differ in embedding model, library version, hardware, split, sample count or
judge model, and nothing in either file would say so.

The capture logic already existed twice — in the release gate and in
`core/pipeline` — in places the evaluation path never reached. These tests cover
the merged version and, as importantly, what it must *not* contain: bundles get
published, so a home directory, a hostname or the name of a credential variable
in one is a disclosure.
"""

from __future__ import annotations

import json
import os

from lasagna.core.provenance import capture, redact


class TestCaptureRecordsWhatMovesAMetric:
    def test_library_versions_are_recorded(self):
        env = capture()
        # Each of these can change a metric without any change to this repo.
        for key in (
            "lasagna_version",
            "torch_version",
            "sentence_transformers_version",
            "ragas_version",
            "datasets_version",
            "numpy_version",
        ):
            assert key in env, f"{key} missing; a version bump would be invisible"

    def test_hardware_is_recorded(self):
        env = capture()
        for key in ("python", "platform", "machine", "cpu_count", "ram_gb", "gpu"):
            assert key in env
        # A p95 or a float-reduction order without the machine is not comparable.
        assert env["platform"]

    def test_install_provenance_distinguishes_source_from_wheel(self):
        env = capture()
        assert env["installed_from"] in {"source", "wheel", "unknown"}

    def test_git_sha_only_claimed_for_source_installs(self):
        """A SHA from the user's cwd says nothing about an installed wheel."""
        env = capture()
        if env["installed_from"] == "wheel":
            assert env["git_commit"] is None

    def test_dirty_tree_is_flagged(self):
        env = capture()
        if env.get("git_commit"):
            assert env["git_dirty"] is not None, (
                "a dirty tree means the recorded SHA does not describe what ran"
            )


class TestCaptureIsSafeToPublish:
    def test_no_credential_names(self):
        """The gate's version records *which* API-key vars were set. Useful in
        a private CI report; in a published bundle it discloses which providers
        a lab uses."""
        blob = json.dumps(capture())
        assert "API_KEY" not in blob
        assert "APIKEY" not in blob

    def test_credential_presence_is_a_boolean(self):
        env = capture()
        assert isinstance(env["llm_credentials_present"], bool)

    def test_no_home_directory_or_absolute_paths(self):
        blob = json.dumps(capture())
        assert os.path.expanduser("~") not in blob
        assert "/Users/" not in blob and "/home/" not in blob

    def test_no_hostname(self):
        import socket

        host = socket.gethostname()
        if host and len(host) > 3:
            assert host not in json.dumps(capture())


class TestRedaction:
    def test_home_and_temp_are_replaced(self):
        home = os.path.expanduser("~")
        out = redact(f"failed at {home}/secret/spot")
        assert home not in out and "<home>" in out

    def test_macos_per_user_temp_token_is_removed(self):
        out = redact("/var/folders/xg/nvyhfcwx78g1djt01/T/lasagna-p1/venv")
        assert "nvyhfcwx78g1djt01" not in out
        assert "lasagna-p1" in out, "the useful part of the path must survive"

    def test_nested_structures_are_redacted(self):
        home = os.path.expanduser("~")
        out = redact({"a": [f"{home}/x"], "b": {"c": f"{home}/y"}})
        assert out["a"][0].startswith("<home>")
        assert out["b"]["c"].startswith("<home>")

    def test_non_strings_pass_through(self):
        assert redact({"n": 1, "f": 1.5, "b": True, "none": None}) == {
            "n": 1,
            "f": 1.5,
            "b": True,
            "none": None,
        }


class TestResultsCarryTheirOwnProvenance:
    def test_task_result_serialises_the_fields_that_were_discarded(self):
        from lasagna.knowledge_evaluation_sdk.runner import TaskResult

        d = TaskResult(task_key="k", benchmark_name="b", config_name="c", run_index=0).to_dict()
        for key in (
            "corpus_fingerprint",  # was computed for cache keys, never emitted
            "resolved_mode",
            "mode_resolution_reason",
            "variance_source",
            "split",
            "sampling_seed",
            "sample_id_digest",
        ):
            assert key in d, f"{key} must reach the results file"

    def test_the_runner_populates_them(self):
        from tests.test_event_loop_safety import executable_source
        from lasagna.knowledge_evaluation_sdk.runner import BenchmarkRunner

        src = executable_source(BenchmarkRunner._execute_task)
        for field in ("corpus_fingerprint", "resolved_mode", "sample_id_digest"):
            assert f"result.{field}" in src, f"{field} is declared but never set"

    def test_sample_digest_identifies_the_question_set_without_shipping_it(self):
        from lasagna.knowledge_evaluation_sdk.runner import _sample_id_digest
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

        samples = benchmark_registry.get("smoke").load()
        digest = _sample_id_digest(samples)
        assert digest and len(digest) == 16
        # Stable, order-independent, and reveals nothing about the questions.
        assert digest == _sample_id_digest(list(reversed(samples)))
        assert not any(s.question[:12] in digest for s in samples)

    def test_run_metadata_records_the_resolved_sweep(self):
        """A one-way config hash cannot be read back, so the sweep itself has
        to be in the file or a reader cannot tell what was varied."""
        from lasagna.knowledge_evaluation_sdk.experiment import (
            ExperimentConfig,
            PipelineSweepSpec,
            _run_metadata,
        )

        cfg = ExperimentConfig(name="x", pipeline=PipelineSweepSpec(rerankers=["a", "b"]))
        meta = _run_metadata(cfg)
        assert meta["pipeline"]["rerankers"] == ["a", "b"]
        assert meta["variance_source"] == "run"
        # seeded_per_task or seeded_global_concurrent: never "reproducible".
        assert meta["determinism"].startswith("seeded"), "never claim bit-reproducibility"
        assert "environment" in meta
        assert os.path.expanduser("~") not in json.dumps(meta)
