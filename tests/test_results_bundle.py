"""What decides whether two researchers' numbers may sit in the same table.

The comparability key hashes everything held fixed and deliberately excludes the
dimension under test. Both halves matter: a key over *everything* makes every
row incomparable with every other (the point of a comparison is that one thing
differs), and a key over *nothing* makes all rows comparable regardless of what
they measured — which is what `compare()` effectively assumed before.
"""

from __future__ import annotations

from lasagna.knowledge_evaluation_sdk.bundle import (
    HELD_FIXED_FIELDS,
    METRICS_VERSION,
    _is_floating_alias,
    comparability_key,
    compare_keys,
    varied_dimensions,
)


def _held(**over):
    base = {
        "metrics_version": METRICS_VERSION,
        "benchmark": "beir/scifact",
        "split": "test",
        "n_samples": 200,
        "sampling_seed": 42,
        "sample_id_digest": "abc123",
        "corpus_fingerprint": "def456",
        "resolved_mode": "live",
        "embedder_model": "BAAI/bge-m3",
        "embedder_dim": 1024,
        "chunker": "fixed",
        "retriever_top_k": 20,
        "vector_backend": "milvus",
        "judge_model": "gpt-4o-2024-08-06",
    }
    base.update(over)
    return base


class TestTheKeySeparatesLikeFromUnlike:
    def test_identical_configurations_share_a_key(self):
        assert comparability_key(_held()) == comparability_key(_held())

    def test_key_is_order_independent(self):
        a = _held()
        b = {k: a[k] for k in reversed(list(a))}
        assert comparability_key(a) == comparability_key(b)

    def test_every_held_fixed_field_changes_the_key(self):
        """If a field can change without changing the key, two runs that differ
        on it would be presented as comparable."""
        baseline = comparability_key(_held())
        for field in HELD_FIXED_FIELDS:
            changed = comparability_key(_held(**{field: "SOMETHING-ELSE"}))
            assert changed != baseline, (
                f"changing {field!r} left the comparability key unchanged, so two "
                f"runs differing on it would be treated as comparable"
            )

    def test_the_dimension_under_test_is_not_in_the_key(self):
        """The reranker is what varies; including it would make every arm of a
        sweep incomparable with every other arm."""
        assert "reranker" not in HELD_FIXED_FIELDS
        assert "query_enhancer" not in HELD_FIXED_FIELDS

    def test_metrics_version_is_in_the_key(self):
        """A changed metric definition must drop old numbers out of comparison
        rather than silently mix them with new ones."""
        assert "metrics_version" in HELD_FIXED_FIELDS
        assert comparability_key(_held(metrics_version="9.9")) != comparability_key(_held())

    def test_a_missing_field_does_not_compare_equal_to_a_present_one(self):
        """Absent must differ from known, or a bundle that failed to capture
        the embedding model would match one that did."""
        partial = _held()
        del partial["embedder_model"]
        assert comparability_key(partial) != comparability_key(_held())


class TestMismatchesAreExplained:
    def test_differences_are_reported_field_by_field(self):
        diffs = compare_keys(_held(), _held(n_samples=100, embedder_model="e5-large"))
        joined = " ".join(diffs)
        assert "n_samples" in joined and "embedder_model" in joined
        assert "200" in joined and "100" in joined

    def test_identical_blocks_report_no_differences(self):
        assert compare_keys(_held(), _held()) == []


class TestVariedDimensions:
    def test_component_overrides_are_reported_as_varied(self):
        assert varied_dimensions({"reranker": "mine"}) == ["reranker"]

    def test_config_flags_are_dimensions_too(self):
        """A sweep expressed as flags is still a sweep.

        This used to filter against an allowlist of the six component slots, so
        an arm that turned HyDE on and a baseline that did not produced the same
        answer -- `[]` -- and the leaderboard labelled both `_baseline_`. Any
        experiment run through the `config_overrides` branch, which is the only
        branch that can express a boolean "off" and therefore the only one a 2x2
        can use, reported that it varied nothing.
        """
        assert varied_dimensions({"enable_hyde": True, "top_k": 5}) == [
            "enable_hyde",
            "top_k",
        ]

    def test_derived_keys_are_not_dimensions(self):
        """`collection_name` is computed from `embedding_model`, not chosen."""
        assert varied_dimensions({"embedding_model": "e5", "collection_name": "rag_x"}) == [
            "embedding_model"
        ]

    def test_harness_keys_are_not_dimensions(self):
        """These steer the runner; they never reach the pipeline."""
        assert varied_dimensions({"allow_inert_overrides": True, "save_trajectories": "full"}) == []

    def test_a_true_baseline_still_varies_nothing(self):
        assert varied_dimensions({}) == []


class TestIntegrityFlags:
    def test_floating_judge_aliases_are_detected(self):
        """`gpt-4o-mini` points at different weights over time; a dated
        snapshot does not."""
        assert _is_floating_alias("gpt-4o-mini") is True
        assert _is_floating_alias("gpt-4o-2024-08-06") is False
        assert _is_floating_alias("gpt-4o-mini-2024-07-18") is False

    def test_bundle_flags_single_run_and_alias(self):
        from lasagna.knowledge_evaluation_sdk.bundle import _integrity

        integ = _integrity(
            [{"benchmark_name": "b", "corpus_fingerprint": "", "split": "test"}],
            {"n_runs": 1, "judge_model": "gpt-4o-mini", "determinism": "seeded"},
        )
        assert "SINGLE_RUN" in integ["warnings"]
        assert "UNRESOLVED_JUDGE_ALIAS" in integ["warnings"]
        assert "NO_CORPUS_FINGERPRINT" in integ["warnings"]

    def test_clean_run_has_no_warnings(self):
        from lasagna.knowledge_evaluation_sdk.bundle import _integrity

        # A clean live row carries its stage manifest; without one, isolation
        # is unverified, which is itself a warning.
        integ = _integrity(
            [
                {
                    "benchmark_name": "b",
                    "corpus_fingerprint": "abc",
                    "split": "test",
                    "error": None,
                    "system": {"stages": {"reranker": {"digest": "d"}}},
                }
            ],
            {"n_runs": 3, "judge_model": "gpt-4o-2024-08-06", "determinism": "seeded"},
        )
        assert integ["warnings"] == []


class TestBundleIdentity:
    def test_editing_a_metric_invalidates_the_run_id(self):
        """The id covers the content, so a hand-edited number no longer matches
        — which is what lets a validator reject tampering without rerunning."""
        from lasagna.knowledge_evaluation_sdk.bundle import _content_hash

        bundle = {"results": [{"metrics": {"ndcg@10": 0.71}}], "run_id": "x"}
        before = _content_hash(bundle)
        bundle["results"][0]["metrics"]["ndcg@10"] = 0.91
        assert _content_hash(bundle) != before


class TestTheKeyCanTellConfigurationsApart:
    """Five of the fourteen key fields were structurally always None.

    `held_fixed_from` sources `embedder_model`, `embedder_dim`, `chunker`,
    `retriever_top_k` and `vector_backend` from `run_meta["system"]`, and
    `_run_metadata` never produced that block. So two runs on *different
    embedding models* hashed to the same comparability key and were ranked in
    the same leaderboard table.

    A key that cannot distinguish those is worse than no key, because the
    grouping looks principled while being arbitrary.
    """

    def test_run_metadata_carries_the_system_under_test(self):
        from lasagna.knowledge_evaluation_sdk.experiment import ExperimentConfig, _run_metadata
        from lasagna.knowledge_evaluation_sdk.experiment import BenchmarkSpec

        meta = _run_metadata(ExperimentConfig(name="t", benchmarks=[BenchmarkSpec(name="smoke")]))
        system = meta.get("system") or {}
        assert system.get("embedder_model"), "no embedder recorded — the key cannot group by it"
        assert system.get("embedder_dim"), "no embedding dimension recorded"
        assert system.get("vector_backend"), "no vector backend recorded"

    def test_no_key_field_is_silently_empty(self):
        from lasagna.knowledge_evaluation_sdk.bundle import HELD_FIXED_FIELDS, held_fixed_from
        from lasagna.knowledge_evaluation_sdk.experiment import (
            BenchmarkSpec,
            ExperimentConfig,
            _run_metadata,
        )

        meta = _run_metadata(ExperimentConfig(name="t", benchmarks=[BenchmarkSpec(name="smoke")]))
        row = {
            "benchmark_name": "smoke",
            "split": "test",
            "n_samples": 12,
            "sampling_seed": 42,
            "sample_id_digest": "abc",
            "corpus_fingerprint": "def",
            # Defaults to "native" on TaskResult, so a real row always carries
            # it; the fixture must too, or this asserts a gap that cannot occur.
            "corpus_provenance": "native",
            "resolved_mode": "live",
        }
        held = held_fixed_from(row, meta)
        # `judge_model` is legitimately absent: a retrieval-only run has no LLM
        # judge, and recording None there is accurate rather than a gap.
        # Everything else describes the pipeline and must be populated.
        optional = {"judge_model"}
        empty = sorted(f for f in HELD_FIXED_FIELDS if held.get(f) is None)
        empty = [f for f in empty if f not in optional]
        assert not empty, (
            f"{len(empty)} of {len(HELD_FIXED_FIELDS)} comparability-key fields are "
            f"None, so runs differing in them share a key: {empty}"
        )

    def test_a_different_embedder_produces_a_different_key(self):
        """The property the whole design rests on."""
        from lasagna.knowledge_evaluation_sdk.bundle import comparability_key, held_fixed_from

        row = {"benchmark_name": "smoke", "split": "test", "n_samples": 12}
        a = comparability_key(held_fixed_from(row, {"system": {"embedder_model": "bge-m3"}}))
        b = comparability_key(held_fixed_from(row, {"system": {"embedder_model": "e5-large"}}))
        assert a != b, "two different embedding models hashed to the same key"


class TestBlockingIntegrityActuallyBlocks:
    """`blocking` was initialised and never appended to, so every bundle passed.

    The hub's stated position is that the validator is the product. A validator
    with no rejection condition is a formatter. These are deliberately narrow:
    a bundle is blocked when it is internally incoherent or asserts more than
    the recorded data supports — a weak experiment stays a warning, because a
    pilot is allowed to be a pilot.
    """

    def _rows(self, **overrides):
        row = {
            "benchmark_name": "beir/scifact",
            "config_name": "baseline",
            "metrics": {"ndcg@10": 0.5},
            "split": "test",
            "resolved_mode": "live",
            "hf_revision": "abc123",
            "corpus_fingerprint": "def456",
            "corpus_truncated": False,
        }
        row.update(overrides)
        return [row]

    def _integrity(self, rows, **meta):
        from lasagna.knowledge_evaluation_sdk.bundle import _integrity

        run_meta = {"n_runs": 3, "determinism": "seeded"}
        run_meta.update(meta)
        return _integrity(rows, run_meta)

    def test_a_clean_bundle_blocks_on_nothing(self):
        assert self._integrity(self._rows())["blocking"] == []

    def test_a_truncated_corpus_blocks(self):
        out = self._integrity(self._rows(corpus_truncated=True))
        assert "TRUNCATED_CORPUS" in out["blocking"]

    def test_an_unpinned_live_dataset_warns_but_does_not_block(self):
        """A rule nobody can satisfy is not a standard, it is an outage.

        Nothing pins revisions yet, so blocking this made every live run
        unpublishable the moment the code landed. `--strict` still promotes it,
        and `--strict` is what the hub runs — so publishing requires a pin while
        local work is unaffected.
        """
        out = self._integrity(self._rows(hf_revision=""))
        assert "UNPINNED_DATASET" in out["warnings"]
        assert "UNPINNED_DATASET" not in out["blocking"]

    def test_generation_only_does_not_need_a_revision(self):
        """No corpus is loaded, so there is no upstream to pin."""
        out = self._integrity(self._rows(resolved_mode="generation_only", hf_revision=""))
        assert "UNPINNED_DATASET" not in out["blocking"]

    def test_significance_on_one_run_blocks(self):
        out = self._integrity(self._rows(significance={"p_value": 0.01}), n_runs=1)
        assert "SIGNIFICANCE_WITHOUT_REPLICATION" in out["blocking"]

    def test_one_run_without_a_significance_claim_is_only_a_warning(self):
        """A pilot is allowed to be a pilot."""
        out = self._integrity(self._rows(), n_runs=1)
        assert out["blocking"] == []
        assert "SINGLE_RUN" in out["warnings"]

    def test_an_empty_bundle_blocks(self):
        assert "NO_RESULTS" in self._integrity([])["blocking"]


class TestTheValidatorRejectsBlockedBundles:
    def test_validate_exits_nonzero_on_a_blocking_code(self, tmp_path):
        """End to end through the CLI, not just the integrity function."""
        import json
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        from lasagna.knowledge_evaluation_sdk.bundle import build_bundle

        class _R:
            _results = [
                {
                    "benchmark_name": "beir/scifact",
                    "config_name": "c",
                    "metrics": {"ndcg@10": 0.5},
                    "resolved_mode": "live",
                    "corpus_truncated": True,
                    "hf_revision": "abc",
                }
            ]
            experiment_config = {
                "n_runs": 3,
                "determinism": "seeded",
                "environment": {
                    "lasagna_version": "0.1.0",
                    "python": "3.12",
                    "platform": "test",
                },
            }

        path = tmp_path / "run.json"
        path.write_text(json.dumps(build_bundle(_R())))

        import os

        env = dict(os.environ, PYTHONPATH=str(repo / "src"))
        r = subprocess.run(
            [sys.executable, "-m", "lasagna", "results", "validate", str(path)],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        assert r.returncode != 0, "a truncated-corpus bundle validated cleanly"
        assert "TRUNCATED_CORPUS" in r.stdout + r.stderr
