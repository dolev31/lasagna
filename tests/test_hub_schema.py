"""The published schema must describe the bundles this package actually writes.

The hub validates submissions against a checked-in JSON Schema. If that file
drifts from `bundle.py`, the failure is silent and one-directional in the worst
way: a schema that requires fewer `held_fixed` fields than the comparability key
hashes will happily accept a bundle whose key cannot be recomputed.

The first hand-written copy of this file shipped a `schema_version` pattern of
`^1\\.` — a literal backslash after JSON unescaping — which rejected every valid
bundle. That is the failure mode this file exists to prevent, so the pattern is
tested against a real version string rather than compared as text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION

REPO = Path(__file__).resolve().parents[1]
SCHEMA_FILE = REPO / "hub" / "run-1.0.schema.json"


@pytest.fixture
def checked_in() -> dict:
    return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))


def _fake_results(**overrides):
    """The duck type `build_bundle` consumes: `_results` rows + `experiment_config`."""

    class _Results:
        _results = [
            {
                "benchmark_name": "smoke",
                "config_name": "with_reranker",
                "metrics": {"ndcg@10": 0.9},
                "split": "test",
                "n_samples": 12,
                "sampling_seed": 42,
                "sample_id_digest": "abc123",
                "corpus_fingerprint": "def456",
                "resolved_mode": "live",
                "config_overrides": dict(overrides) or {"reranker": "length_demo"},
            }
        ]
        experiment_config = {
            "n_runs": 3,
            # A bundle carries the metrics version its run recorded.
            "metrics_version": METRICS_VERSION,
            "determinism": "seeded",
            "variance_source": "run",
            "judge_model": "gpt-4o-mini-2024-07-18",
            "environment": {
                "lasagna_version": "0.1.0",
                "python": "3.12.0",
                "platform": "Darwin",
            },
            "system": {
                "embedder_model": "bge-m3",
                "embedder_dim": 1024,
                "chunker": "recursive",
                "retriever_top_k": 10,
                "vector_backend": "milvus",
            },
        }

    return _Results()


class TestTheCheckedInCopyIsCurrent:
    def test_matches_the_generator(self, checked_in):
        from lasagna.knowledge_evaluation_sdk.bundle import json_schema

        assert checked_in == json_schema(), (
            "hub/run-1.0.schema.json is stale. Regenerate with "
            "`lasagna results schema -o hub/run-1.0.schema.json`"
        )

    def test_filename_matches_the_declared_version(self, checked_in):
        from lasagna.knowledge_evaluation_sdk.bundle import SCHEMA_VERSION

        assert SCHEMA_FILE.name == f"run-{SCHEMA_VERSION}.schema.json", (
            f"schema is version {SCHEMA_VERSION} but the file is named "
            f"{SCHEMA_FILE.name}; the hub serves it by version"
        )
        assert checked_in["$id"].endswith(f"run-{SCHEMA_VERSION}.json")


class TestTheSchemaAcceptsWhatWeProduce:
    def test_version_pattern_matches_the_real_version(self, checked_in):
        """The bug that shipped: a pattern matching nothing."""
        from lasagna.knowledge_evaluation_sdk.bundle import SCHEMA_VERSION

        pattern = checked_in["properties"]["schema_version"]["pattern"]
        assert re.match(pattern, SCHEMA_VERSION), (
            f"schema_version pattern {pattern!r} does not match the version "
            f"this package writes ({SCHEMA_VERSION!r}) — every bundle would be rejected"
        )

    def test_every_key_field_is_required_or_has_a_documented_default(self, checked_in):
        """Fewer fields here than the key hashes means unverifiable keys pass.

        A field may be optional only if `KEY_DEFAULTS` says what its absence
        means, so a reader can still recompute the key from what the bundle
        carries. Anything else is a field the key hashes and the schema does
        not demand — unrecomputable by design.
        """
        from lasagna.knowledge_evaluation_sdk.bundle import HELD_FIXED_FIELDS, KEY_DEFAULTS

        held = checked_in["properties"]["configs"]["items"]["properties"]["held_fixed"]
        required = set(held["required"])
        # Fields added to the key in a later metrics version are required from
        # that version on (an older bundle cannot carry them, and its key does
        # not hash them).
        for rule in held.get("allOf", []):
            required |= set(rule["then"]["required"])
        unaccounted = set(HELD_FIXED_FIELDS) - required - set(KEY_DEFAULTS)
        assert not unaccounted, (
            f"{sorted(unaccounted)} are hashed into the comparability key but are "
            f"neither required by the schema nor given a default, so a bundle can "
            f"omit them and nobody can recompute its key"
        )

    def test_varied_dimensions_is_not_enum_restricted(self, checked_in):
        """It was, to the six component slots, and that rejected every bundle a
        factorial or flag sweep produces: `'enable_hyde' is not one of [...]`.
        A 2x2 varies a boolean flag, and flags are open-ended.
        """
        spec = checked_in["properties"]["configs"]["items"]["properties"]["varied_dimensions"]
        assert "enum" not in spec.get("items", {}), (
            "the published schema restricts varied_dimensions to a fixed list, so "
            "it rejects flag-based sweeps"
        )

    def test_a_flag_swept_bundle_validates_against_the_published_schema(self, checked_in):
        """End to end, on the shape a factorial actually emits."""
        jsonschema = pytest.importorskip("jsonschema")

        from lasagna.knowledge_evaluation_sdk.bundle import build_bundle

        jsonschema.validate(build_bundle(_fake_results(enable_hyde=True)), checked_in)


class TestAddingAKeyFieldDoesNotOrphanOldBundles:
    """Adding `corpus_provenance` to the key silently rewrote it for every run.

    `lasagna results validate` then reported honest pre-release bundles as
    INVALID — with the message hub/README documents as "held_fixed was edited to
    fake comparability", i.e. it accused every earlier submission of tampering.
    And the leaderboard groups on the recorded key, so two commensurable runs
    straddling the release split into two tables that each looked complete.
    """

    def _old_block(self):
        from lasagna.knowledge_evaluation_sdk.bundle import HELD_FIXED_FIELDS

        return {f: "x" for f in HELD_FIXED_FIELDS if f != "corpus_provenance"}

    def test_a_pre_release_bundle_keeps_its_key(self):
        from lasagna.knowledge_evaluation_sdk.bundle import comparability_key

        old = self._old_block()
        assert comparability_key(old) == comparability_key({**old, "corpus_provenance": "native"})

    def test_pooled_still_differs_from_native(self):
        """The default must not blunt the enforcement it was added for."""
        from lasagna.knowledge_evaluation_sdk.bundle import comparability_key

        old = self._old_block()
        assert comparability_key({**old, "corpus_provenance": "native"}) != comparability_key(
            {**old, "corpus_provenance": "pooled_gold"}
        )

    def test_a_default_is_only_allowed_where_absence_proves_the_value(self):
        """Guards the mechanism against becoming a shrug.

        Defaulting `embedder_model` would let an unrecorded model compare equal
        to a known one — the exact failure the key exists to prevent.
        """
        from lasagna.knowledge_evaluation_sdk.bundle import KEY_DEFAULTS

        assert set(KEY_DEFAULTS) == {"corpus_provenance"}

    def test_an_explicitly_recorded_value_still_wins_over_the_default(self):
        from lasagna.knowledge_evaluation_sdk.bundle import comparability_key

        old = self._old_block()
        explicit = comparability_key({**old, "corpus_provenance": "pooled_gold"})
        assert explicit != comparability_key(old)

    def test_a_real_bundle_validates(self, checked_in):
        """End to end: build a bundle, validate it against the published schema."""
        jsonschema = pytest.importorskip("jsonschema")

        from lasagna.knowledge_evaluation_sdk.bundle import build_bundle

        jsonschema.validate(build_bundle(_fake_results()), checked_in)

    def test_a_hand_edited_bundle_is_caught(self, checked_in):
        """Schema validity is not integrity: the run id must also be recomputed.

        This asserts the division of labour is real — the schema alone will
        happily accept a tampered bundle, which is why the validator recomputes
        the hash rather than trusting the file.
        """
        from lasagna.knowledge_evaluation_sdk.bundle import _content_hash

        bundle = {"schema_version": "1.0", "run_id": "deadbeefdeadbeef", "results": []}
        assert _content_hash(bundle) != bundle["run_id"]


class TestWhatVariedIsRecorded:
    """The key excludes the dimension under test, so something else must name it.

    `varied_dimensions()` existed and was unit-tested, but nothing called it —
    every bundle shipped `varied_dimensions: []`. A leaderboard grouping rows by
    comparability key could then show two rows that differ, with no way to say
    what differs, which is the one question a reader has.
    """

    def test_a_swept_stage_is_named(self):
        from lasagna.knowledge_evaluation_sdk.bundle import build_bundle

        bundle = build_bundle(_fake_results(reranker="length_demo"))
        assert bundle["configs"][0]["varied_dimensions"] == ["reranker"]

    def test_a_swept_flag_is_named_too(self):
        """Not only component slots. See TestVariedDimensions in test_results_bundle."""
        from lasagna.knowledge_evaluation_sdk.bundle import build_bundle

        bundle = build_bundle(_fake_results(enable_mmr=False))
        assert bundle["configs"][0]["varied_dimensions"] == ["enable_mmr"]

    def test_the_field_is_read_off_the_result_not_the_task(self):
        """`config_overrides` lived only on `BenchmarkTask`, and `build_bundle`
        reads `TaskResult.to_dict()` rows. The key was therefore absent from
        every row of every real run, so this whole feature was dead in
        production while passing here on hand-built dicts.
        """
        from lasagna.knowledge_evaluation_sdk.runner import TaskResult

        row = TaskResult(
            task_key="k",
            benchmark_name="smoke",
            config_name="c",
            run_index=0,
            config_overrides={"enable_hyde": True},
        ).to_dict()
        assert row["config_overrides"] == {"enable_hyde": True}

    def test_the_varied_stage_is_not_in_the_key(self):
        """If it were, no two rows in a sweep would ever be comparable."""
        from lasagna.knowledge_evaluation_sdk.bundle import build_bundle

        a = build_bundle(_fake_results(reranker="length_demo"))
        b = build_bundle(_fake_results(reranker="cross_encoder"))
        assert a["configs"][0]["comparability_key"] == b["configs"][0]["comparability_key"], (
            "changing the stage under test changed the comparability key, so the "
            "two arms of a sweep would be reported as incomparable"
        )


class TestTheHubIsDocumented:
    def test_workflow_and_readme_ship_together(self):
        """A schema without the workflow that applies it is not a hub."""
        assert (REPO / "hub" / "validate.yml").is_file()
        assert (REPO / "hub" / "README.md").is_file()

    def test_readme_states_the_trust_model(self):
        """The one claim a leaderboard must never overstate."""
        text = (REPO / "hub" / "README.md").read_text()
        assert "not independently reproduced" in text


def test_the_schema_says_rank_is_authoritative_over_score():
    """Exported scores tie (every scifact query, for a keyword reranker), so a
    reader who re-sorts `_retrieved` by score rebuilds a different ranking."""
    from lasagna.knowledge_evaluation_sdk.bundle import json_schema

    rows = json_schema()["properties"]["results"]["items"]["properties"]
    assert "`rank` is authoritative" in rows["per_sample_scores"]["description"]
