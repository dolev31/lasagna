"""Two runs share a comparability key only if they held the same pipeline fixed.

The key hashed 16 fields and nothing about retrieval mode, fusion weights, MMR,
HyDE or first-stage settings. On scifact, bundle X ran `[base, keyword
reranker]` dense-only and bundle Y ran the same two arms with
`enable_hybrid_search: true, hybrid_alpha: 0.3` in both. All four rows got key
`b5fc8dec25964f8e`, Y validated, and `lasagna results leaderboard` put them in
one table: two different first stages (a 0.028 NDCG@10 gap) presented as the
same baseline under "rows in one table held the same configuration fixed".

Every live row already records a digest of what reached each stage. The key now
includes the digests of the stages the run does not vary.
"""

from __future__ import annotations

import hashlib
import json

from lasagna.knowledge_evaluation_sdk.bundle import (
    HELD_FIXED_FIELDS,
    METRICS_VERSION,
    build_bundle,
    comparability_key,
    compare_keys,
)
from lasagna.knowledge_evaluation_sdk.leaderboard import collect_rows, group, mark_verified
from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults
from lasagna.knowledge_evaluation_sdk.runner import TaskResult
from lasagna.knowledge_evaluation_sdk.stages import stage_manifest


class _Config:
    """Enough of a RAGConfig for `stage_manifest`: the raw settings dict."""

    def __init__(self, settings):
        self._config = dict(settings)


BASE = {"enable_hybrid_search": False, "hybrid_alpha": 0.7, "enable_reranking": False}


def _row(name, overrides, settings, run=0, manifest=True, value=0.6):
    component = {"reranker": overrides["reranker"]} if "reranker" in overrides else {}
    system = {"embedder_model": "all-MiniLM-L6-v2", "embedder_dim": 384, "chunker": "semantic"}
    if manifest:
        system["stages"] = stage_manifest(_Config(settings), component)
    return TaskResult(
        task_key=f"{name}|{run}",
        benchmark_name="beir/scifact",
        config_name=name,
        run_index=run,
        metrics={"ndcg@10": value},
        split="test",
        n_samples=100,
        sample_id_digest="ids",
        corpus_fingerprint="corpus",
        resolved_mode="live",
        config_overrides=dict(overrides),
        system=system,
    )


def _sweep(held=None, manifest=True, submitter="labX", values=(0.6684, 0.5709)):
    held = held or {}
    settings = {**BASE, **held}
    rows = []
    for run in (0, 1):
        # A small spread across runs, so a row has an interval to verify inside.
        rows.append(_row("base", dict(held), settings, run, manifest, values[0] + 0.01 * run))
        rows.append(
            _row(
                "kw",
                {**held, "reranker": "my_ideas:keyword_overlap"},
                settings,
                run,
                manifest,
                values[1] + 0.01 * run,
            )
        )
    results = BenchmarkResults(
        rows, "sweep", {"n_runs": 2, "system": {}, "metrics_version": METRICS_VERSION}
    )
    return build_bundle(results, submitter={"name": submitter})


def _keys(bundle):
    return {c["name"]: c["comparability_key"] for c in bundle["configs"]}


HYBRID = {"enable_hybrid_search": True, "hybrid_alpha": 0.3}


class TestSettingsHeldFixedOutsideTheSixteenFields:
    def test_arms_of_one_sweep_share_a_key(self):
        keys = _keys(_sweep())
        assert keys["base"] == keys["kw"]

    def test_a_different_first_stage_gets_a_different_key(self):
        """The review's repro: dense-only and hybrid alpha=0.3 shared a key."""
        assert _keys(_sweep())["base"] != _keys(_sweep(HYBRID))["base"]

    def test_the_varied_stage_stays_out_of_the_key(self):
        """Two labs' reranker sweeps on the same pipeline still compare."""
        assert _keys(_sweep()) == _keys(_sweep(submitter="labY"))

    def test_the_leaderboard_keeps_the_two_systems_apart(self):
        rows = collect_rows([_sweep(), _sweep(HYBRID, submitter="labY")], "ndcg@10")
        groups = group(rows)["beir/scifact"]
        assert len(groups) == 2 and all(len(g) == 2 for g in groups.values())

    def test_the_difference_is_named_by_stage(self):
        x = _sweep()["configs"][0]["held_fixed"]
        y = _sweep(HYBRID)["configs"][0]["held_fixed"]
        diffs = compare_keys(x, y)
        assert any(d.startswith("fixed_stages.retriever") for d in diffs), diffs
        assert not any("reranker" in d for d in diffs), diffs


class TestRowsWithoutAManifest:
    def test_they_are_not_merged_with_another_bundle(self):
        """Nothing records what such a row held fixed, so equal keys prove
        nothing about its retriever."""
        a = _sweep(manifest=False)
        b = _sweep(HYBRID, manifest=False, submitter="labY")
        rows = collect_rows([a, b], "ndcg@10")
        groups = group(rows)["beir/scifact"]
        assert len(groups) == 2, "two bundles with no stage record shared a table"

    def test_they_are_never_verified_across_bundles(self):
        a = _sweep(manifest=False)
        b = _sweep(manifest=False, submitter="labY")
        rows = collect_rows([a, b], "ndcg@10")
        mark_verified(rows)
        assert not any(r["verified"] for r in rows)

    def test_their_own_arms_still_share_a_table(self):
        rows = collect_rows([_sweep(manifest=False)], "ndcg@10")
        assert len(group(rows)["beir/scifact"]) == 1


class TestBundlesMintedBeforeTheField:
    def test_an_older_bundle_still_recomputes_to_its_own_key(self):
        """`results validate` recomputes the key; a 1.3 bundle must not turn
        INVALID because the field set grew."""
        from lasagna.knowledge_evaluation_sdk.bundle import _KEY_FIELDS_SINCE, _KEY_FIELDS_WHEN_SET

        # Every field added to the key after 1.3 is left out of a 1.3 key, and
        # so is a field hashed only when set (corpus_ingest_gap), which no
        # older bundle carries.
        later = set(_KEY_FIELDS_SINCE) | _KEY_FIELDS_WHEN_SET
        held = {f: None for f in HELD_FIXED_FIELDS if f not in later}
        held.update(metrics_version="1.3", benchmark="beir/scifact", corpus_provenance="native")
        payload = {f: held[f] for f in HELD_FIXED_FIELDS if f not in later}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        assert comparability_key(held) == hashlib.sha256(blob.encode()).hexdigest()[:16]

    def test_a_current_bundle_without_the_field_does_not_match_one_with_it(self):
        held = _sweep()["configs"][0]["held_fixed"]
        missing = {k: v for k, v in held.items() if k != "fixed_stages"}
        assert comparability_key(missing) != comparability_key(held)
