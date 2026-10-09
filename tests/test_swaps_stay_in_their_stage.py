"""Swapping one stage must not silently change another.

The paper says "each stage sits behind a fixed interface, so swapping one
method cannot silently change the implementation at another". Two halves:

* every listed built-in implements its stage's interface, so components of a
  stage are interchangeable; and
* every task records a stage manifest, and a sweep whose arms differ at a stage
  neither arm names is flagged, blocking in the bundle. An arm whose named change
  never reached its stage is flagged as inert.
"""

from __future__ import annotations

import copy

import pytest

from lasagna.core.registry import _get_expected_abc, registry
from lasagna.knowledge_evaluation_sdk.bundle import _integrity
from lasagna.knowledge_evaluation_sdk.stages import (
    STAGES,
    declared_stages,
    isolation_report,
    stage_manifest,
    stage_of,
)


@pytest.mark.parametrize(
    "stage", ["reranker", "query_enhancer", "chunker", "embedder", "document_processor"]
)
def test_every_listed_built_in_implements_its_stage(stage):
    abc = _get_expected_abc(stage)
    for name in registry.list(stage)[stage]:
        assert issubclass(registry.get_class(stage, name), abc), f"{stage} {name!r}"


def test_old_adapted_names_still_resolve_but_are_not_listed():
    assert registry.has("reranker", "cross_encoder_adapted")
    assert "cross_encoder_adapted" not in registry.list("reranker")["reranker"]
    assert registry.get_class("reranker", "cross_encoder_adapted") is registry.get_class(
        "reranker", "cross_encoder"
    )


def test_llm_enhancers_without_an_llm_say_so(monkeypatch):
    # Returning the query alone made a keyless HyDE or expansion arm score the
    # control's number with Failed: 0. They raise, and the pipeline decides:
    # fall back when serving, fail the task when evaluating.
    from lasagna.retrieval.enhancement import enhancer

    for cls in ("QueryExpander", "QueryDecomposer", "HyDEGenerator"):
        monkeypatch.setattr(getattr(enhancer, cls), "_get_llm", lambda self, *a, **k: None)
    for name in ("hyde", "query_expansion", "query_decomposition"):
        with pytest.raises(RuntimeError, match="LLM"):
            registry.get("query_enhancer", name).enhance("what is rag")


def test_built_in_enhancers_read_the_arms_settings(monkeypatch):
    from types import SimpleNamespace

    from lasagna.api._search_orchestrator import _query_variations
    from lasagna.retrieval.enhancement import enhancer

    seen = {}

    def fake_expand(e, q, n, max_tokens=300, model_ref=None):
        seen.update(n=n, model=model_ref)
        return [f"{q} {i}" for i in range(n)]

    monkeypatch.setattr(enhancer, "_expand_query_sync", fake_expand)
    arm = SimpleNamespace(max_query_variations=5, hyde_model="arm-model")
    out = _query_variations(registry.get("query_enhancer", "query_expansion"), "q", 5, arm)
    assert len(out) == 5 and seen == {"n": 5, "model": "arm-model"}


def test_settings_map_to_their_stage():
    assert stage_of("reranker_model") == "reranker"
    assert stage_of("enable_hyde") == "query_enhancer"
    assert stage_of("embedding_model") == "embedder"
    assert stage_of("chunk_size") == "chunker"
    assert stage_of("hybrid_alpha") == "retriever"
    assert stage_of("use_agentic_rag") == "agentic_pipeline"
    assert stage_of("docling_enabled") == "document_processor"
    assert stage_of("enable_search_cache") is None  # operational, never compared
    assert stage_of("a_setting_added_later") == "other"


class _Config:
    def __init__(self, **values):
        self._config = values


def _row(name, overrides, manifest, profile="speed", bench="b"):
    return {
        "benchmark_name": bench,
        "config_name": name,
        "config_overrides": overrides,
        "profile": profile,
        "error": None,
        "system": {"stages": manifest},
    }


BASE = dict(reranker_model="m", enable_reranking=False, embedding_model="e", chunk_size=800)


def _manifest(components=None, **changes):
    return stage_manifest(_Config(**{**BASE, **changes}), components or {})


def test_a_clean_swap_passes():
    rows = [
        _row("control", {}, _manifest()),
        _row(
            "mine",
            {"reranker": "mine"},
            _manifest({"reranker": "mine"}, enable_reranking=True),
        ),
    ]
    report = isolation_report(rows)
    assert report == {"undeclared": [], "inert": [], "unverified": []}


def test_a_change_outside_the_named_stage_is_caught_and_blocks():
    rows = [
        _row("control", {}, _manifest()),
        # Names a reranker, but its embedder changed too: not attributable.
        _row(
            "mine",
            {"reranker": "mine"},
            _manifest({"reranker": "mine"}, enable_reranking=True, embedding_model="other"),
        ),
    ]
    report = isolation_report(rows)
    assert len(report["undeclared"]) == 1
    assert "embedder" in report["undeclared"][0] and "embedding_model" in report["undeclared"][0]
    integ = _integrity(rows, {"n_runs": 3, "determinism": "seeded"})
    assert "UNDECLARED_STAGE_CHANGE" in integ["blocking"]


def test_an_arm_whose_change_never_arrived_is_inert():
    rows = [
        _row("control", {}, _manifest()),
        _row("hyde", {"enable_hyde": True}, _manifest()),  # the flag did not reach search
    ]
    report = isolation_report(rows)
    assert report["inert"] and "hyde" in report["inert"][0]
    assert "INERT_ARM" in _integrity(rows, {})["warnings"]


def test_whole_profile_comparisons_are_not_flagged():
    rows = [
        _row("speed", {}, _manifest(), profile="speed"),
        _row("standard", {}, _manifest(embedding_model="big", chunk_size=500), profile="standard"),
    ]
    assert isolation_report(rows)["undeclared"] == []


def test_results_without_manifests_are_unverified_not_passed():
    row = _row("old", {}, None)
    row["system"] = {}
    report = isolation_report([row])
    assert report["unverified"] and not report["undeclared"]


def test_declared_stages_come_from_settings_and_components():
    assert declared_stages({"reranker": "x", "hybrid_alpha": 0.3, "a_new_knob": 10}) == {
        "reranker",
        "retriever",
        "other",
    }


def test_harness_depth_settings_belong_to_retrieval():
    # metric_k and the over-fetch set how deep retrieval goes; an arm that
    # sweeps them must be neither flagged undeclared nor inert.
    assert stage_of("metric_k") == "retriever"
    assert stage_of("retrieval_chunk_overfetch") == "retriever"
    assert stage_of("save_trajectories") is None
    assert stage_of("allow_inert_overrides") is None


def test_the_manifest_covers_every_stage_and_names_values():
    m = _manifest({"reranker": "mine"})
    assert set(m) == {*STAGES, "other"}
    assert m["reranker"]["component"] == "mine"
    assert m["reranker"]["settings"]["reranker_model"] == "m"
    same = copy.deepcopy(m)
    assert same["embedder"]["digest"] == _manifest()["embedder"]["digest"]


def test_a_depth_arm_is_neither_inert_nor_undeclared():
    control = stage_manifest(_Config(**BASE), {}, harness={"metric_k": 10})
    deep = stage_manifest(_Config(**BASE), {}, harness={"metric_k": 100})
    rows = [_row("control", {}, control), _row("deep", {"metric_k": 100}, deep)]
    assert isolation_report(rows) == {"undeclared": [], "inert": [], "unverified": []}
