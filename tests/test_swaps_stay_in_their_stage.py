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
    track_manifest,
)


@pytest.mark.parametrize(
    "stage",
    [
        "reranker",
        "query_enhancer",
        "chunker",
        "embedder",
        "document_processor",
        "retriever",
        "agentic_pipeline",
        "generator",
    ],
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


def _row(name, overrides, manifest, profile="speed", bench="b", tracks=None):
    return {
        "benchmark_name": bench,
        "config_name": name,
        "config_overrides": overrides,
        "profile": profile,
        "error": None,
        "system": {"stages": manifest, **({"tracks": list(tracks)} if tracks else {})},
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


def _with_lists(row, lists):
    row["per_sample_scores"] = [
        {"_sample_id": q, "_retrieved_ids": ids, "ndcg@10": 0.5} for q, ids in lists.items()
    ]
    return row


def test_a_change_that_never_moved_a_ranking_is_inert():
    # MMR is read only when reranking is on: the mmr cell of a rerank x mmr
    # factorial reached its config and returned the neither cell's lists,
    # and the harness then reported a significant "interaction".
    same = {f"q{i}": [f"d{i}", f"e{i}"] for i in range(12)}
    rows = [
        _with_lists(_row("neither", {"enable_mmr": False}, _manifest()), same),
        _with_lists(_row("mmr", {"enable_mmr": True}, _manifest(enable_mmr=True)), same),
    ]
    inert = isolation_report(rows)["inert"]
    assert len(inert) == 1 and "identical ranked lists on all 12 queries" in inert[0]
    assert "enable_mmr" in inert[0]


def test_a_change_that_moved_one_ranking_is_not_inert():
    rows = [
        _with_lists(_row("neither", {"enable_mmr": False}, _manifest()), {"q1": ["d1", "d2"]}),
        _with_lists(
            _row("mmr", {"enable_mmr": True}, _manifest(enable_mmr=True)), {"q1": ["d2", "d1"]}
        ),
    ]
    assert isolation_report(rows)["inert"] == []


def test_generation_only_changes_are_not_judged_by_retrieval():
    same = {"q1": ["d1", "d2"]}
    rows = [
        _with_lists(_row("a", {"max_agent_iterations": 1}, _manifest()), same),
        _with_lists(
            _row("b", {"max_agent_iterations": 3}, _manifest(max_agent_iterations=3)), same
        ),
    ]
    assert isolation_report(rows)["inert"] == []


def test_an_override_without_a_default_reaches_the_manifest():
    class _Overridden(_Config):
        def __init__(self, base, overrides):
            super().__init__(**base)
            self._query_overrides = overrides

        def _get_session_override(self, key):
            return self._query_overrides.get(key)

    base = stage_manifest(_Overridden(BASE, {}), {})
    arm = stage_manifest(_Overridden(BASE, {"document_awareness": {"mode": "first_tokens"}}), {})
    assert base["chunker"]["digest"] != arm["chunker"]["digest"]
    assert arm["chunker"]["settings"]["document_awareness"] == {"mode": "first_tokens"}


# A declared change excuses the settings it names and what the harness sets on
# their account, not the whole stage. A stage either arm named was skipped
# whole: an arm naming hybrid_alpha hid a leaked default_top_k, and a reranker
# sweep, where every arm names a reranker, hid a leaked rerank_top_k.


def test_a_leak_inside_a_named_stage_is_caught():
    rows = [
        _row("control", {}, _manifest(hybrid_alpha=0.7, default_top_k=10)),
        _row(
            "sparse",
            {"hybrid_alpha": 0.3},
            _manifest(hybrid_alpha=0.3, default_top_k=50),  # default_top_k leaked
        ),
    ]
    report = isolation_report(rows)
    assert len(report["undeclared"]) == 1
    line = report["undeclared"][0]
    assert "retriever" in line and "default_top_k" in line and "hybrid_alpha" not in line
    assert "UNDECLARED_STAGE_CHANGE" in _integrity(rows, {})["blocking"]


def test_two_component_arms_that_also_differ_in_a_setting_are_caught():
    rows = [
        _row(
            "a",
            {"reranker": "a"},
            _manifest({"reranker": "a"}, enable_reranking=True, rerank_top_k=5),
        ),
        _row(
            "b",
            {"reranker": "b"},
            _manifest({"reranker": "b"}, enable_reranking=True, rerank_top_k=20),
        ),
    ]
    report = isolation_report(rows)
    assert len(report["undeclared"]) == 1
    assert "rerank_top_k" in report["undeclared"][0]
    assert "component" not in report["undeclared"][0]


_CONTROL = dict(
    BASE,
    chunking_strategy="semantic",
    default_top_k=40,
    embedding_dim=1024,
    collection_name="c",
    use_agentic_rag=False,
)
_CONTROL_HARNESS = {
    "metric_k": 10,
    "context_order": {"retrieval": "ranked", "generation": "ranked"},
}


@pytest.mark.parametrize(
    ("overrides", "components", "changes", "harness"),
    [
        # metric_k raises the depth the harness asks retrieval for
        ({"metric_k": 100}, {}, {"default_top_k": 400}, {"metric_k": 100}),
        # context reordering is recorded per track
        (
            {"enable_context_reordering": True},
            {},
            {},
            {"context_order": {"retrieval": "ranked", "generation": "lost_in_the_middle"}},
        ),
        # a structural chunker setting decides which chunker runs
        ({"enable_hierarchical": True}, {}, {"enable_hierarchical": True}, {}),
        # an embedder component gets a collection of its own, at its dimension
        (
            {"embedder": "mine"},
            {"embedder": "mine"},
            {"embedding_dim": 384, "collection_name": "m"},
            {},
        ),
        # so does an embedding model named as a setting
        (
            {"embedding_model": "other"},
            {},
            {"embedding_model": "other", "embedding_dim": 384, "collection_name": "o"},
            {},
        ),
        # an agentic pipeline named as a component turns the loop on
        (
            {"agentic_pipeline": "self_ask"},
            {"agentic_pipeline": "self_ask"},
            {"use_agentic_rag": True},
            {},
        ),
    ],
)
def test_what_the_harness_sets_for_a_declared_change_is_not_a_leak(
    overrides, components, changes, harness
):
    # Each of these the harness does in experiment.py on the arm's account;
    # the check must charge them to the declaration, or every such arm blocks.
    control = stage_manifest(_Config(**_CONTROL), {}, harness=_CONTROL_HARNESS)
    arm = stage_manifest(
        _Config(**{**_CONTROL, **changes}), components, harness={**_CONTROL_HARNESS, **harness}
    )
    rows = [_row("control", {}, control), _row("arm", overrides, arm)]
    assert isolation_report(rows)["undeclared"] == []


# Inert is judged setting by setting, and track by track.


def test_a_declared_setting_that_never_arrived_is_inert_beside_one_that_did():
    rows = [
        _row("control", {}, _manifest(hybrid_alpha=0.7, rrf_k=60)),
        # Names both; only rrf_k reached the retriever. By stage digest the
        # arm had moved, so nothing said hybrid_alpha never arrived.
        _row("fused", {"hybrid_alpha": 0.3, "rrf_k": 10}, _manifest(hybrid_alpha=0.7, rrf_k=10)),
    ]
    report = isolation_report(rows)
    assert report["undeclared"] == []
    (line,) = report["inert"]
    assert "['hybrid_alpha']" in line and "control" in line and "['rrf_k']" in line


BOTH = ("retrieval", "generation")


def _tracked(components=None, retrieval=None, generation=None, **common):
    """A manifest as the harness records it: the split keys per track."""
    return track_manifest(
        {
            "retrieval": _manifest(components, **common, **(retrieval or {})),
            "generation": _manifest(components, **common, **(generation or {})),
        }
    )


@pytest.mark.parametrize(
    ("overrides", "components"),
    [
        ({"use_agentic_rag": True}, None),
        ({"agentic_pipeline": "self_ask"}, {"agentic_pipeline": "self_ask"}),
    ],
)
def test_an_agentic_arm_is_inert_on_retrieval_and_live_on_generation(overrides, components):
    # The retrieval track forces the loop off for every arm; the generation
    # track runs it. Digesting the retrieval track's values called the arm
    # inert outright; the generation track's alone hid that its retrieval
    # numbers are the control's.
    control = _tracked(retrieval={"use_agentic_rag": False}, generation={"use_agentic_rag": False})
    arm = _tracked(
        components, retrieval={"use_agentic_rag": False}, generation={"use_agentic_rag": True}
    )
    key = next(iter(overrides))

    both = [_row("control", {}, control, tracks=BOTH), _row("agentic", overrides, arm, tracks=BOTH)]
    report = isolation_report(both)
    assert report["undeclared"] == []
    (line,) = report["inert"]
    assert repr(key) in line and "retrieval track" in line
    assert "generation track ran with its own" in line

    only = ["retrieval"]
    retrieval_only = [
        _row("control", {}, control, tracks=only),
        _row("agentic", overrides, arm, tracks=only),
    ]
    (line,) = isolation_report(retrieval_only)["inert"]
    assert "identical to the control" in line and "forces" in line

    only = ["generation"]
    generation_only = [
        _row("control", {}, control, tracks=only),
        _row("agentic", overrides, arm, tracks=only),
    ]
    assert isolation_report(generation_only)["inert"] == []


def test_split_keys_are_recorded_per_track_and_the_rest_once():
    m = _tracked(retrieval={"default_top_k": 40}, generation={"default_top_k": 10}, chunk_size=500)
    assert m["retriever"]["settings"]["default_top_k"] == {"retrieval": 40, "generation": 10}
    assert m["chunker"]["settings"]["chunk_size"] == 500
    # A difference on one track moves the stage's digest, and only its.
    same = _tracked(
        retrieval={"default_top_k": 40}, generation={"default_top_k": 40}, chunk_size=500
    )
    assert same["retriever"]["digest"] != m["retriever"]["digest"]
    assert same["chunker"]["digest"] == m["chunker"]["digest"]


def test_arms_are_compared_only_on_tracks_both_ran():
    # A no_rag arm skips retrieval: what its retrieval track would have
    # searched with changed nothing it measured, so it is no leak.
    control = _tracked(
        retrieval={"similarity_threshold": 0.0}, generation={"similarity_threshold": 0.5}
    )
    arm = _tracked(
        retrieval={"similarity_threshold": 0.9}, generation={"similarity_threshold": 0.5}
    )
    rows = [
        _row("control", {}, control, tracks=BOTH),
        _row("closed", {"no_rag": True}, arm, tracks=["generation"]),
    ]
    assert isolation_report(rows)["undeclared"] == []
    # The same difference on a track both ran is one.
    rows = [
        _row("control", {}, control, tracks=BOTH),
        _row("other", {"no_rag": True}, arm, tracks=BOTH),
    ]
    (line,) = isolation_report(rows)["undeclared"]
    assert "similarity_threshold" in line


def test_a_row_that_did_not_record_its_tracks_is_judged_whole_with_no_track_claim():
    # A results file written by 0.1.0: scalar settings, context_order for
    # both tracks, no record of which ran. Re-bundled, a retrieval-only run's
    # reordering arm was told its generation track ran with its own value.
    order = {"retrieval": "ranked", "generation": "lost_in_the_middle"}
    control = stage_manifest(
        _Config(**BASE), {}, harness={"context_order": {**order, "generation": "ranked"}}
    )
    arm = stage_manifest(_Config(**BASE), {}, harness={"context_order": order})
    rows = [
        _row("control", {}, control),
        _row("reordered", {"enable_context_reordering": True}, arm),
    ]
    report = isolation_report(rows)
    assert report["undeclared"] == []
    assert not any("track" in line for line in report["inert"])
    # Judged whole, a value the arm did change is live, and one it did not is inert.
    rows[1] = _row("reordered", {"enable_context_reordering": True}, control)
    (line,) = isolation_report(rows)["inert"]
    assert "identical to the control" in line and "track" not in line
