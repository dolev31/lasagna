"""A run that generates nothing records no judge and no answer-similarity model.

Both only score answers, on the generation track. A retrieval-only sweep
recorded them anyway, from `RAG_EVAL_JUDGE_MODEL` and the similarity default:
the bundle then raised `UNRESOLVED_JUDGE_ALIAS`, so `validate --strict` failed
a run that judged nothing, and two identical retrieval runs split on a
leaderboard by an environment variable that changed nothing they computed.
`sampling` beside them was already left out of such a run.
"""

from __future__ import annotations

from lasagna.knowledge_evaluation_sdk import bundle as bundle_mod

ROW = {
    "benchmark_name": "smoke",
    "config_name": "speed",
    "corpus_fingerprint": "fp",
    "split": "test",
    "n_samples": 12,
}


def _meta(tracks, judge="llama3.2", similarity="sentence-transformers/all-MiniLM-L6-v2"):
    meta = {"n_runs": 1, "nothing_samples": True, "determinism": "seeded"}
    if tracks is not None:
        meta["tracks"] = tracks
    meta["judge_model"] = judge
    meta["similarity_model"] = similarity
    return meta


def test_a_retrieval_run_is_not_flagged_for_a_judge_it_never_called():
    integrity = bundle_mod._integrity([ROW], _meta(["retrieval"]))
    assert "UNRESOLVED_JUDGE_ALIAS" not in integrity["warnings"]


def test_a_run_that_judged_keeps_the_alias_warning():
    for tracks in (["retrieval", "generation"], ["generation"], None):
        integrity = bundle_mod._integrity([ROW], _meta(tracks))
        assert "UNRESOLVED_JUDGE_ALIAS" in integrity["warnings"], tracks


def test_retrieval_runs_group_whatever_judge_the_environment_named():
    one = bundle_mod.held_fixed_from(ROW, _meta(["retrieval"], judge="llama3.2"))
    other = bundle_mod.held_fixed_from(ROW, _meta(["retrieval"], judge=None, similarity=None))
    assert one["judge_model"] is None and one["similarity_model"] is None
    assert bundle_mod.comparability_key(one) == bundle_mod.comparability_key(other)


def test_generation_runs_still_split_by_judge():
    one = bundle_mod.held_fixed_from(ROW, _meta(["generation"], judge="judge-a"))
    other = bundle_mod.held_fixed_from(ROW, _meta(["generation"], judge="judge-b"))
    assert one["judge_model"] == "judge-a"
    assert bundle_mod.comparability_key(one) != bundle_mod.comparability_key(other)


def _cfg(tracks, judge="llama3.2"):
    from lasagna.knowledge_evaluation_sdk.experiment import _parse_config

    return _parse_config(
        {
            "name": "judge record",
            "benchmarks": [{"name": "smoke"}],
            "tracks": tracks,
            "mode": "live",
            "n_runs": 1,
            "judge_model": judge,
        }
    )


def test_the_run_record_names_a_judge_only_when_a_track_generates(monkeypatch):
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    monkeypatch.setattr(exp, "_active_judge_model", None)
    retrieval = exp._run_metadata(_cfg(["retrieval"]))
    assert retrieval["judge_model"] is None
    assert retrieval["similarity_model"] is None
    generation = exp._run_metadata(_cfg(["retrieval", "generation"]))
    assert generation["judge_model"] == "llama3.2"
