"""A run that writes no answers records no judge and no answer-similarity model.

Both only score answers. A live retrieval-only sweep recorded them anyway,
from `RAG_EVAL_JUDGE_MODEL` (or `judge_model:`) and the similarity default:
the bundle then raised `UNRESOLVED_JUDGE_ALIAS`, so `validate --strict` failed
a run that judged nothing, and two identical retrieval runs split on a
leaderboard by an environment variable that changed nothing they computed.
`sampling` beside them was already left out of such a run.

A benchmark whose mode resolves to `generation_only` (TriviaQA under `auto`)
writes and judges answers whatever the tracks, so it keeps its judge, and an
unnamed default judge there is now recorded too.
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
    # A task that answered offline was judged, whatever the tracks.
    offline = {**ROW, "resolved_mode": "generation_only"}
    integrity = bundle_mod._integrity([offline], _meta(["retrieval"]))
    assert "UNRESOLVED_JUDGE_ALIAS" in integrity["warnings"]


def test_a_bundle_keeps_the_key_its_run_recorded():
    # The key reads the run record as written, so a bundle made before this
    # change gets the same key from `results compare` and the leaderboard.
    held = bundle_mod.held_fixed_from(ROW, _meta(["retrieval"], judge="llama3.2"))
    assert held["judge_model"] == "llama3.2"


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


def test_the_run_record_names_a_judge_only_when_answers_are_written(monkeypatch):
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    monkeypatch.setattr(exp, "_active_judge_model", None)
    retrieval = exp._run_metadata(_cfg(["retrieval"]))
    assert retrieval["judge_model"] is None
    assert retrieval["similarity_model"] is None
    # Two runs that differ only in the judge the environment named.
    other = exp._run_metadata(_cfg(["retrieval"], judge="another-judge"))
    assert other["judge_model"] is None
    generation = exp._run_metadata(_cfg(["retrieval", "generation"]))
    assert generation["judge_model"] == "llama3.2"


def test_a_benchmark_answered_offline_keeps_its_judge(monkeypatch):
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    # TriviaQA has no corpus: under `auto` it resolves to generation_only, and
    # answers and is judged on the retrieval track too.
    assert exp.resolve_mode("auto", "triviaqa")[0] == "generation_only"
    monkeypatch.setattr(exp, "_active_judge_model", None)
    monkeypatch.delenv("RAG_EVAL_JUDGE_MODEL", raising=False)
    cfg = exp._parse_config(
        {
            "name": "offline",
            "benchmarks": [{"name": "triviaqa"}],
            "tracks": ["retrieval"],
            "mode": "auto",
        }
    )
    assert exp._writes_answers(cfg)
    # The default judge is the one that scored it, so the record says so.
    assert exp._resolve_judge_model(cfg) == exp.DEFAULT_JUDGE_MODEL
    named = exp._parse_config(
        {
            "name": "offline",
            "benchmarks": [{"name": "triviaqa"}],
            "tracks": ["retrieval"],
            "mode": "auto",
            "judge_model": "llama3.2",
        }
    )
    meta = exp._run_metadata(named)
    assert meta["judge_model"] == "llama3.2"
    assert meta["similarity_model"] is not None
