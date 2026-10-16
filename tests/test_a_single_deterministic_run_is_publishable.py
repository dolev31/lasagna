"""A single run is a complete result when nothing in the run samples.

The docs, the skills and the end-of-run note recommend ``n_runs: 1`` for a
retrieval-only sweep: fixed models return the same lists, so a repeat adds
wall time and no information. But every bundle with one run carried the
``SINGLE_RUN`` warning, which ``results validate --strict`` (the hub's check)
refused, and the leaderboard never verified a single-run row. Following the
advice made a result unpublishable.

The run now records ``nothing_samples``: only the retrieval track ran, no
task made an LLM call, and repeats (if any) were identical, the check behind
the "all 3 runs were identical" note. Such a run carries no ``SINGLE_RUN``
and can be verified by an exact reproduction. A run with an LLM stage still
needs repeats.
"""

from __future__ import annotations

import asyncio

import pytest

from lasagna.knowledge_evaluation_sdk import bundle as bundle_mod
from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.leaderboard import mark_verified
from lasagna.knowledge_evaluation_sdk.runner import TaskResult


def _result(config="dense", run_index=0, score=0.6, llm_calls=0):
    return TaskResult(
        task_key=f"{config}:{run_index}",
        benchmark_name="scifact",
        config_name=config,
        run_index=run_index,
        metrics={"ndcg@10": score},
        per_sample_scores=[{"_sample_id": "q1", "ndcg@10": score}],
        system_llm_calls=llm_calls,
    )


def test_one_retrieval_run_without_an_llm_call_samples_nothing():
    assert exp.nothing_samples([_result()], ["retrieval"])


@pytest.mark.parametrize(
    "results, tracks",
    [
        ([_result(llm_calls=3)], ["retrieval"]),  # an agentic step or LLM reranker
        ([_result()], ["retrieval", "generation"]),  # a generator and a judge
        ([_result(score=0.6), _result(run_index=1, score=0.61)], ["retrieval"]),
        ([], ["retrieval"]),
    ],
    ids=["llm-call", "generation-track", "repeats-differ", "nothing-ran"],
)
def test_anything_that_samples_is_not_called_deterministic(results, tracks):
    assert not exp.nothing_samples(results, tracks)


def test_identical_repeats_still_count_as_nothing_sampling():
    assert exp.nothing_samples([_result(), _result(run_index=1)], ["retrieval"])


def test_the_run_records_it(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sweep = {
        "name": "retrieval-only",
        "benchmarks": [{"name": "smoke", "n_samples": 2}],
        "tracks": ["retrieval"],
        "n_runs": 1,
        "output_dir": str(tmp_path / "out"),
        "pipeline": {"base_profile": "speed"},
    }

    async def retrieve(task, samples):
        return _result(config=task.config_name)

    results = asyncio.run(exp.Experiment.from_dict(sweep, eval_fn=retrieve).run())

    assert results.experiment_config["nothing_samples"] is True


def _warnings(n_runs, nothing_samples):
    rows = [_result().to_dict()]
    meta = {"n_runs": n_runs, "nothing_samples": nothing_samples}
    return bundle_mod._integrity(rows, meta)["warnings"]


def test_a_single_run_that_samples_nothing_has_no_single_run_warning():
    assert "SINGLE_RUN" not in _warnings(1, True)


def test_a_single_run_with_an_llm_stage_still_warns():
    assert "SINGLE_RUN" in _warnings(1, False)
    assert "SINGLE_RUN" in _warnings(1, None)  # a results file from before the field


def _row(submitter, mean, nothing_samples, n_runs=1):
    return {
        "benchmark": "scifact",
        "config": "dense",
        "submitter": submitter,
        "comparability_key": "k1",
        "held_fixed": {"fixed_stages": {"retriever": "abc"}},
        "varied_levels": {},
        "mean": mean,
        "std": 0.0,
        "n_runs": n_runs,
        "nothing_samples": nothing_samples,
    }


def test_the_hub_verifies_an_exact_reproduction_of_a_deterministic_single_run():
    rows = [_row("lab-a", 0.6123, True), _row("lab-b", 0.6123, True)]
    mark_verified(rows)
    assert all(r["verified"] for r in rows)


def test_the_hub_still_never_verifies_a_single_run_that_samples():
    rows = [_row("lab-a", 0.6123, False), _row("lab-b", 0.6123, False)]
    mark_verified(rows)
    assert not any(r["verified"] for r in rows)


def test_a_different_number_does_not_verify_it():
    rows = [_row("lab-a", 0.6123, True), _row("lab-b", 0.6124, True)]
    mark_verified(rows)
    assert not any(r["verified"] for r in rows)
