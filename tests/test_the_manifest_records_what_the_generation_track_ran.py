"""The stage manifest records what each track searched with.

The retrieval track sends every arm's queries through the standard tier and
fixes the retrieval depth; the generation track reads the arm's own values.
The manifest digested the retrieval track's configuration, so on a generation
track an arm that set ``use_agentic_rag: true`` ran the loop, digested like
its control at the agentic stage, and was called inert; two arms that
differed only in ``routing_default_tier`` could not be told apart. Recording
the generation track's values instead hides the other half: that arm's
retrieval numbers are the control's. The manifest records each such setting
per track, and which tracks ran, and the isolation check judges an arm's
declared change on each.
"""

from __future__ import annotations

import asyncio

from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.stages import TRACK_SPLIT_KEYS, isolation_report, stage_of
from tests.test_retrieval_metrics_score_the_ranked_order import (  # noqa: F401 (fixtures)
    live,
    orchestrator,
)
from tests.test_the_generator_reads_the_top_passages_reordered import llm  # noqa: F401 (fixture)


def test_every_key_the_retrieval_track_forces_is_recorded_per_track():
    # A key the retrieval track starts forcing fails here until it is listed:
    # recorded per track, filed as operational, or recorded otherwise the way
    # context order is.
    written: dict = {}
    exp._fix_retrieval_routing(written)
    exp._fix_retrieval_depth(written, 40)
    assert stage_of("enable_search_cache") is None
    assert stage_of("context_order") == stage_of("enable_context_reordering") == "reranker"
    accounted = exp._RETRIEVAL_TRACK_FORCED | {"enable_search_cache", "enable_context_reordering"}
    assert set(written) <= accounted
    assert exp._RETRIEVAL_TRACK_FORCED <= set(written)
    assert exp._RETRIEVAL_TRACK_FORCED <= TRACK_SPLIT_KEYS


def _run(tmp_path, tracks, arms):
    data = {
        "name": "tracks",
        "benchmarks": [{"name": "smoke"}],
        "mode": "live",
        "tracks": tracks,
        "output_dir": str(tmp_path / "out"),
        "max_concurrency": 1,
        "llm_cache": False,
        "pipeline": {"base_profile": "speed", "config_overrides": arms},
    }
    results = asyncio.run(exp.Experiment.from_dict(data).run())._results
    for r in results:
        assert r.error is None, r.error
    return {r.config_name: r for r in results}


ARMS = [{"name": "control"}, {"name": "shallow", "default_top_k": 3}]


def test_a_depth_setting_is_recorded_per_track_and_inert_where_forced(tmp_path, llm):  # noqa: F811
    results = _run(tmp_path, ["retrieval", "generation"], ARMS)
    control, arm = results["control"], results["shallow"]
    assert arm.system["tracks"] == ["retrieval", "generation"]
    depth = arm.system["stages"]["retriever"]["settings"]["default_top_k"]
    forced = control.system["stages"]["retriever"]["settings"]["default_top_k"]["retrieval"]
    # The harness asks retrieval for metric_k times the over-fetch, whatever the arm says.
    assert depth == {"retrieval": forced, "generation": 3} and forced >= 40
    report = isolation_report([r.to_dict() for r in results.values()])
    assert report["undeclared"] == []
    assert any(
        "'default_top_k'" in line
        and "retrieval track" in line
        and "generation track ran with its own" in line
        for line in report["inert"]
    ), report["inert"]


def test_on_a_retrieval_only_run_the_forced_setting_is_inert_outright(tmp_path, live):  # noqa: F811
    results = _run(tmp_path, ["retrieval"], ARMS)
    assert results["shallow"].system["tracks"] == ["retrieval"]
    report = isolation_report([r.to_dict() for r in results.values()])
    assert any(
        "identical to the control" in line and "forces" in line for line in report["inert"]
    ), report["inert"]
