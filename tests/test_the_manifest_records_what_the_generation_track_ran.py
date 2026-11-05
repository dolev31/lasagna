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
from tests.test_retrieval_metrics_score_the_ranked_order import (  # noqa: F401 - pytest finds these fixtures by name
    live,
    orchestrator,
)
from tests.test_the_generator_reads_the_top_passages_reordered import (  # noqa: F401 - a fixture, found by name
    llm,
)


def test_every_key_the_retrieval_track_forces_is_recorded_per_track():
    # A key the retrieval track starts forcing fails here until it is recorded
    # per track or filed as operational. enable_context_reordering, forced off,
    # was recorded once, as the retrieval track's False, for an arm whose
    # generator read its passages reordered.
    written: dict = {}
    exp._fix_retrieval_routing(written)
    exp._fix_retrieval_depth(written, 40)
    assert stage_of("enable_search_cache") is None
    assert {k for k in written if stage_of(k) is not None} <= TRACK_SPLIT_KEYS


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


def test_a_depth_setting_is_recorded_per_track_and_inert_where_forced(tmp_path, llm):  # noqa: F811 - the fixture, not a redefinition
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


def test_on_a_retrieval_only_run_the_forced_setting_is_inert_outright(tmp_path, live):  # noqa: F811 - the fixture, not a redefinition
    results = _run(tmp_path, ["retrieval"], ARMS)
    assert results["shallow"].system["tracks"] == ["retrieval"]
    report = isolation_report([r.to_dict() for r in results.values()])
    assert any(
        "identical to the control" in line and "forces" in line for line in report["inert"]
    ), report["inert"]


def test_an_arm_digests_alike_whichever_tracks_ran(tmp_path, llm):  # noqa: F811 - the fixture, not a redefinition
    # Each track's values are recorded whether or not the track ran, so a
    # retrieval-only, a generation-only and a full run record one manifest for
    # an arm (``system.stages``, which is all this asserts). stages.py,
    # experiment.py and the CHANGELOG said so; nothing held it.
    arms = [*ARMS, {"name": "reordered", "enable_context_reordering": True}]
    runs = {
        "+".join(tracks): _run(tmp_path / "+".join(tracks), tracks, arms)
        for tracks in (["retrieval"], ["generation"], ["retrieval", "generation"])
    }
    for run, results in runs.items():
        assert {r.system["tracks"] == run.split("+") for r in results.values()} == {True}
    for arm in ("control", "shallow", "reordered"):
        manifests = [results[arm].system["stages"] for results in runs.values()]
        assert manifests[0] == manifests[1] == manifests[2], arm
