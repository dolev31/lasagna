"""The stage manifest records the settings the generation track ran with.

The retrieval track sends every arm's queries through the standard tier and
fixes the retrieval depth, and the manifest digested that forced
configuration. The generation track reads the arm's own values, so on it an
arm that set ``use_agentic_rag: true`` ran the loop, digested like its control
at the agentic stage, and was called inert; two arms that differed only in
``routing_default_tier`` could not be told apart.
"""

from __future__ import annotations

from lasagna.knowledge_evaluation_sdk import experiment as exp
from lasagna.knowledge_evaluation_sdk.stages import stage_manifest, stage_of


class _Config:
    def __init__(self, **values):
        self._config = values


BASE = dict(
    use_agentic_rag=False,
    agentic_pipeline="legacy_react",
    routing_default_tier="standard",
    default_top_k=10,
    similarity_threshold=0.5,
    chunk_cite_retrieval={"strategy": "adaptive"},
    chunk_size=800,
)


def _tracks(arm_config, depth=40):
    """What each track ran with, derived from the arm's config as `_eval_live` does."""
    retrieval = dict(arm_config)
    exp._fix_retrieval_routing(retrieval)
    exp._fix_retrieval_depth(retrieval, depth)
    return retrieval, dict(arm_config)


def _manifest(arm_config, generation_ran=True):
    retrieval, generation = _tracks(arm_config)
    settings = exp._manifest_settings(retrieval, generation if generation_ran else None)
    return stage_manifest(_Config(**settings), {})


def test_an_agentic_arm_differs_from_its_control_when_generation_ran():
    control = _manifest(BASE)
    arm = _manifest({**BASE, "use_agentic_rag": True, "routing_default_tier": "comprehensive"})
    assert arm["agentic_pipeline"]["digest"] != control["agentic_pipeline"]["digest"]
    assert arm["agentic_pipeline"]["settings"]["use_agentic_rag"] is True
    assert arm["query_enhancer"]["settings"]["routing_default_tier"] == "comprehensive"


def test_a_depth_setting_the_generation_track_read_is_recorded():
    control = _manifest(BASE)
    arm = _manifest({**BASE, "default_top_k": 3, "similarity_threshold": 0.9})
    assert arm["retriever"]["settings"]["default_top_k"] == 3
    assert arm["retriever"]["settings"]["similarity_threshold"] == 0.9
    assert arm["retriever"]["digest"] != control["retriever"]["digest"]


def test_without_a_generation_track_the_forced_values_stand():
    # Retrieval alone ran with the loop off and the depth fixed, so an arm
    # that asked for either is inert there, and the manifest must say so.
    arm = _manifest({**BASE, "use_agentic_rag": True, "default_top_k": 3}, generation_ran=False)
    assert arm["agentic_pipeline"]["settings"]["use_agentic_rag"] is False
    assert arm["retriever"]["settings"]["default_top_k"] == 40
    assert (
        arm["agentic_pipeline"]["digest"]
        == _manifest(BASE, generation_ran=False)["agentic_pipeline"]["digest"]
    )


def test_settings_the_retrieval_track_does_not_force_come_from_it():
    retrieval, generation = _tracks(BASE)
    settings = exp._manifest_settings(retrieval, {**generation, "chunk_size": 100})
    assert settings["chunk_size"] == 800


def test_every_key_the_retrieval_track_forces_is_accounted_for():
    # A key the retrieval track starts forcing fails here until it is listed:
    # taken from the generation track, filed as operational, or recorded per
    # track the way context order is.
    written: dict = {}
    exp._fix_retrieval_routing(written)
    exp._fix_retrieval_depth(written, 40)
    assert stage_of("enable_search_cache") is None
    assert stage_of("context_order") == stage_of("enable_context_reordering") == "reranker"
    accounted = exp._RETRIEVAL_TRACK_FORCED | {"enable_search_cache", "enable_context_reordering"}
    assert set(written) <= accounted
    assert exp._RETRIEVAL_TRACK_FORCED <= set(written)
