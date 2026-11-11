"""An agentic arm runs its loop under a profile that turns query routing off.

``pipeline.query`` forces the comprehensive tier for an arm that names an
agentic pipeline. The router answered "routing disabled" before it read the
forced tier, so under ``speed`` (routing off, default tier ``fast``) every
agentic arm searched the fast tier and never ran its loop. The harness then
scored those answers as the loop's: it flagged a fallback only when the
agentic run had failed and stamped its standard results, and these carried
no stamp at all.
"""

from __future__ import annotations

from types import SimpleNamespace

from lasagna.core.session_config._profiles import resolve_ingestion_defaults
from lasagna.knowledge_evaluation_sdk.experiment import _agentic_fell_back
from lasagna.retrieval.routing.router import QueryRouter, RoutingTier


def _config(**settings):
    defaults = dict(
        enable_query_routing=True,
        routing_strategy="rule_based",
        routing_default_tier="standard",
        routing_confidence_threshold=0.5,
        routing_allow_tier_override=True,
        routing_log_decisions=False,
        use_agentic_rag=True,
        enable_query_classification=True,
    )
    return SimpleNamespace(**{**defaults, **settings})


def _speed_arm():
    """The `speed` profile's routing, with the arm's agentic loop turned on."""
    speed = resolve_ingestion_defaults("speed")
    assert speed["enable_query_routing"] is False and speed["routing_default_tier"] == "fast"
    return _config(
        enable_query_routing=False,
        routing_default_tier=speed["routing_default_tier"],
        use_agentic_rag=True,
    )


def test_a_forced_tier_wins_when_routing_is_off():
    decision = QueryRouter().route(
        "Who directed Jaws?", None, None, _speed_arm(), forced_tier="comprehensive"
    )
    assert decision.tier == RoutingTier.COMPREHENSIVE
    assert decision.strategy_used == "forced"


def test_routing_off_still_gives_the_default_tier_when_nothing_is_forced():
    decision = QueryRouter().route("Who directed Jaws?", None, None, _speed_arm())
    assert decision.tier == RoutingTier.FAST and decision.strategy_used == "disabled"


def test_a_session_that_refuses_overrides_keeps_its_default_tier():
    config = _config(
        enable_query_routing=False, routing_default_tier="fast", routing_allow_tier_override=False
    )
    decision = QueryRouter().route("q", None, None, config, forced_tier="comprehensive")
    assert decision.tier == RoutingTier.FAST


def test_an_agentic_request_answered_by_another_tier_is_a_fallback():
    def result(meta):
        return SimpleNamespace(chunk=SimpleNamespace(metadata=meta, content="x"))

    # Fast and standard results carry no tier stamp; only the agentic tier
    # stamps "comprehensive" on what it returns.
    assert _agentic_fell_back(True, [result({})]) is True
    assert _agentic_fell_back(True, [result({"_routing_tier_used": "comprehensive"})]) is False
    assert _agentic_fell_back(False, [result({})]) is False


def test_the_speed_arm_the_harness_builds_reaches_its_loop():
    """The real config: the `speed` profile with an agentic arm's overrides.

    As the harness passes them to `pipeline.query`: the arm names a loop, which
    turns `use_agentic_rag` on, so the forced tier is not downgraded either.
    """
    from lasagna.core.config import rag_config

    speed = resolve_ingestion_defaults("speed")
    arm = rag_config.with_query_overrides(
        {**speed, "agentic_pipeline": "self_ask", "use_agentic_rag": True}
    )
    assert arm.enable_query_routing is False and arm.routing_default_tier == "fast"
    decision = QueryRouter().route("Who directed Jaws?", None, None, arm, "comprehensive")
    assert decision.tier == RoutingTier.COMPREHENSIVE

    # CRAG forces the comprehensive tier too; without the agentic flag it is
    # downgraded to standard, no longer dropped for the profile's fast tier.
    crag = rag_config.with_query_overrides({**speed, "enable_crag": True})
    decision = QueryRouter().route("Who directed Jaws?", None, None, crag, "comprehensive")
    assert decision.tier == RoutingTier.STANDARD
