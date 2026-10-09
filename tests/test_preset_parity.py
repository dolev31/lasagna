"""Verify SDK presets stay aligned with session config profile specs.

Ensures config_for_speed(), config_for_standard(), config_for_balanced(),
and config_for_max_quality() produce values that match _PROFILE_SPECS
for every shared key.  Any drift between the SDK and session config
will cause an immediate test failure with a clear message.

SHARED_KEYS is computed dynamically: the intersection of _PROFILE_SPECS
keys and KnowledgeConfig field names, minus known session-config-only keys
(routing, score normalization) and nested dicts (chunk_cite_retrieval).
"""

import dataclasses

import pytest


# Keys that exist in _PROFILE_SPECS but are NOT KnowledgeConfig fields.
# These are session-config-only internal knobs and should not be compared.
# NOTE: Most keys have been moved to KnowledgeConfig for full parity checking.
# Only agentic_hyde_strategy remains excluded because the SDK uses the
# hyde_strategy enum field, mapped via SPECIAL_MAPPINGS in _config_bridge.py.
_SESSION_CONFIG_ONLY_KEYS = frozenset(
    {
        "agentic_hyde_strategy",
    }
)


def _compute_shared_keys():
    """Compute the intersection of _PROFILE_SPECS keys and KnowledgeConfig fields.

    This is dynamic so that new fields are automatically checked.
    """
    from lasagna.sdk.config import KnowledgeConfig
    from lasagna.core.session_config._profiles import ProfileMixin

    config_fields = {f.name for f in dataclasses.fields(KnowledgeConfig)}
    all_spec_keys = set()
    for spec in ProfileMixin._PROFILE_SPECS.values():
        for key, value in spec.items():
            # Skip nested dicts (chunk_cite_retrieval) — not a flat field
            if not isinstance(value, dict):
                all_spec_keys.add(key)

    return sorted((all_spec_keys & config_fields) - _SESSION_CONFIG_ONLY_KEYS)


def _get_profile_specs():
    from lasagna.core.session_config._profiles import ProfileMixin

    return ProfileMixin._PROFILE_SPECS


def _get_presets():
    from lasagna.sdk.config import (
        config_for_speed,
        config_for_standard,
        config_for_balanced,
        config_for_max_quality,
    )

    return {
        "speed": config_for_speed(),
        "standard": config_for_standard(),
        "balanced": config_for_balanced(),
        "max_quality": config_for_max_quality(),
    }


@pytest.fixture(scope="module")
def profile_specs():
    return _get_profile_specs()


@pytest.fixture(scope="module")
def presets():
    return _get_presets()


@pytest.fixture(scope="module")
def shared_keys():
    return _compute_shared_keys()


# =========================================================================
# Parity Tests
# =========================================================================


class TestPresetParity:
    """SDK presets must match _PROFILE_SPECS for all shared keys."""

    def test_speed_preset_parity(self, profile_specs, presets, shared_keys):
        self._assert_parity("speed", profile_specs, presets, shared_keys)

    def test_standard_preset_parity(self, profile_specs, presets, shared_keys):
        self._assert_parity("standard", profile_specs, presets, shared_keys)

    def test_balanced_preset_parity(self, profile_specs, presets, shared_keys):
        self._assert_parity("balanced", profile_specs, presets, shared_keys)

    def test_max_quality_preset_parity(self, profile_specs, presets, shared_keys):
        self._assert_parity("max_quality", profile_specs, presets, shared_keys)

    def _assert_parity(self, name, profile_specs, presets, shared_keys):
        spec = profile_specs[name]
        preset = presets[name]
        mismatches = []

        for key in shared_keys:
            if key not in spec:
                continue
            spec_val = spec[key]
            preset_val = getattr(preset, key, "__MISSING__")
            # Handle enum values — compare raw value
            if hasattr(preset_val, "value"):
                preset_val = preset_val.value
            if preset_val != spec_val:
                mismatches.append(f"  {key}: SDK={preset_val!r}, profile_spec={spec_val!r}")

        assert not mismatches, (
            f"Drift detected in '{name}' preset vs _PROFILE_SPECS:\n" + "\n".join(mismatches)
        )


# =========================================================================
# Coverage Tests
# =========================================================================


def test_all_profile_names_have_presets():
    """Every name in _PROFILE_SPECS must have a matching SDK preset."""
    specs = _get_profile_specs()
    presets = _get_presets()

    missing = set(specs.keys()) - set(presets.keys())
    assert not missing, f"Missing SDK presets for profiles: {missing}"


def test_shared_keys_is_nonempty(shared_keys):
    """Sanity check: the dynamic SHARED_KEYS computation must find keys."""
    assert len(shared_keys) >= 10, (
        f"Expected at least 10 shared keys, got {len(shared_keys)}: {shared_keys}"
    )


def test_reranker_model_parity(profile_specs, presets):
    """reranker_model should match when set in _PROFILE_SPECS."""
    for name, spec in profile_specs.items():
        if "reranker_model" not in spec:
            continue
        preset = presets.get(name)
        if preset is None:
            continue
        assert preset.reranker_model == spec["reranker_model"], (
            f"Drift: {name}.reranker_model — "
            f"SDK={preset.reranker_model!r}, profile={spec['reranker_model']!r}"
        )


# =========================================================================
# Deep Merge Tests
# =========================================================================


# =========================================================================
# Config Bridge Tests
# =========================================================================


def test_speed_preset_bridges_query_classification_false():
    """Speed preset must bridge enable_query_classification=False to backend."""
    from lasagna.sdk.config import config_for_speed
    from lasagna.sdk._config_bridge import get_runtime_overrides

    cfg = config_for_speed()
    assert cfg.enable_query_classification is False, (
        "config_for_speed() must set enable_query_classification=False"
    )
    overrides = get_runtime_overrides(cfg)
    assert overrides.get("enable_query_classification") is False, (
        "enable_query_classification=False must appear in runtime overrides "
        "(it differs from SDK default=True)"
    )


def test_non_speed_presets_do_not_bridge_query_classification():
    """Standard/balanced/max_quality keep default=True, so bridge omits the key."""
    from lasagna.sdk.config import (
        config_for_standard,
        config_for_balanced,
        config_for_max_quality,
    )
    from lasagna.sdk._config_bridge import get_runtime_overrides

    for name, fn in [
        ("standard", config_for_standard),
        ("balanced", config_for_balanced),
        ("max_quality", config_for_max_quality),
    ]:
        cfg = fn()
        assert cfg.enable_query_classification is True, (
            f"{name} preset should keep enable_query_classification=True (default)"
        )
        overrides = get_runtime_overrides(cfg)
        assert "enable_query_classification" not in overrides, (
            f"{name}: default-valued enable_query_classification should NOT "
            "appear in runtime overrides (bridge only sends non-defaults)"
        )


# =========================================================================
# SDK Validation Rule Tests
# =========================================================================


def test_quality_gates_warns_without_structured_graph():
    """Quality-gates validation must warn when structured graph is disabled."""
    from lasagna.sdk.config_validation import (
        quality_gates_require_structured_graph,
        Severity,
    )
    from lasagna.sdk.config import KnowledgeConfig

    # hallucination check without structured graph
    cfg = KnowledgeConfig(enable_hallucination_check=True, use_structured_graph=False)
    issues = quality_gates_require_structured_graph(cfg)
    assert len(issues) == 1
    assert issues[0].severity == Severity.WARNING
    assert "enable_hallucination_check" in issues[0].message

    # answer quality check without structured graph
    cfg2 = KnowledgeConfig(enable_answer_quality_check=True, use_structured_graph=False)
    issues2 = quality_gates_require_structured_graph(cfg2)
    assert len(issues2) == 1
    assert "enable_answer_quality_check" in issues2[0].message

    # both checks without structured graph
    cfg3 = KnowledgeConfig(
        enable_hallucination_check=True,
        enable_answer_quality_check=True,
        use_structured_graph=False,
    )
    issues3 = quality_gates_require_structured_graph(cfg3)
    assert len(issues3) == 2


def test_quality_gates_silent_with_structured_graph():
    """No warnings when quality gates are used WITH structured graph."""
    from lasagna.sdk.config_validation import (
        quality_gates_require_structured_graph,
    )
    from lasagna.sdk.config import KnowledgeConfig

    cfg = KnowledgeConfig(
        enable_hallucination_check=True,
        enable_answer_quality_check=True,
        use_structured_graph=True,
    )
    issues = quality_gates_require_structured_graph(cfg)
    assert issues == []


# =========================================================================
# Deep Merge Tests
# =========================================================================


def test_deep_merge_preserves_base_nested_keys():
    """_deep_merge should merge nested dicts, not replace them."""
    from lasagna.core.session_config._data_models import _deep_merge

    base = {
        "chunk_cite_retrieval": {
            "strategy": "adaptive",
            "max_chunks": 20,
            "extra_key": True,
        }
    }
    override = {"chunk_cite_retrieval": {"max_chunks": 30}}
    _deep_merge(base, override)
    assert base["chunk_cite_retrieval"]["strategy"] == "adaptive"  # preserved
    assert base["chunk_cite_retrieval"]["max_chunks"] == 30  # overridden
    assert base["chunk_cite_retrieval"]["extra_key"] is True  # preserved


def test_deep_merge_replaces_non_dict_values():
    """_deep_merge should replace non-dict values entirely."""
    from lasagna.core.session_config._data_models import _deep_merge

    base = {"enable_hyde": False, "chunk_size": 800}
    override = {"enable_hyde": True, "chunk_size": 1000}
    _deep_merge(base, override)
    assert base["enable_hyde"] is True
    assert base["chunk_size"] == 1000
