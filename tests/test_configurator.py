"""
Comprehensive tests for the RAG Configurator.

Tests the full lifecycle: schema definition, config validation, cross-field
rules, reindex detection, and profile CRUD (create, modify, delete, rename).

Run with:
    pytest tests/test_configurator.py -v

Sections:
    1. Schema integrity — all parameters defined correctly
    2. Validation — type checks, range checks, cross-field rules
    3. Reindex detection — embedding/chunking param changes flag reindex
    4. Profile lifecycle — create, read, modify, delete, rename, guards
    5. API routes — FastAPI endpoint integration tests
"""

import re

import pytest

# ── Schema & Validation (no external dependencies) ──────────────────────
from lasagna.configurator.schema import (
    PARAMETERS,
    TABS,
    GROUPS,
    EnumOption,
    get_schema_dict,
    get_reindex_keys,
    get_defaults,
)
from lasagna.configurator.validation import (
    validate_config,
    ValidationResult,
)


@pytest.fixture
def fresh_config_store(tmp_path):
    """A profile and session store of this test's own.

    Profiles and each session's active profile persist in the process-wide
    database behind ``get_rag_database()``, and these tests share one session
    id. Without a store per test, ``set_active_profile("speed")`` in one test
    leaked into the next, so the outcome depended on test order. Patching
    ``_resolve_user_data_path`` (what these fixtures used to do) isolates
    nothing: that path feeds only the legacy JSON migration.
    """
    from lasagna.core.database import get_rag_database, reset_rag_database

    reset_rag_database()
    get_rag_database(str(tmp_path / "rag.db"))
    yield
    reset_rag_database()


# =========================================================================
# 1. Schema Integrity
# =========================================================================


class TestSchemaIntegrity:
    """Verify the parameter schema is complete and well-formed."""

    def test_parameters_list_is_non_empty(self):
        """At least 60 parameters should be defined."""
        assert len(PARAMETERS) >= 60, f"Expected 60+ params, got {len(PARAMETERS)}"

    def test_all_parameters_have_required_fields(self):
        """Every parameter must have key, type, label, tab, group, default."""
        for p in PARAMETERS:
            assert p.key, f"Parameter missing key: {p}"
            assert p.type in ("boolean", "integer", "float", "string", "enum", "object"), (
                f"Invalid type '{p.type}' for {p.key}"
            )
            assert p.label, f"Parameter {p.key} missing label"
            assert p.tab in ("essential", "search", "processing", "advanced"), (
                f"Invalid tab '{p.tab}' for {p.key}"
            )
            assert p.group, f"Parameter {p.key} missing group"

    def test_no_duplicate_keys(self):
        """Parameter keys must be unique."""
        keys = [p.key for p in PARAMETERS]
        duplicates = [k for k in keys if keys.count(k) > 1]
        assert not duplicates, f"Duplicate parameter keys: {set(duplicates)}"

    def test_enum_parameters_have_options(self):
        """Enum-type parameters must define at least 2 options."""
        for p in PARAMETERS:
            if p.type == "enum":
                assert p.enum_options and len(p.enum_options) >= 2, (
                    f"Enum param '{p.key}' needs at least 2 options, got {len(p.enum_options or [])}"
                )

    def test_numeric_parameters_have_ranges(self):
        """Integer and float parameters should have min/max defined."""
        for p in PARAMETERS:
            if p.type in ("integer", "float"):
                if p.hidden:
                    continue  # Auto-computed fields may skip ranges
                assert p.min_value is not None or p.max_value is not None, (
                    f"Numeric param '{p.key}' should have min or max defined"
                )

    def test_tabs_definition(self):
        """Four tabs should be defined: essential, search, processing, advanced."""
        tab_keys = {t["key"] for t in TABS}
        assert tab_keys == {"essential", "search", "processing", "advanced"}

    def test_groups_reference_valid_tabs(self):
        """Every group must reference a valid tab."""
        tab_keys = {t["key"] for t in TABS}
        for g in GROUPS:
            assert g["tab"] in tab_keys, f"Group '{g['key']}' references invalid tab '{g['tab']}'"

    def test_parameters_reference_valid_groups(self):
        """Every parameter's group must exist in GROUPS."""
        group_keys = {g["key"] for g in GROUPS}
        for p in PARAMETERS:
            assert p.group in group_keys, (
                f"Parameter '{p.key}' references invalid group '{p.group}'"
            )

    def test_hidden_parameters_exist(self):
        """Auto-computed fields should be marked hidden."""
        hidden_keys = {p.key for p in PARAMETERS if p.hidden}
        # These are known auto-computed / internal fields
        expected_hidden = {"embedding_dim", "batch_size", "reranker_batch_size"}
        assert expected_hidden.issubset(hidden_keys), (
            f"Expected hidden params {expected_hidden - hidden_keys} not marked hidden"
        )

    def test_reindex_keys_are_correct(self):
        """Embedding and chunking params must require reindex."""
        reindex = set(get_reindex_keys())
        # These should always require reindex
        must_reindex = {
            "embedding_model",
            "embedding_dim",
            "chunking_strategy",
            "chunk_size",
            "chunk_overlap",
        }
        assert must_reindex.issubset(reindex), (
            f"Missing from reindex keys: {must_reindex - reindex}"
        )

    def test_get_schema_dict_structure(self):
        """get_schema_dict() should return tabs, groups, and parameters."""
        schema = get_schema_dict()
        assert "tabs" in schema
        assert "groups" in schema
        assert "parameters" in schema
        assert isinstance(schema["parameters"], list)
        assert len(schema["parameters"]) == len(PARAMETERS)

    def test_get_defaults_returns_all_keys(self):
        """get_defaults() should return a value for every non-nested parameter."""
        defaults = get_defaults()
        for p in PARAMETERS:
            if p.parent_key:
                # Nested params are under their parent
                assert p.parent_key in defaults, (
                    f"Parent '{p.parent_key}' for '{p.key}' missing from defaults"
                )
            else:
                assert p.key in defaults, f"'{p.key}' missing from defaults"

    def test_parameter_to_dict_roundtrip(self):
        """ParameterDef.to_dict() should produce valid JSON-serializable dicts."""
        for p in PARAMETERS:
            d = p.to_dict()
            assert d["key"] == p.key
            assert d["type"] == p.type
            assert d["label"] == p.label
            assert d["tab"] == p.tab
            assert d["group"] == p.group
            assert "default" in d

    def test_enum_option_to_dict(self):
        """EnumOption.to_dict() should contain value, label, description."""
        opt = EnumOption("test_val", "Test Label", "A description")
        d = opt.to_dict()
        assert d == {"value": "test_val", "label": "Test Label", "description": "A description"}

    def test_nested_params_have_parent_key(self):
        """Parameters with dotted keys should have parent_key set."""
        for p in PARAMETERS:
            if "." in p.key:
                assert p.parent_key is not None, f"Dotted param '{p.key}' missing parent_key"


# =========================================================================
# 2. Validation — Type Checks, Range Checks, Cross-Field Rules
# =========================================================================


class TestValidationBasic:
    """Test basic type and range validation."""

    def test_valid_config_passes(self):
        """A config with all default values should pass validation."""
        defaults = get_defaults()
        result = validate_config(defaults)
        assert result.valid, f"Default config should be valid, got errors: {result.errors}"
        assert len(result.errors) == 0

    def test_empty_config_passes(self):
        """An empty config should pass (no values to validate)."""
        result = validate_config({})
        assert result.valid

    def test_boolean_type_error(self):
        """Non-boolean value for a boolean param should produce an error."""
        result = validate_config({"enable_hybrid_search": "yes"})
        assert not result.valid
        error_keys = {e["key"] for e in result.errors}
        assert "enable_hybrid_search" in error_keys

    def test_integer_type_error(self):
        """String value for an integer param should produce an error."""
        result = validate_config({"default_top_k": "fifty"})
        assert not result.valid
        error_keys = {e["key"] for e in result.errors}
        assert "default_top_k" in error_keys

    def test_integer_below_minimum(self):
        """Value below min_value should produce an error."""
        result = validate_config({"default_top_k": 0})
        assert not result.valid
        assert any("below minimum" in e["message"] for e in result.errors)

    def test_integer_above_maximum(self):
        """Value above max_value should produce an error."""
        result = validate_config({"default_top_k": 100_001})
        assert not result.valid
        assert any("exceeds maximum" in e["message"] for e in result.errors)

    def test_float_below_minimum(self):
        """Float below its min should error."""
        result = validate_config({"similarity_threshold": -0.5})
        assert not result.valid

    def test_float_above_maximum(self):
        """Float above its max should error."""
        result = validate_config({"similarity_threshold": 1.5})
        assert not result.valid

    def test_enum_invalid_value(self):
        """Invalid enum value should produce an error."""
        result = validate_config({"chunking_strategy": "nonexistent"})
        assert not result.valid
        assert any("Invalid value" in e["message"] for e in result.errors)

    def test_enum_valid_value(self):
        """Valid enum value should pass."""
        result = validate_config({"chunking_strategy": "semantic"})
        assert result.valid

    def test_valid_boolean_values(self):
        """True and False should both be valid for boolean params."""
        for val in (True, False):
            result = validate_config({"enable_hybrid_search": val})
            assert result.valid, f"Boolean {val} should be valid"

    def test_ocr_languages_valid_list(self):
        """Valid list of strings should pass."""
        result = validate_config({"ocr_languages": ["en", "fr"]})
        assert result.valid

    def test_ocr_languages_wrong_type(self):
        """Non-list value should produce an error."""
        result = validate_config({"ocr_languages": 42})
        assert not result.valid
        assert any(e["key"] == "ocr_languages" for e in result.errors)

    def test_ocr_languages_non_string_items(self):
        """List with non-string items should produce an error."""
        result = validate_config({"ocr_languages": ["en", 123]})
        assert not result.valid
        assert any("strings" in e["message"] for e in result.errors)


class TestValidationCrossField:
    """Test cross-field validation rules."""

    def test_chunk_overlap_exceeds_chunk_size_warning(self):
        """chunk_overlap >= chunk_size should produce a warning."""
        result = validate_config({"chunk_size": 500, "chunk_overlap": 600})
        assert any(w["key"] == "chunk_overlap" for w in result.warnings), (
            "Should warn when overlap >= size"
        )

    def test_chunk_overlap_equal_to_size_warning(self):
        """chunk_overlap == chunk_size should also warn."""
        result = validate_config({"chunk_size": 500, "chunk_overlap": 500})
        assert any(w["key"] == "chunk_overlap" for w in result.warnings)

    def test_valid_chunk_overlap(self):
        """chunk_overlap < chunk_size should not warn."""
        result = validate_config({"chunk_size": 800, "chunk_overlap": 150})
        overlap_warnings = [w for w in result.warnings if w["key"] == "chunk_overlap"]
        assert len(overlap_warnings) == 0

    def test_min_chunk_exceeds_max_chunk_error(self):
        """min_chunk_size >= max_chunk_size should error."""
        result = validate_config({"min_chunk_size": 2000, "max_chunk_size": 1000})
        assert not result.valid
        assert any(e["key"] == "min_chunk_size" for e in result.errors)

    def test_child_exceeds_parent_chunk_error(self):
        """child_chunk_size >= parent_chunk_size should error."""
        result = validate_config({"child_chunk_size": 3000, "parent_chunk_size": 2000})
        assert not result.valid
        assert any(e["key"] == "child_chunk_size" for e in result.errors)

    def test_embedding_dim_model_mismatch_warning(self):
        """Mismatched embedding_dim for a known model should warn."""
        result = validate_config(
            {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dim": 1024,  # Should be 384
            }
        )
        assert any(w["key"] == "embedding_dim" for w in result.warnings), (
            "Should warn on dim mismatch"
        )

    def test_embedding_dim_model_match_no_warning(self):
        """Correct dim for a known model should not warn."""
        result = validate_config(
            {
                "embedding_model": "BAAI/bge-large-en-v1.5",
                "embedding_dim": 1024,
            }
        )
        dim_warnings = [w for w in result.warnings if w["key"] == "embedding_dim"]
        assert len(dim_warnings) == 0

    def test_chunk_retrieval_min_exceeds_max_error(self):
        """min_chunks > max_chunks in nested chunk_cite_retrieval should error."""
        result = validate_config(
            {
                "chunk_cite_retrieval": {
                    "min_chunks": 25,
                    "max_chunks": 10,
                }
            }
        )
        assert not result.valid
        assert any("min_chunks" in e["key"] for e in result.errors)

    def test_structured_graph_without_agentic_rag_warning(self):
        """use_structured_graph=True without use_agentic_rag should warn."""
        result = validate_config(
            {
                "use_structured_graph": True,
                "use_agentic_rag": False,
            }
        )
        assert any(w["key"] == "use_structured_graph" for w in result.warnings), (
            "Should warn when structured graph enabled without agentic RAG"
        )

    def test_chunk_retrieval_valid_bounds(self):
        """Valid min_chunks < max_chunks should pass."""
        result = validate_config(
            {
                "chunk_cite_retrieval": {
                    "min_chunks": 2,
                    "max_chunks": 20,
                }
            }
        )
        ccr_errors = [e for e in result.errors if "chunk_cite_retrieval" in e["key"]]
        assert len(ccr_errors) == 0


class TestValidationResult:
    """Test the ValidationResult dataclass."""

    def test_fresh_result_is_valid(self):
        result = ValidationResult()
        assert result.valid
        assert result.errors == []
        assert result.warnings == []
        assert not result.requires_reindex

    def test_add_error_invalidates(self):
        result = ValidationResult()
        result.add_error("key", "msg")
        assert not result.valid
        assert len(result.errors) == 1

    def test_add_warning_keeps_valid(self):
        result = ValidationResult()
        result.add_warning("key", "msg")
        assert result.valid
        assert len(result.warnings) == 1

    def test_to_dict_structure(self):
        result = ValidationResult()
        result.add_error("k1", "err")
        result.add_warning("k2", "warn")
        result.requires_reindex = True
        result.reindex_keys = ["embedding_model"]
        d = result.to_dict()
        assert d["valid"] is False
        assert len(d["errors"]) == 1
        assert len(d["warnings"]) == 1
        assert d["requires_reindex"] is True
        assert d["reindex_keys"] == ["embedding_model"]


# =========================================================================
# 3. Reindex Detection
# =========================================================================


class TestReindexDetection:
    """Verify reindex is flagged when embedding/chunking params change."""

    def test_embedding_model_change_triggers_reindex(self):
        """Changing embedding_model should require reindex."""
        baseline = {"embedding_model": "all-MiniLM-L6-v2"}
        config = {"embedding_model": "BAAI/bge-large-en-v1.5"}
        result = validate_config(config, baseline)
        assert result.requires_reindex
        assert "embedding_model" in result.reindex_keys

    def test_chunk_size_change_triggers_reindex(self):
        """Changing chunk_size should require reindex."""
        baseline = {"chunk_size": 800}
        config = {"chunk_size": 1200}
        result = validate_config(config, baseline)
        assert result.requires_reindex
        assert "chunk_size" in result.reindex_keys

    def test_chunking_strategy_change_triggers_reindex(self):
        """Changing chunking_strategy should require reindex."""
        baseline = {"chunking_strategy": "semantic"}
        config = {"chunking_strategy": "fixed"}
        result = validate_config(config, baseline)
        assert result.requires_reindex
        assert "chunking_strategy" in result.reindex_keys

    def test_non_reindex_change_no_flag(self):
        """Changing a non-reindex param should NOT flag reindex."""
        baseline = {"default_top_k": 75}
        config = {"default_top_k": 100}
        result = validate_config(config, baseline)
        assert not result.requires_reindex
        assert result.reindex_keys == []

    def test_same_value_no_reindex(self):
        """Same value should not trigger reindex even for reindex-sensitive keys."""
        baseline = {"embedding_model": "all-MiniLM-L6-v2"}
        config = {"embedding_model": "all-MiniLM-L6-v2"}
        result = validate_config(config, baseline)
        assert not result.requires_reindex

    def test_no_baseline_no_reindex(self):
        """Without baseline, reindex should never be flagged."""
        config = {"embedding_model": "BAAI/bge-large-en-v1.5", "chunk_size": 2000}
        result = validate_config(config, baseline=None)
        assert not result.requires_reindex

    def test_multiple_reindex_keys(self):
        """Multiple reindex-sensitive changes should all be reported."""
        baseline = {
            "embedding_model": "BAAI/bge-m3",
            "chunk_size": 800,
            "chunk_overlap": 0,
        }
        config = {
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "chunk_size": 1200,
            "chunk_overlap": 200,
        }
        result = validate_config(config, baseline)
        assert result.requires_reindex
        assert len(result.reindex_keys) >= 3


# =========================================================================
# 4. Profile Lifecycle (CRUD) via KnowledgeConfigManager
# =========================================================================


class TestProfileLifecycle:
    """Full lifecycle tests: create, read, modify, delete, rename profiles.

    Each test gets a fresh KnowledgeConfigManager over a store of its own
    (``fresh_config_store``), so no test sees another's profiles.
    """

    @pytest.fixture
    def manager(self, fresh_config_store):
        """A KnowledgeConfigManager over a store of this test's own."""
        from lasagna.core.session_config import KnowledgeConfigManager

        return KnowledgeConfigManager()

    @pytest.fixture
    def session_id(self):
        # Sessions persist in the suite's one database, and one restored with a
        # saved active profile keeps it: a shared id leaked between tests.
        import uuid

        return f"test-session-{uuid.uuid4().hex[:8]}"

    # ── List built-in profiles ───────────────────────────────────────────

    def test_list_builtin_profiles(self, manager, session_id):
        """Built-in profiles (default, speed, standard, balanced, max_quality) should exist."""
        profiles = manager.list_profiles(session_id=session_id)
        names = {p["name"] for p in profiles}
        expected = {"default", "speed", "standard", "balanced", "max_quality"}
        assert expected.issubset(names), f"Missing built-in profiles: {expected - names}"

    def test_builtin_profiles_marked_as_builtin(self, manager, session_id):
        """Built-in profiles should have builtin=True."""
        profiles = manager.list_profiles(session_id=session_id)
        for p in profiles:
            if p["name"] in ("default", "speed", "standard", "balanced", "max_quality"):
                assert p["builtin"] is True, f"'{p['name']}' should be builtin"

    def test_default_active_profile(self, manager, session_id):
        """Default active profile should be 'default'."""
        active = manager.get_active_profile(session_id=session_id)
        assert active == "default"

    # ── Create a profile ─────────────────────────────────────────────────

    def test_create_custom_profile(self, manager, session_id):
        """Creating a custom profile should add it to the profile list."""
        config = {"enable_hybrid_search": True, "default_top_k": 50}
        manager.save_profile("my_custom", config, session_id=session_id)

        profiles = manager.list_profiles(session_id=session_id)
        names = {p["name"] for p in profiles}
        assert "my_custom" in names

    def test_create_profile_sets_active(self, manager, session_id):
        """By default, saving a profile makes it the active one."""
        config = {"enable_hybrid_search": True}
        manager.save_profile("new_active", config, session_id=session_id)

        active = manager.get_active_profile(session_id=session_id)
        assert active == "new_active"

    def test_save_as_active_changes_what_search_runs(self, manager, session_id):
        """The configurator's "Save As" must reach search, not just the store.

        It reported "Saved!" while retrieval kept its old settings, because
        saving never published the profile to the config search reads.
        """
        from lasagna.core.config import rag_config

        manager.save_profile("no_rerank", {"enable_reranking": False}, session_id=session_id)
        assert rag_config.with_session(session_id).enable_reranking is False

    def test_create_profile_without_set_active(self, manager, session_id):
        """With set_active=False, the active profile should not change."""
        manager.save_profile(
            "background_profile",
            {"default_top_k": 100},
            session_id=session_id,
            set_active=False,
        )
        active = manager.get_active_profile(session_id=session_id)
        assert active == "default"

    def test_create_profile_with_full_config(self, manager, session_id):
        """Creating a profile with the full default config should succeed."""
        defaults = get_defaults()
        manager.save_profile("full_config", defaults, session_id=session_id)

        profiles = manager.list_profiles(session_id=session_id)
        names = {p["name"] for p in profiles}
        assert "full_config" in names

    # ── Modify a profile ─────────────────────────────────────────────────

    def test_modify_custom_profile(self, manager, session_id):
        """Saving a profile with the same name should update its config."""
        config_v1 = {"default_top_k": 50}
        manager.save_profile("mutable", config_v1, session_id=session_id)

        config_v2 = {"default_top_k": 100, "enable_hyde": True}
        manager.save_profile("mutable", config_v2, session_id=session_id)

        # Should still have only one profile named 'mutable'
        profiles = manager.list_profiles(session_id=session_id)
        mutable_count = sum(1 for p in profiles if p["name"] == "mutable")
        assert mutable_count == 1

    def test_overwrite_default_profile_raises(self, manager, session_id):
        """Saving to the 'default' profile name should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be overwritten"):
            manager.save_profile("default", {"default_top_k": 50}, session_id=session_id)

    def test_profile_name_validation_empty(self, manager, session_id):
        """Empty profile name should raise ValueError."""
        with pytest.raises(ValueError, match="1-100 characters"):
            manager.save_profile("", {"default_top_k": 50}, session_id=session_id)

    def test_profile_name_validation_too_long(self, manager, session_id):
        """Profile name > 100 chars should raise ValueError."""
        long_name = "x" * 101
        with pytest.raises(ValueError, match="1-100 characters"):
            manager.save_profile(long_name, {"default_top_k": 50}, session_id=session_id)

    def test_profile_config_must_be_dict(self, manager, session_id):
        """Non-dict config should raise ValueError."""
        with pytest.raises(ValueError, match="must be an object"):
            manager.save_profile("bad", "not-a-dict", session_id=session_id)

    # ── Delete a profile ─────────────────────────────────────────────────

    def test_delete_custom_profile(self, manager, session_id):
        """Deleting a custom profile should remove it from the list."""
        manager.save_profile("to_delete", {"default_top_k": 10}, session_id=session_id)

        profiles_before = manager.list_profiles(session_id=session_id)
        assert any(p["name"] == "to_delete" for p in profiles_before)

        manager.delete_profile("to_delete", session_id=session_id)

        profiles_after = manager.list_profiles(session_id=session_id)
        assert not any(p["name"] == "to_delete" for p in profiles_after)

    def test_delete_active_profile_falls_back_to_default(self, manager, session_id):
        """Deleting the active profile should switch back to 'default'."""
        manager.save_profile("active_then_delete", {"default_top_k": 10}, session_id=session_id)
        assert manager.get_active_profile(session_id=session_id) == "active_then_delete"

        manager.delete_profile("active_then_delete", session_id=session_id)
        assert manager.get_active_profile(session_id=session_id) == "default"

    def test_delete_builtin_profile_raises(self, manager, session_id):
        """Deleting a built-in profile should raise ValueError."""
        for name in ("default", "speed", "standard", "balanced", "max_quality"):
            with pytest.raises(ValueError, match="cannot be deleted"):
                manager.delete_profile(name, session_id=session_id)

    def test_delete_nonexistent_profile_is_safe(self, manager, session_id):
        """Deleting a non-existent profile should not error (idempotent)."""
        # Should not raise
        manager.delete_profile("never_existed", session_id=session_id)

    # ── Rename a profile ─────────────────────────────────────────────────

    def test_rename_custom_profile(self, manager, session_id):
        """Renaming a custom profile should update the list."""
        manager.save_profile("old_name", {"default_top_k": 40}, session_id=session_id)
        manager.rename_profile("old_name", "new_name", session_id=session_id)

        profiles = manager.list_profiles(session_id=session_id)
        names = {p["name"] for p in profiles}
        assert "new_name" in names
        assert "old_name" not in names

    def test_rename_updates_active_profile(self, manager, session_id):
        """Renaming the active profile should update the active reference."""
        manager.save_profile("before_rename", {"default_top_k": 40}, session_id=session_id)
        assert manager.get_active_profile(session_id=session_id) == "before_rename"

        manager.rename_profile("before_rename", "after_rename", session_id=session_id)
        assert manager.get_active_profile(session_id=session_id) == "after_rename"

    def test_rename_builtin_raises(self, manager, session_id):
        """Renaming a built-in profile should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be renamed"):
            manager.rename_profile("speed", "my_speed", session_id=session_id)

    def test_rename_to_existing_raises(self, manager, session_id):
        """Renaming to an existing profile name should raise ValueError."""
        manager.save_profile("profile_a", {"default_top_k": 10}, session_id=session_id)
        manager.save_profile("profile_b", {"default_top_k": 20}, session_id=session_id)
        with pytest.raises(ValueError, match="already exists"):
            manager.rename_profile("profile_a", "profile_b", session_id=session_id)

    def test_rename_nonexistent_raises(self, manager, session_id):
        """Renaming a profile that doesn't exist should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            manager.rename_profile("ghost", "new_ghost", session_id=session_id)

    # ── Switch active profile ────────────────────────────────────────────

    def test_switch_to_builtin_profile(self, manager, session_id):
        """Switching to a built-in profile should update the active profile."""
        manager.set_active_profile("speed", session_id=session_id)
        assert manager.get_active_profile(session_id=session_id) == "speed"

    def test_switch_active_profiles_round_trip(self, manager, session_id):
        """Switching between profiles should work bidirectionally."""
        manager.set_active_profile("balanced", session_id=session_id)
        assert manager.get_active_profile(session_id=session_id) == "balanced"

        manager.set_active_profile("speed", session_id=session_id)
        assert manager.get_active_profile(session_id=session_id) == "speed"

        manager.set_active_profile("default", session_id=session_id)
        assert manager.get_active_profile(session_id=session_id) == "default"

    # ── Full lifecycle scenario ──────────────────────────────────────────

    def test_full_profile_lifecycle(self, manager, session_id):
        """End-to-end: create → activate → modify → rename → delete."""
        # 1. Create
        config = {"enable_hybrid_search": True, "default_top_k": 60}
        manager.save_profile("lifecycle_test", config, session_id=session_id)
        assert manager.get_active_profile(session_id=session_id) == "lifecycle_test"

        # 2. Verify it exists
        profiles = manager.list_profiles(session_id=session_id)
        assert any(p["name"] == "lifecycle_test" for p in profiles)

        # 3. Modify (save with same name, new config)
        config_v2 = {"enable_hybrid_search": False, "default_top_k": 100, "enable_hyde": True}
        manager.save_profile("lifecycle_test", config_v2, session_id=session_id)

        # Still only one instance
        profiles = manager.list_profiles(session_id=session_id)
        lifecycle_profiles = [p for p in profiles if p["name"] == "lifecycle_test"]
        assert len(lifecycle_profiles) == 1

        # 4. Rename
        manager.rename_profile("lifecycle_test", "lifecycle_renamed", session_id=session_id)
        assert manager.get_active_profile(session_id=session_id) == "lifecycle_renamed"

        profiles = manager.list_profiles(session_id=session_id)
        names = {p["name"] for p in profiles}
        assert "lifecycle_renamed" in names
        assert "lifecycle_test" not in names

        # 5. Delete
        manager.delete_profile("lifecycle_renamed", session_id=session_id)
        assert manager.get_active_profile(session_id=session_id) == "default"

        profiles = manager.list_profiles(session_id=session_id)
        assert not any(p["name"] == "lifecycle_renamed" for p in profiles)


# =========================================================================
# 5. Profile Stats & Global Profiles
# =========================================================================


class TestProfileStats:
    """Test profile statistics and global profile management."""

    @pytest.fixture
    def manager(self, fresh_config_store):
        """A KnowledgeConfigManager over a store of this test's own."""
        from lasagna.core.session_config import KnowledgeConfigManager

        return KnowledgeConfigManager()

    def test_profile_stats_structure(self, manager):
        """get_profile_stats() should return expected fields."""
        stats = manager.get_profile_stats()
        assert "global_custom_profiles_count" in stats
        assert "global_custom_profiles_limit" in stats
        assert "default_profiles_count" in stats
        assert stats["default_profiles_count"] >= 5  # 5 built-in profiles

    def test_global_profiles_track_custom(self, manager):
        """Custom profiles should appear in global profile list."""
        manager.save_profile("global_test", {"default_top_k": 50})
        globals_list = manager.list_global_profiles()
        names = {p["name"] for p in globals_list}
        assert "global_test" in names

    def test_session_isolation(self, manager):
        """Different sessions should have independent active profiles."""
        manager.set_active_profile("speed", session_id="session-a")
        manager.set_active_profile("balanced", session_id="session-b")

        assert manager.get_active_profile(session_id="session-a") == "speed"
        assert manager.get_active_profile(session_id="session-b") == "balanced"


# =========================================================================
# 6. Configurator API Routes (FastAPI integration)
# =========================================================================


class TestConfiguratorAPI:
    """Test the configurator's FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create a TestClient for the configurator router."""
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi not installed")

        from lasagna.configurator.routes import router

        app = FastAPI()
        app.include_router(router, prefix="/configurator")
        return TestClient(app)

    def test_get_schema(self, client):
        """GET /configurator/api/schema should return full parameter schema."""
        resp = client.get("/configurator/api/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "tabs" in data
        assert "groups" in data
        assert "parameters" in data
        assert len(data["parameters"]) >= 60

    def test_schema_parameter_structure(self, client):
        """Each parameter in the schema response should have required fields."""
        resp = client.get("/configurator/api/schema")
        for param in resp.json()["parameters"]:
            assert "key" in param
            assert "type" in param
            assert "label" in param
            assert "tab" in param
            assert "group" in param
            assert "default" in param

    def test_validate_valid_config(self, client):
        """POST /configurator/api/validate with valid config should return valid=true."""
        resp = client.post(
            "/configurator/api/validate",
            json={
                "config": get_defaults(),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_validate_invalid_config(self, client):
        """POST /configurator/api/validate with invalid config should return errors."""
        resp = client.post(
            "/configurator/api/validate",
            json={
                "config": {"default_top_k": -10},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_validate_with_baseline_detects_reindex(self, client):
        """POST /configurator/api/validate with baseline should detect reindex needs."""
        resp = client.post(
            "/configurator/api/validate",
            json={
                "config": {"embedding_model": "BAAI/bge-large-en-v1.5"},
                "baseline": {"embedding_model": "all-MiniLM-L6-v2"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["requires_reindex"] is True
        assert "embedding_model" in data["reindex_keys"]

    def test_validate_cross_field_errors(self, client):
        """POST /configurator/api/validate should catch cross-field violations."""
        resp = client.post(
            "/configurator/api/validate",
            json={
                "config": {"min_chunk_size": 2000, "max_chunk_size": 500},
            },
        )
        data = resp.json()
        assert data["valid"] is False
        assert any("min_chunk_size" in e["key"] for e in data["errors"])

    def test_validate_nested_params(self, client):
        """Nested params like chunk_cite_retrieval should be validated."""
        resp = client.post(
            "/configurator/api/validate",
            json={
                "config": {
                    "chunk_cite_retrieval": {
                        "strategy": "invalid_strategy",
                    }
                },
            },
        )
        data = resp.json()
        assert data["valid"] is False


# =========================================================================
# 7. Edge Cases & Robustness
# =========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_validate_unknown_keys_ignored(self):
        """Unknown config keys should be silently ignored."""
        result = validate_config({"totally_unknown_key": "value"})
        assert result.valid

    def test_validate_none_values_skipped(self):
        """None values should be skipped (not validated)."""
        result = validate_config({"default_top_k": None})
        assert result.valid

    def test_validate_mixed_valid_and_invalid(self):
        """Mix of valid and invalid params should report only errors."""
        result = validate_config(
            {
                "enable_hybrid_search": True,  # valid
                "default_top_k": -10,  # invalid (below min)
                "similarity_threshold": 0.5,  # valid
                "chunking_strategy": "nonexistent",  # invalid (bad enum)
            }
        )
        assert not result.valid
        error_keys = {e["key"] for e in result.errors}
        assert "default_top_k" in error_keys
        assert "chunking_strategy" in error_keys
        # Valid keys should NOT be in errors
        assert "enable_hybrid_search" not in error_keys
        assert "similarity_threshold" not in error_keys

    def test_nested_param_validation(self):
        """Nested params under parent_key should be validated correctly."""
        # chunk_cite_retrieval.strategy is a nested enum
        result = validate_config(
            {
                "chunk_cite_retrieval": {
                    "strategy": "adaptive",  # valid
                    "max_chunks": 20,  # valid
                }
            }
        )
        ccr_errors = [e for e in result.errors if "chunk_cite_retrieval" in e["key"]]
        assert len(ccr_errors) == 0

    def test_all_enum_options_are_valid(self):
        """Every enum option's value should pass validation."""
        for p in PARAMETERS:
            if p.type == "enum" and p.enum_options:
                for opt in p.enum_options:
                    if p.parent_key:
                        child_key = p.key.split(".")[-1]
                        config = {p.parent_key: {child_key: opt.value}}
                    else:
                        config = {p.key: opt.value}
                    result = validate_config(config)
                    enum_errors = [e for e in result.errors if e["key"] == p.key]
                    assert len(enum_errors) == 0, (
                        f"Enum option '{opt.value}' for '{p.key}' should be valid but got: {enum_errors}"
                    )

    def test_default_values_all_valid(self):
        """All default values should pass validation."""
        defaults = get_defaults()
        result = validate_config(defaults)
        assert result.valid, f"Default config invalid: {result.errors}"

    def test_boundary_values_integer(self):
        """Min and max boundary values for integer params should be valid.

        Cross-field rules (e.g. child_chunk_size < parent_chunk_size) can
        legitimately reject a boundary value when tested in isolation, so
        we only check that range errors ("below minimum" / "exceeds maximum")
        are absent — not cross-field errors.
        """
        for p in PARAMETERS:
            if p.type == "integer" and p.min_value is not None and p.max_value is not None:
                # Test min boundary
                if p.parent_key:
                    child_key = p.key.split(".")[-1]
                    config = {p.parent_key: {child_key: int(p.min_value)}}
                else:
                    config = {p.key: int(p.min_value)}
                result = validate_config(config)
                range_errors = [
                    e
                    for e in result.errors
                    if e["key"] == p.key
                    and ("below minimum" in e["message"] or "exceeds maximum" in e["message"])
                ]
                assert len(range_errors) == 0, (
                    f"Min boundary {p.min_value} for '{p.key}' should be valid: {range_errors}"
                )

                # Test max boundary
                if p.parent_key:
                    config = {p.parent_key: {child_key: int(p.max_value)}}
                else:
                    config = {p.key: int(p.max_value)}
                result = validate_config(config)
                range_errors = [
                    e
                    for e in result.errors
                    if e["key"] == p.key
                    and ("below minimum" in e["message"] or "exceeds maximum" in e["message"])
                ]
                assert len(range_errors) == 0, (
                    f"Max boundary {p.max_value} for '{p.key}' should be valid: {range_errors}"
                )


# =========================================================================
# 8. Validation Enforcement — mutation methods reject invalid configs
# =========================================================================


class TestValidationEnforcement:
    """Verify that all config mutation paths enforce validation.

    These tests ensure that invalid configurations are rejected by the
    manager methods (patch_config, save_profile, set_active_profile,
    import_config) — not just the UI validate endpoint.
    """

    @pytest.fixture
    def manager(self, fresh_config_store):
        """A KnowledgeConfigManager over a store of this test's own."""
        from lasagna.core.session_config import KnowledgeConfigManager

        return KnowledgeConfigManager()

    @pytest.fixture
    def session_id(self):
        return "enforcement-test-session"

    # ── patch_config enforcement ──────────────────────────────────────

    def test_patch_config_rejects_invalid_type(self, manager, session_id):
        """patch_config should reject non-numeric value for integer param."""
        with pytest.raises(ValueError, match="Invalid configuration"):
            manager.patch_config({"default_top_k": "not_a_number"}, session_id=session_id)

    def test_patch_config_rejects_out_of_range(self, manager, session_id):
        """patch_config should reject value exceeding maximum (a session's, the lower)."""
        with pytest.raises(ValueError, match="default_top_k is at most 1000"):
            manager.patch_config({"default_top_k": 100_001}, session_id=session_id)

    def test_patch_config_rejects_cross_field_violation(self, manager, session_id):
        """patch_config should reject cross-field violation after merging with effective config.

        The effective config has parent_chunk_size=2000 (default). Patching
        child_chunk_size=3000 should fail because child must be < parent.
        """
        with pytest.raises(ValueError, match="Invalid configuration"):
            manager.patch_config({"child_chunk_size": 3000}, session_id=session_id)

    def test_patch_config_accepts_valid(self, manager, session_id):
        """patch_config should succeed with valid values."""
        result = manager.patch_config({"default_top_k": 50}, session_id=session_id)
        assert isinstance(result, dict)

    def test_patch_merges_with_effective_config(self, manager, session_id):
        """Partial patch should be validated against merged effective config."""
        # chunk_overlap=100 is valid when effective chunk_size=800 (default)
        result = manager.patch_config({"chunk_overlap": 100}, session_id=session_id)
        assert isinstance(result, dict)

    def test_patch_config_rejects_invalid_enum(self, manager, session_id):
        """patch_config should reject invalid enum value."""
        with pytest.raises(ValueError, match="Invalid configuration"):
            manager.patch_config({"chunking_strategy": "nonexistent"}, session_id=session_id)

    # ── save_profile enforcement ──────────────────────────────────────

    def test_save_profile_rejects_invalid(self, manager, session_id):
        """save_profile should reject config with out-of-range values."""
        with pytest.raises(ValueError, match="Invalid configuration"):
            manager.save_profile("bad_profile", {"default_top_k": -10}, session_id=session_id)

    def test_save_profile_accepts_valid(self, manager, session_id):
        """save_profile should succeed with valid config."""
        result = manager.save_profile("good_profile", {"default_top_k": 50}, session_id=session_id)
        assert isinstance(result, dict)
        profiles = manager.list_profiles(session_id=session_id)
        assert any(p["name"] == "good_profile" for p in profiles)

    # ── set_active_profile enforcement ────────────────────────────────

    def test_set_active_rejects_invalid_config(self, manager, session_id):
        """set_active_profile should reject invalid config dict."""
        with pytest.raises(ValueError, match="Invalid configuration"):
            manager.set_active_profile(
                "custom", config={"default_top_k": -10}, session_id=session_id
            )

    def test_set_active_accepts_valid_config(self, manager, session_id):
        """set_active_profile with valid config should succeed."""
        result = manager.set_active_profile(
            "speed", config={"default_top_k": 50}, session_id=session_id
        )
        assert isinstance(result, dict)

    def test_set_active_no_config_skips_validation(self, manager, session_id):
        """set_active_profile without config should not validate."""
        result = manager.set_active_profile("speed", session_id=session_id)
        assert isinstance(result, dict)

    # ── import_config enforcement ─────────────────────────────────────

    def test_import_rejects_invalid_profile_config(self, manager, session_id):
        """import_config should reject profiles with invalid config values."""
        import_data = {
            "schema_version": 2,
            "profiles": [
                {
                    "name": "bad_import",
                    "config": {"default_top_k": -10},
                }
            ],
        }
        with pytest.raises(ValueError, match="Invalid configuration"):
            manager.import_config(import_data, session_id=session_id)

    def test_import_accepts_valid_profile_config(self, manager, session_id):
        """import_config should accept profiles with valid config."""
        import_data = {
            "schema_version": 2,
            "profiles": [
                {
                    "name": "good_import",
                    "config": {"default_top_k": 50, "enable_hybrid_search": True},
                }
            ],
        }
        result = manager.import_config(import_data, session_id=session_id)
        assert isinstance(result, dict)

    # ── Warning behavior ──────────────────────────────────────────────

    def test_warnings_dont_reject(self, manager, session_id):
        """Config with auto-correctable warnings should be accepted (not rejected)."""
        # embedding_dim mismatch for a known model produces a warning, not an error
        result = manager.patch_config(
            {"embedding_model": "all-MiniLM-L6-v2", "embedding_dim": 1024},
            session_id=session_id,
        )
        assert isinstance(result, dict)

    # ── Error message format ──────────────────────────────────────────

    def test_validation_error_message_format(self, manager, session_id):
        """ValueError message should contain the parameter key and description."""
        with pytest.raises(ValueError) as exc_info:
            manager.patch_config({"default_top_k": -10}, session_id=session_id)
        msg = str(exc_info.value)
        assert "default_top_k" in msg
        assert "below minimum" in msg


# =========================================================================
# 9. Reindex Key Drift Guard
# =========================================================================


class TestReindexKeySync:
    """Verify backend reindex keys match the frontend static list.

    Prevents silent drift between schema.py get_reindex_keys() and
    configurator/frontend/src/schema/reindexKeys.ts.
    """

    @staticmethod
    def _parse_reindex_keys_ts() -> set[str]:
        """Parse REINDEX_KEYS from the frontend TypeScript source.

        Uses a regex anchored to the ``export const REINDEX_KEYS``
        declaration, then extracts quoted strings with a pattern that
        handles both single and double quotes and ignores inline
        comments.  No Prettier/ESLint enforces formatting in this
        frontend package, so the parser must be quote-style-agnostic.
        """
        from tests._paths import CONFIGURATOR_DIR

        ts_path = CONFIGURATOR_DIR / "frontend" / "src" / "schema" / "reindexKeys.ts"
        assert ts_path.exists(), f"Frontend source not found: {ts_path}"
        content = ts_path.read_text()
        # Match the export declaration: export const REINDEX_KEYS ... = [ ... ];
        m = re.search(
            r"export\s+const\s+REINDEX_KEYS[^=]*=\s*\[(.*?)];",
            content,
            re.DOTALL,
        )
        assert m, "Could not find 'export const REINDEX_KEYS = [...]' in reindexKeys.ts"
        array_body = m.group(1)
        # Strip single-line comments (// ...) that may appear inline
        array_body = re.sub(r"//[^\n]*", "", array_body)
        # Extract all single- or double-quoted strings
        keys = re.findall(r"""["']([^"']+)["']""", array_body)
        assert keys, "No quoted strings found in REINDEX_KEYS array body"
        return set(keys)

    def test_reindex_keys_match_frontend(self):
        """get_reindex_keys() must match reindexKeys.ts REINDEX_KEYS exactly."""
        backend_keys = set(get_reindex_keys())
        frontend_keys = self._parse_reindex_keys_ts()
        assert backend_keys == frontend_keys, (
            f"Reindex key drift! "
            f"Backend-only: {backend_keys - frontend_keys}, "
            f"Frontend-only: {frontend_keys - backend_keys}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestResearchSettingsAreValid:
    """Session validation stops corrupt values, not research choices.

    Every session write is validated now, and the UI's ranges refused settings
    a sweep needs: 60-word chunks, 64-dimension Matryoshka embeddings, TREC's
    retrieval depth of 1000, and any reranker or embedder not on a short list.
    """

    def test_research_values_pass(self):
        result = validate_config(
            {
                "chunk_size": 60,
                "embedding_dim": 64,
                "default_top_k": 1000,
                "reranker_model": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            }
        )
        assert result.valid, result.errors

    def test_a_model_name_must_still_be_a_name(self):
        assert not validate_config({"reranker_model": ""}).valid
        assert not validate_config({"embedding_model": 3}).valid
