"""Verify registry compatibility contract.

Ensures:
- core/registry.py and retrieval/registry.py export the same singleton
- KnowledgePipeline imports still resolve
- Generic API works (has, get, list, get_class)
- Decorator registration works with ABC validation
- enhancer/query_enhancer alias resolves both directions
- Error messages are clear and actionable
"""

import pytest


# =========================================================================
# Singleton & Import Path Tests
# =========================================================================


def test_core_and_retrieval_share_singleton():
    """Both import paths must return the same registry instance."""
    from lasagna.core.registry import registry as core_registry
    from lasagna.retrieval.registry import (
        component_registry as retrieval_registry,
    )

    assert core_registry is retrieval_registry


def test_component_types_constant_exists():
    """COMPONENT_TYPES must be importable from core.registry."""
    from lasagna.core.registry import COMPONENT_TYPES

    assert "reranker" in COMPONENT_TYPES
    assert "embedder" in COMPONENT_TYPES
    assert "chunker" in COMPONENT_TYPES
    assert "retriever" in COMPONENT_TYPES
    assert "document_processor" in COMPONENT_TYPES
    assert "query_enhancer" in COMPONENT_TYPES


def test_sdk_component_registry_reexport():
    """SDK module re-exports the same registry."""
    from lasagna.sdk.component_registry import component_registry
    from lasagna.core.registry import registry

    assert component_registry is registry


def test_pipeline_imports_resolve():
    """KnowledgePipeline's registry imports must not break."""
    from lasagna.core.pipeline import KnowledgePipeline

    assert hasattr(KnowledgePipeline, "replace")
    assert hasattr(KnowledgePipeline, "get_component")


# =========================================================================
# Generic API Tests
# =========================================================================


def test_registry_list_returns_dict_of_lists():
    """list() returns {component_type: [names]} with correct types."""
    from lasagna.core.registry import registry

    result = registry.list()
    assert isinstance(result, dict)
    for key, names in result.items():
        assert isinstance(key, str)
        assert isinstance(names, list)
        for name in names:
            assert isinstance(name, str)


def test_registry_has_builtin():
    """has() returns True for built-in components."""
    from lasagna.core.registry import registry

    assert registry.has("reranker", "cross_encoder")


def test_registry_get_class_returns_class():
    """get_class() returns the actual class, not an instance."""
    from lasagna.core.registry import registry

    cls = registry.get_class("reranker", "cross_encoder")
    assert cls is not None
    assert isinstance(cls, type)
    assert hasattr(cls, "rerank")


def test_registry_list_with_filter():
    """list(component_type) returns only that type."""
    from lasagna.core.registry import registry

    result = registry.list("reranker")
    assert set(result.keys()) == {"reranker"}
    assert "cross_encoder" in result["reranker"]


# =========================================================================
# Alias Tests
# =========================================================================


def test_query_enhancer_alias_both_directions():
    """Both 'enhancer' and 'query_enhancer' resolve to the same store."""
    from lasagna.core.registry import registry

    # Both aliases must return the same result
    qe_list = registry.list("query_enhancer")
    e_list = registry.list("enhancer")
    assert qe_list == e_list

    # has() works with both aliases
    for name in qe_list.get("query_enhancer", []):
        assert registry.has("enhancer", name), f"has('enhancer', '{name}') should be True"
        assert registry.has("query_enhancer", name), (
            f"has('query_enhancer', '{name}') should be True"
        )


# =========================================================================
# Registration Tests
# =========================================================================


def test_decorator_registration():
    """@registry.register() decorator works for custom components."""
    from lasagna.core.registry import registry
    from lasagna.core.base_components import BaseReranker

    @registry.register("reranker", "_test_reranker")
    class TestReranker(BaseReranker):
        def rerank(self, query, documents, top_k=5, **kwargs):
            return documents[:top_k]

    try:
        assert registry.has("reranker", "_test_reranker")
        cls = registry.get_class("reranker", "_test_reranker")
        assert cls is TestReranker
    finally:
        registry.unregister_reranker("_test_reranker")

    assert not registry.has("reranker", "_test_reranker")


def test_adapter_registrations_present():
    """Adapter-wrapped builtins must be registered with _adapted suffix."""
    from lasagna.core.registry import registry

    assert registry.has("reranker", "cross_encoder_adapted")
    assert registry.has("reranker", "combined_adapted")
    assert registry.has("chunker", "semantic_adapted")
    assert registry.has("embedder", "default_adapted")
    assert registry.has("query_enhancer", "combined_adapted")
    assert registry.has("document_processor", "default_adapted")


def test_register_class_with_override():
    """register_class() with override=True replaces existing."""
    from lasagna.core.registry import registry

    class V1:
        pass

    class V2:
        pass

    registry.register_class("reranker", "_override_test", V1, override=True)
    registry.register_class("reranker", "_override_test", V2, override=True)
    try:
        assert registry.get_class("reranker", "_override_test") is V2
    finally:
        registry.unregister_reranker("_override_test")


# =========================================================================
# Error Path Tests
# =========================================================================


def test_get_nonexistent_raises_key_error():
    """get() with unknown name must raise KeyError with available list."""
    from lasagna.core.registry import registry

    with pytest.raises(KeyError, match="Available"):
        registry.get("reranker", "_does_not_exist_12345")


def test_get_class_nonexistent_raises_key_error():
    """get_class() with unknown name must raise KeyError with available list."""
    from lasagna.core.registry import registry

    with pytest.raises(KeyError, match="Available"):
        registry.get_class("reranker", "_does_not_exist_12345")


def test_invalid_component_type_raises_value_error():
    """Operations on invalid component types must raise ValueError."""
    from lasagna.core.registry import registry

    with pytest.raises(ValueError, match="Valid types"):
        registry.get("not_a_type", "foo")

    with pytest.raises(ValueError, match="Valid types"):
        registry.get_class("not_a_type", "foo")

    with pytest.raises(ValueError, match="Valid types"):
        registry.list("not_a_type")

    with pytest.raises(ValueError, match="Valid types"):
        registry.has("not_a_type", "foo")


def test_register_duplicate_without_override_raises():
    """register_class() without override=True on existing name must raise."""
    from lasagna.core.registry import registry

    class DummyReranker:
        def rerank(self, query, documents, top_k=5):
            return documents

    registry.register_class("reranker", "_dup_test", DummyReranker, override=True)
    try:
        with pytest.raises(ValueError, match="already registered"):
            registry.register_class("reranker", "_dup_test", DummyReranker)
    finally:
        registry.unregister_reranker("_dup_test")


class TestResearchProvenance:
    """Components may declare where they came from and what was claimed.

    This is what lets a results table show a paper's reported metric next to
    the one actually reproduced here, so the fields must survive registration
    and come back out of the listing API.
    """

    def test_paper_and_claims_round_trip(self):
        from lasagna.core.registry import registry
        from lasagna.core.base_components import BaseReranker

        @registry.register(
            "reranker",
            "_provenance_probe",
            paper="arxiv:1234.56789",
            claims={"beir/scifact": {"ndcg@10": 0.456}},
            version="1.2.3",
        )
        class _ProbeReranker(BaseReranker):
            def rerank(self, query, documents, top_k=None):
                return documents

        try:
            entry = next(r for r in registry.list_rerankers() if r["name"] == "_provenance_probe")
            assert entry["paper"] == "arxiv:1234.56789"
            assert entry["claims"] == {"beir/scifact": {"ndcg@10": 0.456}}
            assert entry["version"] == "1.2.3"
        finally:
            registry._components["reranker"].pop("_provenance_probe", None)

    def test_components_without_provenance_stay_uncluttered(self):
        """Built-ins declare no paper, so those keys must be absent, not None."""
        from lasagna.core.registry import registry

        entry = next(r for r in registry.list_rerankers() if r["name"] == "cross_encoder")
        assert "paper" not in entry
        assert "claims" not in entry
        assert {"name", "description", "class"} <= set(entry)
