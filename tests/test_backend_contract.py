"""Contract tests verifying SDK managers work with both local and remote backends.

These tests ensure the SDK's managers (DocumentManager, SearchManager) don't
crash on the data shapes returned by either the local RAGBackend or the
RemoteRAGBackend.  They use lightweight mocks — no heavy ML deps required.
"""

from __future__ import annotations

import dataclasses
from typing import Optional


from lasagna.sdk.types import DocumentStatus


# =========================================================================
# Fixtures: fake backend responses
# =========================================================================


@dataclasses.dataclass
class _FakeDocLocal:
    """Simulates a local RAGBackend document (status is enum)."""

    document_id: str = "doc_001"
    session_id: str = "sess_001"
    filename: str = "report.pdf"
    file_type: str = "pdf"
    file_size: int = 1024
    status: object = dataclasses.field(default_factory=lambda: _EnumStatus("ready"))
    chunks_count: int = 5
    upload_date: str = "2025-01-01"
    error_message: Optional[str] = None


@dataclasses.dataclass
class _EnumStatus:
    """Simulates an enum-like status with .value attribute."""

    value: str


@dataclasses.dataclass
class _FakeDocRemote:
    """Simulates a RemoteDocument (status is plain string)."""

    document_id: str = "doc_002"
    session_id: str = "sess_002"
    filename: str = "report.pdf"
    file_type: str = "pdf"
    file_size: int = 2048
    status: str = "ready"
    chunks_count: int = 3
    chunk_count: int = 3
    upload_date: str = "2025-06-01"
    error_message: Optional[str] = None


# =========================================================================
# Contract: DocumentStatus resolution
# =========================================================================


class TestDocumentStatusResolution:
    """DocumentStatus must be constructable from both enum and string status."""

    def test_status_from_enum(self):
        doc = _FakeDocLocal(status=_EnumStatus("ready"))
        status_str = doc.status.value if hasattr(doc.status, "value") else doc.status
        assert DocumentStatus(status_str) == DocumentStatus.READY

    def test_status_from_string(self):
        doc = _FakeDocRemote(status="ready")
        status_str = doc.status.value if hasattr(doc.status, "value") else doc.status
        assert DocumentStatus(status_str) == DocumentStatus.READY

    def test_resolve_status_helper(self):
        from lasagna.sdk.managers.document_manager import _resolve_status

        assert _resolve_status(_EnumStatus("ready")) == DocumentStatus.READY
        assert _resolve_status("ready") == DocumentStatus.READY
        assert _resolve_status(_EnumStatus("processing")) == DocumentStatus.PROCESSING


# =========================================================================
# Contract: RemoteUploadResult has chunks_count
# =========================================================================


class TestUploadResultContract:
    """Upload result must expose both chunk_count and chunks_count."""

    def test_remote_upload_result_chunks_count(self):
        from lasagna.sdk.remote_backend import RemoteUploadResult

        result = RemoteUploadResult(
            document_id="doc_001",
            filename="test.pdf",
            chunk_count=10,
        )
        assert result.chunk_count == 10
        assert result.chunks_count == 10


# =========================================================================
# Contract: RemoteSearchResult has .document
# =========================================================================


class TestSearchResultContract:
    """Search results must expose a .document object with standard fields."""

    def test_remote_search_result_has_document(self):
        from lasagna.sdk.remote_backend import (
            RemoteSearchResult,
            RemoteChunk,
            RemoteDocumentRef,
        )

        chunk = RemoteChunk(chunk_id="c1", content="test", document_id="d1")
        doc_ref = RemoteDocumentRef(document_id="d1", filename="test.pdf")
        result = RemoteSearchResult(
            chunk=chunk,
            score=0.9,
            document_id="d1",
            filename="test.pdf",
            document=doc_ref,
        )
        assert result.document is not None
        assert result.document.document_id == "d1"
        assert result.document.filename == "test.pdf"

    def test_parse_search_results_populates_document(self):
        from lasagna.sdk.remote_backend import RemoteRAGBackend

        data = {
            "results": [
                {
                    "chunk": {
                        "chunk_id": "c1",
                        "content": "hello",
                        "document_id": "d1",
                    },
                    "score": 0.85,
                    "document_id": "d1",
                    "filename": "report.pdf",
                },
            ],
        }
        results = RemoteRAGBackend._parse_search_results(data)
        assert len(results) == 1
        assert results[0].document is not None
        assert results[0].document.document_id == "d1"
        assert results[0].document.filename == "report.pdf"


# =========================================================================
# Contract: upload_document accepts document_id
# =========================================================================


class TestUploadDocumentIdContract:
    """upload_document() must accept an optional document_id kwarg."""

    def test_remote_upload_signature_accepts_document_id(self):
        """RemoteRAGBackend.upload_document accepts document_id."""
        from lasagna.sdk.remote_backend import RemoteRAGBackend
        import inspect

        sig = inspect.signature(RemoteRAGBackend.upload_document)
        assert "document_id" in sig.parameters

    def test_remote_async_upload_signature_accepts_document_id(self):
        """RemoteRAGBackend._async_upload_document accepts document_id."""
        from lasagna.sdk.remote_backend import RemoteRAGBackend
        import inspect

        sig = inspect.signature(RemoteRAGBackend._async_upload_document)
        assert "document_id" in sig.parameters


# =========================================================================
# Contract: Config bridge dirty-field tracking
# =========================================================================


class TestConfigBridge:
    """Config bridge must only forward non-default fields."""

    def test_default_config_produces_no_overrides(self):
        from lasagna.sdk.config import KnowledgeConfig
        from lasagna.sdk._config_bridge import (
            get_init_overrides,
            get_runtime_overrides,
        )

        config = KnowledgeConfig()
        assert get_init_overrides(config) == {}
        assert get_runtime_overrides(config) == {}

    def test_single_field_produces_single_override(self):
        from lasagna.sdk.config import KnowledgeConfig
        from lasagna.sdk._config_bridge import get_runtime_overrides

        config = KnowledgeConfig(enable_hyde=True)
        overrides = get_runtime_overrides(config)
        assert "enable_hyde" in overrides
        assert overrides["enable_hyde"] is True

    def test_init_time_field_not_in_runtime(self):
        from lasagna.sdk.config import KnowledgeConfig
        from lasagna.sdk._config_bridge import get_runtime_overrides

        config = KnowledgeConfig(database_backend="postgresql")
        overrides = get_runtime_overrides(config)
        assert "database_backend" not in overrides

    def test_special_mapping_hyde_strategy(self):
        from lasagna.sdk.config import KnowledgeConfig
        from lasagna.sdk._config_bridge import get_runtime_overrides

        config = KnowledgeConfig(hyde_strategy="full")
        overrides = get_runtime_overrides(config)
        assert "agentic_hyde_strategy" in overrides
        assert overrides["agentic_hyde_strategy"] == "full"


# =========================================================================
# Contract: SessionConfigProvider.set_config_overrides
# =========================================================================


class TestSessionProviderContract:
    """SessionConfigProvider ABC and InMemorySessionProvider must support set_config_overrides."""

    def test_abc_has_set_config_overrides(self):
        from lasagna.providers.session_provider import SessionConfigProvider

        assert hasattr(SessionConfigProvider, "set_config_overrides")

    def test_in_memory_set_config_overrides(self):
        from lasagna.providers.session_provider import InMemorySessionProvider

        provider = InMemorySessionProvider()
        provider.set_config_overrides("s1", {"enable_hyde": True, "top_k": 10})
        resolved = provider.get_resolved_config("s1")
        assert resolved["enable_hyde"] is True
        assert resolved["top_k"] == 10

    def test_in_memory_overrides_merge(self):
        from lasagna.providers.session_provider import InMemorySessionProvider

        provider = InMemorySessionProvider()
        provider.set_config_overrides("s1", {"a": 1})
        provider.set_config_overrides("s1", {"b": 2})
        resolved = provider.get_resolved_config("s1")
        assert resolved == {"a": 1, "b": 2}

    def test_in_memory_remove_config_overrides(self):
        from lasagna.providers.session_provider import InMemorySessionProvider

        provider = InMemorySessionProvider()
        provider.set_config_overrides("s1", {"a": 1, "b": 2, "c": 3})
        provider.remove_config_overrides("s1", ["a", "c"])
        resolved = provider.get_resolved_config("s1")
        assert resolved == {"b": 2}

    def test_abc_has_remove_config_overrides(self):
        from lasagna.providers.session_provider import SessionConfigProvider

        assert hasattr(SessionConfigProvider, "remove_config_overrides")

    def test_rrf_k_override_reaches_resolved_config(self):
        from lasagna.providers.session_provider import InMemorySessionProvider

        provider = InMemorySessionProvider()
        provider.set_config_overrides("s1", {"rrf_k": 80})
        resolved = provider.get_resolved_config("s1")
        assert resolved["rrf_k"] == 80

    def test_document_awareness_nested_override(self):
        from lasagna.providers.session_provider import InMemorySessionProvider

        provider = InMemorySessionProvider()
        provider.set_config_overrides(
            "s1", {"document_awareness": {"enabled": False, "first_tokens_count": 100}}
        )
        resolved = provider.get_resolved_config("s1")
        assert resolved["document_awareness"]["enabled"] is False
        assert resolved["document_awareness"]["first_tokens_count"] == 100


# =========================================================================
# Contract: REST flat→nested canonicalization
# =========================================================================


class TestCanonicalPatch:
    """_canonicalize_patch() translates flat doc-awareness keys to nested."""

    def test_flat_to_nested_translation(self):
        from lasagna.api.rest.routes.config import _canonicalize_patch

        result = _canonicalize_patch(
            {
                "document_awareness_enabled": False,
                "first_tokens_count": 100,
                "rrf_k": 80,
            }
        )
        assert result == {
            "document_awareness": {"enabled": False, "first_tokens_count": 100},
            "rrf_k": 80,
        }

    def test_no_doc_awareness_keys_passes_through(self):
        from lasagna.api.rest.routes.config import _canonicalize_patch

        result = _canonicalize_patch({"rrf_k": 80, "enable_hyde": True})
        assert result == {"rrf_k": 80, "enable_hyde": True}

    def test_empty_patch(self):
        from lasagna.api.rest.routes.config import _canonicalize_patch

        assert _canonicalize_patch({}) == {}


# =========================================================================
# Contract: SDK bridge maps new fields
# =========================================================================


class TestBridgeNewFields:
    """RUNTIME_FIELD_MAP includes rrf_k and coverage_scorer_mode."""

    def test_rrf_k_in_runtime_map(self):
        from lasagna.sdk._config_bridge import RUNTIME_FIELD_MAP

        assert "rrf_k" in RUNTIME_FIELD_MAP

    def test_coverage_scorer_mode_in_runtime_map(self):
        from lasagna.sdk._config_bridge import RUNTIME_FIELD_MAP

        assert "coverage_scorer_mode" in RUNTIME_FIELD_MAP


# =========================================================================
# Critical: session_provider write-through (dual-store bug prevention)
# =========================================================================


class TestSessionProviderWriteThrough:
    """PATCH /config must write to BOTH KnowledgeConfigManager AND session provider.

    Without the write-through, RAGConfig.with_session(sid).rrf_k reads from
    InMemorySessionProvider (empty) while the value only lives in
    KnowledgeConfigManager — the exact dual-store bug.
    """

    def test_session_provider_receives_patch_overrides(self):
        """Simulate the PATCH route: write to session provider, read via RAGConfig."""
        from lasagna.providers.session_provider import InMemorySessionProvider

        sp = InMemorySessionProvider()
        # Simulate what the fixed PATCH handler does:
        sp.set_config_overrides("test_session", {"rrf_k": 80})

        resolved = sp.get_resolved_config("test_session")
        assert resolved.get("rrf_k") == 80, (
            "session_provider.set_config_overrides must make values "
            "readable via get_resolved_config (dual-store prevention)"
        )

    def test_nested_document_awareness_via_session_provider(self):
        """Canonicalized nested doc-awareness must survive provider round-trip."""
        from lasagna.providers.session_provider import InMemorySessionProvider

        sp = InMemorySessionProvider()
        sp.set_config_overrides(
            "test_session",
            {
                "document_awareness": {"enabled": False, "enrichment_strategy": "llm_summary"},
            },
        )

        resolved = sp.get_resolved_config("test_session")
        da = resolved.get("document_awareness", {})
        assert da.get("enabled") is False
        assert da.get("enrichment_strategy") == "llm_summary"

    def test_sequential_nested_patches_preserve_siblings(self):
        """Two partial PATCH calls on different document_awareness keys must preserve both.

        Regression test for shallow dict.update overwriting nested siblings.
        Uses the atomic deep_merge_config_overrides (same path as route).
        """
        from lasagna.api.rest.routes.config import _canonicalize_patch
        from lasagna.providers.session_provider import InMemorySessionProvider

        sp = InMemorySessionProvider()

        # First PATCH: set enrichment_strategy
        patch1 = _canonicalize_patch({"enrichment_strategy": "llm_summary"})
        sp.deep_merge_config_overrides("s1", patch1)

        # Second PATCH: set first_tokens_count (must NOT lose enrichment_strategy)
        patch2 = _canonicalize_patch({"first_tokens_count": 100})
        sp.deep_merge_config_overrides("s1", patch2)

        resolved = sp.get_resolved_config("s1")
        da = resolved.get("document_awareness", {})
        assert da.get("enrichment_strategy") == "llm_summary", (
            "enrichment_strategy lost after second PATCH (shallow merge bug)"
        )
        assert da.get("first_tokens_count") == 100

    def test_deep_merge_with_scalar_and_nested_mix(self):
        """Mixed scalar + nested patches don't interfere."""
        from lasagna.providers.session_provider import InMemorySessionProvider

        sp = InMemorySessionProvider()
        sp.deep_merge_config_overrides("s1", {"rrf_k": 80, "document_awareness": {"enabled": True}})
        sp.deep_merge_config_overrides(
            "s1", {"rrf_k": 100, "document_awareness": {"enrichment_strategy": "hybrid"}}
        )

        resolved = sp.get_resolved_config("s1")
        assert resolved["rrf_k"] == 100
        assert resolved["document_awareness"]["enabled"] is True  # preserved
        assert resolved["document_awareness"]["enrichment_strategy"] == "hybrid"  # added

    def test_delegates_to_manager_flag(self):
        """PersistentSessionProvider must have _delegates_to_manager=True."""
        from lasagna.providers.session_provider import (
            InMemorySessionProvider,
            PersistentSessionProvider,
        )

        assert getattr(PersistentSessionProvider, "_delegates_to_manager", False) is True
        assert getattr(InMemorySessionProvider, "_delegates_to_manager", False) is False


# =========================================================================
# Regression: P0 — Override revert to default clears session state
# =========================================================================


class TestOverrideRevertRegression:
    """Setting a field back to default must remove the session override."""

    def test_revert_to_default_clears_override(self):
        """Simulate: enable_hyde=True, then enable_hyde=False (default)."""
        from lasagna.sdk.config import KnowledgeConfig
        from lasagna.sdk._config_bridge import get_runtime_overrides
        from lasagna.providers.session_provider import InMemorySessionProvider

        provider = InMemorySessionProvider()
        session_id = "test_revert"

        # Step 1: Apply config with enable_hyde=True (non-default)
        config1 = KnowledgeConfig(enable_hyde=True)
        overrides1 = get_runtime_overrides(config1)
        assert "enable_hyde" in overrides1
        provider.set_config_overrides(session_id, overrides1)
        bridged_keys = set(overrides1.keys())

        # Step 2: Revert to default (enable_hyde=False is the default)
        config2 = KnowledgeConfig()
        overrides2 = get_runtime_overrides(config2)
        assert "enable_hyde" not in overrides2

        # Remove stale keys (simulating _apply_runtime_config logic)
        stale = bridged_keys - set(overrides2.keys())
        provider.remove_config_overrides(session_id, stale)
        if overrides2:
            provider.set_config_overrides(session_id, overrides2)

        # Verify the override was removed
        resolved = provider.get_resolved_config(session_id)
        assert "enable_hyde" not in resolved


# =========================================================================
# Regression: P0 — Init-time override leakage across clients
# =========================================================================


class TestInitTimeLeakageRegression:
    """Init-time overrides from one client must not leak to the next."""

    def test_apply_init_overrides_cleans_stale_keys(self):
        """Simulate: client A sets database_backend=postgresql, client B is default."""
        from lasagna.sdk.config import KnowledgeConfig
        from lasagna.sdk._config_bridge import (
            apply_init_overrides,
        )
        import lasagna.sdk._config_bridge as bridge_mod

        env_overrides = {}

        # Client A: sets database_backend=postgresql
        config_a = KnowledgeConfig(database_backend="postgresql")
        apply_init_overrides(env_overrides, config_a)
        assert env_overrides.get("database_backend") == "postgresql"
        assert "database_backend" in bridge_mod._applied_init_keys

        # Client B: default config — should clean up client A's override
        config_b = KnowledgeConfig()
        apply_init_overrides(env_overrides, config_b)
        assert "database_backend" not in env_overrides
        assert bridge_mod._applied_init_keys == set()


# =========================================================================
# Regression: P1 — Missing bridge fields
# =========================================================================


class TestMissingBridgeFields:
    """All backend-consumed SDK fields must be mapped in the bridge."""

    def test_agentic_fields_mapped(self):
        from lasagna.sdk.config import KnowledgeConfig
        from lasagna.sdk._config_bridge import get_runtime_overrides

        config = KnowledgeConfig(use_agentic_rag=False, max_agent_iterations=5)
        overrides = get_runtime_overrides(config)
        assert overrides["use_agentic_rag"] is False
        assert overrides["max_agent_iterations"] == 5

    def test_document_awareness_fields_nested(self):
        from lasagna.sdk.config import KnowledgeConfig
        from lasagna.sdk._config_bridge import get_runtime_overrides

        config = KnowledgeConfig(
            document_awareness_enabled=False,
            first_tokens_count=200,
            max_documents_in_context=5,
        )
        overrides = get_runtime_overrides(config)
        assert "document_awareness" in overrides
        nested = overrides["document_awareness"]
        assert nested["enabled"] is False
        assert nested["first_tokens_count"] == 200
        assert nested["max_documents_in_context"] == 5


# =========================================================================
# Regression: P2 — Document.chunks_count alias
# =========================================================================


class TestDocumentChunksCountAlias:
    """SDK Document type must expose chunks_count for backend compatibility."""

    def test_chunks_count_alias(self):
        from lasagna.sdk.types import Document, DocumentStatus

        doc = Document(
            document_id="d1",
            filename="test.pdf",
            session_id="s1",
            status=DocumentStatus.READY,
            chunk_count=42,
        )
        assert doc.chunk_count == 42
        assert doc.chunks_count == 42
