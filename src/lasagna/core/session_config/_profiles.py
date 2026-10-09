"""Profile management mixin for session configuration.

Handles profile CRUD, built-in preset creation (speed/standard/balanced/max_quality),
profile defaults loading from TOML, and profile listing/statistics.
"""

import copy
import os
import sys
from typing import Any, Dict, List, Optional

from loguru import logger

if sys.version_info >= (3, 11):
    import tomllib as toml
else:
    try:
        import tomli as toml  # type: ignore
    except ImportError:
        try:
            import toml  # type: ignore
        except ImportError:
            toml = None  # type: ignore

from ._data_models import (
    KNOWLEDGE_MODULE_DIR,
    MAX_GLOBAL_PROFILES,
    SESSION_TTL_SECONDS,
    MAX_CACHED_SESSIONS,
    RAGProfile,
    _deep_merge,
)


# Module-level cache for parsed TOML defaults.  In PostgreSQL mode a new
# KnowledgeConfigManager is created per request; without this cache every
# request would re-read and re-parse settings.lasagna.toml from disk.
_cached_default_profiles: Optional[List[RAGProfile]] = None


class ProfileMixin:
    """Mixin for profile management.

    Expects the host class to provide:
    - self._lock: RLock
    - self._default_profiles: List[RAGProfile]
    - self._global_custom_profiles: List[RAGProfile]
    - self._global_profile_names: set
    - self._session_states: Dict[str, KnowledgeConfigState]
    - self._persist_global_profiles(): method
    - self._persist_session_states(): method
    - self.get_session_state(): method
    """

    def _load_defaults(self) -> None:
        """Load default profiles from settings.lasagna.toml.

        Results are cached at module level so request-scoped instances
        (PostgreSQL mode) skip disk I/O after the first load.
        """
        global _cached_default_profiles
        if _cached_default_profiles is not None:
            self._default_profiles = copy.deepcopy(_cached_default_profiles)
            return

        knowledge_toml_path = KNOWLEDGE_MODULE_DIR / "settings.lasagna.toml"

        if not knowledge_toml_path.exists():
            logger.warning("settings.lasagna.toml not found in knowledge module")
            self._default_profiles = [self._create_minimal_default_profile()]
            _cached_default_profiles = copy.deepcopy(self._default_profiles)
            return

        knowledge_toml_path = str(knowledge_toml_path)

        try:
            if toml is None:
                logger.warning("No TOML parser available, using minimal defaults")
                self._default_profiles = [self._create_minimal_default_profile()]
                _cached_default_profiles = copy.deepcopy(self._default_profiles)
                return

            # Use binary mode for tomllib/tomli, text mode for toml
            if getattr(toml, "__name__", "") in {"tomllib", "tomli"}:
                with open(knowledge_toml_path, "rb") as f:
                    toml_data = toml.load(f)
            else:
                with open(knowledge_toml_path, "r") as f:
                    toml_data = toml.load(f)

            # Extract RAG configuration
            rag_config = toml_data.get("rag", {})

            # Convert Dynaconf-style nested objects to plain dicts
            rag_config = self._to_plain_dict(rag_config)
            if not isinstance(rag_config, dict):
                rag_config = {}
            if not rag_config:
                logger.warning("settings.lasagna.toml missing [rag] config; using minimal defaults")

            base_defaults = copy.deepcopy(self._create_minimal_default_profile().config)
            _deep_merge(base_defaults, rag_config)
            rag_config = base_defaults

            # Create default profile from TOML
            default_profile = RAGProfile(
                name="default",
                description="Default configuration from settings.lasagna.toml",
                config=rag_config,
            )

            # Create preset profiles based on common configurations
            speed_profile = RAGProfile(
                name="speed",
                description="Fast, pure vector search - no reranking/MMR/HyDE",
                config=self._create_speed_config(rag_config),
            )

            standard_profile = RAGProfile(
                name="standard",
                description="SOTA features with BGE-M3 embedding model (recommended starter)",
                config=self._create_standard_config(rag_config),
            )

            balanced_profile = RAGProfile(
                name="balanced",
                description="SOTA features with BGE-M3, hierarchical chunking, structured graph",
                config=self._create_balanced_config(rag_config),
            )

            max_quality_profile = RAGProfile(
                name="max_quality",
                description="All features enabled - HyDE, query expansion, max recall",
                config=self._create_max_quality_config(rag_config),
            )

            self._default_profiles = [
                default_profile,
                speed_profile,
                standard_profile,
                balanced_profile,
                max_quality_profile,
            ]

            logger.info(
                f"Loaded {len(self._default_profiles)} RAG profiles from settings.lasagna.toml"
            )

        except Exception as e:
            logger.error(f"Failed to load settings.lasagna.toml: {e}")
            self._default_profiles = [self._create_minimal_default_profile()]

        _cached_default_profiles = copy.deepcopy(self._default_profiles)

    def _ensure_default_profile(self, profiles: List[RAGProfile]) -> List[RAGProfile]:
        """Ensure the default profile exists and matches settings.lasagna.toml."""
        default_profile = None
        for profile in self._default_profiles:
            if profile.name == "default":
                default_profile = profile
                break

        if default_profile is None:
            return profiles

        updated_profiles = []
        found_default = False
        for profile in profiles:
            if profile.name == "default":
                updated_profiles.append(copy.deepcopy(default_profile))
                found_default = True
            else:
                updated_profiles.append(profile)

        if not found_default:
            updated_profiles.insert(0, copy.deepcopy(default_profile))

        return updated_profiles

    def _is_protected_profile_name(self, profile_name: str) -> bool:
        """Return True when the profile is one of the built-in defaults."""
        return any(profile.name == profile_name for profile in self._default_profiles)

    def _to_plain_dict(self, obj: Any) -> Any:
        """Convert nested objects to plain dictionaries."""
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._to_plain_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_plain_dict(item) for item in obj]
        else:
            return obj

    def _create_minimal_default_profile(self) -> RAGProfile:
        """Create a minimal default profile when TOML is unavailable."""
        return RAGProfile(
            name="default",
            description="Default RAG configuration",
            config={
                # Vector backend configuration (BYOVDB)
                "vector_backend": os.getenv("RAG_VECTOR_BACKEND", "milvus"),
                "vector_backend_config": {},
                # Feature toggles
                "enable_hybrid_search": True,
                "enable_reranking": True,
                "enable_mmr": True,
                "enable_hyde": False,
                "agentic_hyde_strategy": "adaptive",
                "enable_ocr": False,
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dim": 384,
                "chunking_strategy": "semantic",
                "chunk_size": 800,
                "chunk_overlap": 0,
                "default_top_k": 20,
                "rerank_top_k": 5,
                "similarity_threshold": 0.3,
                "hybrid_alpha": 0.7,
                "mmr_lambda": 0.5,
                "mmr_score_normalization": "sigmoid",
                "use_agentic_rag": True,
                "max_agent_iterations": 3,
                "chunk_refinement_method": "auto",
                # SOTA retrieval defaults
                "enable_context_reordering": True,
                "enable_contextual_retrieval": False,
                "enable_rag_fusion": False,
                "rag_fusion_n_queries": 3,
                "enable_crag": False,
                "crag_web_search_fallback": False,
                "embedding_backend": "auto",
                "docling_enabled": False,
                "enable_query_routing": True,
                "routing_strategy": "rule_based",
                "routing_default_tier": "standard",
                # Adaptive chunk retrieval (default profile)
                "chunk_cite_retrieval": {
                    "strategy": "adaptive",
                    "fixed_size": 5,
                    "max_chunks": 20,
                    "min_chunks": 2,
                    "threshold_score": 0.5,
                    "score_gap_buffer": 2,
                    "min_gap_ratio": 0.15,
                },
            },
        )

    # -- Profile specs (only differences from base) ---------------------------
    _PROFILE_SPECS: Dict[str, Dict[str, Any]] = {
        "speed": {
            "enable_hybrid_search": False,
            "enable_reranking": False,
            "enable_hyde": False,
            "enable_mmr": False,
            "enable_query_expansion": False,
            "enable_query_decomposition": False,
            "use_agentic_rag": False,
            "use_structured_graph": False,
            "agentic_pipeline": "legacy_react",
            "docling_enabled": False,
            "enable_context_reordering": False,
            "enable_contextual_retrieval": False,
            "enable_crag": False,
            "enable_rag_fusion": False,
            "enable_hallucination_check": False,
            "enable_answer_quality_check": False,
            "enable_query_classification": False,
            "enable_query_routing": False,
            "routing_default_tier": "fast",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "default_top_k": 20,
            "rerank_top_k": 5,
            "similarity_threshold": 0.3,
            "enable_hierarchical": False,
            "parent_chunk_size": 2000,
            "child_chunk_size": 500,
            "auto_merge_threshold": 0.5,
            "chunk_cite_retrieval": {
                "strategy": "fixed",
                "fixed_size": 5,
                "max_chunks": 10,
                "min_chunks": 2,
                "threshold_score": 0.5,
                "score_gap_buffer": 2,
                "min_gap_ratio": 0.15,
            },
        },
        "standard": {
            "enable_hybrid_search": True,
            "enable_reranking": True,
            "enable_mmr": True,
            "use_agentic_rag": True,
            "agentic_pipeline": "default",
            "enable_hyde": False,
            "enable_query_expansion": False,
            "enable_query_decomposition": False,
            "docling_enabled": True,
            "enable_context_reordering": True,
            "enable_rag_fusion": False,
            "enable_crag": False,
            "enable_query_routing": True,
            "routing_default_tier": "standard",
            "reranker_model": "mixedbread-ai/mxbai-rerank-large-v1",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
            "chunking_strategy": "semantic",
            "chunk_size": 800,
            "chunk_overlap": 0,
            "default_top_k": 20,
            "rerank_top_k": 5,
            "mmr_lambda": 0.5,
            "mmr_score_normalization": "sigmoid",
            "hybrid_alpha": 0.7,
            "similarity_threshold": 0.3,
            "max_agent_iterations": 3,
            "enable_hierarchical": False,
            "parent_chunk_size": 2000,
            "child_chunk_size": 500,
            "auto_merge_threshold": 0.5,
            "chunk_refinement_method": "auto",
            "chunk_cite_retrieval": {
                "strategy": "adaptive",
                "fixed_size": 5,
                "max_chunks": 15,
                "min_chunks": 2,
                "threshold_score": 0.5,
                "score_gap_buffer": 2,
                "min_gap_ratio": 0.15,
            },
        },
        "balanced": {
            "enable_hybrid_search": True,
            "enable_reranking": True,
            "enable_mmr": True,
            "use_agentic_rag": True,
            "agentic_pipeline": "default",
            "use_structured_graph": True,
            "enable_hallucination_check": True,
            "enable_hyde": False,
            "enable_query_expansion": False,
            "enable_query_decomposition": False,
            "docling_enabled": True,
            "enable_context_reordering": True,
            "enable_rag_fusion": False,
            "enable_crag": False,
            "enable_query_routing": True,
            "routing_strategy": "adaptive",
            "routing_confidence_threshold": 0.5,
            "reranker_model": "mixedbread-ai/mxbai-rerank-large-v1",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
            "chunking_strategy": "semantic",
            "chunk_size": 800,
            "chunk_overlap": 0,
            "enable_hierarchical": True,
            "parent_chunk_size": 2000,
            "child_chunk_size": 500,
            "auto_merge_threshold": 0.5,
            "default_top_k": 20,
            "rerank_top_k": 5,
            "mmr_lambda": 0.5,
            "mmr_score_normalization": "sigmoid",
            "hybrid_alpha": 0.7,
            "similarity_threshold": 0.3,
            "max_agent_iterations": 3,
            "chunk_refinement_method": "auto",
            "chunk_cite_retrieval": {
                "strategy": "adaptive",
                "fixed_size": 5,
                "max_chunks": 20,
                "min_chunks": 2,
                "threshold_score": 0.5,
                "score_gap_buffer": 2,
                "min_gap_ratio": 0.15,
            },
        },
        "max_quality": {
            "enable_hybrid_search": True,
            "enable_reranking": True,
            "enable_mmr": True,
            "enable_hyde": True,
            "enable_query_expansion": True,
            "enable_query_decomposition": False,
            "use_agentic_rag": True,
            "agentic_pipeline": "default",
            "use_structured_graph": True,
            "enable_hallucination_check": True,
            "enable_answer_quality_check": True,
            "docling_enabled": True,
            "enable_context_reordering": True,
            "enable_rag_fusion": True,
            "enable_crag": True,
            "enable_contextual_retrieval": True,
            "enable_query_routing": False,
            "routing_default_tier": "comprehensive",
            "routing_confidence_threshold": 0.3,
            "reranker_model": "mixedbread-ai/mxbai-rerank-large-v1",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
            "chunking_strategy": "semantic",
            "chunk_size": 800,
            "chunk_overlap": 0,
            "enable_hierarchical": True,
            "parent_chunk_size": 2000,
            "child_chunk_size": 500,
            "auto_merge_threshold": 0.4,
            "default_top_k": 20,
            "rerank_top_k": 5,
            "mmr_lambda": 0.5,
            "mmr_score_normalization": "sigmoid",
            "hybrid_alpha": 0.7,
            "similarity_threshold": 0.2,
            "max_agent_iterations": 5,
            "agentic_hyde_strategy": "adaptive",
            "chunk_refinement_method": "auto",
            "chunk_cite_retrieval": {
                "strategy": "adaptive",
                "fixed_size": 5,
                "max_chunks": 30,
                "min_chunks": 2,
                "threshold_score": 0.5,
                "score_gap_buffer": 2,
                "min_gap_ratio": 0.15,
            },
        },
    }

    def _create_profile_config(
        self,
        profile_name: str,
        base_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a profile configuration by applying a spec to the base config.

        Uses _deep_merge so nested dicts (e.g. chunk_cite_retrieval) are merged
        key-by-key rather than replaced wholesale.
        """
        config = copy.deepcopy(base_config)
        _deep_merge(config, self._PROFILE_SPECS[profile_name])
        return config

    # Backward-compatible aliases
    def _create_speed_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        return self._create_profile_config("speed", base_config)

    def _create_standard_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        return self._create_profile_config("standard", base_config)

    def _create_balanced_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        return self._create_profile_config("balanced", base_config)

    def _create_max_quality_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        return self._create_profile_config("max_quality", base_config)

    def list_profiles(
        self,
        include_builtin: bool = True,
        session_id: Optional[str] = None,
        user_id: str = "__anonymous__",
    ) -> List[Dict[str, Any]]:
        """List available profiles.

        Returns built-in profiles (visible to all) plus the user's custom profiles.
        When ``user_id`` is ``"__anonymous__"``, all custom profiles are shown
        (backward compatibility for single-user deployments).
        """
        builtin_names = {p.name for p in self._default_profiles}
        result = []

        if include_builtin:
            for p in self._default_profiles:
                result.append(
                    {
                        "name": p.name,
                        "description": p.description,
                        "builtin": True,
                    }
                )

        for p in self._global_custom_profiles:
            if p.name not in builtin_names:
                # In multi-user mode, only show profiles owned by this user
                # or by __anonymous__ (shared). In single-user mode (__anonymous__),
                # show all profiles.
                if user_id != "__anonymous__" and p.user_id not in (user_id, "__anonymous__"):
                    continue
                result.append(
                    {
                        "name": p.name,
                        "description": p.description,
                        "builtin": False,
                        "user_id": p.user_id,
                    }
                )

        return result

    def get_active_profile(
        self,
        session_id: Optional[str] = None,
    ) -> str:
        """Get the name of the active profile for a session."""
        state = self.get_session_state(session_id)
        return state.active_profile

    def set_active_profile(
        self,
        profile_name: str,
        config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Set the active profile and optionally update its config."""
        if config:
            self._validate_or_raise(config, session_id)

        state = self.get_session_state(session_id)

        with self._lock:
            profile_found = False
            for profile in state.profiles:
                if profile.name == profile_name:
                    profile_found = True
                    if config:
                        state.session_overrides = config
                    break

            if not profile_found:
                if config:
                    new_profile = RAGProfile(
                        name=profile_name,
                        description=f"Custom profile: {profile_name}",
                        config=config,
                    )
                    state.profiles.append(new_profile)

            state.active_profile = profile_name
            self._persist_session_states()

        # Without this the profile switch is recorded and never applied:
        # search resolves through the session provider, not this manager.
        self._publish_to_search(session_id)
        return state.to_dict()

    def save_profile(
        self,
        profile_name: str,
        config: Dict[str, Any],
        session_id: Optional[str] = None,
        set_active: bool = True,
        user_id: str = "__anonymous__",
    ) -> Dict[str, Any]:
        """Save or update a named profile.

        The ``user_id`` is stored on the profile for ownership tracking.
        In multi-user mode, users can only update their own profiles.
        """
        if profile_name == "default":
            raise ValueError("The default profile cannot be overwritten.")

        if not isinstance(config, dict):
            raise ValueError("Profile config must be an object.")

        if not profile_name or len(profile_name) > 100:
            raise ValueError("Profile name must be 1-100 characters.")

        if config:
            self._validate_or_raise(config)

        state = self.get_session_state(session_id)

        with self._lock:
            is_new = profile_name not in self._global_profile_names
            if is_new and len(self._global_custom_profiles) >= MAX_GLOBAL_PROFILES:
                raise ValueError(
                    f"Maximum number of profiles ({MAX_GLOBAL_PROFILES}) reached. "
                    "Delete unused profiles before creating new ones."
                )

            # Ownership check: prevent overwriting profiles owned by other users.
            # Profiles owned by "__anonymous__" are shared/legacy and editable by anyone.
            if not is_new:
                existing_global = next(
                    (p for p in self._global_custom_profiles if p.name == profile_name),
                    None,
                )
                if (
                    existing_global
                    and existing_global.user_id != "__anonymous__"
                    and existing_global.user_id != user_id
                ):
                    raise ValueError(f"Profile '{profile_name}' is owned by another user.")

            new_profile = RAGProfile(
                name=profile_name,
                description=f"Saved profile: {profile_name}",
                config=config,
                user_id=user_id,
            )

            existing = None
            for profile in state.profiles:
                if profile.name == profile_name:
                    existing = profile
                    break

            if existing:
                existing.config = config
                existing.description = new_profile.description
            else:
                state.profiles.append(copy.deepcopy(new_profile))

            if profile_name in self._global_profile_names:
                for i, gp in enumerate(self._global_custom_profiles):
                    if gp.name == profile_name:
                        self._global_custom_profiles[i] = copy.deepcopy(new_profile)
                        break
            else:
                self._global_custom_profiles.append(copy.deepcopy(new_profile))
                self._global_profile_names.add(profile_name)

            if set_active:
                state.active_profile = profile_name
                state.session_overrides = {}

            state.profiles = self._ensure_default_profile(state.profiles)

            self._persist_global_profiles()
            if set_active:
                self._persist_session_states()

            logger.info(
                f"Saved profile '{profile_name}' globally "
                f"({len(self._global_custom_profiles)} total custom profiles)"
            )

        # Saving as the active profile (the configurator's "Save As") must
        # change what search runs, exactly as set_active_profile does; it used
        # to report "Saved!" and leave retrieval as it was.
        if set_active or state.active_profile == profile_name:
            self._publish_to_search(session_id)
        return state.to_dict()

    def rename_profile(
        self,
        old_name: str,
        new_name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rename an existing profile.

        ``user_id`` is checked against the global profile's owner, as
        `save_profile` and `delete_profile` both do. This method did not, while
        mutating the same global store they guard -- so one caller could rename
        another's profile out from under them, and the REST route did not even
        resolve a ``user_id`` to pass.
        """
        if self._is_protected_profile_name(old_name) or self._is_protected_profile_name(new_name):
            raise ValueError("Built-in profiles cannot be renamed.")
        if not new_name or len(new_name) > 100:
            raise ValueError("New profile name must be 1-100 characters.")

        state = self.get_session_state(session_id)

        with self._lock:
            if new_name in self._global_profile_names or any(
                p.name == new_name for p in state.profiles
            ):
                raise ValueError(f"Profile '{new_name}' already exists.")

            profile = next((p for p in state.profiles if p.name == old_name), None)
            if not profile:
                raise ValueError(f"Profile '{old_name}' not found.")

            profile.name = new_name
            if profile.description:
                profile.description = profile.description.replace(old_name, new_name)

            if state.active_profile == old_name:
                state.active_profile = new_name

            for gp in self._global_custom_profiles:
                if gp.name == old_name:
                    # Same ownership rule as save_profile and delete_profile.
                    if user_id and gp.user_id != "__anonymous__" and gp.user_id != user_id:
                        raise ValueError(f"Profile '{old_name}' is owned by another user.")
                    gp.name = new_name
                    if gp.description:
                        gp.description = gp.description.replace(old_name, new_name)
                    self._global_profile_names.discard(old_name)
                    self._global_profile_names.add(new_name)
                    break

            state.profiles = self._ensure_default_profile(state.profiles)
            self._persist_global_profiles()
            self._persist_session_states()

            logger.info(f"Renamed profile '{old_name}' to '{new_name}' globally")

        return state.to_dict()

    def delete_profile(
        self,
        profile_name: str,
        session_id: Optional[str] = None,
        user_id: str = "__anonymous__",
    ) -> Dict[str, Any]:
        """Delete a named profile (default is protected).

        In multi-user mode, verifies ownership before deleting.
        """
        if self._is_protected_profile_name(profile_name):
            raise ValueError("Built-in profiles cannot be deleted.")

        state = self.get_session_state(session_id)

        with self._lock:
            # Ownership check: prevent deleting profiles owned by other users.
            # Profiles owned by "__anonymous__" are shared/legacy and deletable by anyone.
            existing = next(
                (p for p in self._global_custom_profiles if p.name == profile_name),
                None,
            )
            if existing and existing.user_id != "__anonymous__" and existing.user_id != user_id:
                raise ValueError(f"Profile '{profile_name}' is owned by another user.")
            state.profiles = [p for p in state.profiles if p.name != profile_name]
            if state.active_profile == profile_name:
                state.active_profile = "default"
                state.session_overrides = {}

            self._global_custom_profiles = [
                p for p in self._global_custom_profiles if p.name != profile_name
            ]
            self._global_profile_names.discard(profile_name)

            state.profiles = self._ensure_default_profile(state.profiles)
            self._persist_global_profiles()
            self._persist_session_states()

            logger.info(f"Deleted profile '{profile_name}' globally")

        # The session may have fallen back to default; search must follow.
        self._publish_to_search(session_id)
        return state.to_dict()

    def update_profile_description(
        self,
        profile_name: str,
        description: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a profile description (default is protected)."""
        if profile_name == "default":
            raise ValueError("The default profile description cannot be edited.")

        state = self.get_session_state(session_id)

        with self._lock:
            profile = next((p for p in state.profiles if p.name == profile_name), None)
            if not profile:
                raise ValueError(f"Profile '{profile_name}' not found.")

            profile.description = description

            for gp in self._global_custom_profiles:
                if gp.name == profile_name:
                    gp.description = description
                    break

            state.profiles = self._ensure_default_profile(state.profiles)
            self._persist_global_profiles()
            self._persist_session_states()

        return state.to_dict()

    def reload_defaults(self) -> None:
        """Reload default profiles from settings.lasagna.toml."""
        with self._lock:
            self._load_defaults()
        logger.info("Reloaded default RAG profiles from settings.lasagna.toml")

    def get_profile_stats(self) -> Dict[str, Any]:
        """Get statistics about profiles and sessions."""
        with self._lock:
            return {
                "global_custom_profiles_count": len(self._global_custom_profiles),
                "global_custom_profiles_limit": MAX_GLOBAL_PROFILES,
                "default_profiles_count": len(self._default_profiles),
                "active_sessions_count": len(self._session_states),
                "max_cached_sessions": MAX_CACHED_SESSIONS,
                "session_ttl_hours": SESSION_TTL_SECONDS / 3600,
            }

    def list_global_profiles(self, user_id: str = "__anonymous__") -> List[Dict[str, Any]]:
        """List global custom profiles visible to the given user."""
        with self._lock:
            result = []
            for p in self._global_custom_profiles:
                if user_id != "__anonymous__" and p.user_id not in (user_id, "__anonymous__"):
                    continue
                result.append(
                    {
                        "name": p.name,
                        "description": p.description,
                        "user_id": p.user_id,
                    }
                )
            return result


# ---------------------------------------------------------------------------
# Standalone helpers (used by experiment runner for ingestion signatures)
# ---------------------------------------------------------------------------


def resolve_ingestion_defaults(profile_name: str) -> Dict[str, Any]:
    """Return the ingestion-time config values for a given profile.

    Reads ``_PROFILE_SPECS`` directly and returns only the keys that are
    relevant to ingestion (embedding model, chunking, hierarchical, etc.).
    This is used by the experiment runner to compute ingestion signatures
    without instantiating the full config manager.

    Args:
        profile_name: One of "speed", "standard", "balanced", "max_quality".

    Returns:
        Dict of profile config values (unfiltered — caller should intersect
        with ``_INGESTION_TIME_KEYS``).
    """
    specs = ProfileMixin._PROFILE_SPECS
    if profile_name not in specs:
        return {}
    return dict(specs[profile_name])
