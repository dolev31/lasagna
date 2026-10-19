"""Profile management mixin for session configuration.

Handles profile CRUD, built-in preset creation (speed/standard/balanced/max_quality),
profile defaults loading from TOML, and profile listing/statistics.
"""

import copy
import os
import sys
import time
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
    ANONYMOUS,
    KNOWLEDGE_MODULE_DIR,
    MAX_GLOBAL_PROFILES,
    MAX_CACHED_SESSIONS,
    RAGProfile,
    UnknownProfileError,
    _check_profile_size,
    _deep_merge,
    _resolve,
    SettingsNotSaved,
    kept_in_process,
    partly_saved,
    session_idle_seconds,
)


#: How stale this process's list of saved profiles may be before it is read again.
_PROFILES_RECHECK_SECONDS = 1.0

#: How many profile changes this process has saved, over all of its managers.
_profile_writes_here = 0

# Module-level cache for parsed TOML defaults.  In PostgreSQL mode a new
# KnowledgeConfigManager is created per request; without this cache every
# request would re-read and re-parse settings.lasagna.toml from disk.
_cached_default_profiles: Optional[List[RAGProfile]] = None


#: What each built-in profile does, as every surface describes it: the REST
#: profile list and the SDK's ``list_profiles`` repeated their own copies,
#: which used marketing words and named embedding models the profiles do not use.
BUILTIN_PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "default": "Default configuration from settings.lasagna.toml",
    "speed": (
        "Vector search only, with all-MiniLM-L6-v2: no reranking, MMR or HyDE "
        "(start here: a 92 MB download, no LLM)"
    ),
    "standard": (
        "Hybrid search, reranking and agentic retrieval with BGE-M3 (for quality: "
        "a 3.2 GB download, and an LLM for complex queries)"
    ),
    "balanced": "Standard plus hierarchical chunking and the structured agentic graph, with BGE-M3",
    "max_quality": "Every retrieval feature on: HyDE, query expansion, RAG fusion and CRAG",
}


class ProfileMixin:
    """Mixin for profile management.

    Expects the host class to provide:
    - self._lock: RLock
    - self._default_profiles: List[RAGProfile]
    - self._global_custom_profiles: List[RAGProfile]
    - self._global_profile_names: set
    - self._session_states: Dict[str, KnowledgeConfigState]
    - self._read_saved_profiles(), self._save_global_profile(),
      self._delete_global_profile(): methods (from PersistenceMixin)
    - self._persist_session_states(): method
    - self.get_session_state(): method
    """

    def _refresh_saved_profiles(self, *, force: bool = False) -> None:
        """Read the saved profiles again, when another manager may have changed them.

        Each process read them once, when it started: a profile another
        process saved later was unknown here (a session on it came back as
        default), and one it changed kept its old settings here. Read at most
        once a second, unless ``force`` or this process saved a change since;
        the sessions' copies of a saved profile follow its stored settings.
        """
        now = time.monotonic()
        if (
            not force
            and now - getattr(self, "_profiles_checked_at", float("-inf"))
            < _PROFILES_RECHECK_SECONDS
            and getattr(self, "_profile_writes_seen", 0) == _profile_writes_here
        ):
            return
        self._profiles_checked_at = now
        self._profile_writes_seen = _profile_writes_here
        reader = getattr(self, "_read_saved_profiles", None)
        profiles = reader() if reader is not None else None
        if profiles is None:
            return
        with self._lock:
            self._global_custom_profiles = profiles
            self._global_profile_names = {p.name for p in profiles}
            # By owner and name: by name alone, a session on the shared
            # profile of a name took another owner's of that name.
            saved = {
                (p.user_id or ANONYMOUS, p.name): p
                for p in profiles
                if not self._is_protected_profile_name(p.name)
            }
            for state in self._session_states.values():
                for i, profile in enumerate(state.profiles):
                    current = saved.get((profile.user_id or ANONYMOUS, profile.name))
                    if current is not None and (
                        current.config != profile.config
                        or current.description != profile.description
                    ):
                        state.profiles[i] = copy.deepcopy(current)

    def _mark_saved_profiles_read(self) -> None:
        """Note that the saved profiles were just read (the manager's first load)."""
        self._profiles_checked_at = time.monotonic()
        self._profile_writes_seen = _profile_writes_here

    def _saved_profiles_changed(self) -> None:
        """Note that this manager saved a profile change: the process's others read it at once."""
        global _profile_writes_here
        _profile_writes_here += 1
        self._profile_writes_seen = _profile_writes_here

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
                description=BUILTIN_PROFILE_DESCRIPTIONS["default"],
                config=rag_config,
            )

            # Create preset profiles based on common configurations
            speed_profile = RAGProfile(
                name="speed",
                description=BUILTIN_PROFILE_DESCRIPTIONS["speed"],
                config=self._create_speed_config(rag_config),
            )

            standard_profile = RAGProfile(
                name="standard",
                description=BUILTIN_PROFILE_DESCRIPTIONS["standard"],
                config=self._create_standard_config(rag_config),
            )

            balanced_profile = RAGProfile(
                name="balanced",
                description=BUILTIN_PROFILE_DESCRIPTIONS["balanced"],
                config=self._create_balanced_config(rag_config),
            )

            max_quality_profile = RAGProfile(
                name="max_quality",
                description=BUILTIN_PROFILE_DESCRIPTIONS["max_quality"],
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
                "chunking_strategy": "hierarchical",  # enable_hierarchical is on by default
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
                # Retrieval defaults
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
            # Used only when an arm or a user turns reranking on: without it the
            # global 881 MB default loaded, against what this profile is for.
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
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
            # What runs: headers and tables are kept (preserve_headers and
            # preserve_tables, from settings.lasagna.toml), so sections are
            # split structurally and each is chunked semantically. It said
            # "semantic", and every upload warned that it was overridden.
            "chunking_strategy": "structure_aware",
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
            "chunking_strategy": "structure_aware",  # as in speed
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
            "chunking_strategy": "hierarchical",  # enable_hierarchical wins
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
            "chunking_strategy": "hierarchical",  # enable_hierarchical wins
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
        self._refresh_saved_profiles()
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

    def _is_visible_to(self, profile: Any, user_id: str) -> bool:
        """Whether ``user_id`` may see and use a saved profile (a ``RAGProfile`` or its dict).

        Anonymous is the single-user identity (auth off): it sees everything,
        as `list_profiles` always allowed.
        """
        owner = profile.get("user_id") if isinstance(profile, dict) else profile.user_id
        return user_id == ANONYMOUS or (owner or ANONYMOUS) in (user_id, ANONYMOUS)

    def _saved(self, name: str, owner: str) -> Optional[RAGProfile]:
        """The saved profile ``name`` of ``owner``, exactly."""
        return next(
            (
                p
                for p in self._global_custom_profiles
                if p.name == name and (p.user_id or ANONYMOUS) == owner
            ),
            None,
        )

    def _saved_for(self, name: str, user_id: str, *, besides: Any = None) -> Optional[RAGProfile]:
        """The saved profile ``name`` that ``user_id`` gets (``besides`` one excluded).

        Its own, else the shared one (saved without a login). Anonymous, the
        single user, gets another owner's when it has none of its own (the
        first by owner). Profiles were looked up by name alone, so a tenant on
        the shared profile of a name was handed another tenant's of that name.
        """
        candidates = [
            p for p in self._global_custom_profiles if p.name == name and p is not besides
        ]
        for owner in (user_id, ANONYMOUS):
            mine = next((p for p in candidates if (p.user_id or ANONYMOUS) == owner), None)
            if mine is not None:
                return mine
        if user_id == ANONYMOUS and candidates:
            return min(candidates, key=lambda p: p.user_id or "")
        return None

    @staticmethod
    def _put_profile(state: Any, profile: RAGProfile) -> None:
        """Make ``profile`` the session's profile of its name (one per name)."""
        state.profiles = [p for p in state.profiles if p.name != profile.name] + [profile]

    def _refuse_past_limit(self, names: List[str], user_id: str) -> None:
        """Raise when saving ``names`` as ``user_id``'s passes its limit of saved profiles."""
        new = [n for n in dict.fromkeys(names) if self._saved(n, user_id) is None]
        # Counted per owner. One shared count let a tenant fill it and lock
        # every other tenant out of saving any profile at all.
        owned = sum(1 for p in self._global_custom_profiles if (p.user_id or ANONYMOUS) == user_id)
        if new and owned + len(new) > MAX_GLOBAL_PROFILES:
            raise ValueError(
                f"Maximum number of profiles ({MAX_GLOBAL_PROFILES}) reached. "
                "Delete unused profiles before creating new ones."
            )

    def _keep_saved(self, profile: RAGProfile) -> None:
        """Store ``profile`` as its owner's saved profile of its name. Call in ``_session_write``.

        A profile only a session's copy held (an imported one, or one a switch
        made from settings) was lost to every other manager: the session row
        keeps the active profile's name and owner, not its settings, so each
        other process, and on PostgreSQL each other request, put the session
        back on ``default``.
        """
        owner = profile.user_id or ANONYMOUS
        mine = self._saved(profile.name, owner)
        self._save_global_profile(profile)
        if mine is not None:
            at = self._global_custom_profiles.index(mine)
            self._global_custom_profiles[at] = copy.deepcopy(profile)
        else:
            self._global_custom_profiles.append(copy.deepcopy(profile))
            self._global_profile_names.add(profile.name)
        self._saved_profiles_changed()

    def _refuse_builtin_name(self, profile_name: str, verb: str = "overwritten") -> None:
        """Raise for a built-in profile's name: search runs its own settings, whatever is saved."""
        if self._is_protected_profile_name(profile_name):
            raise ValueError(
                f"Built-in profile {profile_name!r} cannot be {verb}: search always runs a "
                f"built-in profile's own settings, so other settings saved under its name "
                f"would be shown and not used. Nothing was changed. Use another name (for "
                f"example {profile_name + '-custom'!r})."
            )

    def _not_yours(self, name: str, verb: str) -> ValueError:
        """Why a caller may not ``verb`` the saved profile ``name`` someone else owns."""
        shared = any(
            p.name == name and (p.user_id or ANONYMOUS) == ANONYMOUS
            for p in self._global_custom_profiles
        )
        whose = (
            "shared (saved without a login, by the CLI or a local MCP server), so with auth "
            "on only the server's own CLI changes it"
            if shared
            else "another user's"
        )
        return ValueError(
            f"Profile {name!r} is not yours to {verb}: it is {whose}. Nothing was changed. "
            f"Save a profile of your own, under that name or another."
        )

    def _find_profile(self, state, profile_name: str, user_id: str) -> Optional[RAGProfile]:
        """The profile ``profile_name`` the caller gets: built-in, saved, or the session's own.

        A profile saved from another session after this one was created is in
        the saved list and was listed by `list_profiles`, but activating it
        answered "Unknown profile", because only the session's copy was looked
        at.
        """
        if self._is_protected_profile_name(profile_name):
            return next((p for p in state.profiles if p.name == profile_name), None) or next(
                (p for p in self._default_profiles if p.name == profile_name), None
            )
        if not any(p.name == profile_name for p in self._global_custom_profiles):
            # Perhaps saved by another process since this one read the list.
            self._refresh_saved_profiles(force=True)
        saved = self._saved_for(profile_name, user_id)
        if saved is not None:
            return saved
        # One of this session's own, made by a switch with settings and not saved.
        return next(
            (
                p
                for p in state.profiles
                if p.name == profile_name
                and self._is_visible_to(p, user_id)
                and not any(g.name == profile_name for g in self._global_custom_profiles)
            ),
            None,
        )

    def set_active_profile(
        self,
        profile_name: str,
        config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        *,
        force: bool = False,
        user_id: str = ANONYMOUS,
        reset: bool = False,
    ) -> Dict[str, Any]:
        """Set the active profile and optionally update its config.

        Refused, with nothing changed, when the switch would embed at another
        size than the stored documents (``force`` skips that check). ``reset``
        drops the session's own settings (its overrides), so the profile's
        apply as they are. A name no profile has, with ``config``, is saved as
        ``user_id``'s profile first, with the checks `save_profile` makes.
        """
        if config:
            config = self._session_settings(config, document=True)
            self._validate_or_raise(config, session_id)

        sid = session_id or "default"
        with self._session_write(session_id) as state:
            profile, profile_config, overrides = self._switch_plan(
                state, profile_name, config, user_id, reset=reset
            )
            self._refuse_unsearchable(
                state,
                _resolve(profile_config, overrides),
                change=f"Profile {profile_name!r}",
                force=force,
                session_id=session_id,
                profile_config=profile_config,
            )

            previous = state.active_profile
            created = profile is None
            if created:
                _check_profile_size(config)
                self._refuse_past_limit([profile_name], user_id)
                profile = RAGProfile(
                    name=profile_name,
                    description=f"Saved profile: {profile_name}",
                    config=config,
                    user_id=user_id,
                )
                self._keep_saved(profile)
                self._put_profile(state, copy.deepcopy(profile))
            elif profile not in state.profiles:
                self._put_profile(state, copy.deepcopy(profile))
            state.session_overrides = overrides
            state.active_profile = profile_name
            # Whose profile of that name the session is on, so every process
            # reads the same one back.
            state.user_id = profile.user_id or ANONYMOUS
            try:
                self._persist_single_session(sid)
            except SettingsNotSaved as exc:
                if not created:
                    raise
                raise partly_saved(
                    exc,
                    f"Profile {profile_name!r} was saved; session {sid!r} is still on "
                    f"profile {previous!r}",
                ) from exc.__cause__

        # Without this the profile switch is recorded and never applied:
        # search resolves through the session provider, not this manager.
        self._publish_to_search(session_id)
        return state.to_dict()

    def _switch_plan(
        self,
        state: Any,
        profile_name: str,
        config: Optional[Dict[str, Any]],
        user_id: str,
        *,
        reset: bool = False,
    ) -> tuple:
        """``(profile, profile_config, overrides)`` a switch to ``profile_name`` would leave.

        Worked out before any of it is applied, so a refusal changes nothing.
        The old order switched, checked, and switched back, and the way back
        dropped the user's overrides or kept the refused request's. Call with
        ``self._lock`` held.
        """
        previous = state.get_active_profile()
        profile = self._find_profile(state, profile_name, user_id)
        if profile is None and not config:
            # Recording a name no profile has made `configure --profile
            # bogus` print "Applied profile: bogus" and change nothing.
            known = sorted(
                {p.name for p in state.profiles if self._is_visible_to(p, user_id)}
                | {p.name for p in self._global_custom_profiles if self._is_visible_to(p, user_id)}
            )
            raise UnknownProfileError(
                f"Unknown profile {profile_name!r}. Available: {', '.join(known)}"
            )
        if profile is None:
            return profile, config, {} if reset else state.session_overrides
        if config:
            return profile, profile.config, config
        if reset:
            return profile, profile.config, {}
        if previous is not None and previous.name != profile_name:
            return (
                profile,
                profile.config,
                _without_echoes(state.session_overrides, previous.config),
            )
        return profile, profile.config, state.session_overrides

    def preview_active_profile(
        self,
        profile_name: str,
        session_id: Optional[str] = None,
        *,
        config: Optional[Dict[str, Any]] = None,
        user_id: str = ANONYMOUS,
        reset: bool = False,
    ) -> tuple:
        """``(resolved, chosen)``: the session's settings after ``set_active_profile``.

        Takes what ``set_active_profile`` takes. ``resolved`` is what the
        refusal checks compare (the profile's whole configuration and the
        overrides); ``chosen`` is what search would read as the session's own
        (see ``get_chosen_config``). Nothing is changed: a reindex into the new
        profile runs on these before the switch.
        """
        if config:
            config = self._session_settings(config, document=True)
            self._validate_or_raise(config, session_id)
        state = self.get_session_state(session_id)
        with self._lock:
            profile, profile_config, overrides = self._switch_plan(
                state, profile_name, config, user_id, reset=reset
            )
            resolved = _resolve(profile_config, overrides)
            chosen: Dict[str, Any] = {}
            if profile_name != "default":
                spec = (
                    self._PROFILE_SPECS.get(profile_name)
                    if self._is_protected_profile_name(profile_name)
                    else None
                )
                chosen = copy.deepcopy(spec if spec is not None else profile_config)
            _deep_merge(chosen, copy.deepcopy(overrides))
        return resolved, chosen

    def save_profile(
        self,
        profile_name: str,
        config: Dict[str, Any],
        session_id: Optional[str] = None,
        set_active: bool = True,
        user_id: str = ANONYMOUS,
        *,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Save or update the caller's profile ``profile_name``.

        Each owner (``user_id``) has profiles of its own, by name: saving
        never changes another owner's, the shared ones included. Refused when
        the new settings would move another session on this profile off its
        documents' collection; ``force`` (the server's operator, in Python)
        applies it anyway, as it skips the check for this session.
        """
        self._refuse_builtin_name(profile_name)

        if not isinstance(config, dict):
            raise ValueError("Profile config must be an object.")

        if not profile_name or len(profile_name) > 100:
            raise ValueError("Profile name must be 1-100 characters.")

        config = self._session_settings(config, document=True)
        _check_profile_size(config)
        if config:
            self._validate_or_raise(config)

        sid = session_id or "default"
        with self._session_write(session_id) as state:
            self._refresh_saved_profiles(force=True)
            mine = self._saved(profile_name, user_id)
            self._refuse_past_limit([profile_name], user_id)
            if mine is not None:
                self._refuse_moving_others(
                    mine,
                    config,
                    change=f"Saving profile {profile_name!r}",
                    exclude=sid,
                    force=force,
                )

            # Saving as (or over) the active profile changes what this session
            # embeds with, exactly as switching to it would.
            active = state.get_active_profile()
            on_it = (
                active is not None
                and active.name == profile_name
                and (active.user_id or ANONYMOUS) == user_id
            )
            if set_active or on_it:
                overrides = {} if set_active else state.session_overrides
                self._refuse_unsearchable(
                    state,
                    _resolve(config, overrides),
                    change=f"Profile {profile_name!r}",
                    force=force,
                    session_id=session_id,
                )

            new_profile = RAGProfile(
                name=profile_name,
                description=f"Saved profile: {profile_name}",
                config=config,
                user_id=user_id,
            )
            previous = state.active_profile
            if set_active or on_it:
                self._put_profile(state, copy.deepcopy(new_profile))
            if set_active:
                state.active_profile = profile_name
                state.session_overrides = {}
                state.user_id = user_id

            state.profiles = self._ensure_default_profile(state.profiles)

            # The profile first: a session row naming a profile not saved yet
            # puts the session on default for every other process.
            self._keep_saved(new_profile)
            if set_active:
                try:
                    self._persist_single_session(sid)
                except SettingsNotSaved as exc:
                    raise partly_saved(
                        exc,
                        f"Profile {profile_name!r} was saved; session {sid!r} is still on "
                        f"profile {previous!r}",
                    ) from exc.__cause__

            logger.info(
                f"Saved profile '{profile_name}' globally "
                f"({len(self._global_custom_profiles)} total custom profiles)"
            )

        # Saving as the active profile (the configurator's "Save As") must
        # change what search runs, exactly as set_active_profile does; it used
        # to report "Saved!" and leave retrieval as it was.
        if set_active or on_it:
            self._publish_to_search(session_id)
        return state.to_dict()

    def _refuse_moving_others(
        self,
        profile: RAGProfile,
        config: Optional[Dict[str, Any]],
        *,
        change: str,
        exclude: str,
        force: bool,
    ) -> None:
        """Raise when ``change`` to ``profile`` would leave another session's documents unsearched.

        ``config`` is the profile's new settings; None when it goes (a delete
        or a rename), and each session falls back to what its active profile's
        name then finds, else ``default``. Only the caller's session was
        checked: re-saved settings reached every session on the profile, and
        a delete put them on default, both searching a collection their
        documents were not in. ``force`` skips this (the server's operator).
        """
        if force:
            return
        lister = getattr(self, "_sessions_on_profile", None)
        rows = lister(profile.name) if lister is not None else None
        if rows is None:
            return  # a stand-in store
        from lasagna.api._profile_guard import DocumentsLeftBehind, _plural, _searched_counts
        from lasagna.core.config import collection_name_for_settings

        default = next((p for p in self._default_profiles if p.name == "default"), None)
        stranded = []
        for row in rows:
            sid = row.get("session_id")
            if not sid or sid == exclude or kept_in_process(sid):
                continue
            owner = row.get("user_id") or ANONYMOUS
            if self._saved_for(profile.name, owner) is not profile:
                continue  # its name finds another owner's profile
            overrides = row.get("session_overrides") or {}
            if config is not None:
                after = config
            else:
                instead = self._saved_for(profile.name, owner, besides=profile)
                after = (
                    instead.config if instead is not None else (default.config if default else {})
                )
            before = collection_name_for_settings(_resolve(profile.config, overrides))
            if collection_name_for_settings(_resolve(after, overrides)) == before:
                continue
            counts = _searched_counts(sid)
            held = counts.get(before, 0) + counts.get(None, 0) + counts.get("", 0)
            if held:
                stranded.append((sid, held))
        if not stranded:
            return
        named = ", ".join(f"{sid} ({_plural(n, 'document')})" for sid, n in stranded)
        moves = "; ".join(
            f"lasagna configure --profile <another> -s {sid} --reindex" for sid, _ in stranded
        )
        one = len(stranded) == 1
        instead = (
            "Save it under a new name instead, or first move "
            if config is not None
            else "Do it after moving "
        )
        raise DocumentsLeftBehind(
            f"{change} would move {_plural(len(stranded), 'other session')} on it off the "
            f"collection holding {'its' if one else 'their'} documents: {named}. "
            f"{'It' if one else 'They'} would then search another model's collection and "
            f"find none of them. Nothing was changed. {instead}"
            f"{'that session' if one else 'those sessions'} to another profile with "
            f"{'its' if one else 'their'} documents ({moves}). The server's operator can "
            f"apply it anyway from Python, with force=True."
        )

    def rename_profile(
        self,
        old_name: str,
        new_name: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        *,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Rename the caller's profile ``old_name``.

        ``user_id`` owns it, as in `save_profile` and `delete_profile`: no one
        renames another owner's profile. Refused when a session on it holds
        documents it would no longer find (``force`` skips that).
        """
        if self._is_protected_profile_name(old_name) or self._is_protected_profile_name(new_name):
            raise ValueError("Built-in profiles cannot be renamed.")
        if not new_name or len(new_name) > 100:
            raise ValueError("New profile name must be 1-100 characters.")
        owner = user_id or ANONYMOUS
        sid = session_id or "default"

        with self._session_write(session_id) as state:
            self._refresh_saved_profiles(force=True)
            if self._saved(new_name, owner) is not None or any(
                p.name == new_name for p in state.profiles
            ):
                raise ValueError(f"Profile '{new_name}' already exists.")
            mine = self._saved(old_name, owner)
            local = next((p for p in state.profiles if p.name == old_name), None)
            if mine is None and any(p.name == old_name for p in self._global_custom_profiles):
                raise self._not_yours(old_name, "rename")
            if mine is None and local is None:
                raise ValueError(f"Profile '{old_name}' not found.")
            if mine is not None:
                self._refuse_moving_others(
                    mine, None, change=f"Renaming profile {old_name!r}", exclude=sid, force=force
                )

            renamed, previous = None, state.active_profile
            if mine is not None:
                renamed = copy.deepcopy(mine)
                renamed.name = new_name
                if renamed.description:
                    renamed.description = renamed.description.replace(old_name, new_name)
                self._global_custom_profiles[self._global_custom_profiles.index(mine)] = renamed
                self._global_profile_names = {p.name for p in self._global_custom_profiles}
            if local is not None and (local.user_id or ANONYMOUS) == owner:
                moved = copy.deepcopy(renamed) if renamed is not None else copy.deepcopy(local)
                moved.name = new_name
                state.profiles = [p for p in state.profiles if p is not local] + [moved]
                if state.active_profile == old_name:
                    state.active_profile = new_name

            state.profiles = self._ensure_default_profile(state.profiles)
            # The new row, then the session, then the old row: stopped at any
            # point, the session names a profile that exists. The old row went
            # before the session was saved, and a failed save left the session
            # on a name with no profile (default, for every other process).
            if renamed is not None:
                self._save_global_profile(renamed)
                self._saved_profiles_changed()
            try:
                self._persist_single_session(sid)
            except SettingsNotSaved as exc:
                if renamed is None:
                    raise
                raise partly_saved(
                    exc,
                    f"Profile {new_name!r} was saved as a copy of {old_name!r}, which is "
                    f"kept, and session {sid!r} is still on profile {previous!r}",
                ) from exc.__cause__
            if renamed is not None:
                try:
                    self._delete_global_profile(
                        RAGProfile(name=old_name, description="", config={}, user_id=owner)
                    )
                except SettingsNotSaved as exc:
                    raise partly_saved(
                        exc,
                        f"Profile {old_name!r} was saved as {new_name!r} and session {sid!r} "
                        f"uses the new name; the old name is kept until it is deleted "
                        f"(delete profile {old_name!r})",
                    ) from exc.__cause__

            logger.info(f"Renamed profile '{old_name}' to '{new_name}' globally")

        return state.to_dict()

    def delete_profile(
        self,
        profile_name: str,
        session_id: Optional[str] = None,
        user_id: str = "__anonymous__",
        *,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Delete the caller's profile ``profile_name`` (built-in profiles are protected).

        No one deletes another owner's profile, the shared ones included under
        auth. Refused when a session on it holds documents its fallback would
        not search (``force``: the server's operator, in Python, deletes it
        anyway).
        """
        if self._is_protected_profile_name(profile_name):
            raise ValueError("Built-in profiles cannot be deleted.")

        sid = session_id or "default"
        with self._session_write(session_id) as state:
            self._refresh_saved_profiles(force=True)
            mine = self._saved(profile_name, user_id)
            if mine is None and any(p.name == profile_name for p in self._global_custom_profiles):
                raise self._not_yours(profile_name, "delete")
            if mine is not None:
                self._refuse_moving_others(
                    mine,
                    None,
                    change=f"Deleting profile {profile_name!r}",
                    exclude=sid,
                    force=force,
                )
            active = state.get_active_profile()
            on_it = (
                state.active_profile == profile_name
                and active is not None
                and (active.user_id or ANONYMOUS) == user_id
            )
            if on_it:
                # The session falls back to default, which may embed at another size.
                fallback = next((p for p in state.profiles if p.name == "default"), None)
                self._refuse_unsearchable(
                    state,
                    _resolve(fallback.config if fallback else {}, {}),
                    change=f"Deleting the active profile {profile_name!r} falls back to 'default', which",
                    force=force,
                    session_id=session_id,
                )
            state.profiles = [
                p
                for p in state.profiles
                if not (p.name == profile_name and (p.user_id or ANONYMOUS) == user_id)
            ]
            if on_it:
                state.active_profile = "default"
                state.session_overrides = {}
                state.user_id = ANONYMOUS

            if mine is not None:
                self._global_custom_profiles = [
                    p for p in self._global_custom_profiles if p is not mine
                ]
                self._global_profile_names = {p.name for p in self._global_custom_profiles}

            state.profiles = self._ensure_default_profile(state.profiles)
            # The session first: deleted first, a failed save of the session
            # left it naming a profile that no longer exists.
            self._persist_single_session(sid)
            if mine is not None:
                try:
                    self._delete_global_profile(mine)
                except SettingsNotSaved as exc:
                    if not on_it:
                        raise
                    raise partly_saved(
                        exc, f"Session {sid!r} was moved to profile 'default'"
                    ) from exc.__cause__
                self._saved_profiles_changed()

            logger.info(f"Deleted profile '{profile_name}' globally")

        # The session may have fallen back to default; search must follow.
        self._publish_to_search(session_id)
        return state.to_dict()

    def update_profile_description(
        self,
        profile_name: str,
        description: str,
        session_id: Optional[str] = None,
        user_id: str = ANONYMOUS,
    ) -> Dict[str, Any]:
        """Update the description of the caller's profile (default is protected)."""
        if profile_name == "default":
            raise ValueError("The default profile description cannot be edited.")

        with self._session_write(session_id) as state:
            mine = self._saved(profile_name, user_id)
            if mine is None and any(p.name == profile_name for p in self._global_custom_profiles):
                raise self._not_yours(profile_name, "change")
            profile = next((p for p in state.profiles if p.name == profile_name), None)
            if mine is None and not profile:
                raise ValueError(f"Profile '{profile_name}' not found.")

            if profile is not None and (profile.user_id or ANONYMOUS) == user_id:
                profile.description = description
            state.profiles = self._ensure_default_profile(state.profiles)
            if mine is not None:
                gp = copy.deepcopy(mine)
                gp.description = description
                self._keep_saved(gp)

        return state.to_dict()

    def reload_defaults(self) -> None:
        """Reload default profiles from settings.lasagna.toml."""
        with self._lock:
            self._load_defaults()
        logger.info("Reloaded default RAG profiles from settings.lasagna.toml")

    def get_profile_stats(self) -> Dict[str, Any]:
        """Get statistics about profiles and sessions."""
        self._refresh_saved_profiles()
        with self._lock:
            return {
                "global_custom_profiles_count": len(self._global_custom_profiles),
                "global_custom_profiles_limit": MAX_GLOBAL_PROFILES,
                "default_profiles_count": len(self._default_profiles),
                "active_sessions_count": len(self._session_states),
                "max_cached_sessions": MAX_CACHED_SESSIONS,
                # None: session_idle_days = 0 keeps idle sessions' settings.
                "session_ttl_hours": (
                    None if session_idle_seconds() is None else session_idle_seconds() / 3600
                ),
            }

    def list_global_profiles(self, user_id: str = "__anonymous__") -> List[Dict[str, Any]]:
        """List global custom profiles visible to the given user."""
        self._refresh_saved_profiles()
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
def _without_echoes(overrides: Dict[str, Any], profile_config: Dict[str, Any]) -> Dict[str, Any]:
    """Session overrides minus those that only repeat the profile being left.

    The CLI and SDK clients write the active profile's values back as session
    overrides, and overrides win over the profile. So after `configure
    --profile speed` and an upload, switching to another profile was recorded
    and changed nothing: search kept running speed's copied values. Values the
    user actually changed differ from the old profile's and are kept.
    """
    return {
        k: v
        for k, v in (overrides or {}).items()
        if not (k in (profile_config or {}) and profile_config[k] == v)
    }


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
