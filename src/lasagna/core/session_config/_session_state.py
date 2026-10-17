"""Session state mixin for session configuration.

Handles session state retrieval, config patching, reset, import/export,
RAG master toggle, and document focus management.
"""

import copy
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional

from loguru import logger

from ._data_models import (
    ANONYMOUS,
    SCHEMA_VERSION,
    RAGProfile,
    KnowledgeConfigState,
    _check_profile_size,
    _deep_merge,
    _resolve,
)


#: How stale a cached session may be before it is compared with the store.
_RECHECK_SECONDS = 1.0

#: How often a process records in the store that it still uses a session.
_USE_RECORD_SECONDS = 3600.0

#: When this process last recorded each session's use (``time.monotonic``).
_use_recorded_at: Dict[str, float] = {}

#: How many changes this process has saved to each session, over all of its
#: managers. A manager whose count differs reads the session again at once.
_writes_here: Dict[str, int] = {}

#: Keys each session last published to an in-memory session provider, so a
#: key the session no longer chooses (a profile switched away from) is
#: removed rather than left behind. Per process, like that provider.
_published_keys: Dict[str, set] = {}


def session_lock(session_id: str):
    """The session's lock, held against every process that shares the store.

    Every change to a session's settings holds it from reading the session
    as stored to writing it back. Taken before any document's lock, never
    after.
    """
    from lasagna.core.store_locks import store_lock

    return store_lock(f"session:{session_id or 'default'}")


def _persisted_view(state: Any) -> tuple:
    """The fields the session table keeps, from a state or a stored row."""
    get = state.get if isinstance(state, dict) else lambda k, d=None: getattr(state, k, d)
    return (
        get("active_profile", "default"),
        get("session_overrides", {}),
        bool(get("rag_enabled", True)),
        list(get("focused_document_ids", []) or []),
    )


class SessionStateMixin:
    """Mixin for session state management.

    Expects the host class to provide:
    - self._lock: RLock
    - self._session_states: Dict[str, KnowledgeConfigState]
    - self._default_profiles: List[RAGProfile]
    - self._global_custom_profiles: List[RAGProfile]
    - self._reload_if_changed(): method (from PersistenceMixin)
    - self._cleanup_old_sessions(): method (from PersistenceMixin)
    - self._ensure_default_profile(): method (from ProfileMixin)
    - self._persist_session_states(): method (from PersistenceMixin)
    - self._delete_persisted_session(): method (from PersistenceMixin)
    - self._get_citations_file_path(): method (from CitationStoreMixin)
    """

    def _state_from_dict(self, data: Dict[str, Any]) -> KnowledgeConfigState:
        """Build a KnowledgeConfigState from persisted data."""
        profiles_data = data.get("profiles", [])
        profiles = []
        if isinstance(profiles_data, list):
            for item in profiles_data:
                if isinstance(item, dict) and item.get("name"):
                    profiles.append(RAGProfile.from_dict(item))

        if not profiles:
            profiles = [copy.deepcopy(p) for p in self._default_profiles]

        profiles = self._ensure_default_profile(profiles)
        profile_names = {p.name for p in profiles}
        active_profile = data.get("active_profile") or "default"
        if active_profile not in profile_names:
            # The session table keeps the active profile's name, not the
            # profile: a saved custom profile is found among the saved ones.
            # Without this every restart put such a session back on default.
            saved = next(
                (p for p in self._global_custom_profiles if p.name == active_profile), None
            )
            if saved is not None:
                profiles.append(copy.deepcopy(saved))
            else:
                active_profile = "default"

        session_overrides = data.get("session_overrides", {})
        if not isinstance(session_overrides, dict):
            session_overrides = {}

        defaults_source = data.get("defaults_source", "settings.lasagna.toml")

        rag_enabled = data.get("rag_enabled", True)
        if not isinstance(rag_enabled, bool):
            rag_enabled = True

        focused_document_ids = data.get("focused_document_ids", [])
        if not isinstance(focused_document_ids, list):
            focused_document_ids = []

        last_accessed = data.get("last_accessed", time.time())

        return KnowledgeConfigState(
            profiles=profiles,
            active_profile=active_profile,
            session_overrides=session_overrides,
            defaults_source=defaults_source,
            rag_enabled=rag_enabled,
            focused_document_ids=focused_document_ids,
            last_accessed=last_accessed,
        )

    def get_session_state(self, session_id: Optional[str] = None) -> KnowledgeConfigState:
        """Get or create session state, as last written by any process."""
        session_id = session_id or "default"

        with self._lock:
            self._reload_if_changed()
            self._cleanup_old_sessions()
            self._adopt_persisted_session(session_id)
            self._record_use(session_id)

            if session_id not in self._session_states:
                all_profiles = [copy.deepcopy(p) for p in self._default_profiles]

                # Only the shared ones: a profile with an owner is found when its
                # owner asks for it, and copying every tenant's into every new
                # session showed them (configs included) to all of them.
                default_names = {p.name for p in all_profiles}
                for custom_profile in self._global_custom_profiles:
                    if (
                        custom_profile.name not in default_names
                        and custom_profile.user_id == ANONYMOUS
                    ):
                        all_profiles.append(copy.deepcopy(custom_profile))

                state = KnowledgeConfigState(
                    profiles=all_profiles,
                    active_profile="default",
                    session_overrides={},
                    defaults_source="settings.lasagna.toml",
                    rag_enabled=True,
                    focused_document_ids=[],
                    last_accessed=time.time(),
                )
                state.profiles = self._ensure_default_profile(state.profiles)
                self._session_states[session_id] = state
            else:
                self._session_states[session_id].last_accessed = time.time()

            return self._session_states[session_id]

    def _record_use(self, session_id: str) -> None:
        """Record in the store that the session is in use, at most once an hour per process.

        Only writes recorded a use, and the store deletes the settings of a
        session unused for ``session_idle_days``: a session that was only
        searched lost its profile while it was in use.
        """
        now = time.monotonic()
        if now - _use_recorded_at.get(session_id, float("-inf")) < _USE_RECORD_SECONDS:
            return
        if len(_use_recorded_at) > 100_000:
            _use_recorded_at.clear()
        _use_recorded_at[session_id] = now
        touch = getattr(self, "_touch_persisted_session", None)
        if touch is not None:
            touch(session_id)

    def _adopt_persisted_session(self, session_id: str, *, fresh: bool = False) -> None:
        """Take the stored session when another process changed it since we looked.

        `lasagna configure` and a running `lasagna serve` share one store, and
        each read it only when it started. So a profile set from the CLI never
        reached the running server (it answered "default" for that session),
        and the server's next write put its stale copy back over the CLI's.
        Checked at most once a second per session: search reads the session's
        config once per setting. ``fresh`` reads the store now, past any cache
        (a write about to save the session).
        """
        now = time.monotonic()
        cached = self._session_states.get(session_id)
        checked = self._session_checked_at.get(session_id, float("-inf"))
        # A change another manager of this process saved is read at once: with
        # PostgreSQL each request has its own manager, and the session
        # provider's (which search reads) kept its copy for up to a second
        # after a REST patch had returned.
        seen = getattr(self, "_session_writes_seen", None)
        if seen is None:
            seen = self._session_writes_seen = {}
        writes = _writes_here.get(session_id, 0)
        if (
            not fresh
            and cached is not None
            and now - checked < _RECHECK_SECONDS
            and seen.get(session_id, 0) == writes
        ):
            return
        self._session_checked_at[session_id] = now
        seen[session_id] = writes
        stored = self._read_persisted_session(session_id, fresh=fresh)
        if stored is None:
            return
        if cached is not None and _persisted_view(cached) == _persisted_view(stored):
            return
        state = self._state_from_dict(stored)
        if cached is not None:
            # Per-request results, never persisted.
            state.last_citations, state.last_chunks = cached.last_citations, cached.last_chunks
        self._session_states[session_id] = state

    @contextmanager
    def _session_write(self, session_id: Optional[str]) -> Iterator[KnowledgeConfigState]:
        """The session as stored now, held against every process until the change is saved.

        A change was applied to this process's copy of the session, up to a
        second old, and the whole row written back with no lock across
        processes: two api workers patching different keys of one session at
        once kept only the second patch, and a change of ``rag_enabled`` put
        back the overrides it had read. Every change to a session's settings
        goes through this.
        """
        sid = session_id or "default"
        with session_lock(sid):
            with self._lock:
                self._adopt_persisted_session(sid, fresh=True)
                try:
                    yield self.get_session_state(sid)
                finally:
                    _writes_here[sid] = _writes_here.get(sid, 0) + 1
                    self._session_writes_seen[sid] = _writes_here[sid]

    def get_config(
        self, session_id: Optional[str] = None, user_id: str = ANONYMOUS
    ) -> Dict[str, Any]:
        """Get the full configuration response for the API, as ``user_id`` may see it."""
        state = self.get_session_state(session_id)
        return self._view(state, user_id)

    def _view(self, state: KnowledgeConfigState, user_id: str) -> Dict[str, Any]:
        """The session's state with only the profiles ``user_id`` may see.

        ``to_dict`` is also what gets persisted, so it keeps everything; what a
        caller is shown drops other owners' saved profiles and adds the
        caller's own saved ones that this session has not used yet.
        """
        data = state.to_dict()
        with self._lock:
            foreign = self._foreign_profile_names(user_id)
            shown = [p for p in data["profiles"] if p["name"] not in foreign]
            names = {p["name"] for p in shown}
            for profile in self._global_custom_profiles:
                if profile.name not in names and self._is_visible_to(profile, user_id):
                    shown.append(profile.to_dict())
        data["profiles"] = shown
        return data

    def get_resolved_config(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the resolved configuration for RAG execution."""
        state = self.get_session_state(session_id)
        return state.get_resolved_config()

    def get_chosen_config(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """What this session chose: its profile's own settings and its overrides.

        This is what search reads as the session's settings, which rank above
        the environment's. `get_resolved_config` starts from the profile's
        whole configuration: for ``default`` the packaged settings file, for
        a built-in profile that file plus the few settings that define the
        profile. Handed to search, the file's ``vector_backend``,
        ``embedding_model`` and the rest beat ``RAG_VECTOR_BACKEND=qdrant`` and
        every other environment setting, for every session: the compose stack
        configured for Qdrant searched a Milvus Lite file. So a built-in
        profile contributes only its spec (``_PROFILE_SPECS``), ``default``
        nothing, and a saved profile everything that was saved.
        """
        state = self.get_session_state(session_id)
        active = state.get_active_profile()
        chosen: Dict[str, Any] = {}
        if active is not None and active.name != "default":
            spec = (
                self._PROFILE_SPECS.get(active.name)
                if self._is_protected_profile_name(active.name)
                else None
            )
            chosen = copy.deepcopy(spec if spec is not None else active.config)
        _deep_merge(chosen, copy.deepcopy(state.session_overrides))
        return chosen

    def _publish_to_search(self, session_id: Optional[str]) -> None:
        """Mirror the resolved config into the store the search path reads.

        This manager and ``SessionConfigProvider`` were two disconnected
        stores. Writing here updated the manager's own SQLite-backed state and
        nothing else, while ``rag_config.with_session()`` -- what the search
        orchestrator actually resolves through -- read from the provider. So
        every config-changing tool returned success and changed nothing:
        ``patch_config({"rerank_top_k": 17})`` left search using 5.

        That is the worst failure shape available. An error would have been
        noticed; silent success means an agent tuning retrieval believes it
        tuned retrieval, and every downstream number is quietly wrong.

        Publishing rather than merging the two stores keeps this a contained
        fix: the manager stays the source of truth for profiles and history,
        and the provider stays the thing search reads.

        ``PersistentSessionProvider`` already resolves *through* this manager,
        and its ``set_config_overrides`` calls ``patch_config`` — so publishing
        to it from inside ``patch_config`` recurses. Measured before the guard:
        one user-level patch became 49 nested calls and a ``RecursionError``,
        each level persisting to SQLite. The provider advertises this with
        ``_delegates_to_manager``, which the REST config route already honours
        for the same reason.
        """
        try:
            from lasagna.providers import get_session_provider

            provider = get_session_provider()
            if getattr(provider, "_delegates_to_manager", False):
                # It reads from here already; writing back would recurse.
                return
            sid = session_id or "default"
            chosen = self.get_chosen_config(session_id)
            stale = _published_keys.get(sid, set()) - set(chosen)
            if stale and hasattr(provider, "remove_config_overrides"):
                provider.remove_config_overrides(sid, stale)
            provider.set_config_overrides(sid, chosen)
            _published_keys[sid] = set(chosen)
        except Exception:
            # Never let publishing break the write itself -- but say so, since
            # the symptom otherwise is a config change that does not apply.
            logger.exception(
                "Config was saved but could not be published to the search path; "
                "retrieval will keep using the previous settings."
            )

    def patch_config(
        self,
        patch: Dict[str, Any],
        session_id: Optional[str] = None,
        *,
        force: bool = False,
        trusted: bool = False,
    ) -> Dict[str, Any]:
        """Apply a patch to the session overrides.

        Only settings a session may set are accepted (see ``_session_keys``);
        ``trusted`` is for in-process callers writing the process's own
        configuration, never for a value a client sent.
        """
        patch = self._session_settings(patch, trusted=trusted)

        with self._session_write(session_id) as state:
            self._validate_or_raise(patch, session_id, is_patch=True)
            # A patch of embedding_dim is a profile switch by another name, and
            # was the route by which a tenant set up the drop of the shared
            # collection.
            self._refuse_unsearchable(
                state,
                _resolve(state.get_resolved_config(), patch),
                change="This change",
                force=force,
                session_id=session_id,
                # In-process writes of the process's own configuration: the
                # harness names a collection per arm on its own sessions.
                documents=not trusted,
            )
            _deep_merge(state.session_overrides, patch)
            self._persist_single_session(session_id or "default")

        self._publish_to_search(session_id)
        return state.to_dict()

    def reset_config(
        self,
        scope: str = "active",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Reset configuration to defaults."""
        with self._session_write(session_id) as state:
            if scope == "all":
                default = next((p for p in self._default_profiles if p.name == "default"), None)
                after = _resolve(default.config if default else {}, {})
            else:
                active = state.get_active_profile()
                after = _resolve(active.config if active else {}, {})
            self._refuse_unsearchable(
                state, after, change="Resetting the configuration", session_id=session_id
            )
            if scope == "all":
                state.profiles = [copy.deepcopy(p) for p in self._default_profiles]
                state.active_profile = "default"
                state.session_overrides = {}
            else:
                state.session_overrides = {}
            self._persist_single_session(session_id or "default")

        self._publish_to_search(session_id)
        active = state.get_active_profile()
        result = state.to_dict()
        if active:
            result["config"] = active.config
        return result

    def export_config(
        self, session_id: Optional[str] = None, user_id: str = ANONYMOUS
    ) -> Dict[str, Any]:
        """Export the full configuration for download, as ``user_id`` may see it."""
        state = self.get_session_state(session_id)
        return self._view(state, user_id)

    def import_config(
        self,
        import_data: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Import configuration from uploaded JSON.

        Everything is checked before anything is applied: every value, and
        what the session would embed with afterwards. Session overrides used
        to be stored as they came, so ``{"chunk_size": "lots"}`` answered 200
        and every later read of the session's config failed with a 500.
        """
        schema_version = import_data.get("schema_version")
        if not schema_version:
            raise ValueError("Missing schema_version in import data")

        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ValueError("schema_version must be an integer")
        if schema_version > SCHEMA_VERSION:
            raise ValueError(
                f"Import schema version {schema_version} is newer than supported version {SCHEMA_VERSION}"
            )

        if "profiles" not in import_data:
            raise ValueError("Missing 'profiles' in import data")

        profiles_data = import_data["profiles"]
        if not isinstance(profiles_data, list):
            raise ValueError("'profiles' must be an array")
        if not all(isinstance(p, dict) for p in profiles_data):
            raise ValueError("Each profile must be an object")

        profile_names = [p.get("name") for p in profiles_data]
        if len(profile_names) != len(set(profile_names)):
            raise ValueError("Profile names must be unique")

        imported_profiles = []
        for p_data in profiles_data:
            if not p_data.get("name") or not isinstance(p_data["name"], str):
                raise ValueError("Each profile must have a 'name'")
            if not isinstance(p_data.get("config", {}), dict):
                raise ValueError(f"Profile '{p_data['name']}' config must be an object")
            builtin = next((p for p in self._default_profiles if p.name == p_data["name"]), None)
            profile_config = self._session_settings(
                p_data.get("config", {}),
                document=True,
                builtin=builtin.config if builtin else None,
            )
            if builtin is not None:
                # A built-in profile keeps the server's settings; an import
                # changes only what a session may change in it.
                profile_config = {**copy.deepcopy(builtin.config), **profile_config}
            _check_profile_size(profile_config)
            if profile_config:
                self._validate_or_raise(profile_config)
            imported_profiles.append(RAGProfile.from_dict({**p_data, "config": profile_config}))

        overrides = import_data.get("session_overrides")
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError("'session_overrides' must be an object")
        if overrides:
            overrides = self._session_settings(overrides, document=True)

        active_profile = import_data.get("active_profile")
        if active_profile is not None and not isinstance(active_profile, str):
            raise ValueError("'active_profile' must be a string")

        with self._session_write(session_id) as state:
            profiles = [copy.deepcopy(p) for p in state.profiles]
            for imported in imported_profiles:
                for i, p in enumerate(profiles):
                    if p.name == imported.name:
                        profiles[i] = imported
                        break
                else:
                    profiles.append(imported)

            active = state.active_profile
            if active_profile and active_profile in {p.name for p in profiles}:
                active = active_profile
            new_overrides = state.session_overrides if overrides is None else overrides

            target = next((p for p in profiles if p.name == active), None)
            after = _resolve(target.config if target else {}, new_overrides)
            if overrides:
                # Against the profile they will apply to, so cross-field rules
                # see what search will see.
                self._validate_or_raise(after)
            self._refuse_unsearchable(
                state, after, change="The imported configuration", session_id=session_id
            )

            state.profiles = profiles
            state.active_profile = active
            state.session_overrides = new_overrides
            self._persist_single_session(session_id or "default")

        # Without this an import was recorded and search kept the old settings.
        self._publish_to_search(session_id)
        return state.to_dict()

    def set_rag_enabled(
        self,
        enabled: bool,
        session_id: Optional[str] = None,
    ) -> None:
        """Set whether RAG is enabled for a session."""
        with self._session_write(session_id) as state:
            state.rag_enabled = enabled
            logger.debug(f"Set rag_enabled for session {session_id}: {enabled}")

            if not enabled:
                state.focused_document_ids = []
                state.last_citations = []
                state.last_chunks = []
                logger.debug(
                    f"Cleared focused_document_ids and cached results for session "
                    f"{session_id} (RAG disabled)"
                )

            self._persist_single_session(session_id or "default")

    def is_rag_enabled(
        self,
        session_id: Optional[str] = None,
    ) -> bool:
        """Check if RAG is enabled for a session."""
        state = self.get_session_state(session_id)
        return state.rag_enabled

    def set_focused_document_ids(
        self,
        document_ids: List[str],
        session_id: Optional[str] = None,
    ) -> None:
        """Set focused document IDs for a session."""
        with self._session_write(session_id) as state:
            state.focused_document_ids = document_ids or []
            logger.debug(
                f"Set focused_document_ids for session {session_id}: "
                f"{len(state.focused_document_ids)} documents"
            )
            self._persist_single_session(session_id or "default")

    def get_focused_document_ids(
        self,
        session_id: Optional[str] = None,
    ) -> List[str]:
        """Get focused document IDs for a session."""
        state = self.get_session_state(session_id)
        return state.focused_document_ids or []

    def clear_focused_document_ids(
        self,
        session_id: Optional[str] = None,
    ) -> None:
        """Clear focused document IDs for a session."""
        self.set_focused_document_ids([], session_id)

    def remove_session_overrides(
        self, keys: Iterable[str], session_id: Optional[str] = None
    ) -> None:
        """Drop ``keys`` from the session's overrides; the profile's values apply again."""
        with self._session_write(session_id) as state:
            removed = [k for k in keys if k in state.session_overrides]
            for key in removed:
                del state.session_overrides[key]
            if removed:
                self._persist_single_session(session_id or "default")

    def delete_session_state(self, session_id: str) -> bool:
        """Completely remove a session from config state."""
        deleted = False

        with self._lock:
            if session_id in self._session_states:
                del self._session_states[session_id]
                deleted = True
                logger.debug(f"Removed session {session_id} from in-memory state")

            # Remove persisted citations file
            try:
                citations_file = self._get_citations_file_path(session_id)
                import os

                if os.path.exists(citations_file):
                    os.remove(citations_file)
                    logger.debug(f"Removed citations file: {citations_file}")
            except Exception as e:
                logger.warning(f"Could not delete citations file for session {session_id}: {e}")

            if deleted:
                # Delete from database (SQLite/PostgreSQL) so it doesn't
                # get resurrected on next restart.
                self._delete_persisted_session(session_id)
                logger.info(f"Session state deleted: {session_id}")

        return deleted

    def _persist_sessions(self) -> None:
        """Alias for _persist_session_states for backward compatibility."""
        self._persist_session_states()
