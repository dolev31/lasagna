"""Session configuration provider interface.

This interface abstracts session-scoped RAG configuration,
allowing different session management strategies.

Example usage for external agents:

    from lasagna.providers import InMemorySessionProvider

    session = InMemorySessionProvider()

    # Enable RAG for a session
    session.set_rag_enabled("session123", True)

    # Focus on specific documents
    session.set_focused_document_ids("session123", ["doc1", "doc2"])
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Any, Iterable, List
from threading import Lock

from loguru import logger


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> None:
    """Recursively merge *patch* into *base* (mutates *base*)."""
    for key, value in patch.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


class SessionConfigProvider(ABC):
    """Abstract interface for session-scoped RAG configuration.

    Sessions allow different users/conversations to have isolated:
    - RAG enabled/disabled state
    - Focused document IDs (documents to search within)
    - Custom configuration overrides
    - RAG profiles

    Default: InMemorySessionProvider (auto-configured)
    Custom: Implement this ABC and pass to configure_providers()
    """

    @abstractmethod
    def get_resolved_config(self, session_id: str) -> Dict[str, Any]:
        """Get resolved config for a session (base config + session overrides).

        This returns any session-specific configuration overrides that
        should be merged with the base RAG configuration.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary of configuration overrides for this session
        """
        ...

    @abstractmethod
    def is_rag_enabled(self, session_id: str) -> bool:
        """Check if RAG is enabled for this session.

        Args:
            session_id: Session identifier

        Returns:
            True if RAG is enabled, False otherwise
        """
        ...

    @abstractmethod
    def set_rag_enabled(self, session_id: str, enabled: bool) -> None:
        """Enable or disable RAG for a session.

        Args:
            session_id: Session identifier
            enabled: Whether RAG should be enabled
        """
        ...

    @abstractmethod
    def get_focused_document_ids(self, session_id: str) -> List[str]:
        """Get document IDs that this session is focused on.

        When focused documents are set, RAG searches are filtered
        to only include chunks from these documents.

        Args:
            session_id: Session identifier

        Returns:
            List of focused document IDs (empty = search all documents)
        """
        ...

    @abstractmethod
    def set_focused_document_ids(self, session_id: str, doc_ids: List[str]) -> None:
        """Set focused document IDs for a session.

        Args:
            session_id: Session identifier
            doc_ids: List of document IDs to focus on (empty = all documents)
        """
        ...

    def get_excluded_document_ids(self, session_id: str) -> List[str]:
        """Get document IDs excluded from RAG searches.

        Override this method if your provider supports document exclusion.

        Args:
            session_id: Session identifier

        Returns:
            List of excluded document IDs (empty = none excluded)
        """
        return []

    def set_excluded_document_ids(self, session_id: str, doc_ids: List[str]) -> None:
        """Set excluded document IDs for a session.

        Override this method if your provider supports document exclusion.

        Args:
            session_id: Session identifier
            doc_ids: List of document IDs to exclude
        """
        pass

    def set_config_override(self, session_id: str, key: str, value: Any) -> None:
        """Set a single config override for a session.

        Convenience wrapper around set_config_overrides().

        Args:
            session_id: Session identifier
            key: Configuration key
            value: Configuration value
        """
        self.set_config_overrides(session_id, {key: value})

    def set_config_overrides(self, session_id: str, overrides: Dict[str, Any]) -> None:
        """Set multiple config overrides for a session.

        Default implementation: no-op. Subclasses should override.
        Used by the SDK config bridge to write per-session settings.

        Args:
            session_id: Session identifier
            overrides: Dictionary of backend config keys to values
        """
        pass

    def deep_merge_config_overrides(self, session_id: str, overrides: Dict[str, Any]) -> None:
        """Deep-merge *overrides* into existing session overrides.

        Unlike ``set_config_overrides`` (shallow update), nested dicts
        are merged recursively so that partial patches on keys like
        ``document_awareness`` don't clobber sibling sub-keys.

        Default implementation falls back to ``set_config_overrides``.
        Providers that store overrides in-memory should override this
        with an atomic implementation.
        """
        self.set_config_overrides(session_id, overrides)

    def remove_config_overrides(self, session_id: str, keys: Iterable[str]) -> None:
        """Remove specific config override keys for a session.

        Used by the SDK config bridge to clear overrides when a field
        reverts to its default value.

        Default implementation: no-op. Subclasses should override.

        Args:
            session_id: Session identifier
            keys: Config keys to remove from overrides
        """
        pass

    def clear_session(self, session_id: str) -> None:
        """Clear all session data.

        Override this method if your provider needs cleanup.

        Args:
            session_id: Session identifier
        """
        pass


class InMemorySessionProvider(SessionConfigProvider):
    """In-memory session configuration (no persistence).

    Suitable for:
    - Stateless deployments
    - Testing
    - Single-process applications
    - Short-lived sessions

    Note: All session data is lost on process restart.

    Args:
        max_sessions: Maximum number of sessions to retain (0 = unlimited).
            When the limit is exceeded, the least-recently-accessed session
            is evicted. Default: 0 (unlimited, backward-compatible).

    Example:
        provider = InMemorySessionProvider()
        provider.set_rag_enabled("user123", True)
        provider.set_focused_document_ids("user123", ["doc1", "doc2"])

        # Later...
        if provider.is_rag_enabled("user123"):
            docs = provider.get_focused_document_ids("user123")
    """

    def __init__(self, max_sessions: int = 0):
        """Initialize in-memory session provider.

        Args:
            max_sessions: Maximum sessions to keep (0 = unlimited).
        """
        self._sessions: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_sessions = max_sessions
        self._lock = Lock()

    def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create session data.

        Must be called with self._lock held. Moves accessed sessions to the
        end (most-recently-used) and evicts the oldest if over limit.

        Args:
            session_id: Session identifier

        Returns:
            Session data dictionary
        """
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
        else:
            self._sessions[session_id] = {
                "rag_enabled": True,
                "focused_document_ids": [],
                "excluded_document_ids": [],
                "overrides": {},
            }
            # Evict oldest if over limit
            if self._max_sessions > 0 and len(self._sessions) > self._max_sessions:
                evicted_id, _ = self._sessions.popitem(last=False)
                logger.debug(
                    f"Session evicted (LRU, max_sessions={self._max_sessions}): {evicted_id}"
                )
        return self._sessions[session_id]

    def get_resolved_config(self, session_id: str) -> Dict[str, Any]:
        """Get session configuration overrides.

        Args:
            session_id: Session identifier

        Returns:
            Configuration overrides dictionary
        """
        with self._lock:
            session = self._get_or_create_session(session_id)
            return session.get("overrides", {}).copy()

    def is_rag_enabled(self, session_id: str) -> bool:
        """Check if RAG is enabled for session.

        Args:
            session_id: Session identifier

        Returns:
            True if RAG enabled
        """
        with self._lock:
            session = self._get_or_create_session(session_id)
            return session.get("rag_enabled", True)

    def set_rag_enabled(self, session_id: str, enabled: bool) -> None:
        """Set RAG enabled state for session.

        Args:
            session_id: Session identifier
            enabled: Whether RAG should be enabled
        """
        with self._lock:
            session = self._get_or_create_session(session_id)
            session["rag_enabled"] = enabled

    def get_focused_document_ids(self, session_id: str) -> List[str]:
        """Get focused document IDs for session.

        Args:
            session_id: Session identifier

        Returns:
            List of focused document IDs
        """
        with self._lock:
            session = self._get_or_create_session(session_id)
            return session.get("focused_document_ids", []).copy()

    def set_focused_document_ids(self, session_id: str, doc_ids: List[str]) -> None:
        """Set focused document IDs for session.

        Args:
            session_id: Session identifier
            doc_ids: Document IDs to focus on
        """
        with self._lock:
            session = self._get_or_create_session(session_id)
            session["focused_document_ids"] = list(doc_ids)

    def get_excluded_document_ids(self, session_id: str) -> List[str]:
        """Get excluded document IDs for session.

        Args:
            session_id: Session identifier

        Returns:
            List of excluded document IDs
        """
        with self._lock:
            session = self._get_or_create_session(session_id)
            return session.get("excluded_document_ids", []).copy()

    def set_excluded_document_ids(self, session_id: str, doc_ids: List[str]) -> None:
        """Set excluded document IDs for session.

        Args:
            session_id: Session identifier
            doc_ids: Document IDs to exclude
        """
        with self._lock:
            session = self._get_or_create_session(session_id)
            session["excluded_document_ids"] = list(doc_ids)

    def set_config_override(self, session_id: str, key: str, value: Any) -> None:
        """Set a configuration override for session.

        Args:
            session_id: Session identifier
            key: Configuration key
            value: Configuration value
        """
        with self._lock:
            session = self._get_or_create_session(session_id)
            session["overrides"][key] = value

    def set_config_overrides(self, session_id: str, overrides: Dict[str, Any]) -> None:
        """Set multiple config overrides for a session.

        Args:
            session_id: Session identifier
            overrides: Dictionary of config keys to values
        """
        with self._lock:
            session = self._get_or_create_session(session_id)
            session["overrides"].update(overrides)

    def deep_merge_config_overrides(self, session_id: str, overrides: Dict[str, Any]) -> None:
        """Atomically deep-merge *overrides* into session overrides.

        Nested dicts are merged recursively under a single lock
        acquisition, preventing race conditions between concurrent
        PATCH requests.
        """
        with self._lock:
            session = self._get_or_create_session(session_id)
            _deep_merge(session["overrides"], overrides)

    def remove_config_overrides(self, session_id: str, keys: Iterable[str]) -> None:
        """Remove specific config override keys for a session.

        Args:
            session_id: Session identifier
            keys: Config keys to remove from overrides
        """
        with self._lock:
            session = self._get_or_create_session(session_id)
            for k in keys:
                session["overrides"].pop(k, None)

    def clear_session(self, session_id: str) -> None:
        """Clear all data for a session.

        Args:
            session_id: Session identifier
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def clear_all_sessions(self) -> None:
        """Clear all session data."""
        with self._lock:
            self._sessions.clear()


class PersistentSessionProvider(SessionConfigProvider):
    """File-backed session provider using KnowledgeConfigManager.

    Opt-in via ``RAG_SESSION_BACKEND=persistent`` env var.

    All manager access is **lazy** (``_get_manager()`` pattern) to avoid
    deadlock: this class may be instantiated inside
    ``_auto_configure_from_env()`` which holds the global ``_provider_lock``.
    Eager initialisation would re-enter the lock via ``get_config_provider()``,
    causing a deadlock.

    Example::

        provider = PersistentSessionProvider()
        provider.set_rag_enabled("session123", True)
        # State is persisted to disk via KnowledgeConfigManager.
    """

    # Signal to callers (e.g. REST PATCH handler) that this provider
    # delegates to KnowledgeConfigManager, so they should not write
    # through a second time.
    _delegates_to_manager = True

    def __init__(self):
        self._manager = None
        self._lock = Lock()

    def _get_manager(self):
        if self._manager is not None:
            return self._manager
        with self._lock:
            if self._manager is None:
                from lasagna.core.session_config import (
                    get_knowledge_config_manager,
                )

                self._manager = get_knowledge_config_manager()
        return self._manager

    def get_resolved_config(self, session_id: str) -> Dict[str, Any]:
        return self._get_manager().get_resolved_config(session_id)

    def is_rag_enabled(self, session_id: str) -> bool:
        return self._get_manager().is_rag_enabled(session_id)

    def set_rag_enabled(self, session_id: str, enabled: bool) -> None:
        self._get_manager().set_rag_enabled(enabled=enabled, session_id=session_id)

    def get_focused_document_ids(self, session_id: str) -> List[str]:
        return self._get_manager().get_focused_document_ids(session_id=session_id)

    def set_focused_document_ids(self, session_id: str, doc_ids: List[str]) -> None:
        self._get_manager().set_focused_document_ids(document_ids=doc_ids, session_id=session_id)

    def get_excluded_document_ids(self, session_id: str) -> List[str]:
        manager = self._get_manager()
        if hasattr(manager, "get_excluded_document_ids"):
            return manager.get_excluded_document_ids(session_id)
        return []

    def set_excluded_document_ids(self, session_id: str, doc_ids: List[str]) -> None:
        manager = self._get_manager()
        if hasattr(manager, "set_excluded_document_ids"):
            manager.set_excluded_document_ids(doc_ids, session_id)

    def set_config_overrides(self, session_id: str, overrides: Dict[str, Any]) -> None:
        manager = self._get_manager()
        if hasattr(manager, "patch_config"):
            manager.patch_config(overrides, session_id=session_id)

    def remove_config_overrides(self, session_id: str, keys: Iterable[str]) -> None:
        manager = self._get_manager()
        try:
            state = manager.get_session_state(session_id)
            changed = False
            for k in keys:
                if k in state.session_overrides:
                    del state.session_overrides[k]
                    changed = True
            if changed and hasattr(manager, "_persist_session_states"):
                manager._persist_session_states()
        except Exception as e:
            logger.debug(f"Failed to remove config overrides for session {session_id}: {e}")

    def clear_session(self, session_id: str) -> None:
        manager = self._get_manager()
        if hasattr(manager, "clear_session"):
            manager.clear_session(session_id)
