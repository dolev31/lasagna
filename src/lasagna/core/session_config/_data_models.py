"""Data models, constants, and utilities for session configuration.

Contains RAGProfile, KnowledgeConfigState, schema constants,
and the _deep_merge utility function.
"""

import copy
import os
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from lasagna.sdk.exceptions import BackendError, ProfileNotFoundError

# Knowledge module folder (for self-contained config)
KNOWLEDGE_MODULE_DIR = Path(__file__).parent.parent.parent

# =============================================================================
# Schema Version for Import/Export Compatibility
# =============================================================================
SCHEMA_VERSION = 2

# How long a process keeps an unused session in its memory (24 hours). The
# store keeps a session's settings for ``session_idle_seconds()``.
SESSION_TTL_SECONDS = 24 * 60 * 60

#: The GC policy's default ``session_idle_days``.
DEFAULT_SESSION_IDLE_DAYS = 30


def session_idle_seconds() -> Optional[float]:
    """How long the store keeps the settings of an unused session without documents.

    The GC policy's ``session_idle_days`` (``RAG_GC_SESSION_IDLE_DAYS``, else
    ``[rag.lifecycle]`` in the settings file, else 30), read as
    ``LifecyclePolicy.load()`` reads it without importing the lifecycle
    package, which loads the vector store's client. Every process used to
    delete settings unused for 24 hours, while reads never recorded a use: a
    session configured one day searched with the default model the next.
    None for 0, which keeps them, as ``document_ttl_days = 0`` disables expiry:
    0 deleted every idle session's settings within a minute, in use or not.
    """
    raw = os.environ.get("RAG_GC_SESSION_IDLE_DAYS")
    if raw is None:
        try:
            from lasagna.core.config import rag_config

            raw = rag_config._get_knowledge_setting(
                "rag", "lifecycle", "session_idle_days", default=DEFAULT_SESSION_IDLE_DAYS
            )
        except Exception:
            raw = DEFAULT_SESSION_IDLE_DAYS
    try:
        days = int(raw)
    except (TypeError, ValueError):
        days = DEFAULT_SESSION_IDLE_DAYS
    return days * 86400 if days > 0 else None


# Maximum number of sessions to keep in memory
MAX_CACHED_SESSIONS = 100

# Maximum number of saved custom profiles per owner
MAX_GLOBAL_PROFILES = 50

# A saved profile's config, serialized; the built-in ones are about 4.3 KB.
MAX_PROFILE_CONFIG_BYTES = 10_000

# The owner of profiles saved without a verified identity (auth off: one user).
ANONYMOUS = "__anonymous__"


#: Sessions whose settings stay in the process that set them (``keep_in_process``).
_kept_in_process: set = set()


def keep_in_process(session_id: str) -> None:
    """Keep ``session_id``'s settings in this process: never read from or written to the store.

    For the evaluation harness. Its sessions are named after the corpus
    (``bench_corpus_<hash>``) so that every run on a corpus reuses its vectors;
    with the persistent store two experiment processes on one corpus then
    shared one row of settings, and each read the other's arm settings in the
    agentic loop and at ingestion. Every manager of the process shares these
    sessions' settings, and they are exempt from the check that a change keeps
    the session's documents searchable: the harness names a collection per arm.
    """
    _kept_in_process.add(session_id)


def kept_in_process(session_id: Optional[str]) -> bool:
    """Whether ``session_id``'s settings stay in this process (see ``keep_in_process``)."""
    return bool(session_id) and session_id in _kept_in_process


class UnknownProfileError(ProfileNotFoundError, ValueError):
    """No profile of that name exists for the caller."""


class SettingsNotSaved(BackendError):
    """The settings store could not record a change; nothing was changed.

    Every surface answers it plainly (REST with a 503): a failed write used
    to be logged and answered as a success, and the process read the old
    row back within a second.
    """


def not_saved(
    what: str, exc: BaseException, *, verb: str = "saved", why: str = ""
) -> SettingsNotSaved:
    """``SettingsNotSaved`` for ``what`` (``the settings of session 'r1'``), naming why.

    ``verb``: what could not be done to it; ``why``: what that leaves unknown.
    """
    from lasagna.core.provenance import redact_paths

    reason = (redact_paths(str(exc)).strip().splitlines() or [""])[0][:300]
    return SettingsNotSaved(
        f"{what[0].upper()}{what[1:]} could not be {verb} ({type(exc).__name__}"
        f"{': ' + reason if reason else ''}){why}, so nothing was changed. Retry once the "
        f"metadata store answers (lasagna doctor and GET /api/v1/health/ready name it)."
    )


# Memory limits for session state
# Without these limits, a malicious or buggy client could cause OOM by storing
# unlimited citations/chunks. These limits are generous for normal use but
# prevent memory exhaustion attacks.
MAX_CITATIONS_PER_SESSION = 500  # Typical queries return 5-20 citations
MAX_CHUNKS_PER_SESSION = 500  # Matches citations limit
MAX_CHUNK_CONTENT_LENGTH = 10000  # ~10KB per chunk content


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RAGProfile:
    """A named RAG configuration profile."""

    name: str
    description: str
    config: Dict[str, Any]
    user_id: str = "__anonymous__"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config,
            "user_id": self.user_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGProfile":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            config=data.get("config", {}),
            user_id=data.get("user_id", "__anonymous__"),
        )


@dataclass
class KnowledgeConfigState:
    """Session state for knowledge/RAG configuration."""

    profiles: List[RAGProfile] = field(default_factory=list)
    active_profile: str = "default"
    session_overrides: Dict[str, Any] = field(default_factory=dict)
    defaults_source: str = "settings.lasagna.toml"
    user_id: str = "__anonymous__"
    # Master toggle for RAG - when False, all RAG operations should be disabled
    rag_enabled: bool = True
    # Focused document IDs from "/" picker - enforced at search time
    # Cleared each request in the main server, persisted only to share across processes
    focused_document_ids: List[str] = field(default_factory=list)
    # Last RAG search results - citations and chunks for frontend display
    # These are stored in memory only (not persisted) and updated on each search
    last_citations: List[Dict[str, Any]] = field(default_factory=list)
    last_chunks: List[Dict[str, Any]] = field(default_factory=list)
    # Internal: track file mtime to detect cross-process citation updates (not persisted)
    _citations_file_mtime: float = field(default=0.0, repr=False)
    # Timestamp of last access (for TTL-based cleanup)
    last_accessed: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "defaults_source": self.defaults_source,
            "profiles": [p.to_dict() for p in self.profiles],
            "active_profile": self.active_profile,
            "session_overrides": self.session_overrides,
            "rag_enabled": self.rag_enabled,
            "focused_document_ids": self.focused_document_ids,
            "last_accessed": self.last_accessed,
            "user_id": self.user_id,
            # Note: last_citations and last_chunks are intentionally not persisted
            # They are ephemeral per-request data
        }

    def get_active_profile(self) -> Optional[RAGProfile]:
        """Get the currently active profile."""
        for profile in self.profiles:
            if profile.name == self.active_profile:
                return profile
        return self.profiles[0] if self.profiles else None

    def get_resolved_config(self) -> Dict[str, Any]:
        """Get the resolved configuration (active profile + overrides)."""
        active = self.get_active_profile()
        if not active:
            return self.session_overrides.copy()

        resolved = copy.deepcopy(active.config)
        # Apply session overrides (deep merge)
        _deep_merge(resolved, self.session_overrides)
        return resolved


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Deep merge override into base dict (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _resolve(profile_config: Optional[Dict[str, Any]], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """The resolved config a profile and overrides would give, without applying them."""
    resolved = copy.deepcopy(profile_config or {})
    _deep_merge(resolved, copy.deepcopy(overrides or {}))
    return resolved


def _check_profile_size(config: Dict[str, Any]) -> None:
    """Refuse a profile config too large to keep for every owner."""
    import json

    size = len(json.dumps(config, default=str))
    if size > MAX_PROFILE_CONFIG_BYTES:
        raise ValueError(
            f"Profile config is {size} bytes; the limit is {MAX_PROFILE_CONFIG_BYTES}."
        )
