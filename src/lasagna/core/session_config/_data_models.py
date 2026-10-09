"""Data models, constants, and utilities for session configuration.

Contains RAGProfile, KnowledgeConfigState, schema constants,
and the _deep_merge utility function.
"""

import copy
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Knowledge module folder (for self-contained config)
KNOWLEDGE_MODULE_DIR = Path(__file__).parent.parent.parent

# =============================================================================
# Schema Version for Import/Export Compatibility
# =============================================================================
SCHEMA_VERSION = 2

# Session TTL: sessions older than this are candidates for cleanup (24 hours)
SESSION_TTL_SECONDS = 24 * 60 * 60

# Maximum number of sessions to keep in memory
MAX_CACHED_SESSIONS = 100

# Maximum number of saved custom profiles per owner
MAX_GLOBAL_PROFILES = 50

# A saved profile's config, serialized; the built-in ones are about 4.3 KB.
MAX_PROFILE_CONFIG_BYTES = 10_000

# The owner of profiles saved without a verified identity (auth off: one user).
ANONYMOUS = "__anonymous__"


class UnknownProfileError(ValueError):
    """No profile of that name exists for the caller."""


# MEDIUM PRIORITY FIX: Memory limits for session state
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
