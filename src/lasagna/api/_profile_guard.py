"""Refuse a profile switch that would leave stored documents unsearchable.

Every session shares one collection, embedded at one size. Switching a session
to a profile with another embedding size was accepted on every surface (CLI,
REST, MCP), and every later upload and search in it failed with an error whose
remedies exist only in Python.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional, Tuple


class EmbeddingSizeConflict(ValueError):
    """The profile embeds at a size the stored documents were not embedded at."""


def stored_dimension_mismatch(
    session_id: str, vector_backend: Any = None
) -> Optional[Tuple[int, int]]:
    """(stored, configured) embedding sizes when stored documents disagree, else None."""
    from lasagna.api._collection_manager import CollectionManager
    from lasagna.core.config import rag_config

    config = rag_config.with_session(session_id)
    backend, owned = vector_backend, vector_backend is None
    try:
        if owned:
            backend = CollectionManager.create_vector_backend()
            backend.connect()
        # Both readers use only the backend they are handed.
        shim = SimpleNamespace(_default_backend=backend)
        if not CollectionManager.collection_has_documents(shim, config.collection_name):
            return None
        stored = CollectionManager.get_collection_embedding_dim(shim, config.collection_name)
    except Exception:
        return None  # no store yet, or unreadable: nothing to protect
    finally:
        if owned and backend is not None:
            try:
                backend.disconnect()
            except Exception:
                pass
    if stored is not None and int(stored) != int(config.embedding_dim):
        return int(stored), int(config.embedding_dim)
    return None


def set_profile_checked(
    manager: Any,
    profile_name: str,
    session_id: str,
    *,
    config: Optional[dict] = None,
    force: bool = False,
    vector_backend: Any = None,
) -> dict:
    """Switch profiles, or keep the current one and raise EmbeddingSizeConflict."""
    previous = manager.get_active_profile(session_id=session_id)
    result = manager.set_active_profile(profile_name, config=config, session_id=session_id)
    if force:
        return result
    mismatch = stored_dimension_mismatch(session_id, vector_backend)
    if mismatch is None:
        return result
    manager.set_active_profile(previous, session_id=session_id)
    stored, wanted = mismatch
    raise EmbeddingSizeConflict(
        f"Profile {profile_name!r} embeds at {wanted} dimensions, but the stored "
        f"documents were embedded at {stored}, so every search would fail. Kept "
        f"profile {previous!r}. To use {profile_name!r}, point RAG_DB_DIR at a new "
        f"directory and upload again, or switch with force."
    )
