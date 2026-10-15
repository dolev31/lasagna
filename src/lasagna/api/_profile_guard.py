"""Refuse a config change that would leave stored documents unsearchable.

Every session shares one collection, embedded at one size. Switching a session
to a profile with another embedding size was accepted on every surface (CLI,
REST, MCP), and every later upload and search in it failed with an error whose
remedies exist only in Python.

`KnowledgeConfigManager` calls `refuse_embedding_change` before it applies any
write that changes the session's embedding size (a profile switch, a patch, a
save, an import, a reset), so no surface can reach that state by a route the
check does not see. It used to be called by three surfaces after the switch,
then undone, which missed the configurator's save and import, and lost the
user's overrides on the way back.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional


class EmbeddingSizeConflict(ValueError):
    """The change embeds at a size the stored documents were not embedded at."""


class EmbeddingSizeUnverified(EmbeddingSizeConflict):
    """The store could not be read, so the change could not be checked."""


def _in_process_backend() -> Any:
    """The vector backend this process already serves from, if it has one.

    Milvus Lite admits one process per store. Inside a server, opening a second
    handle would only work by accident of the in-process server cache, and a
    CLI next to a running server cannot open one at all.
    """
    from lasagna.api import rag_backend

    backend = rag_backend._instance
    if backend is not None and getattr(backend, "_initialized", False):
        return getattr(backend, "_vector_backend", None)
    return None


def stored_embedding_dim(collection_name: str, vector_backend: Any = None) -> Optional[int]:
    """The size the stored documents were embedded at; None when nothing is stored.

    Raises when the store exists and cannot be read. The caller must treat that
    as "cannot tell", never as "nothing to protect": a Milvus Lite store locked
    by a running server read as empty, and the switch went through.
    """
    from lasagna.api._collection_manager import CollectionManager

    backend = vector_backend or _in_process_backend()
    owned = backend is None
    if owned:
        backend = CollectionManager.create_vector_backend()
        if getattr(backend, "is_lite", False) and not _lite_store_exists(backend):
            return None  # never created, so nothing was ever stored
        backend.connect()
    try:
        if not backend.has_collection(collection_name):
            return None
        stats = backend.get_collection_stats(collection_name)
        rows = stats.get("row_count", 0) if isinstance(stats, dict) else stats.row_count
        if int(rows or 0) == 0:
            return None
        # The dimension reader takes a manager; it uses only the backend.
        shim = SimpleNamespace(_default_backend=backend)
        stored = CollectionManager.get_collection_embedding_dim(shim, collection_name)
        if stored is None:
            raise RuntimeError(f"could not read the embedding size of {collection_name!r}")
        return int(stored)
    finally:
        if owned:
            try:
                backend.disconnect()
            except Exception:
                pass


def _lite_store_exists(backend: Any) -> bool:
    import os

    path = getattr(backend, "_db_path", None)
    return bool(path) and os.path.exists(path)


def _effective_dim(resolved: Dict[str, Any]) -> int:
    from lasagna.core.config import get_rag_config

    dim = resolved.get("embedding_dim")
    return int(dim if dim is not None else get_rag_config().embedding_dim)


def refuse_embedding_change(
    *,
    before: Dict[str, Any],
    after: Dict[str, Any],
    change: str,
    kept: str,
    vector_backend: Any = None,
) -> None:
    """Raise unless the session would still search what is stored after a change.

    ``before`` and ``after`` are the session's resolved config now and as the
    change would leave it; ``change`` names the change for the message and
    ``kept`` the profile that stays active when it is refused. A change that
    keeps the embedding size never touches the store.
    """
    wanted, current = _effective_dim(after), _effective_dim(before)
    if wanted == current:
        return
    from lasagna.core.config import collection_name_for_settings

    # The collection the session would use after the change: each embedding
    # model has its own, so this is usually not the one it uses now.
    collection = collection_name_for_settings(after)
    remedy = (
        "Nothing was changed; profile {kept!r} is still active. To embed at "
        "{wanted}, give it its own store (point RAG_DB_DIR at a new directory) "
        "and upload again."
    ).format(kept=kept, wanted=wanted)
    try:
        stored = stored_embedding_dim(collection, vector_backend)
    except Exception as exc:
        raise EmbeddingSizeUnverified(
            f"{change} would embed at {wanted} dimensions, and the vector store could "
            f"not be read to check that against the stored documents ({exc}). "
            f"{remedy}"
        ) from exc
    if stored is not None and stored != wanted:
        raise EmbeddingSizeConflict(
            f"{change} would embed at {wanted} dimensions, but the stored documents "
            f"were embedded at {stored}, so every search and upload would fail. "
            f"{remedy}"
        )
