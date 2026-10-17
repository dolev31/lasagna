"""Refuse a config change that would leave stored documents unsearchable.

Each embedding model has its own collection, which every session on that
model shares, and a collection holds vectors of one size. Switching a session to
a profile with another embedding size, onto a collection already holding
vectors of a different size (an explicit ``collection_name``, or another model
at the same name), was accepted on every surface (CLI, REST, MCP), and every
later upload and search in it failed with an error whose remedies exist only in
Python.

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


class DocumentsLeftBehind(EmbeddingSizeConflict):
    """The change would search another collection than the one holding the session's documents."""


#: Documents with vectors, or about to have them; a failed upload stored none.
_SEARCHED_STATUSES = frozenset({"uploading", "processing", "indexing", "ready"})


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def session_has_documents(session_id: str) -> bool:
    """Whether the metadata database holds documents of ``session_id`` with vectors.

    Read without the vector store. A database that cannot be read counts as
    holding some, so a check that relies on this is never skipped by mistake.
    """
    from lasagna.core.database import get_rag_database

    try:
        rows = get_rag_database().get_documents(session_id)
    except Exception:  # noqa: BLE001 - unknown is "may hold some"
        return True
    return any(str(row.get("status") or "").lower() in _SEARCHED_STATUSES for row in rows)


def refuse_leaving_documents(
    *,
    session_id: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
    change: str,
    kept: str,
) -> None:
    """Raise when the session's documents are in a collection the change would stop searching.

    Each embedding model has its own collection, so a switch from ``speed``
    (all-MiniLM-L6-v2) to ``standard`` (bge-m3), or from bge-small to
    ``speed`` at the same 384 dimensions, moved search to an empty collection:
    `configure` printed "Applied profile" and every search returned nothing
    from the documents `status` still counted. The documents' collection is
    the one the metadata database recorded for each (the session's current
    one for a document from before per-model collections).
    """
    from lasagna.core.config import collection_name_for_settings
    from lasagna.core.database import get_rag_database

    db = get_rag_database()
    rows = [
        row
        for row in db.get_documents(session_id)
        if str(row.get("status") or "").lower() in _SEARCHED_STATUSES
    ]
    if not rows:
        return
    target = collection_name_for_settings(after)
    current = collection_name_for_settings(before)
    left = [row for row in rows if (row.get("collection_name") or current) != target]
    if not left:
        return
    collection = left[0].get("collection_name") or current
    try:
        stamp = db.get_collection_meta(collection, after.get("vector_backend") or "milvus")
    except Exception:  # noqa: BLE001 - the stamp only names the model
        stamp = None
    old_model = (stamp or {}).get("embedding_model") or before.get("embedding_model")
    new_model = after.get("embedding_model")
    one = len(left) == 1
    them = "it" if one else "them"
    raise DocumentsLeftBehind(
        f"{change} would search with {new_model}, but the "
        f"{_plural(len(left), 'document')} in session {session_id!r} "
        f"{'was' if one else 'were'} embedded with {old_model}, in that model's "
        f"collection, so no search would find {them}. Nothing was changed; profile "
        f"{kept!r} is still active. Either use a new session for {new_model} and "
        f"upload there, or switch and reindex {them} in one step (lasagna configure "
        f"--reindex, or reindex=true over REST and MCP), which reindexes every document "
        f"with {new_model} first and keeps the session as it is if one fails. To switch "
        f"anyway: lasagna configure --force, or force=true over REST and MCP; until "
        f"each document is reindexed (client.documents.reindex(document_id) in the "
        f"SDK, POST /api/v1/documents/<id>/reindex or POST /api/knowledge/reindex over "
        f"REST, or the MCP tool reindex_document), searches return nothing from {them}."
    )


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
    from lasagna.api._collection_manager import CollectionManager, _flush_before_counting

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
        _flush_before_counting(backend, collection_name)  # see collection_is_empty
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
