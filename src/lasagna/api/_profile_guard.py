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

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional


class EmbeddingSizeConflict(ValueError):
    """The change embeds at a size the stored documents were not embedded at."""


class EmbeddingSizeUnverified(EmbeddingSizeConflict):
    """The store could not be read, so the change could not be checked."""


class DocumentsLeftBehind(EmbeddingSizeConflict):
    """The change would search another collection than the one holding the session's documents."""


class ProfileSwitchInProgress(EmbeddingSizeConflict):
    """A switch of the session that moves its documents is running, or stopped before it ended."""


#: Sessions whose switch this thread is running: its own writes go through.
_switching = threading.local()


@contextmanager
def switching(session_id: str) -> Iterator[None]:
    """Let this thread change ``session_id`` while its switch is recorded as running."""
    held: Dict[str, int] = getattr(_switching, "held", None) or {}
    _switching.held = held
    held[session_id] = held.get(session_id, 0) + 1
    try:
        yield
    finally:
        held[session_id] -= 1
        if not held[session_id]:
            del held[session_id]


def unfinished_switch(session_id: str) -> Optional[Dict[str, Any]]:
    """The recorded switch of ``session_id`` that has not ended, or None."""
    from lasagna.core.database import get_rag_database

    read = getattr(get_rag_database(), "get_profile_switch", None)
    if read is None:
        return None
    try:
        record = read(session_id)
    except Exception:  # noqa: BLE001 - a store without the table has no switch
        return None
    return record if isinstance(record, dict) else None


def switch_in_progress_message(record: Dict[str, Any], change: str) -> str:
    """Why ``change`` waits for the switch in ``record``, and what ends an interrupted one."""
    sid, profile, previous = record["session_id"], record["profile"], record["previous_profile"]
    started = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(float(record["started_at"])))
    return (
        f"{change} was refused: session {sid!r} is switching from profile {previous!r} to "
        f"{profile!r} (started {started}), which reindexes its documents. Retry when the "
        f"switch has ended. If it was interrupted (the process running it stopped), finish "
        f"it with lasagna configure --resume-switch -s {sid}, or go back with lasagna "
        f"configure --profile {previous} -s {sid} --reindex (over REST and MCP: that "
        f"switch with reindex=true)."
    )


def refuse_during_switch(session_id: str, change: str) -> None:
    """Raise ``ProfileSwitchInProgress`` while another thread or process switches the session.

    An upload, a reindex, a delete or a settings change during a switch that
    moves the session's documents raced it: an upload recorded the old
    collection and was left there when the switch ended, and a patch was
    overwritten by the switch's own. The thread running the switch is let
    through.
    """
    if session_id in (getattr(_switching, "held", None) or {}):
        return
    record = unfinished_switch(session_id)
    if record is not None:
        raise ProfileSwitchInProgress(switch_in_progress_message(record, change))


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


class DocumentsElsewhereWarning(UserWarning):
    """A search missed the session's documents that sit in another model's collection."""


#: ``(session, collection) -> (checked at, message)``: searches read this at
#: most once every few seconds per session, from the metadata database.
_elsewhere: Dict[tuple, tuple] = {}
_ELSEWHERE_TTL_SECONDS = 3.0


def model_of_collection(collection: str) -> tuple:
    """``(model, profile)`` that writes to ``collection``: its stamp, or the built-in
    profile whose model it is named after; ``profile`` is None when no built-in is."""
    from lasagna.core.config import collection_name_for_settings, get_rag_config
    from lasagna.core.database import get_rag_database
    from lasagna.core.session_config._profiles import ProfileMixin

    config = get_rag_config()
    model, profile = None, None
    for name, spec in ProfileMixin._PROFILE_SPECS.items():
        try:
            if collection_name_for_settings(spec) == collection:
                model = spec.get("embedding_model") or config.embedding_model
                profile = name
                break
        except Exception:  # noqa: BLE001 - only names a profile
            continue
    try:
        stamp = get_rag_database().get_collection_meta(
            collection, config.vector_backend or "milvus"
        )
    except Exception:  # noqa: BLE001 - the stamp only names the model
        stamp = None
    if stamp and stamp.get("embedding_model"):
        model = stamp["embedding_model"]
    return str(model or collection), profile


#: Settings that say which model a session embeds with. A switch to another
#: profile keeps a session's override of one, as it keeps any override that
#: does not only repeat the profile it leaves.
_EMBEDDING_KEYS = ("embedding_model", "embedding_dim", "embedding_truncate_dim")


def fixes_for_documents_elsewhere(session_id: str, back: Optional[str]) -> tuple:
    """``(reindex, switch_back)``: what every surface offers for documents in another
    model's collection; ``switch_back`` is None when no built-in profile indexed them.

    ``reindex`` reindexes them with the session's settings as they are, its
    overrides included (a switch to the profile it is on keeps them).
    ``switch_back`` makes the session search where they are: a switch to
    ``back``, the built-in profile whose model indexed them. A session that
    overrides its embedding model keeps that override across a switch, so the
    advice was a switch that changed nothing; then its overrides are reset too.
    """
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        state = get_knowledge_config_manager(session_id=session_id).get_session_state(session_id)
        active = state.active_profile or "default"
        overridden = [k for k in _EMBEDDING_KEYS if k in state.session_overrides]
    except Exception:  # noqa: BLE001 - only shapes the advice
        active, overridden = "<profile>", []
    reindex = f"lasagna configure --profile {active} -s {session_id} --reindex"
    if back is None:
        return reindex, None
    switch = f"lasagna configure --profile {back} -s {session_id}"
    if not overridden:
        return reindex, None if back == active else switch
    named = " and ".join(
        [", ".join(overridden[:-1]), overridden[-1]] if len(overridden) > 1 else overridden
    )
    reset = (
        f"reset the session's overrides, which set {named}, with POST "
        f"/api/config/knowledge/reset, client.config_manager.reset() in the SDK, or the "
        f"MCP tool reset_rag_config"
    )
    return reindex, reset if back == active else f"{reset}, then {switch}"


def _described(collection: str, model: str, other_model: str) -> str:
    """``model``'s collection, or the collection by name when both sides name one model."""
    return f"{model}'s collection" if model != other_model else f"collection {collection!r}"


def documents_elsewhere(session_id: str, collection: str, model: str) -> Optional[str]:
    """Why a search of ``collection`` cannot find some of the session's documents.

    None when every document of the session is in ``collection``. A reindex
    into another model through the SDK, or a switch with ``force``, left the
    documents in the old model's collection while the session searched the
    new one's, and every search returned an empty list with no word of why.
    """
    key = (session_id, collection)
    now = time.monotonic()
    cached = _elsewhere.get(key)
    if cached is not None and now - cached[0] < _ELSEWHERE_TTL_SECONDS:
        return cached[1]
    from lasagna.core.database import get_rag_database

    rows = [
        row
        for row in get_rag_database().get_documents(session_id)
        if str(row.get("status") or "").lower() in _SEARCHED_STATUSES
    ]
    missed = [r for r in rows if r.get("collection_name") not in (None, "", collection)]
    message = None
    if missed:
        others = sorted({str(r["collection_name"]) for r in missed})
        named = [model_of_collection(c) for c in others]
        back = next((profile for _, profile in named if profile), None)
        reindex, switch_back = fixes_for_documents_elsewhere(session_id, back)
        where = ", ".join(_described(c, m, model) for c, (m, _) in zip(others, named))
        here = _described(collection, model, named[0][0]) if len(named) == 1 else f"{model}'s"
        message = (
            f"{len(missed)} of {_plural(len(rows), 'document')} in session {session_id!r} "
            f"{'is' if len(missed) == 1 else 'are'} in {where}, and this search uses "
            f"{here}, so it cannot find {'it' if len(missed) == 1 else 'them'}. "
            f"Reindex them with {model} ({reindex}, or client.documents.reindex(document_id) "
            f"from a client on that model), or search with the model that indexed them"
            + (f" ({switch_back})." if switch_back else ".")
        )
    if message and (cached is None or cached[1] != message):
        from loguru import logger

        logger.warning(message)
    _elsewhere[key] = (now, message)
    return message


def forget_documents_elsewhere(session_id: Optional[str] = None) -> None:
    """Drop what searches remember about where a session's documents are."""
    for key in [k for k in _elsewhere if session_id is None or k[0] == session_id]:
        _elsewhere.pop(key, None)


def _model_of_documents(collection: str, before: Dict[str, Any], vector_backend: str) -> tuple:
    """``(model, size)`` the documents in ``collection`` were embedded with.

    From the collection's stamp, else the built-in profile it is named after,
    else the session's settings before the change. Naming the session's model
    instead made a refusal say the documents were embedded with the very model
    it refused.
    """
    from lasagna.core.database import get_rag_database
    from lasagna.core.session_config._profiles import ProfileMixin

    try:
        stamp = get_rag_database().get_collection_meta(collection, vector_backend)
    except Exception:  # noqa: BLE001 - the stamp only names the model
        stamp = None
    if stamp and stamp.get("embedding_model"):
        return stamp["embedding_model"], stamp.get("embedding_dim")
    model, profile = model_of_collection(collection)
    if profile is not None:
        spec = ProfileMixin._PROFILE_SPECS[profile]
        return spec.get("embedding_model") or model, spec.get("embedding_dim")
    return before.get("embedding_model"), _effective_dim(before)


def refuse_leaving_documents(
    *,
    session_id: str,
    before: Dict[str, Any],
    after: Dict[str, Any],
    change: str,
    kept: str,
    profile_config: Optional[Dict[str, Any]] = None,
    remedy: Optional[str] = None,
) -> None:
    """Raise when the session's documents are in a collection the change would stop searching.

    Each embedding model has its own collection, so a switch from ``speed``
    (all-MiniLM-L6-v2) to ``standard`` (bge-m3), or from bge-small to
    ``speed`` at the same 384 dimensions, moved search to an empty collection:
    `configure` printed "Applied profile" and every search returned nothing
    from the documents `status` still counted. The documents' collection is
    the one the metadata database recorded for each (the session's current
    one for a document from before per-model collections).

    ``profile_config`` is the profile a switch goes to, so the message names
    its model apart from a session override that would replace it.
    ``remedy`` replaces the fixes named (for one surface); it may name
    ``{new_model}`` and ``{them}``.
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
    old_model, old_dim = _model_of_documents(
        collection, before, after.get("vector_backend") or "milvus"
    )
    new_model, new_dim = after.get("embedding_model"), _effective_dim(after)
    new, old = new_model, old_model
    if old_model == new_model:
        # One model on both sides: the size, or else the collection, differs.
        if old_dim is not None and int(old_dim) != new_dim:
            new, old = (
                f"{new_model} at {new_dim} dimensions",
                f"{old_model} at {old_dim} dimensions",
            )
        else:
            new = f"{new_model} in collection {target!r}"
            old = f"{old_model} in collection {collection!r}"
    one = len(left) == 1
    them = "it" if one else "them"
    why = ""
    profile_model = (profile_config or {}).get("embedding_model")
    if profile_model and profile_model != new_model:
        why = (
            f" (the profile embeds with {profile_model}, and this session's override "
            f"embedding_model={new_model} stays in force across a switch)"
        )
    fixes = (
        remedy.format(new_model=new_model, them=them)
        if remedy
        else (
            f"Either use a new session for {new_model} and upload there, or switch and "
            f"reindex {them} in one step (lasagna configure --reindex, or reindex=true over "
            f"REST and MCP), which reindexes every document with {new_model} first and keeps "
            f"the session as it is if one fails. To switch anyway: lasagna configure --force, "
            f"or force=true over REST and MCP; until each document is reindexed "
            f"(client.documents.reindex(document_id) in the SDK, POST "
            f"/api/v1/documents/<id>/reindex or POST /api/knowledge/reindex over REST, or the "
            f"MCP tool reindex_document), searches return nothing from {them}."
        )
    )
    if why:
        fixes += (
            " To use the profile's own model instead, reset the session's overrides first "
            "(POST /api/config/knowledge/reset, client.config_manager.reset() in the SDK, or "
            "the MCP tool reset_rag_config)."
        )
    raise DocumentsLeftBehind(
        f"{change} would search with {new}{why}, but the "
        f"{_plural(len(left), 'document')} in session {session_id!r} "
        f"{'was' if one else 'were'} embedded with {old}, in that model's collection, so "
        f"no search would find {them}. Nothing was changed; profile {kept!r} is still "
        f"active. {fixes}"
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
