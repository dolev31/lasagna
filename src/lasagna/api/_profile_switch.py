"""Switch a session to another profile, reindexing its documents first when asked.

`lasagna configure --profile standard -s r1 --reindex`, with BGE-M3 not
downloaded, switched the session, failed to reindex any of its three
documents, and left it on a profile whose model could not load: every search
then failed, and `doctor` reported nothing blocking. A switch that moves
documents now runs in this order, on every surface (the CLI, REST and MCP):

1. The new profile's models must load here: cached, or downloadable under
   the download policy. Otherwise nothing changes.
2. Every document is reindexed with the new profile's settings while the
   session stays on its old profile (``RAGConfig.with_session_settings``).
3. Only when all of them are reindexed does the session switch. If one
   fails, those already moved are reindexed back with the old settings, the
   session keeps its old profile, and the error names each document that
   failed and why.

The switch holds the session's lock in the store from start to end, so two
switches of one session run one after the other, in any processes. While it
runs, a row in ``profile_switches`` says so: an upload, a reindex, a delete or
a settings change of the session from anywhere else is refused with
``ProfileSwitchInProgress``. Interrupted (Ctrl-C, a stopped pod), the documents
already moved are moved back before the interruption goes on; a process that
died leaves the row, which ``doctor`` reports and every write names, and the
next switch of the session with reindex (``lasagna configure --resume-switch``
repeats the interrupted one) moves those documents too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from lasagna.core.session_config._data_models import ANONYMOUS, SettingsNotSaved
from lasagna.sdk.exceptions import ConfigurationError, KnowledgeError


class ProfileCannotLoad(ConfigurationError, ValueError):
    """The new profile's models cannot be loaded here; nothing was changed."""


class ReindexFailed(KnowledgeError, RuntimeError):
    """A document could not be reindexed with the new profile; the session kept its old one."""

    def __init__(self, message: str, report: "SwitchReport"):
        super().__init__(message)
        self.report = report


@dataclass
class SwitchReport:
    """What a switch did: the documents reindexed, and any that failed (with why)."""

    profile: str
    previous: str
    session_id: str
    embedding_model: str = ""
    reindexed: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)
    #: Moved back to the old profile's settings after a failure.
    restored: List[str] = field(default_factory=list)
    #: Could not be moved back: still in the new profile's collection, or without chunks.
    stranded: List[Tuple[str, str]] = field(default_factory=list)
    #: Failed after their old chunks were removed: they have none until reindexed back.
    unrecorded: List[str] = field(default_factory=list)
    total: int = 0
    #: Whether the switch was asked to reindex.
    reindex: bool = False
    #: The session's own embedding settings that stay in force over the profile's.
    own_settings: Dict[str, Any] = field(default_factory=dict)
    #: What the switch went through with: a reranker that is not downloaded.
    warnings: List[str] = field(default_factory=list)

    def applied(self) -> str:
        """``Applied profile: speed``, naming the session's own model settings that stay."""
        line = f"Applied profile: {self.profile}"
        if not self.own_settings:
            return line
        named = ", ".join(f"{k}={v}" for k, v in self.own_settings.items())
        return (
            f"{line} with {named} from this session's settings (for the profile's own "
            f"model: lasagna configure --profile {self.profile} --reset -s {self.session_id} "
            f"--reindex)"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "previous_profile": self.previous,
            "session_id": self.session_id,
            "embedding_model": self.embedding_model,
            "documents": self.total,
            "reindexed": list(self.reindexed),
            "failed": [{"document_id": d, "error": e} for d, e in self.failed],
            "restored": list(self.restored),
            "stranded": [{"document_id": d, "error": e} for d, e in self.stranded],
            "own_settings": dict(self.own_settings),
            "warnings": list(self.warnings),
        }


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _setting(settings: Dict[str, Any], key: str) -> Any:
    from lasagna.core.config import get_rag_config

    return settings[key] if key in settings else getattr(get_rag_config(), key, None)


def _download_fix(problem: Any, model: str) -> str:
    from lasagna.core.setup_text import download_command

    if problem.offline:
        return f"download it once with network access: {download_command(model)}"
    return (
        f"download it first ({download_command(model)}), or set "
        f"RAG_ALLOW_MODEL_DOWNLOAD_IN_REQUESTS=true for the server"
    )


def unloadable_models(settings: Dict[str, Any]) -> List[str]:
    """Why the embedding model ``settings`` need cannot load here; empty when it can.

    Cached, or downloadable under the download policy (offline mode, a server
    that does not download inside a request). A cached fallback embedder
    counts, as the embedder loads it then. The reranker is not one of them: a
    search skips one that is not downloaded and says so (``degraded_stages``
    ``reranker``, results in retrieval order), so a switch to its profile goes
    through with the same warning (``reranker_warning``).
    """
    from lasagna.core import model_downloads

    model = _setting(settings, "embedding_model")
    problem = model_downloads.refusal(model) if model else None
    if problem is None:
        return []
    fallback = _setting(settings, "embedding_model_fallback")
    if problem.offline and fallback and model_downloads.is_cached(fallback):
        return []
    return [f"{problem} To use it, {_download_fix(problem, model)}."]


def reranker_warning(settings: Dict[str, Any]) -> Optional[str]:
    """What searches with ``settings`` will lack when their reranker cannot load here, or None."""
    from lasagna.core import model_downloads

    if not _setting(settings, "enable_reranking"):
        return None
    reranker = _setting(settings, "reranker_model")
    problem = model_downloads.refusal(reranker, kind="reranker model") if reranker else None
    if problem is None:
        return None
    return (
        f"{problem} Until it is, searches skip it and return results in retrieval order, "
        f"not reranked (degraded_stages: reranker); to rerank, {_download_fix(problem, reranker)}."
    )


def _searched_documents(session_id: str) -> List[Dict[str, Any]]:
    from lasagna.api._profile_guard import _SEARCHED_STATUSES
    from lasagna.core.database import get_rag_database

    return [
        row
        for row in get_rag_database().get_documents(session_id)
        if str(row.get("status") or "").lower() in _SEARCHED_STATUSES
    ]


def switch_profile(
    manager: Any,
    profile: str,
    *,
    session_id: str,
    user_id: str = ANONYMOUS,
    force: bool = False,
    reindex: bool = False,
    config: Optional[Dict[str, Any]] = None,
    backend: Any = None,
    reset: bool = False,
) -> SwitchReport:
    """Switch ``session_id`` to ``profile``; with ``reindex``, move its documents first.

    ``config`` and ``reset`` are what ``set_active_profile`` takes (``reset``
    drops the session's own settings). Without ``reindex`` this is
    ``manager.set_active_profile`` (``force`` skips the check that the
    session's documents stay searchable), plus, when forced, the check that
    the new profile's models load. With ``reindex``, see the module
    docstring. ``backend`` is the ``RAGBackend`` that reindexes (default:
    this process's).

    Raises:
        UnknownProfileError: no such profile.
        ProfileCannotLoad: a model of the new profile cannot load; nothing changed.
        ReindexFailed: a document could not be reindexed; the session kept
            its old profile (``exc.report`` says what happened to each).
        EmbeddingSizeConflict: refused without ``force``/``reindex``.
        ProfileSwitchInProgress: without ``reindex``, while another switch
            of the session runs or after one stopped before it ended.
    """
    if not reindex:
        return _switch_without_reindex(
            manager, profile, session_id, user_id=user_id, force=force, config=config, reset=reset
        )
    from lasagna.api._profile_guard import switching
    from lasagna.core.session_config._session_state import session_lock

    # The whole switch holds the session's lock: a second switch waits for
    # this one, and an upload's row is saved before it starts or after it ends.
    with session_lock(session_id), switching(session_id):
        return _switch_with_reindex(
            manager,
            profile,
            session_id,
            user_id=user_id,
            config=config,
            backend=backend,
            reset=reset,
        )


def _report(
    manager: Any, profile: str, session_id: str, user_id: str, config, reset: bool = False
) -> tuple:
    state = manager.get_session_state(session_id)
    previous = state.active_profile
    resolved, chosen = manager.preview_active_profile(
        profile, session_id, config=config, user_id=user_id, reset=reset
    )
    # What search will run with: the session's choice over the process's own
    # settings. The profile's whole configuration carries the packaged
    # settings file's models, so a server started with RAG_RERANKER_MODEL set
    # refused a forced switch for a reranker it never loads.
    report = SwitchReport(
        profile=profile,
        previous=previous,
        session_id=session_id,
        embedding_model=str(_setting(chosen, "embedding_model") or ""),
    )
    profile_config = manager.preview_active_profile(
        profile, session_id, config=config, user_id=user_id, reset=True
    )[0]
    from lasagna.api._profile_guard import _EMBEDDING_KEYS

    report.own_settings = {
        k: resolved[k]
        for k in _EMBEDDING_KEYS
        if k in resolved and profile_config.get(k) != resolved[k] and not config
    }
    return report, resolved, chosen


def _refuse_unloadable(
    report: SwitchReport, resolved: Dict[str, Any], now: Dict[str, Any], *, settings: bool = False
) -> None:
    """Raise ``ProfileCannotLoad`` when a model ``resolved`` needs cannot load here.

    ``now`` is the session's resolved config, to say what is true of its
    documents (see ``documents_now``); ``settings``: the switch brings
    settings of its own (an SDK client's).
    """
    problems = unloadable_models(resolved)
    rerank = reranker_warning(resolved)
    if problems:
        from lasagna.api._profile_guard import documents_now

        state = documents_now(report.session_id, now)
        what = f"Profile {report.profile!r}" + (" with these settings" if settings else "")
        raise ProfileCannotLoad(
            f"{what} cannot be used here: {' '.join(problems)}"
            + (f" {rerank}" if rerank else "")
            + f" Nothing was changed; session {report.session_id!r} stays on profile "
            f"{report.previous!r}." + (f" {state}" if state else "")
        )
    if rerank:
        logger.warning(rerank)
        report.warnings.append(rerank)


def _switch_without_reindex(
    manager: Any,
    profile: str,
    session_id: str,
    *,
    user_id: str,
    force: bool,
    config,
    reset: bool = False,
) -> SwitchReport:
    report, _, chosen = _report(manager, profile, session_id, user_id, config, reset)
    if force:
        _refuse_unloadable(
            report, chosen, manager.get_resolved_config(session_id), settings=bool(config)
        )
    else:
        rerank = reranker_warning(chosen)
        if rerank:
            logger.warning(rerank)
            report.warnings.append(rerank)
    manager.set_active_profile(
        profile, config=config, session_id=session_id, user_id=user_id, force=force, reset=reset
    )
    return report


def _store():
    from lasagna.core.database import get_rag_database

    return get_rag_database()


def _switch_with_reindex(
    manager: Any,
    profile: str,
    session_id: str,
    *,
    user_id: str,
    config,
    backend: Any,
    reset: bool = False,
) -> SwitchReport:
    """Under the session's lock: reindex every document, then switch (see the module docstring)."""
    from lasagna.api._profile_guard import forget_documents_elsewhere
    from lasagna.core.config import get_rag_config

    report, _, chosen = _report(manager, profile, session_id, user_id, config, reset)
    _refuse_unloadable(
        report, chosen, manager.get_resolved_config(session_id), settings=bool(config)
    )

    # Oldest first, so a rerun after a failure meets the documents in the same order.
    rows = sorted(
        _searched_documents(session_id),
        key=lambda r: (str(r.get("upload_date") or ""), str(r["document_id"])),
    )
    report.total, report.reindex = len(rows), True
    if backend is None:
        from lasagna.api.rag_backend import get_rag_backend

        backend = get_rag_backend()
    target = get_rag_config().with_session_settings(session_id, chosen)
    if target.collection_name != get_rag_config().with_session(session_id).collection_name:
        from lasagna.api._profile_guard import DocumentsLeftBehind, archived_left_behind

        archived = archived_left_behind(
            session_id, target.collection_name, f"Profile {profile!r}", report.previous
        )
        if archived:
            raise DocumentsLeftBehind(archived)
    # A switch that stopped before it ended (its process died) left documents
    # in its own target's collection: this switch moves them too, and back if
    # it fails.
    interrupted = getattr(_store(), "get_profile_switch", lambda _sid: None)(session_id)
    left_behind = []
    if interrupted and interrupted.get("target_collection"):
        left_behind = [
            r["document_id"]
            for r in rows
            if r.get("collection_name") == interrupted["target_collection"]
        ]
    begin = getattr(_store(), "begin_profile_switch", None)
    if begin is not None:
        begin(
            session_id,
            profile,
            report.previous,
            {
                "config": config or None,
                "user_id": user_id,
                "target_collection": target.collection_name,
                "reset": reset,
            },
        )
    settled = False
    try:
        for row in rows:
            document_id = row["document_id"]
            try:
                backend.reindex_document(document_id, session_id, config=target)
                report.reindexed.append(document_id)
            except Exception as exc:  # noqa: BLE001 - reported per document
                logger.warning(f"Reindex of {document_id} with profile {profile!r} failed: {exc}")
                report.failed.append((document_id, _plain_reason(exc, document_id, session_id)))
                if _chunks_lost(exc):
                    report.unrecorded.append(document_id)
                break
        if report.failed:
            _restore(backend, session_id, report, left_behind)
            settled = True
            raise ReindexFailed(_failure_message(report), report)
        manager.set_active_profile(
            profile, config=config, session_id=session_id, user_id=user_id, force=True, reset=reset
        )
        settled = True
    except BaseException as exc:
        # Interrupted (Ctrl-C, SystemExit), or the switch itself failed (its
        # settings were not saved): the documents already moved go back
        # first. Interrupted again while they do, the row stays, for doctor
        # and the next switch.
        if not settled:
            _restore(backend, session_id, report, left_behind)
            settled = True
            if isinstance(exc, SettingsNotSaved):
                raise SettingsNotSaved(f"{exc} {_undone(report)}") from exc
        raise
    finally:
        forget_documents_elsewhere(session_id)
        end = getattr(_store(), "end_profile_switch", None)
        if settled and end is not None:
            try:
                end(session_id)
            except Exception as exc:  # noqa: BLE001 - never hides why the switch stopped
                logger.warning(
                    f"The record of the switch of session {session_id!r} could not be "
                    f"removed ({type(exc).__name__}), so its writes are refused until it is: "
                    f"lasagna configure --resume-switch -s {session_id} ends it."
                )
    return report


def resume_switch(manager: Any, session_id: str, *, backend: Any = None) -> SwitchReport:
    """Repeat the switch of ``session_id`` that stopped before it ended.

    Raises ``ValueError`` when the session has none.
    """
    from lasagna.api._profile_guard import unfinished_switch

    record = unfinished_switch(session_id)
    if record is None:
        raise ValueError(f"Session {session_id!r} has no profile switch to resume.")
    return switch_profile(
        manager,
        record["profile"],
        session_id=session_id,
        user_id=record.get("user_id") or ANONYMOUS,
        reindex=True,
        config=record.get("config") or None,
        backend=backend,
        reset=bool(record.get("reset")),
    )


def _plain_reason(exc: BaseException, document_id: str, session_id: str) -> str:
    """Why a document's reindex failed, as REST's reindex route says it: never a trace.

    The raw exception text reached the 409 of every surface, internal paths and
    library messages included; it is in the log line above instead.
    """
    from lasagna.api.rest.routes.documents import reindex_failure

    return str(reindex_failure(exc, document_id, session_id).detail)


def _restore(
    backend: Any, session_id: str, report: SwitchReport, left_behind: List[str] = ()
) -> None:
    """Reindex the documents already moved back with the session's own settings.

    ``left_behind``: those an interrupted switch had moved, which go back too.
    """
    moved = list(report.reindexed) + [
        d for d in [*left_behind, *report.unrecorded] if d not in report.reindexed
    ]
    # A document whose reindex failed is where it was, unless that was where
    # the interrupted switch had left it, or its old chunks went with the
    # failure (they were removed before the new copy failed to be recorded).
    failed = {d for d, _ in report.failed} - set(left_behind) - set(report.unrecorded)
    for document_id in moved:
        if document_id in failed:
            continue
        try:
            backend.reindex_document(document_id, session_id)
            report.restored.append(document_id)
        except Exception as exc:  # noqa: BLE001 - reported per document
            logger.warning(f"Could not move {document_id} back after a failed switch: {exc}")
            report.stranded.append((document_id, _plain_reason(exc, document_id, session_id)))
    report.reindexed = []


def _undone(report: SwitchReport) -> str:
    """What a switch stopped by a failed write left: the documents moved back, or not."""
    lines = [f"The switch to profile {report.profile!r} was undone."]
    if report.restored:
        lines.append(
            f"The {_plural(len(report.restored), 'document')} already reindexed "
            f"{'was' if len(report.restored) == 1 else 'were'} moved back"
            + (", so every document is searchable as before." if not report.stranded else ".")
        )
    for stranded, why in report.stranded:
        lines.append(
            f"{stranded} could not be moved back ({why}) and is searched only with "
            f"{report.embedding_model}: run lasagna configure --profile {report.previous} "
            f"-s {report.session_id} --reindex to reindex it."
        )
    return " ".join(lines)


def _chunks_lost(exc: BaseException) -> bool:
    """Whether a reindex failed after removing the document's old chunks."""
    from lasagna.api._document_pipeline import ReplacementNotRecorded

    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ReplacementNotRecorded):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _failure_message(report: SwitchReport) -> str:
    """Why the switch was not applied, and what is true of each document now.

    It said "every document is searchable as before" whenever one had been
    moved back, also when another could not be, or had lost its chunks.
    """
    document_id, error = report.failed[0]
    lines = [
        f"Profile {report.profile!r} was not applied; session {report.session_id!r} stays "
        f"on profile {report.previous!r}. Reindexing its "
        f"{_plural(report.total, 'document')} with {report.embedding_model} failed for "
        f"{document_id}: {error}"
    ]
    back = [d for d in report.restored if d not in report.unrecorded]
    if back:
        lines.append(
            f"The {_plural(len(back), 'document')} already reindexed "
            f"{'was' if len(back) == 1 else 'were'} moved back"
            + (", so every document is searchable as before." if not report.stranded else ".")
        )
    for lost in report.unrecorded:
        if lost in report.restored:
            lines.append(
                f"{lost} lost its chunks when its new copy could not be recorded, and was "
                f"reindexed back with profile {report.previous!r}."
            )
    for stranded, why in report.stranded:
        where = (
            "has no chunks, so no search finds it"
            if stranded in report.unrecorded
            else f"is searched only with {report.embedding_model}"
        )
        lines.append(
            f"{stranded} could not be moved back ({why}) and {where}: run lasagna configure "
            f"--profile {report.previous} -s {report.session_id} --reindex to reindex it."
        )
    lines.append("Fix the cause above and run the switch again.")
    return " ".join(lines)


def summary(report: SwitchReport) -> Optional[str]:
    """``Reindexed 3 of 3 documents in session r1 with BAAI/bge-m3.``, and any warning.

    None without a reindex or a warning.
    """
    warned = " ".join(report.warnings)
    if not report.reindex:
        return warned or None
    done = _reindexed(report)
    return f"{done} {warned}" if warned else done


def _reindexed(report: SwitchReport) -> str:
    if not report.total:
        return f"Session {report.session_id} has no documents to reindex."
    own = (
        f" (embedding_model from this session's settings, over profile {report.profile}'s)"
        if "embedding_model" in report.own_settings
        else ""
    )
    return (
        f"Reindexed {len(report.reindexed)} of {_plural(report.total, 'document')} in "
        f"session {report.session_id} with {report.embedding_model}{own}."
    )
