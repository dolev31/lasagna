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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from lasagna.core.session_config._data_models import ANONYMOUS


class ProfileCannotLoad(ValueError):
    """The new profile's models cannot be loaded here; nothing was changed."""


class ReindexFailed(RuntimeError):
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
    #: Could not be moved back: still in the new profile's collection.
    stranded: List[Tuple[str, str]] = field(default_factory=list)
    total: int = 0
    #: Whether the switch was asked to reindex.
    reindex: bool = False

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
        }


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _setting(settings: Dict[str, Any], key: str) -> Any:
    from lasagna.core.config import get_rag_config

    return settings[key] if key in settings else getattr(get_rag_config(), key, None)


def unloadable_models(settings: Dict[str, Any]) -> List[str]:
    """Why each model ``settings`` need cannot load here; empty when all can.

    The embedding model, and the reranker when reranking is on: cached, or
    downloadable under the download policy (offline mode, a server that does
    not download inside a request). A cached fallback embedder counts, as the
    embedder loads it then.
    """
    from lasagna.core import model_downloads
    from lasagna.core.setup_text import download_command

    def fix(problem: Any, model: str) -> str:
        if problem.offline:
            return f"{problem} To use it, download it once with network access: {download_command(model)}."
        return (
            f"{problem} To use it, download it first ({download_command(model)}), or set "
            f"RAG_ALLOW_MODEL_DOWNLOAD_IN_REQUESTS=true for the server."
        )

    problems = []
    model = _setting(settings, "embedding_model")
    problem = model_downloads.refusal(model) if model else None
    if problem is not None:
        fallback = _setting(settings, "embedding_model_fallback")
        if not (problem.offline and fallback and model_downloads.is_cached(fallback)):
            problems.append(fix(problem, model))
    if _setting(settings, "enable_reranking"):
        reranker = _setting(settings, "reranker_model")
        problem = model_downloads.refusal(reranker, kind="reranker model") if reranker else None
        if problem is not None:
            problems.append(fix(problem, reranker))
    return problems


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
) -> SwitchReport:
    """Switch ``session_id`` to ``profile``; with ``reindex``, move its documents first.

    ``config`` is what ``set_active_profile`` takes with a profile. Without
    ``reindex`` this is ``manager.set_active_profile`` (``force``
    skips the check that the session's documents stay searchable), plus, when
    forced, the check that the new profile's models load. With ``reindex``,
    see the module docstring. ``backend`` is the ``RAGBackend`` that
    reindexes (default: this process's).

    Raises:
        UnknownProfileError: no such profile.
        ProfileCannotLoad: a model of the new profile cannot load; nothing changed.
        ReindexFailed: a document could not be reindexed; the session kept
            its old profile (``exc.report`` says what happened to each).
        EmbeddingSizeConflict: refused without ``force``/``reindex``.
    """
    state = manager.get_session_state(session_id)
    previous = state.active_profile
    resolved, chosen = manager.preview_active_profile(
        profile, session_id, config=config, user_id=user_id
    )
    report = SwitchReport(
        profile=profile,
        previous=previous,
        session_id=session_id,
        embedding_model=str(_setting(resolved, "embedding_model") or ""),
    )
    if force or reindex:
        problems = unloadable_models(resolved)
        if problems:
            raise ProfileCannotLoad(
                f"Profile {profile!r} cannot be used here: {' '.join(problems)} Nothing was "
                f"changed; session {session_id!r} stays on profile {previous!r}, and its "
                f"documents stay searchable."
            )
    if not reindex:
        manager.set_active_profile(
            profile, config=config, session_id=session_id, user_id=user_id, force=force
        )
        return report

    # Oldest first, so a rerun after a failure meets the documents in the same order.
    rows = sorted(
        _searched_documents(session_id),
        key=lambda r: (str(r.get("upload_date") or ""), str(r["document_id"])),
    )
    report.total, report.reindex = len(rows), True
    if backend is None:
        from lasagna.api.rag_backend import get_rag_backend

        backend = get_rag_backend()
    from lasagna.core.config import get_rag_config

    target = get_rag_config().with_session_settings(session_id, chosen)
    for row in rows:
        document_id = row["document_id"]
        try:
            backend.reindex_document(document_id, session_id, config=target)
            report.reindexed.append(document_id)
        except Exception as exc:  # noqa: BLE001 - reported per document
            logger.warning(f"Reindex of {document_id} with profile {profile!r} failed: {exc}")
            report.failed.append((document_id, str(exc)))
            break
    if report.failed:
        _restore(backend, session_id, report)
        raise ReindexFailed(_failure_message(report), report)
    manager.set_active_profile(
        profile, config=config, session_id=session_id, user_id=user_id, force=True
    )
    return report


def _restore(backend: Any, session_id: str, report: SwitchReport) -> None:
    """Reindex the documents already moved back with the session's own settings."""
    for document_id in report.reindexed:
        try:
            backend.reindex_document(document_id, session_id)
            report.restored.append(document_id)
        except Exception as exc:  # noqa: BLE001 - reported per document
            logger.warning(f"Could not move {document_id} back after a failed switch: {exc}")
            report.stranded.append((document_id, str(exc)))
    report.reindexed = []


def _failure_message(report: SwitchReport) -> str:
    document_id, error = report.failed[0]
    lines = [
        f"Profile {report.profile!r} was not applied; session {report.session_id!r} stays "
        f"on profile {report.previous!r}. Reindexing its "
        f"{_plural(report.total, 'document')} with {report.embedding_model} failed for "
        f"{document_id}: {error}"
    ]
    if report.restored:
        lines.append(
            f"The {_plural(len(report.restored), 'document')} already reindexed "
            f"{'was' if len(report.restored) == 1 else 'were'} moved back, so every document "
            f"is searchable as before."
        )
    for stranded, why in report.stranded:
        lines.append(
            f"{stranded} could not be moved back ({why}) and is searched only with "
            f"{report.embedding_model}: run lasagna configure --profile {report.previous} "
            f"-s {report.session_id} --reindex to reindex it."
        )
    lines.append("Fix the cause above and run the switch again.")
    return " ".join(lines)


def summary(report: SwitchReport) -> Optional[str]:
    """``Reindexed 3 of 3 documents in session r1 with BAAI/bge-m3.``; None without a reindex."""
    if not report.reindex:
        return None
    if not report.total:
        return f"Session {report.session_id} has no documents to reindex."
    return (
        f"Reindexed {len(report.reindexed)} of {_plural(report.total, 'document')} in "
        f"session {report.session_id} with {report.embedding_model}."
    )
