"""Whether a model may be downloaded while a server handles a request.

The CLI and the SDK download a model on first use, and say so. A server must
not: an unconfigured session searched over REST started a 2.3 GB BGE-M3
download inside the request, or, offline, answered ``500 Internal server
error`` while ``/health/ready`` reported the model as up. ``lasagna serve`` and
the MCP server switch the refusal on for their lifetime;
``allow_model_download_in_requests`` (``RAG_ALLOW_MODEL_DOWNLOAD_IN_REQUESTS``)
switches it off again.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional

_refusing = 0

_TRUTHY = {"1", "true", "yes", "on"}


class ModelNotDownloaded(RuntimeError):
    """A request needs a model that is not on disk, and the server will not fetch it."""

    def __init__(self, model: str, offline: bool, kind: str = "embedding model"):
        self.model = model
        self.offline = offline
        why = (
            "and Hugging Face offline mode is on (HF_HUB_OFFLINE is set)"
            if offline
            else "and a server does not download models inside a request"
        )
        super().__init__(f"The {kind} {model} is not downloaded, {why}.")


@contextmanager
def refuse_downloads_in_requests() -> Iterator[None]:
    """For a server's lifetime: a model that is not cached is refused, not fetched."""
    global _refusing
    _refusing += 1
    try:
        yield
    finally:
        _refusing -= 1


def offline() -> bool:
    """Whether Hugging Face offline mode is set."""
    return any(
        os.environ.get(var, "").strip().lower() in _TRUTHY
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def is_cached(model: str) -> bool:
    """Whether ``model`` loads without a download: a local folder or a cached hub copy."""
    if not model:
        return False
    if Path(model).expanduser().is_dir():
        return True
    # sentence-transformers reads a bare name ("all-MiniLM-L6-v2") as its own org's.
    names = [model] if "/" in model else [model, f"sentence-transformers/{model}"]
    try:
        from huggingface_hub import try_to_load_from_cache

        return any(isinstance(try_to_load_from_cache(n, "config.json"), str) for n in names)
    except Exception:
        return False


def _allowed() -> bool:
    try:
        from lasagna.core.config import rag_config

        return bool(rag_config.allow_model_download_in_requests)
    except Exception:
        return False


def refusal(model: str, kind: str = "embedding model") -> Optional[ModelNotDownloaded]:
    """Why ``model`` cannot be loaded here without a download, or None when it can."""
    if is_cached(model):
        return None
    if offline():
        return ModelNotDownloaded(model, offline=True, kind=kind)
    if _refusing and not _allowed():
        return ModelNotDownloaded(model, offline=False, kind=kind)
    return None


def check(model: str, kind: str = "embedding model") -> None:
    """Raise :class:`ModelNotDownloaded` when a server would have to download ``model``."""
    if not _refusing:
        return
    problem = refusal(model, kind)
    if problem is not None:
        raise problem


def session_configured(session_id: str) -> bool:
    """Whether ``session_id`` has a profile of its own (``lasagna configure``)."""
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        mgr = get_knowledge_config_manager(session_id=session_id)
        return mgr.get_active_profile(session_id=session_id) not in (None, "", "default")
    except Exception:
        return False


def sessions_with_documents() -> Optional[Dict[str, int]]:
    """``{session_id: document count}`` for every session with documents; None if unknown."""
    try:
        from lasagna.core.database import get_rag_database

        counters = get_rag_database().get_all_session_quota_counters()
    except Exception:
        return None
    return {s: c.get("documents", 0) for s, c in counters.items() if c.get("documents")}


def fix_for_session(session_id: str, model: str = "") -> str:
    """What to run so that ``session_id`` can be searched, naming the sessions that can."""
    populated = sessions_with_documents() or {}
    others = {s: n for s, n in populated.items() if s != session_id}
    parts = []
    if others and session_id not in populated:
        listed = ", ".join(f"{s} ({n})" for s, n in sorted(others.items()))
        parts.append(
            f"Session {session_id!r} has no documents. Sessions with documents: {listed}; "
            f"name one with the X-Session-ID header, RAG_SESSION_ID or -s."
        )
    parts.append(fix_for_model(session_id, model))
    return " ".join(parts)


def fix_for_model(session_id: str, model: str = "") -> str:
    """What to run so that ``session_id`` can use a model, or a smaller one.

    Offline, the way to keep the model is to download it while online; a
    server that refuses downloads can also be told to allow them.
    """
    configured = (
        f"Session {session_id!r} is configured with this model."
        if session_configured(session_id)
        else f"Session {session_id!r} has no profile, so it uses the default model."
    )
    download = f"hf download {model}" if model else "hf download <model>"
    allow = "set RAG_ALLOW_MODEL_DOWNLOAD_IN_REQUESTS=true for the server"
    if offline():
        unset = "unset HF_HUB_OFFLINE" + (f" and {allow}" if _refusing else "")
        keep = (
            f"To keep this model, download it once with network access ({download}), "
            f"or {unset} so it downloads on first use."
        )
    else:
        keep = f"To keep this model, download it first ({download}), or {allow}."
    return (
        f"{configured} For the 91 MB keyless model, run: lasagna configure --profile "
        f"speed -s {session_id} (in Python: KnowledgeClient(config=config_for_speed())). "
        f"{keep}"
    )
