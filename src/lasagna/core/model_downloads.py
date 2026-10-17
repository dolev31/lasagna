"""Whether a model may be downloaded while a server handles a request.

The CLI and the SDK download a model on first use, and say so. A server must
not: an unconfigured session searched over REST started a 2.3 GB BGE-M3
download inside the request, or, offline, answered ``500 Internal server
error`` while ``/health/ready`` reported the model as up. ``lasagna serve`` and
the MCP server switch the refusal on for their lifetime;
``allow_model_download_in_requests`` (``RAG_ALLOW_MODEL_DOWNLOAD_IN_REQUESTS``)
switches it off again.

Every model load also runs under :func:`one_weights_copy`, so a first run
downloads each model's weights once.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

_refusing = 0

#: transformers' switch for the background copy :func:`one_weights_copy` stops.
NO_CONVERSION_VAR = "DISABLE_SAFETENSORS_CONVERSION"
_conversion_lock = threading.Lock()
_conversion_loads = 0

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


@contextmanager
def one_weights_copy() -> Iterator[None]:
    """Load a model without transformers downloading a second copy of its weights.

    When a repository has only ``pytorch_model.bin`` (BAAI/bge-m3, and the
    default SPLADE and monoT5 checkpoints), transformers loads it and then
    starts a thread that downloads ``model.safetensors`` from the repository's
    conversion pull request. No later load reads that file: they resolve the
    main branch, where it does not exist, and load the ``.bin`` again. For
    bge-m3 it was 2.3 GB more on the first run (4.6 GB of downloads for one
    2.3 GB model), and the thread kept ``lasagna search`` from exiting until
    it finished.

    transformers has no argument for this, only the environment variable
    ``DISABLE_SAFETENSORS_CONVERSION``, read while the load runs. It is set
    for the duration of the load and removed afterwards; a value the
    environment already has is left alone.
    """
    global _conversion_loads
    with _conversion_lock:
        # Counted only while Lasagna's own "1" is in place, so concurrent
        # loads share it and the last one to finish removes it.
        counted = _conversion_loads > 0 or NO_CONVERSION_VAR not in os.environ
        if counted:
            _conversion_loads += 1
            os.environ[NO_CONVERSION_VAR] = "1"
    try:
        yield
    finally:
        if counted:
            with _conversion_lock:
                _conversion_loads -= 1
                if _conversion_loads == 0:
                    os.environ.pop(NO_CONVERSION_VAR, None)


def offline() -> bool:
    """Whether Hugging Face offline mode is set."""
    return any(
        os.environ.get(var, "").strip().lower() in _TRUTHY
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def is_cached(model: str, filename: str = "config.json") -> bool:
    """Whether ``model`` loads without a download: a local folder or a cached hub copy.

    ``filename`` is a file every copy of that kind of repository holds
    (``tokenizer_config.json`` for a tokenizer).
    """
    if not model:
        return False
    if Path(model).expanduser().is_dir():
        return True
    # sentence-transformers reads a bare name ("all-MiniLM-L6-v2") as its own org's.
    names = [model] if "/" in model else [model, f"sentence-transformers/{model}"]
    try:
        from huggingface_hub import try_to_load_from_cache

        return any(isinstance(try_to_load_from_cache(n, filename), str) for n in names)
    except Exception:
        return False


def _allowed() -> bool:
    try:
        from lasagna.core.config import rag_config

        return bool(rag_config.allow_model_download_in_requests)
    except Exception:
        return False


def refusal(
    model: str,
    kind: str = "embedding model",
    filename: str = "config.json",
    *,
    cached: Optional[bool] = None,
) -> Optional[ModelNotDownloaded]:
    """Why ``model`` cannot be loaded here without a download, or None when it can.

    ``cached`` is for a model that is not a Hugging Face repository (FlashRank
    downloads its own files): whether it is on disk, as its loader tells.
    """
    if cached is None:
        cached = is_cached(model) if filename == "config.json" else is_cached(model, filename)
    if cached:
        return None
    if offline():
        return ModelNotDownloaded(model, offline=True, kind=kind)
    if _refusing and not _allowed():
        return ModelNotDownloaded(model, offline=False, kind=kind)
    return None


def check(
    model: str,
    kind: str = "embedding model",
    filename: str = "config.json",
    *,
    cached: Optional[bool] = None,
) -> None:
    """Raise :class:`ModelNotDownloaded` when a server would have to download ``model``.

    Every loader of a model a session can name calls it before it loads:
    the embedder, the cross-encoder, monoT5, FlashRank, ColBERT, SPLADE and
    the chunk tokenizer.
    """
    if not _refusing:
        return
    problem = refusal(model, kind, filename, cached=cached)
    if problem is not None:
        raise problem


def refuse_offline(model: str, kind: str = "embedding model") -> None:
    """Offline, raise :class:`ModelNotDownloaded` for a model that is not on disk.

    Called when a loader fails, so its "We couldn't connect to
    'https://huggingface.co'" is reported as the model not being downloaded,
    with the fix, and the caller can fall back (a search skips its reranker).
    The embedder checks before it loads, since it also accepts a cached
    fallback.
    """
    if model and offline() and not is_cached(model):
        raise ModelNotDownloaded(model, offline=True, kind=kind)


_process_refusal = False


def refuse_downloads_for_this_process() -> None:
    """For the rest of the process: a model that is not cached is refused, not fetched.

    For a process that serves requests and has no ``with`` block around its
    life, the Celery worker: it processed uploads with the same models and
    downloaded any it lacked, whatever ``allow_model_download_in_requests``
    said. Calling it again changes nothing.
    """
    global _refusing, _process_refusal
    if not _process_refusal:
        _process_refusal = True
        _refusing += 1


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


#: How each surface names a session, for the advice that names another one.
#: The MCP server said "the X-Session-ID header ... or -s", neither of which an
#: MCP caller has.
_NAME_A_SESSION = {
    "rest": "the X-Session-ID header (or a session token)",
    "mcp": "the tool's session_id argument, or RAG_SESSION_ID in the server's env block",
    "cli": "-s",
    "sdk": "KnowledgeClient(session_id=...)",
}


def fix_for_session(session_id: str, model: str = "", *, surface: str = "rest") -> str:
    """What to run so that ``session_id`` can be searched, naming the sessions that can.

    ``surface`` (``rest``, ``mcp``, ``cli`` or ``sdk``) decides how the advice
    says to name another session.
    """
    populated = sessions_with_documents() or {}
    others = {s: n for s, n in populated.items() if s != session_id}
    parts = []
    if others and session_id not in populated:
        listed = ", ".join(f"{s} ({n})" for s, n in sorted(others.items()))
        parts.append(
            f"Session {session_id!r} has no documents. Sessions with documents: {listed}; "
            f"name one with {_NAME_A_SESSION.get(surface, _NAME_A_SESSION['rest'])}."
        )
    parts.append(fix_for_model(session_id, model))
    return " ".join(parts)


def still_to_download(session_id: str, model: str = "") -> List[str]:
    """The models ``session_id`` loads that are not on disk, ``model`` first.

    Its embedder, and its reranker when reranking is on (``session_models``).
    Every hint that says what to download names all of them: the hint for a
    missing BGE-M3 named only it (2.3 of the default profile's 3.2 GB), and
    the next search then failed on the reranker.
    """
    from lasagna.core.setup_text import session_models

    try:
        needed = [m for m in session_models(session_id) if not is_cached(m)]
    except Exception:  # noqa: BLE001 - the hint still names the model that failed
        needed = []
    return list(dict.fromkeys([model, *needed] if model else needed))


def fix_for_model(session_id: str, model: str = "") -> str:
    """What to run so that ``session_id`` can use a model, or a smaller one.

    Offline, the way to keep the model is to download it while online; a
    server that refuses downloads can also be told to allow them. The download
    commands cover every model the session still needs (`still_to_download`).
    """
    configured = (
        f"Session {session_id!r} is configured with this model."
        if session_configured(session_id)
        else f"Session {session_id!r} has no profile, so it uses the default configuration's models."
    )
    from lasagna.core.setup_text import downloads_for

    models = still_to_download(session_id, model)
    download = downloads_for(models) if models else "hf download <model>"
    one = len(models) <= 1
    these = "this model" if one else "the session's models"
    it, they = ("it", "it downloads") if one else ("them", "they download")
    allow = "set RAG_ALLOW_MODEL_DOWNLOAD_IN_REQUESTS=true for the server"
    if offline():
        unset = "unset HF_HUB_OFFLINE" + (f" and {allow}" if _refusing else "")
        keep = (
            f"To keep {these}, download {it} once with network access ({download}), "
            f"or {unset} so {they} on first use."
        )
    else:
        keep = f"To keep {these}, download {it} first ({download}), or {allow}."
    from lasagna.core.setup_text import profile_download_size

    return (
        f"{configured} For the {profile_download_size('speed')} keyless model, run: "
        f"lasagna configure --profile speed -s {session_id} "
        f"(in Python: KnowledgeClient(config=config_for_speed())). {keep}"
    )
