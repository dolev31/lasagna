"""Request limits, the search deadline and the log format of the served API.

These are settings of the server process rather than of a retrieval pipeline,
so they are read from the environment at call time, like ``RAG_REQUIRE_AUTH``,
and never vary per session:

======================  ===========  ==========================================
Variable                Default      Meaning
======================  ===========  ==========================================
``RAG_MAX_BATCH_FILES`` 20           Files in one ``POST /documents/batch``
``RAG_MAX_BATCH_BYTES`` 209715200    Total body of one batch upload (200 MiB)
``RAG_SEARCH_TIMEOUT``  120          Seconds a REST search may run before 504
``LASAGNA_LOG_FORMAT``  ``text``     ``json`` writes one JSON object per line
======================  ===========  ==========================================

A single upload is limited by ``RAG_MAX_FILE_SIZE`` (the existing
``max_file_size`` setting). An invalid value is an error at startup that names
the variable, not a warning that leaves the default in force.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

from loguru import logger

#: Room for multipart boundaries and part headers on top of the file bytes.
MULTIPART_OVERHEAD = 1 << 20
#: Every other body is JSON a few kilobytes long.
JSON_BODY_LIMIT = 1 << 20

_SETTINGS = {
    "RAG_MAX_BATCH_FILES": (20, int, 1, 1000),
    "RAG_MAX_BATCH_BYTES": (200 * 1024 * 1024, int, 1, 1 << 40),
    "RAG_SEARCH_TIMEOUT": (120.0, float, 1.0, 3600.0),
}


def _setting(name: str):
    default, kind, low, high = _SETTINGS[name]
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = kind(raw)
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} is not a number. Set a value from {low} to {high}, "
            f"or unset it for the default ({default})."
        ) from None
    if not low <= value <= high:
        raise ValueError(
            f"{name}={raw!r} is out of range. Set a value from {low} to {high}, "
            f"or unset it for the default ({default})."
        )
    return value


def max_batch_files() -> int:
    return _setting("RAG_MAX_BATCH_FILES")


def max_batch_bytes() -> int:
    return _setting("RAG_MAX_BATCH_BYTES")


def search_timeout() -> float:
    return _setting("RAG_SEARCH_TIMEOUT")


LOG_FORMAT_ENV = "LASAGNA_LOG_FORMAT"


def log_format() -> str:
    raw = os.environ.get(LOG_FORMAT_ENV, "")
    value = raw.strip().lower() or "text"
    if value not in ("text", "json"):
        raise ValueError(f"{LOG_FORMAT_ENV}={raw!r} is not a log format. Use json or text.")
    return value


def apply_log_format(level: Optional[str] = None) -> bool:
    """Switch loguru to one JSON object per line when ``LASAGNA_LOG_FORMAT=json``.

    Only a server calls this, never the library. Each line carries the
    request id from the request-id middleware in ``record.extra``. Returns
    whether the format was changed.
    """
    if log_format() != "json":
        return False
    logger.remove()
    logger.add(
        sys.stderr,
        serialize=True,
        level=(level or os.environ.get("RAG_LOG_LEVEL") or "INFO").upper(),
        backtrace=False,
        # diagnose prints local variables into tracebacks; in a log shipped to
        # a collector that includes whatever secrets were in scope.
        diagnose=False,
    )
    return True


def validate() -> None:
    """Raise on any invalid setting above, naming it."""
    for name in _SETTINGS:
        _setting(name)
    log_format()


def body_limit(path: str) -> Tuple[int, str]:
    """The largest body a request to ``path`` may send, and the setting behind it."""
    route = path.rstrip("/")
    if route == "/api/v1/documents":
        from lasagna.core.config import rag_config

        size = rag_config.max_file_size
        return (
            size + MULTIPART_OVERHEAD,
            f"Files up to {size} bytes are accepted; RAG_MAX_FILE_SIZE raises it.",
        )
    if route == "/api/v1/documents/batch":
        size = max_batch_bytes()
        return (
            size + MULTIPART_OVERHEAD,
            f"A batch may hold {size} bytes in total; RAG_MAX_BATCH_BYTES raises it.",
        )
    return JSON_BODY_LIMIT, "Request bodies other than uploads are limited to 1 MiB."


class _BodyTooLarge(Exception):
    pass


def _declared_length(scope) -> Optional[int]:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


class BodySizeLimitMiddleware:
    """Answer 413 before reading a body that is too large.

    A route's own check came too late: Starlette parses a multipart body into
    temporary files before a route with an ``UploadFile`` parameter runs, so a
    10 GB upload was written to disk in full and only then refused, and a
    batch had no total limit at all. A declared ``Content-Length`` over the
    limit is refused without reading the body; a chunked body is counted as it
    arrives and refused once it passes the limit.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit, hint = body_limit(scope["path"])
        declared = _declared_length(scope)
        if declared is not None and declared > limit:
            await self._refuse(scope, receive, send, limit, hint)
            return

        seen = 0
        exceeded = False
        started = False

        async def counted_receive():
            nonlocal seen, exceeded
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    exceeded = True
                    raise _BodyTooLarge()
            return message

        async def guarded_send(message):
            nonlocal started
            # Once the body is over the limit, whatever the app makes of the
            # interrupted read (FastAPI answers 400 "error parsing the body")
            # is replaced by the 413 below.
            if exceeded and not started:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counted_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded and not started:
            await self._refuse(scope, receive, send, limit, hint)

    @staticmethod
    async def _refuse(scope, receive, send, limit: int, hint: str) -> None:
        from starlette.responses import JSONResponse

        response = JSONResponse(
            status_code=413,
            content={
                "detail": f"The request body is larger than {limit} bytes, the limit "
                f"for {scope['path']}. {hint}"
            },
        )
        await response(scope, receive, send)
