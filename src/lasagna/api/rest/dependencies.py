"""Shared FastAPI dependencies for the RAG REST API.

Provides dependency-injection functions for the RAGBackend singleton and
session-aware configuration.  These are used via ``Depends(...)`` in route
handlers so that every endpoint gets a correctly-initialised backend
without duplicating setup logic.
"""

from __future__ import annotations

import os
import uuid
from typing import Optional, Tuple

from fastapi import Header, HTTPException, Query, Request
from loguru import logger

from lasagna.api.rag_backend import RAGBackend, get_rag_backend

_ALLOW_HEADER_IDENTITY = os.environ.get("RAG_ALLOW_HEADER_IDENTITY", "").lower() in (
    "1",
    "true",
    "yes",
)


def get_backend() -> RAGBackend:
    """Return the RAGBackend singleton.

    Raises ``503 Service Unavailable`` if the backend has not been
    initialised yet (e.g. during startup while models are loading).
    """
    try:
        return get_rag_backend()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"RAG backend not ready: {exc}",
        )


def _get_cached_token(request: Request) -> Optional[Tuple[str, str]]:
    """Return the (session_id, user_id) cached by rate-limit middleware, if any."""
    return getattr(request.state, "decoded_session_token", None)


def get_session_id(
    request: Request,
    x_session_token: Optional[str] = Header(None, alias="X-Session-Token"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    session_id: Optional[str] = Query(None),
) -> str:
    """Extract session ID from token, header, or query parameter.

    Uses the middleware-cached decode when available (zero-cost).
    Falls back to :func:`session_token.extract_session_id`.
    """
    # Fast path: reuse middleware decode. A cached value came from a *verified*
    # token, so it is an authenticated identity by construction.
    cached = _get_cached_token(request)
    if cached:
        return cached[0]

    from .auth_mode import reject_unverified_identity
    from .session_token import decode_session_token, extract_session_id

    from_token = bool(x_session_token and decode_session_token(x_session_token))

    # A credential that was presented and did not verify is an error, in both
    # modes. `extract_session_id` falls through to the header/query path when a
    # token fails to decode, so an expired or forged token silently resolved to
    # the anonymous `default` session -- and the client carried on writing
    # documents into the wrong namespace with a 200 back. Data going quietly to
    # the wrong place is worse than a refusal, and this only affects requests
    # that were already broken.
    if x_session_token and not from_token:
        raise HTTPException(
            status_code=401,
            detail="X-Session-Token is invalid or expired.",
        )

    try:
        resolved = extract_session_id(
            session_token=x_session_token,
            session_id_header=x_session_id,
            session_id_query=session_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # `session_id` is a namespace unless something verifies it.
    #
    # With RAG_REQUIRE_AUTH off this is a no-op and behaviour is unchanged. With
    # it on, an unsigned `X-Session-ID` stops being an identity claim -- which
    # is what stops `DELETE /documents/{id} -H "X-Session-ID: victim"` from
    # working against a deployed instance.
    reason = reject_unverified_identity(resolved, from_token=from_token)
    if reason:
        raise HTTPException(status_code=401, detail=reason)

    return resolved


def get_user_id(
    request: Request,
    x_session_token: Optional[str] = Header(None, alias="X-Session-Token"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    authorization: Optional[str] = Header(None),
) -> str:
    """Extract user identity from request.

    Priority:
        1. JWT Bearer ``sub`` claim (upstream auth — highest trust)
        2. ``X-Session-Token`` → decoded ``sub`` claim
        3. ``X-User-ID`` header
        4. Auto-assign unique ``anon_<uuid>`` (prevents __anonymous__ collision)
    """
    # 1. JWT bearer token (upstream auth)
    if authorization and authorization.startswith("Bearer "):
        try:
            payload = _decode_jwt(authorization[7:])
            user_id = payload.get("sub", "")
            if user_id and len(user_id) <= 128:
                return user_id
        except Exception:
            logger.debug("Bearer JWT decode failed, falling through to next source")

    # 2. Session token — reuse middleware cache
    cached = _get_cached_token(request)
    if cached:
        return cached[1]

    if x_session_token:
        from .session_token import decode_session_token

        result = decode_session_token(x_session_token)
        if result:
            return result[1]  # user_id

    # 3. Explicit header — only accepted when opt-in via RAG_ALLOW_HEADER_IDENTITY
    if x_user_id:
        if not _ALLOW_HEADER_IDENTITY:
            raise HTTPException(
                status_code=401,
                detail="X-User-ID header rejected; use JWT or session token, "
                "or set RAG_ALLOW_HEADER_IDENTITY=true",
            )
        if len(x_user_id) > 128:
            raise HTTPException(status_code=400, detail="user_id exceeds 128 characters")
        return x_user_id

    # 4. Auto-assign unique anonymous identity
    return f"anon_{uuid.uuid4().hex[:16]}"


def get_profile_owner(
    request: Request,
    x_session_token: Optional[str] = Header(None, alias="X-Session-Token"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    authorization: Optional[str] = Header(None),
) -> str:
    """Who owns the profiles a request saves, and whose it may see.

    With auth on, the verified user. With it off the server is loopback-only and
    single-user, and `get_user_id` hands out a new ``anon_`` id on every
    request, so a profile owned by it could not be overwritten or deleted by
    the next request from the same client.
    """
    from lasagna.core.session_config._data_models import ANONYMOUS

    from .auth_mode import require_auth

    if not require_auth():
        return ANONYMOUS
    return get_user_id(request, x_session_token, x_user_id, authorization)


def _decode_jwt(token: str) -> dict:
    """Decode a JWT token using the configured secret.

    Uses ``RAG_JWT_SECRET`` environment variable.  If not set or if the
    ``pyjwt`` package is not installed, raises an exception (caller
    falls back to header-based identity).
    """
    secret = os.environ.get("RAG_JWT_SECRET")
    if not secret:
        raise ValueError("RAG_JWT_SECRET not configured")

    import jwt  # pyjwt

    return jwt.decode(
        token,
        secret,
        algorithms=["HS256"],
        options={"require": ["sub"], "verify_signature": True},
    )
