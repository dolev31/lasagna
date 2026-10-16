"""Session handshake endpoint.

- ``POST /api/v1/sessions`` — create a signed session token

With ``RAG_REQUIRE_AUTH`` on, minting a token needs an ``X-API-Key`` from
``RAG_API_KEYS`` (or the admin token). Without that, anyone who could reach the
endpoint minted as many sessions as they liked, and since rate limits and
tokens were counted per session, each new session was a fresh allowance.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from ..models import ErrorResponse

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _may_claim_an_identity(admin_token: Optional[str]) -> bool:
    """Whether this caller may choose its own ``session_id``/``user_id``.

    Minting a token for an *arbitrary* identity is impersonation: sessions are
    the only isolation boundary in this system, and the data-plane routes do not
    check ownership beyond the session in the token. So a caller who can name
    someone else's ``session_id`` can list, read and delete their documents.

    Same ``RAG_ADMIN_TOKEN`` mechanism the GC/admin routes already use, and it
    fails closed the same way: with no token configured, nobody may claim an
    identity and the endpoint still issues freshly generated ones.
    """
    from lasagna.api.rest.auth_mode import configured_secret, secret_matches

    expected = configured_secret("RAG_ADMIN_TOKEN")
    if not expected or not admin_token:
        return False
    # Constant-time: a token check that leaks its prefix by timing is not a
    # token check.
    return secret_matches(admin_token, expected)


class SessionRequest(BaseModel):
    """Request body for session creation."""

    session_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Desired session ID. Auto-generated if omitted.",
    )
    user_id: Optional[str] = Field(
        None,
        max_length=128,
        description="Desired user ID. Auto-generated if omitted.",
    )


class SessionResponse(BaseModel):
    """Response with signed session token."""

    token: str
    session_id: str
    user_id: str
    expires_at: str  # ISO-8601


def _principal_or_refuse(x_api_key: Optional[str], x_admin_token: Optional[str]) -> Optional[str]:
    """Who is minting this token, or an HTTP error when auth is on and no one is.

    ``None`` with auth off: the server is loopback-only and single-user, and
    tokens then carry no principal, exactly as before.
    """
    from lasagna.api.rest.auth_mode import (
        API_KEYS_ENV,
        MIN_SECRET_LENGTH,
        REQUIRE_AUTH_ENV,
        configured_api_keys,
        principal_for_api_key,
        require_auth,
    )

    if not require_auth():
        return None
    if _may_claim_an_identity(x_admin_token):
        return "admin"
    if not configured_api_keys():
        raise HTTPException(
            status_code=503,
            detail=(
                f"{REQUIRE_AUTH_ENV} is on and {API_KEYS_ENV} lists no usable key, so "
                f"no session can be issued. Set {API_KEYS_ENV} to one or more "
                f"comma-separated keys of at least {MIN_SECRET_LENGTH} characters "
                f"(openssl rand -hex 32) and send one as X-API-Key."
            ),
        )
    principal = principal_for_api_key(x_api_key)
    if principal is None:
        raise HTTPException(
            status_code=401,
            detail=(
                f"{REQUIRE_AUTH_ENV} is on, so creating a session needs an API key: "
                f"send X-API-Key with one of the keys in {API_KEYS_ENV}."
            ),
        )
    return principal


@router.post(
    "",
    response_model=SessionResponse,
    status_code=201,
    responses={
        401: {
            "model": ErrorResponse,
            "description": "Auth is on and X-API-Key is missing or wrong",
        },
        403: {
            "model": ErrorResponse,
            "description": "Naming a session_id or user_id needs X-Admin-Token",
        },
        503: {
            "model": ErrorResponse,
            "description": "Auth is on and RAG_API_KEYS lists no usable key",
        },
    },
)
async def create_session(
    body: SessionRequest = SessionRequest(),
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Create a signed session token encoding (session_id, user_id).

    The token is self-contained: the server decodes it on each request
    without any database lookup.  Pass it via the ``X-Session-Token``
    header on subsequent requests.

    Both ``session_id`` and ``user_id`` are optional and are generated when
    omitted, which is the normal path and needs no credentials.

    **Naming them requires ``X-Admin-Token``.** Without that check this endpoint
    hands out a token for any identity the caller asks for, and since sessions
    are the isolation boundary, that is read/write access to another tenant's
    documents. Requested identities are therefore refused rather than silently
    replaced with generated ones: quietly returning a different session than
    the one asked for would look like success and fail later in a confusing
    place.

    **With ``RAG_REQUIRE_AUTH`` on, an ``X-API-Key`` from ``RAG_API_KEYS`` is
    required** (or ``X-Admin-Token``). The key's principal is written into the
    token, and rate limits are counted per principal rather than per session.
    """
    principal = _principal_or_refuse(x_api_key, x_admin_token)
    requested_identity = body.session_id is not None or body.user_id is not None
    if requested_identity and not _may_claim_an_identity(x_admin_token):
        raise HTTPException(
            status_code=403,
            detail=(
                "Naming a session_id or user_id requires a valid X-Admin-Token. "
                "Omit both to be issued a fresh session. Sessions are the "
                "isolation boundary, so an unauthenticated caller must not be "
                "able to choose one."
            ),
        )

    session_id = body.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    user_id = body.user_id or f"user_{uuid.uuid4().hex[:12]}"

    from ..session_token import create_session_token

    token, exp_ts = create_session_token(session_id, user_id, principal=principal)

    return SessionResponse(
        token=token,
        session_id=session_id,
        user_id=user_id,
        expires_at=datetime.fromtimestamp(exp_ts, tz=timezone.utc).isoformat(),
    )
