"""Session handshake endpoint.

- ``POST /api/v1/sessions`` — create a signed session token
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

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
    from lasagna.api.rest.auth_mode import configured_secret

    expected = configured_secret("RAG_ADMIN_TOKEN")
    if not expected or not admin_token:
        return False
    # Constant-time: a token check that leaks its prefix by timing is not a
    # token check.
    return secrets.compare_digest(admin_token, expected)


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


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionRequest = SessionRequest(),
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
):
    """Create a signed session token encoding (session_id, user_id).

    The token is self-contained — the server decodes it on each request
    without any database lookup.  Pass it via the ``X-Session-Token``
    header on subsequent requests.

    Both ``session_id`` and ``user_id`` are optional and are generated when
    omitted, which is the normal path and needs no credentials.

    **Naming them requires ``X-Admin-Token``.** Without that check this endpoint
    hands out a token for any identity the caller asks for, and since sessions
    are the isolation boundary, that is read/write access to another tenant's
    documents. Requested identities are therefore refused rather than silently
    replaced with generated ones — quietly returning a different session than
    the one asked for would look like success and fail later in a confusing
    place.
    """
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

    token, exp_ts = create_session_token(session_id, user_id)

    return SessionResponse(
        token=token,
        session_id=session_id,
        user_id=user_id,
        expires_at=datetime.fromtimestamp(exp_ts, tz=timezone.utc).isoformat(),
    )
