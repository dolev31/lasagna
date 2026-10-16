"""Session token encoding and decoding.

Provides stateless session identity via signed JWT tokens.  A token
encodes ``(session_id, user_id)`` and can be verified without any
database lookup — the identity is self-contained.

Tokens are signed with ``RAG_JWT_SECRET``.  If the env var is unset a
random ephemeral secret is generated, which is fine for development —
tokens are short-lived and a client can re-handshake after a restart — but
is a hard error once ``RAG_REQUIRE_AUTH`` is on.  The secret is per
*process*, and the shipped k8s manifests run ``replicas: 2`` behind one
Service, so a token minted by one pod does not verify on the other: roughly
half of all authenticated requests would 401, intermittently, and read as
flakiness rather than as the missing configuration it is.

Usage::

    from lasagna.api.rest.session_token import (
        create_session_token,
        decode_session_token,
    )

    token, exp = create_session_token("sess_abc", "alice")
    result = decode_session_token(token)   # ("sess_abc", "alice")
"""

from __future__ import annotations

import os
import secrets
import time
from typing import Optional, Tuple

_DEFAULT_TTL = int(os.environ.get("RAG_SESSION_TOKEN_TTL_SECONDS", "604800"))  # 7 days

# Ephemeral fallback secret — generated once per process, lost on restart.
_fallback_secret: Optional[str] = None


class MissingSigningSecret(RuntimeError):
    """``RAG_REQUIRE_AUTH`` is on and there is no shared signing secret."""


def _get_secret() -> str:
    """Return the signing secret, auto-generating one if ``RAG_JWT_SECRET`` is unset.

    The auto-generated secret is a development convenience and is refused
    outright when identity is being enforced — see the module docstring for why
    a per-process secret behind a load balancer is worse than no auth at all.
    """
    global _fallback_secret
    from lasagna.api.rest.auth_mode import (
        MIN_SECRET_LENGTH,
        REQUIRE_AUTH_ENV,
        configured_secret,
        require_auth,
    )

    secret = configured_secret("RAG_JWT_SECRET")
    if secret:
        return secret

    if require_auth():
        if os.environ.get("RAG_JWT_SECRET", "").strip():
            raise MissingSigningSecret(
                f"{REQUIRE_AUTH_ENV} is set and RAG_JWT_SECRET is a placeholder or "
                f"shorter than {MIN_SECRET_LENGTH} characters, so anyone who has "
                f"read the template could forge session tokens. Generate one with "
                f"`openssl rand -hex 32` and set it identically across every replica."
            )
        raise MissingSigningSecret(
            f"{REQUIRE_AUTH_ENV} is set but RAG_JWT_SECRET is not. Tokens would "
            f"be signed with a secret generated fresh in each process, so any "
            f"deployment with more than one replica -- including the shipped "
            f"k8s manifests -- would reject roughly half of its own tokens. "
            f"Generate one with `openssl rand -hex 32` and set it identically "
            f"across every replica."
        )

    if _fallback_secret is None:
        _fallback_secret = secrets.token_hex(32)
        from loguru import logger

        logger.warning(
            "RAG_JWT_SECRET not set: using ephemeral secret (tokens won't survive restart)"
        )
    return _fallback_secret


#: The issuer this server writes into a session token, and the ones it accepts:
#: tokens minted before 0.1.0 say "rag-knowledge" and stay valid until they expire.
TOKEN_ISSUER = "lasagna"
ACCEPTED_ISSUERS = frozenset({TOKEN_ISSUER, "rag-knowledge"})


def create_session_token(
    session_id: str,
    user_id: str,
    ttl_seconds: int = _DEFAULT_TTL,
    principal: Optional[str] = None,
) -> Tuple[str, float]:
    """Create a signed session token.

    ``principal`` names who minted it (an API key's digest, or ``admin``).
    Rate limits are counted per principal, so minting more sessions does not
    buy more requests.

    Returns ``(token_string, expiry_unix_timestamp)``.
    """
    import jwt  # pyjwt

    now = time.time()
    exp = now + ttl_seconds
    payload = {
        "sub": user_id,
        "sid": session_id,
        "iat": int(now),
        "exp": int(exp),
        "iss": TOKEN_ISSUER,
        "typ": "session",
    }
    if principal:
        payload["prn"] = principal
    token = jwt.encode(payload, _get_secret(), algorithm="HS256")
    return token, exp


def decode_session_claims(token: str) -> Optional[dict]:
    """The verified claims of a session token (``sid``, ``sub``, maybe ``prn``).

    ``None`` if the token is invalid, expired, or not a session token, or if
    it was minted with an API key that is no longer in ``RAG_API_KEYS``. Every
    REST and MCP path that reads a token comes through here, so removing a key
    ends its sessions on both.

    The secret is fetched outside the ``try``: the blanket ``except`` below
    exists to turn a bad *token* into ``None``, and a misconfigured *server*
    must not be laundered through it. Swallowing `MissingSigningSecret` here
    would reject every token instead, which presents as "nobody can log in" with
    nothing in the logs above DEBUG.
    """
    secret = _get_secret()
    try:
        import jwt  # pyjwt

        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"require": ["sub", "sid", "exp", "typ"]},
        )
        if payload.get("typ") != "session" or payload.get("iss") not in ACCEPTED_ISSUERS:
            return None
        if payload.get("sid") and payload.get("sub"):
            from lasagna.api.rest.auth_mode import principal_is_revoked

            if principal_is_revoked(payload.get("prn")):
                from loguru import logger

                logger.debug("Session token refused: its API key is no longer configured")
                return None
            return payload
    except Exception:
        from loguru import logger

        logger.debug("Failed to decode session token")
    return None


def decode_session_token(token: str) -> Optional[Tuple[str, str]]:
    """Decode a session token.

    Returns ``(session_id, user_id)`` or ``None`` if the token is
    invalid, expired, or not a session token.
    """
    claims = decode_session_claims(token)
    if claims is None:
        return None
    return claims["sid"], claims["sub"]


_MAX_SESSION_ID_LEN = 128


def extract_session_id(
    session_token: Optional[str] = None,
    session_id_header: Optional[str] = None,
    session_id_query: Optional[str] = None,
) -> str:
    """Extract session ID from token, header, or query parameter.

    Shared logic used by both FastAPI dependencies and middleware.
    No FastAPI imports — safe to call from any context.

    Priority:
        1. ``session_token`` → decoded ``sid`` claim
        2. ``session_id_header`` (X-Session-ID)
        3. ``session_id_query`` (query param)
        4. ``RAG_SESSION_ID`` (``lasagna serve -s`` sets it), else ``"default"``

    Raises:
        ValueError: If the resolved session ID exceeds 128 characters.
    """
    # 1. Session token (highest priority)
    if session_token:
        result = decode_session_token(session_token)
        if result:
            return result[0]  # session_id from token

    # 2. Explicit header / query param (backward compatible)
    resolved = (
        session_id_header
        or session_id_query
        or os.environ.get("RAG_SESSION_ID", "").strip()
        or "default"
    )
    if len(resolved) > _MAX_SESSION_ID_LEN:
        raise ValueError(f"session_id exceeds {_MAX_SESSION_ID_LEN} characters")
    return resolved
