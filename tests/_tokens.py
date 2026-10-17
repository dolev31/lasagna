"""Session tokens as ``POST /api/v1/sessions`` mints them with auth on.

With ``RAG_REQUIRE_AUTH`` on, a token that names neither an API key nor the
admin token (one minted while auth was off) is refused. A test that stands in
for a verified caller mints its token the way the server does: naming the
first key in ``RAG_API_KEYS``, or ``admin`` when no key is configured.
"""

from __future__ import annotations

from lasagna.api.rest import auth_mode, session_token


def minted_token(session_id: str, user_id: str = "u1") -> str:
    keys = auth_mode.configured_api_keys()
    principal = auth_mode.principal_for_api_key(keys[0]) if keys else "admin"
    token, _expiry = session_token.create_session_token(session_id, user_id, principal=principal)
    return token
