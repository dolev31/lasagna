"""A session token minted by a clock a little ahead of the checking one is accepted.

The CI container's clock is stepped back to its host's every 30 seconds or so
(about 0.1 s each time), and API replicas' clocks differ. A token is stamped
with the second it was minted (``iat``), and PyJWT, with no leeway, refused one
stamped "after" the checking clock's now as not yet valid: a valid token failed
on its next use (401), and the SDK client minted another
(``test_with_auth_on_the_key_mints_one_session_token`` saw two mints).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from lasagna.api.rest import session_token


def _minted_ahead(monkeypatch, seconds: float) -> str:
    monkeypatch.setenv("RAG_JWT_SECRET", "s" * 64)
    monkeypatch.setattr(session_token, "_fallback_secret", None)
    ahead = SimpleNamespace(time=lambda: time.time() + seconds)
    monkeypatch.setattr(session_token, "time", ahead)  # the minting replica's clock
    token, _ = session_token.create_session_token("sess_skew", "user_1", principal="admin")
    monkeypatch.setattr(session_token, "time", time)
    return token


def test_a_token_stamped_a_few_seconds_ahead_is_valid(monkeypatch):
    token = _minted_ahead(monkeypatch, 5)
    assert session_token.decode_session_token(token) == ("sess_skew", "user_1")


def test_one_from_a_clock_far_ahead_is_still_refused(monkeypatch):
    token = _minted_ahead(monkeypatch, 3600)
    assert session_token.decode_session_token(token) is None
