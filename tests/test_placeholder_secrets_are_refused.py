"""The shipped placeholder secrets must not work as secrets.

deploy/k8s/base/secrets.yaml sets RAG_JWT_SECRET and RAG_ADMIN_TOKEN to
CHANGE_ME. They were accepted as-is, so anyone who had read the manifest could
forge session tokens and open the admin routes, while SECURITY.md promises no
default credentials. A placeholder or short value now counts as unset.
"""

from __future__ import annotations

import pytest

from lasagna.api.rest import session_token
from lasagna.api.rest.auth_mode import configured_secret


@pytest.mark.parametrize("value", ["CHANGE_ME", "changeme", "secret", "short-but-real"])
def test_placeholders_and_short_values_count_as_unset(monkeypatch, value):
    monkeypatch.setenv("RAG_ADMIN_TOKEN", value)
    assert configured_secret("RAG_ADMIN_TOKEN") == ""


def test_a_long_random_value_is_used(monkeypatch):
    token = "f" * 64
    monkeypatch.setenv("RAG_ADMIN_TOKEN", token)
    assert configured_secret("RAG_ADMIN_TOKEN") == token


def test_auth_refuses_to_sign_with_the_manifest_placeholder(monkeypatch):
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", "CHANGE_ME")
    monkeypatch.setattr(session_token, "_fallback_secret", None)
    with pytest.raises(session_token.MissingSigningSecret, match="placeholder"):
        session_token._get_secret()
