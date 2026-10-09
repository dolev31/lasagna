"""`session_id` must be a claim someone verified, not a string the caller picked.

Until `auth_mode` existed it was the latter. `extract_session_id` accepted an
`X-Session-ID` header or a `?session_id=` query parameter as identity with no
credential, so this worked against any deployed instance:

    DELETE /api/v1/documents/{id}  -H "X-Session-ID: victim-tenant"

The signed-token machinery in `session_token.py` is well built and was simply
bypassed. `routes/sessions.py` has a constant-time 403 stopping an anonymous
caller from *minting* a token that names someone else — and 13 tests covering
it — but a token was never needed, so the guard protected a door with no wall
around it.

`tests/stress/test_tenant_isolation.py` encodes the intended model and tests the
wrong half: it asserts session B's *search* cannot surface session A's
documents, and its own client authenticates by setting `X-Session-ID`. It never
tests session B *claiming to be* session A. It also never runs.

These tests are the missing half. They exercise the header attack directly.
"""

from __future__ import annotations

import pytest

from lasagna.api.rest import auth_mode


@pytest.fixture(autouse=True)
def _auth_off(monkeypatch):
    """Default state, so each test opts in explicitly."""
    monkeypatch.delenv(auth_mode.REQUIRE_AUTH_ENV, raising=False)


def _enable(monkeypatch):
    monkeypatch.setenv(auth_mode.REQUIRE_AUTH_ENV, "true")


class TestTheDefaultIsUnchanged:
    """Local development, the README quickstart and existing scripts must keep
    working. What closes the hole is the startup refusal, not the default."""

    def test_auth_is_off_unless_asked_for(self):
        assert auth_mode.require_auth() is False

    def test_an_unverified_session_is_accepted_when_off(self):
        assert auth_mode.reject_unverified_identity("victim", from_token=False) is None


class TestTheHeaderAttack:
    def test_an_unverified_session_is_refused_when_auth_is_on(self, monkeypatch):
        """The attack, stated directly."""
        _enable(monkeypatch)
        reason = auth_mode.reject_unverified_identity("victim-tenant", from_token=False)
        assert reason is not None
        assert "victim-tenant" in reason
        assert "token" in reason.lower()

    def test_a_token_backed_session_is_accepted(self, monkeypatch):
        """Enforcement must not break the authenticated path."""
        _enable(monkeypatch)
        assert auth_mode.reject_unverified_identity("tenant-a", from_token=True) is None

    def test_the_anonymous_namespace_is_not_a_free_pass(self, monkeypatch):
        """This asserted the opposite, and the assertion encoded a mistake.

        The `default` session was exempt on the stated grounds that refusing it
        "would break health checks". `routes/health.py` depends only on
        `get_backend` and never resolves a session, so that was simply false --
        and the exemption meant RAG_REQUIRE_AUTH did not stop unauthenticated
        ingest, read or delete in the shared namespace, which is exactly what
        SECURITY.md and the k8s ConfigMap present it as doing.
        """
        _enable(monkeypatch)
        assert auth_mode.reject_unverified_identity("default", from_token=False) is not None

    def test_a_public_corpus_can_still_opt_in(self, monkeypatch):
        """A genuinely public read-only corpus is a real use case; it just has
        to be asked for rather than assumed."""
        _enable(monkeypatch)
        monkeypatch.setenv(auth_mode.ALLOW_ANONYMOUS_ENV, "true")
        assert auth_mode.reject_unverified_identity("default", from_token=False) is None

    def test_the_dependency_raises_401_not_403(self, monkeypatch):
        """401 says "authenticate"; 403 says "you may not". The caller has not
        authenticated, so 401 is the one that tells them what to do."""
        from fastapi import HTTPException

        from lasagna.api.rest import dependencies

        _enable(monkeypatch)

        class _Req:
            state = type("S", (), {})()

        with pytest.raises(HTTPException) as exc:
            dependencies.get_session_id(
                _Req(), x_session_token=None, x_session_id="victim", session_id=None
            )
        assert exc.value.status_code == 401

    def test_a_verified_token_passes_the_dependency(self, monkeypatch):
        from lasagna.api.rest import dependencies
        from lasagna.api.rest.session_token import create_session_token

        monkeypatch.setenv("RAG_JWT_SECRET", "test-secret-not-a-real-one")
        _enable(monkeypatch)
        token, _expiry = create_session_token(session_id="tenant-a", user_id="u1")

        class _Req:
            state = type("S", (), {})()

        assert (
            dependencies.get_session_id(
                _Req(), x_session_token=token, x_session_id=None, session_id=None
            )
            == "tenant-a"
        )

    def test_a_token_does_not_let_a_header_override_it(self, monkeypatch):
        """Token wins, so a caller cannot present a valid token for their own
        session and a header naming someone else's."""
        from lasagna.api.rest import dependencies
        from lasagna.api.rest.session_token import create_session_token

        monkeypatch.setenv("RAG_JWT_SECRET", "test-secret-not-a-real-one")
        _enable(monkeypatch)
        token, _expiry = create_session_token(session_id="mine", user_id="u1")

        class _Req:
            state = type("S", (), {})()

        assert (
            dependencies.get_session_id(
                _Req(), x_session_token=token, x_session_id="victim", session_id=None
            )
            == "mine"
        )


class TestTheServerRefusesToExposeItselfUnauthenticated:
    """A warning is not enough, and this codebase already decided that.

    `mcp/server.py` refuses to start when `--enable-docs-api` is combined with a
    non-loopback host, and its comment explains why: the person harmed by an
    exposed service is not the person reading the startup log.
    """

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
    def test_loopback_starts(self, host):
        auth_mode.require_auth_or_refuse_to_start(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "192.168.1.10", "::"])
    def test_a_public_bind_without_auth_refuses(self, host):
        with pytest.raises(auth_mode.UnsafeBindError):
            auth_mode.require_auth_or_refuse_to_start(host)

    def test_a_public_bind_with_auth_starts(self, monkeypatch):
        _enable(monkeypatch)
        auth_mode.require_auth_or_refuse_to_start("0.0.0.0")

    def test_the_message_says_what_to_do(self):
        with pytest.raises(auth_mode.UnsafeBindError) as exc:
            auth_mode.require_auth_or_refuse_to_start("0.0.0.0")
        message = str(exc.value)
        assert auth_mode.REQUIRE_AUTH_ENV in message
        assert "RAG_JWT_SECRET" in message
        assert "reverse proxy" in message

    def test_an_unresolvable_host_is_treated_as_public(self):
        """The safe direction for an unknown value. A hostname this cannot
        parse is likelier to be a public interface than the local one."""
        assert auth_mode.is_loopback("db.internal.example") is False
        with pytest.raises(auth_mode.UnsafeBindError):
            auth_mode.require_auth_or_refuse_to_start("db.internal.example")

    def test_the_cli_wires_the_refusal(self):
        """The check is only worth having if `serve` calls it."""
        import inspect

        from lasagna import cli

        source = inspect.getsource(cli.serve)
        assert "require_auth_or_refuse_to_start" in source

    def test_uvicorn_is_told_to_trust_forwarding_headers(self):
        """Without this the app sees the proxy as the client for every request,
        which makes any per-IP control meaningless and the access log wrong."""
        import inspect

        from lasagna import cli

        source = inspect.getsource(cli.serve)
        assert "proxy_headers=True" in source
        assert "forwarded_allow_ips" in source


class TestOverRealHTTP:
    """The tests above call `get_session_id` with a hand-built Request.

    That is the shape of test this project keeps getting burned by — written at
    the seam the fix touched, passing whether or not the fix is wired into
    anything. These drive a real FastAPI app through a real client, so header
    parsing, query parsing and HTTPException-to-response conversion are all
    genuinely exercised.

    A full app is deliberately NOT constructed: its lifespan initialises the
    backend, which downloads several gigabytes of model weights. The dependency
    is the unit under test and it is mounted here exactly as the real routes
    mount it.
    """

    @pytest.fixture
    def client(self, monkeypatch):
        fastapi = pytest.importorskip("fastapi")
        testclient = pytest.importorskip("fastapi.testclient")

        monkeypatch.setenv("RAG_JWT_SECRET", "test-secret-not-a-real-one")
        _enable(monkeypatch)

        from lasagna.api.rest.dependencies import get_session_id

        app = fastapi.FastAPI()

        @app.get("/whoami")
        def whoami(session_id: str = fastapi.Depends(get_session_id)):
            return {"session_id": session_id}

        return testclient.TestClient(app)

    def _token(self, session_id="tenant-a"):
        from lasagna.api.rest.session_token import create_session_token

        token, _expiry = create_session_token(session_id=session_id, user_id="u1")
        return token

    def test_the_header_attack_gets_401(self, client):
        """`DELETE /documents/{id} -H "X-Session-ID: victim"` -- the whole point."""
        assert client.get("/whoami", headers={"X-Session-ID": "victim"}).status_code == 401

    def test_the_query_parameter_attack_gets_401(self, client):
        """`?session_id=` is the same hole through a different door, and it is
        the one that ends up in browser history and access logs."""
        assert client.get("/whoami?session_id=victim").status_code == 401

    def test_a_valid_token_resolves_to_its_own_session(self, client):
        response = client.get("/whoami", headers={"X-Session-Token": self._token()})
        assert response.status_code == 200
        assert response.json()["session_id"] == "tenant-a"

    def test_a_header_cannot_override_a_valid_token(self, client):
        """Otherwise a caller authenticates as themselves and then names someone
        else, which would make the token theatre."""
        response = client.get(
            "/whoami",
            headers={"X-Session-Token": self._token("mine"), "X-Session-ID": "victim"},
        )
        assert response.json()["session_id"] == "mine"

    def test_a_forged_token_does_not_authenticate(self, client):
        """A token signed with the wrong secret must not be trusted, and must
        not fall through to the header path either."""
        import jwt

        forged = jwt.encode(
            {"sid": "victim", "sub": "u1", "typ": "session", "exp": 9999999999},
            "the-wrong-secret",
            algorithm="HS256",
        )
        assert client.get("/whoami", headers={"X-Session-Token": forged}).status_code == 401

    def test_no_credential_at_all_is_401_over_http(self, client):
        """The hole this class exists to close, in its most basic form.

        Sending nothing used to return 200 and the `default` session, so an
        instance with RAG_REQUIRE_AUTH on still accepted unauthenticated writes.
        """
        assert client.get("/whoami").status_code == 401

    def test_with_auth_off_the_header_is_honoured_again(self, monkeypatch):
        """The default path, unchanged. If this fails, every existing local
        script broke."""
        fastapi = pytest.importorskip("fastapi")
        testclient = pytest.importorskip("fastapi.testclient")

        monkeypatch.delenv(auth_mode.REQUIRE_AUTH_ENV, raising=False)

        from lasagna.api.rest.dependencies import get_session_id

        app = fastapi.FastAPI()

        @app.get("/whoami")
        def whoami(session_id: str = fastapi.Depends(get_session_id)):
            return {"session_id": session_id}

        response = testclient.TestClient(app).get("/whoami", headers={"X-Session-ID": "anything"})
        assert response.status_code == 200
        assert response.json()["session_id"] == "anything"

    def test_an_expired_token_is_refused_even_with_auth_off(self, monkeypatch):
        """A presented credential that does not verify is an error in both modes.

        `extract_session_id` falls through to the header path when a token fails
        to decode, so an expired token resolved to the anonymous `default`
        session and the client carried on writing documents into the wrong
        namespace with a 200 back. Silent downgrade sends data to the wrong
        place; a refusal sends the client to re-authenticate.
        """
        import time

        fastapi = pytest.importorskip("fastapi")
        testclient = pytest.importorskip("fastapi.testclient")
        jwt = pytest.importorskip("jwt")

        monkeypatch.delenv(auth_mode.REQUIRE_AUTH_ENV, raising=False)
        monkeypatch.setenv("RAG_JWT_SECRET", "test-secret-not-a-real-one")

        from lasagna.api.rest.dependencies import get_session_id

        expired = jwt.encode(
            {
                "sid": "tenant-a",
                "sub": "u1",
                "typ": "session",
                "exp": int(time.time()) - 60,
            },
            "test-secret-not-a-real-one",
            algorithm="HS256",
        )

        app = fastapi.FastAPI()

        @app.get("/whoami")
        def whoami(session_id: str = fastapi.Depends(get_session_id)):
            return {"session_id": session_id}

        response = testclient.TestClient(app).get("/whoami", headers={"X-Session-Token": expired})
        assert response.status_code == 401
        assert "expired" in response.json()["detail"].lower()
