"""Three holes found by an adversarial audit, two of them introduced by recent changes.

**Cross-tenant document takeover.** ``POST /api/v1/documents`` accepted
``document_id`` as a form field and passed it through unchecked.
``document_id`` is the GLOBAL primary key of ``rag_documents``, so uploading
with another session's id overwrote that row and rewrote its ``session_id``.
Verified end to end against a running server before the fix::

    A uploads probeA.txt as probe_takeover_1   -> 201
    B uploads probeB.txt as probe_takeover_1   -> 201
    A's listing: {"documents": [], "total": 0}          <- destroyed
    B's listing: [{... "session_id": "probe-tenant-b"}] <- taken over

**``RAG_REQUIRE_AUTH`` did much less than advertised.** The ``default`` session
was exempt, on the stated grounds that refusing it "would break health checks".
That was false — ``routes/health.py`` depends only on ``get_backend`` and never
resolves a session, and every health path is in ``_EXEMPT_PREFIXES`` besides. So
with the switch on, a caller sending no credential still landed in ``default``
and could ingest, read and delete there. SECURITY.md and the k8s ConfigMap
present that switch as what makes the service safe to expose.

**The startup bind check broke the shipped compose stack.** A process in a
container always sees ``0.0.0.0`` no matter how its port is published, so
``--host=0.0.0.0`` in ``docker-compose.yml`` — published on ``127.0.0.1:8005``,
i.e. genuinely safe — was refused outright. The check cannot see the publish, so
the operator asserts it.
"""

from __future__ import annotations

import pytest

from lasagna.api.rest import auth_mode


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        auth_mode.REQUIRE_AUTH_ENV,
        auth_mode.ALLOW_ANONYMOUS_ENV,
        auth_mode.BIND_IS_PROTECTED_ENV,
    ):
        monkeypatch.delenv(var, raising=False)


class TestTheAnonymousSessionIsNotAFreePass:
    """What `RAG_REQUIRE_AUTH` is advertised as doing, it must actually do."""

    def test_no_credential_is_refused_when_auth_is_on(self, monkeypatch):
        monkeypatch.setenv(auth_mode.REQUIRE_AUTH_ENV, "true")
        reason = auth_mode.reject_unverified_identity("default", from_token=False)
        assert reason is not None
        assert "POST /api/v1/sessions" in reason

    def test_a_named_session_is_still_refused(self, monkeypatch):
        """The half that already worked must keep working."""
        monkeypatch.setenv(auth_mode.REQUIRE_AUTH_ENV, "true")
        assert auth_mode.reject_unverified_identity("victim", from_token=False) is not None

    def test_a_token_still_passes(self, monkeypatch):
        monkeypatch.setenv(auth_mode.REQUIRE_AUTH_ENV, "true")
        assert auth_mode.reject_unverified_identity("default", from_token=True) is None

    def test_the_old_behaviour_can_be_asked_for(self, monkeypatch):
        """A genuinely public read-only corpus is a real use case. It just has
        to be requested rather than assumed."""
        monkeypatch.setenv(auth_mode.REQUIRE_AUTH_ENV, "true")
        monkeypatch.setenv(auth_mode.ALLOW_ANONYMOUS_ENV, "true")
        assert auth_mode.reject_unverified_identity("default", from_token=False) is None

    def test_auth_off_is_completely_unchanged(self):
        assert auth_mode.reject_unverified_identity("default", from_token=False) is None
        assert auth_mode.reject_unverified_identity("anything", from_token=False) is None

    def test_health_never_resolves_a_session(self):
        """The premise of the original mistake, checked rather than assumed.

        The exemption existed because refusing `default` was believed to break
        health checks. If health ever does take a session, this fails and the
        decision gets revisited instead of silently becoming wrong.
        """
        from fastapi.routing import APIRoute

        from lasagna.api.rest.dependencies import get_session_id
        from lasagna.api.rest.routes import health

        def calls(dependant):
            for sub in dependant.dependencies:
                yield sub.call
                yield from calls(sub)

        probes = [
            r for r in health.router.routes if isinstance(r, APIRoute) and "/health" in r.path
        ]
        assert len(probes) == 3
        for route in probes:
            assert get_session_id not in set(calls(route.dependant)), route.path


class TestAContainerCanStillStart:
    def test_a_published_loopback_container_starts(self, monkeypatch):
        """docker-compose publishes 127.0.0.1:8005 and passes --host=0.0.0.0.
        Before this, the shipped stack's mcp service could not boot."""
        monkeypatch.setenv(auth_mode.BIND_IS_PROTECTED_ENV, "true")
        auth_mode.require_auth_or_refuse_to_start("0.0.0.0", surface="MCP server")

    def test_it_must_be_asked_for(self):
        """Unset, the refusal still fires — the assertion is the whole point."""
        with pytest.raises(auth_mode.UnsafeBindError):
            auth_mode.require_auth_or_refuse_to_start("0.0.0.0")

    def test_the_message_offers_the_escape_hatch(self):
        with pytest.raises(auth_mode.UnsafeBindError) as exc:
            auth_mode.require_auth_or_refuse_to_start("0.0.0.0")
        assert auth_mode.BIND_IS_PROTECTED_ENV in str(exc.value)

    def test_the_compose_file_sets_it(self):
        import pathlib

        yaml = pytest.importorskip("yaml")
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "src"
            / "lasagna"
            / "deploy"
            / "docker-compose.yml"
        )
        text = path.read_text()
        assert auth_mode.BIND_IS_PROTECTED_ENV in text, (
            "the shipped stack binds 0.0.0.0 in every container and will refuse "
            "to start without this"
        )
        yaml.safe_load(text)  # still valid YAML

    def test_the_mcp_service_still_publishes_on_loopback(self):
        """What makes the assertion above true. If this changes, the assertion
        becomes a lie and this test is the one that should fail."""
        import pathlib

        yaml = pytest.importorskip("yaml")
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "src"
            / "lasagna"
            / "deploy"
            / "docker-compose.yml"
        )
        compose = yaml.safe_load(path.read_text())
        for name, service in compose["services"].items():
            for port in service.get("ports", []):
                assert str(port).count(":") >= 2, f"{name} publishes {port} to every interface"


class TestDocumentIdCannotNameSomeoneElsesRow:
    """`document_id` is a global primary key that the caller chooses."""

    class _Doc:
        def __init__(self, session_id):
            self.session_id = session_id

    class _Backend:
        def __init__(self, owner):
            self._owner = owner

        def _get_document_metadata(self, document_id):
            if self._owner is None:
                return None
            return TestDocumentIdCannotNameSomeoneElsesRow._Doc(self._owner)

    def _guard(self, owner, requester):
        from lasagna.api.rest.routes.documents import _reject_foreign_document_id

        return _reject_foreign_document_id(self._Backend(owner), "doc_x", requester)

    def test_an_unused_id_is_allowed(self):
        assert self._guard(None, "tenant-a") is None

    def test_reusing_your_own_id_is_allowed(self):
        """The feature the field exists for: idempotent re-upload."""
        assert self._guard("tenant-a", "tenant-a") is None

    def test_another_sessions_id_is_refused(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            self._guard("tenant-a", "tenant-b")

    def test_it_is_a_404_not_a_403(self):
        """403 would confirm the id exists, turning this into an enumeration
        oracle over other tenants' document ids."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            self._guard("tenant-a", "tenant-b")
        assert exc.value.status_code == 404

    def test_the_upload_route_calls_it(self):
        """The guard is worthless if the route does not use it, and the route
        is where the whole defect lived."""
        import ast
        import inspect
        import textwrap

        from lasagna.api.rest.routes import documents

        tree = ast.parse(textwrap.dedent(inspect.getsource(documents.upload_document)))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "_reject_foreign_document_id" in called

    def test_the_check_precedes_the_upload(self):
        import ast
        import inspect
        import textwrap

        from lasagna.api.rest.routes import documents

        tree = ast.parse(textwrap.dedent(inspect.getsource(documents.upload_document)))
        guard = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_reject_foreign_document_id"
        ]
        assign = [
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Subscript) and getattr(n.value, "id", None) == "upload_kwargs"
        ]
        assert guard and assign
        assert min(guard) < max(assign)
