"""The MCP surface must not perform what the REST API refuses.

`api/rest/app.py` mounts the MCP server at ``/mcp`` in the same process that
serves the admin routes. Five tools are operations REST gates behind
``X-Admin-Token``, and they had no authentication of any kind:

    DELETE /api/v1/sessions/{id}   -> 401 without X-Admin-Token
    MCP bulk_delete_session        -> deleted every document, no credential

``rebuild_collection`` is the sharpest case and has no REST equivalent at all.
``collection_name`` resolves to the single global ``rag_documents_v2`` — sessions
are a metadata filter inside it — so a drop takes every session with it, and it
had no ``confirm`` flag where the strictly less destructive
``bulk_delete_session`` did.

These tests call the tool functions themselves rather than `transport_guard`'s
helpers. A test written against the helper passes whether or not the helper is
wired into anything, which is the failure mode this project keeps repeating.
The backend is never reached: every assertion here is about a refusal that
happens before it, and the one test that must reach the backend says so.
"""

from __future__ import annotations

import pytest

from lasagna.api.rest.auth_mode import REQUIRE_AUTH_ENV
from lasagna.mcp import transport_guard
from lasagna.mcp import server as mcp_server

ADMIN = "an-admin-token-for-tests"
SECRET = "a-jwt-secret-for-tests"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _restore_transport():
    """`set_transport` is process-global; leaking it would corrupt other tests."""
    before = transport_guard.get_transport()
    yield
    transport_guard.set_transport(before)


@pytest.fixture
def exposed(monkeypatch):
    """A networked server that has been told it is exposed."""
    transport_guard.set_transport("sse")
    monkeypatch.setenv(REQUIRE_AUTH_ENV, "true")
    monkeypatch.setenv("RAG_ADMIN_TOKEN", ADMIN)
    monkeypatch.setenv("RAG_JWT_SECRET", SECRET)


@pytest.fixture
def local(monkeypatch):
    """The default: a client-launched subprocess."""
    transport_guard.set_transport("stdio")
    monkeypatch.delenv(REQUIRE_AUTH_ENV, raising=False)


def _guard_admin_labels():
    """{enclosing function name: the `tool=` string it passes to guard_admin}."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(mcp_server))
    found = {}
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "guard_admin"
            ):
                label = next((kw.value.value for kw in node.keywords if kw.arg == "tool"), None)
                found[func.name] = label
    return found


def _functions_calling_guard_admin():
    return set(_guard_admin_labels())


def _token(session_id="tenant-a"):
    from lasagna.api.rest.session_token import create_session_token

    token, _expiry = create_session_token(session_id=session_id, user_id="u1")
    return token


#: (callable, kwargs) for each tool that REST gates behind X-Admin-Token.
ADMIN_TOOLS = [
    pytest.param(
        mcp_server.bulk_delete_session,
        {"session_id": "victim", "confirm": True},
        id="bulk_delete_session",
    ),
    pytest.param(
        mcp_server.get_lifecycle_policy, {"session_id": "victim"}, id="get_lifecycle_policy"
    ),
    pytest.param(
        mcp_server.set_lifecycle_policy,
        {"session_id": "victim", "policy_updates": '{"gc_cycle_hours": 1}'},
        id="set_lifecycle_policy",
    ),
    pytest.param(mcp_server.get_gc_status, {"session_id": "victim"}, id="get_gc_status"),
    pytest.param(
        mcp_server.rebuild_collection,
        {"session_id": "victim", "confirm": True},
        id="rebuild_collection",
    ),
]


class TestTheAdminScopeToolsRefuseWithoutTheAdminToken:
    @pytest.mark.parametrize("tool,kwargs", ADMIN_TOOLS)
    def test_no_admin_token_is_refused(self, tool, kwargs, exposed):
        with pytest.raises(transport_guard.MCPAuthError):
            tool(**kwargs)

    @pytest.mark.parametrize("tool,kwargs", ADMIN_TOOLS)
    def test_a_wrong_admin_token_is_refused(self, tool, kwargs, exposed):
        with pytest.raises(transport_guard.MCPAuthError):
            tool(admin_token="not-the-admin-token", **kwargs)

    @pytest.mark.parametrize("tool,kwargs", ADMIN_TOOLS)
    def test_a_session_token_is_not_an_admin_token(self, tool, kwargs, exposed):
        """Holding a valid session must not confer instance-wide authority --
        that is the whole distinction the admin token exists to draw."""
        with pytest.raises(transport_guard.MCPAuthError):
            tool(admin_token=_token(), **kwargs)

    @pytest.mark.parametrize("tool,kwargs", ADMIN_TOOLS)
    def test_an_unset_admin_token_refuses_rather_than_admits(
        self, tool, kwargs, exposed, monkeypatch
    ):
        """REST returns 503 when RAG_ADMIN_TOKEN is unset. Opening up because
        the operator forgot to configure a secret is the wrong direction for an
        operation that can delete every session."""
        monkeypatch.delenv("RAG_ADMIN_TOKEN", raising=False)
        with pytest.raises(transport_guard.MCPAuthError, match="not configured"):
            tool(admin_token="anything", **kwargs)

    @pytest.mark.parametrize("tool,kwargs", ADMIN_TOOLS)
    def test_the_refusal_raises_rather_than_returning_a_string(self, tool, kwargs, exposed):
        """These tools return JSON error strings for ordinary failures, which a
        model may summarise, retry past, or ignore. A refusal must not be
        something the caller can talk its way around."""
        try:
            result = tool(**kwargs)
        except transport_guard.MCPAuthError:
            return
        pytest.fail(f"returned {result!r} instead of raising")

    @pytest.mark.parametrize("tool,kwargs", ADMIN_TOOLS)
    def test_every_admin_tool_accepts_the_parameter(self, tool, kwargs):
        """A tool in ADMIN_SCOPE_TOOLS with no way to pass the credential would
        be permanently unusable on an exposed server."""
        import inspect

        assert "admin_token" in inspect.signature(tool).parameters

    def test_the_declared_set_is_exactly_the_guarded_set(self):
        """The first draft of this had an ``or len(...) == len(...)`` escape
        hatch, which made it pass for any two sets of the same size. Read the
        source: a name in ``ADMIN_SCOPE_TOOLS`` that no function guards is a
        false sense of security, and a function that guards without being
        declared makes the constant a lie."""
        assert _functions_calling_guard_admin() == set(transport_guard.ADMIN_SCOPE_TOOLS)

    def test_the_declared_set_is_exactly_what_this_file_exercises(self):
        assert {p.id for p in ADMIN_TOOLS} == set(transport_guard.ADMIN_SCOPE_TOOLS)

    @pytest.mark.parametrize("tool,kwargs", ADMIN_TOOLS)
    def test_the_operator_path_is_reachable(self, tool, kwargs, exposed):
        """Both guards apply: the admin token authorises the *operation*, the
        session token establishes *who*. An operator holding only the admin
        token would be locked out if the second had no answer -- it does, the
        anonymous ``default`` namespace, which is also what the REST admin
        routes use (they take no session at all).

        Reaching the backend is the assertion; what it then does is not this
        test's business, so a backend failure counts as a pass.
        """
        kwargs = {**kwargs, "session_id": "default"}
        try:
            tool(admin_token=ADMIN, **kwargs)
        except transport_guard.MCPAuthError as exc:
            pytest.fail(f"a correctly credentialled operator was refused: {exc}")
        except Exception:
            pass

    def test_each_guard_names_its_own_function(self):
        """`guard_admin(..., tool="...")` takes a hand-written string, and the
        message it produces is the only thing telling a caller which tool to
        fix. A rename would silently misdirect them."""
        for func_name, declared in _guard_admin_labels().items():
            assert declared == func_name, f"{func_name} reports itself as {declared!r}"


class TestStdioIsUnchanged:
    """The client launched this process and owns its pipes. Requiring a token
    would break every local MCP client config and protect nobody."""

    @pytest.mark.parametrize("tool,kwargs", ADMIN_TOOLS)
    def test_admin_tools_are_not_gated_over_stdio(self, tool, kwargs, local, monkeypatch):
        """Reaching the guard is the assertion; whatever the backend does after
        is not this test's business, so a backend failure counts as a pass."""
        try:
            tool(**kwargs)
        except transport_guard.MCPAuthError:
            pytest.fail("stdio must not require a credential")
        except Exception:
            pass

    def test_an_asserted_session_id_still_works_over_stdio(self, local):
        sid, user_id = mcp_server._resolve_session("whatever-i-like")
        assert sid == "whatever-i-like"
        assert user_id.startswith("anon_")

    def test_networked_with_auth_off_is_also_unchanged(self, monkeypatch):
        """`require_auth_or_refuse_to_start` makes this combination a startup
        error on any non-loopback bind, so "auth off" means "loopback" -- the
        case stdio's reasoning already covers."""
        transport_guard.set_transport("sse")
        monkeypatch.delenv(REQUIRE_AUTH_ENV, raising=False)
        assert mcp_server._resolve_session("whatever-i-like")[0] == "whatever-i-like"


class TestSessionIdIsAClaimHereToo:
    def test_an_asserted_session_is_refused_when_exposed(self, exposed):
        with pytest.raises(transport_guard.MCPAuthError, match="victim"):
            mcp_server._resolve_session("victim")

    def test_a_verified_token_is_accepted(self, exposed):
        assert mcp_server._resolve_session("ignored", _token("tenant-a"))[0] == "tenant-a"

    def test_a_token_wins_over_an_asserted_id(self, exposed):
        """Otherwise a caller authenticates as themselves and then names someone
        else, which would make the token theatre."""
        assert mcp_server._resolve_session("victim", _token("mine"))[0] == "mine"

    def test_the_anonymous_namespace_is_not_a_free_pass(self, exposed):
        """Matches the REST side: `default` stopped being exempt once it turned
        out the health-check justification for exempting it was false."""
        with pytest.raises(transport_guard.MCPAuthError):
            mcp_server._resolve_session("default")

    def test_an_operator_with_the_admin_token_is_not_blocked(self, exposed):
        """The admin token proves instance-wide authority, which subsumes "who
        am I". Requiring a session token on top would make MCP stricter than
        REST -- whose admin routes take no session at all -- and would leave an
        operator holding the right credential unable to use it."""
        assert mcp_server._resolve_session("default", admin_verified=True)[0] == "default"

    def test_an_invalid_token_does_not_fall_through_to_the_asserted_id(self, exposed):
        """`_resolve_session` used to swallow a decode failure and continue to
        the session_id the caller typed, so presenting garbage was strictly
        better than presenting nothing."""
        with pytest.raises(transport_guard.MCPAuthError, match="invalid or expired"):
            mcp_server._resolve_session("victim", "not-a-jwt")

    def test_an_invalid_token_is_refused_over_stdio_too(self, local):
        """A credential that was presented and did not verify is an error in
        every mode. Silently downgrading sends writes to the wrong namespace."""
        with pytest.raises(transport_guard.MCPAuthError):
            mcp_server._resolve_session("victim", "not-a-jwt")


class TestTheDefaultDirectionIsStrict:
    def test_an_unset_transport_counts_as_networked(self):
        """`api/rest/app.py` mounts this server without `main()` ever running.
        If the default were stdio, mounting it would silently disable the guard.
        """
        import importlib

        module = importlib.reload(transport_guard)
        try:
            assert module.is_networked() is True
        finally:
            importlib.reload(transport_guard)

    @pytest.mark.parametrize("name", ["sse", "http", "streamable-http", "", "anything-else"])
    def test_only_stdio_is_treated_as_local(self, name):
        transport_guard.set_transport(name)
        assert transport_guard.is_networked() is True

    def test_the_rest_mount_declares_the_transport(self):
        """Belt and braces: the default already covers it, but the mount site
        should not depend on a default in another module holding."""
        import inspect

        from lasagna.api.rest import app as app_module

        source = inspect.getsource(app_module.create_app)
        assert 'transport_guard.set_transport("http")' in source

    def test_main_declares_the_transport_before_serving(self):
        """Parsed, not grepped.

        The first draft of this compared `source.index("set_transport")` against
        `source.index("mcp.run")` and failed against correct code, because
        `mcp.run` appears in a comment 30 lines above the call. Substring
        matching over source has produced a false pass and now a false failure
        in this project; the call graph is what the assertion is about, so read
        that.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(mcp_server.main)))

        def lines_calling(name):
            return [
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == name
            ]

        declared = lines_calling("set_transport")
        served = lines_calling("run")
        assert declared, "main() never records the transport"
        assert served, "main() never serves -- this test is checking nothing"
        assert max(declared) < min(served)

    def test_main_passes_the_real_transport_argument(self):
        """`set_transport("stdio")` hardcoded anywhere in main() would satisfy
        the ordering test above while disabling the guard entirely."""
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(mcp_server.main)))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "set_transport"
            ):
                assert node.args, "set_transport called with no argument"
                assert ast.unparse(node.args[0]) == "args.transport"
                return
        pytest.fail("main() never calls set_transport")

    def test_the_sse_server_refuses_an_unsafe_bind(self):
        import inspect

        source = inspect.getsource(mcp_server.main)
        assert "require_auth_or_refuse_to_start" in source


class TestRebuildCollectionIsGuardedLikeWhatItDoes:
    """It drops the shared collection, so it removes every session's documents
    -- strictly more destructive than `bulk_delete_session`, which has always
    had a confirm flag."""

    def test_it_requires_confirmation(self, local):
        result = mcp_server.rebuild_collection(session_id="s1")
        assert "confirm must be true" in result

    def test_the_refusal_says_the_blast_radius(self, local):
        result = mcp_server.rebuild_collection(session_id="s1")
        assert "every session" in result

    def test_the_collection_really_is_shared(self):
        """The premise. If collections were ever per-session this guard would
        be overcautious, and this test should fail rather than the docstring
        quietly becoming wrong."""
        from lasagna.core.config import rag_config

        assert rag_config.collection_name == "rag_documents_v2"
