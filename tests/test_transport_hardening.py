"""Transport defaults that decide what a browser and a flood can do.

Two defaults were wrong in ways that only show up from outside the process, so
nothing in the suite could see them.

**CORS defaulted to `["*"]`** with `allow_methods=["*"]` and
`allow_headers=["*"]`. Because this API authenticates with a custom header
rather than a cookie, `allow_credentials=False` bought nothing — the preflight
for `X-Session-ID` succeeded, so any web page a developer visited could issue
`DELETE /api/v1/documents/{id}` against their local instance and the browser
would carry it out. Verified here: the old configuration answers that preflight
`200` with `Access-Control-Allow-Origin: *`.

**`/mcp` and `/configurator` were exempt from rate limiting.** Both carry
destructive operations — `bulk_delete_session` and `clear_session` on the
embedded MCP surface, profile deletion and a full delete-and-reingest on the
configurator — so the most expensive and least authenticated routes in the app
were the only unthrottled ones. The comment that (correctly) kept
`/api/v1/sessions` out of that list applies at least as strongly to both.
"""

from __future__ import annotations

import pytest


def _client(origins):
    fastapi = pytest.importorskip("fastapi")
    testclient = pytest.importorskip("fastapi.testclient")
    from fastapi.middleware.cors import CORSMiddleware

    app = fastapi.FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=(bool(origins) and origins != ["*"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.delete("/doc")
    def delete_doc():
        return {"ok": True}

    return testclient.TestClient(app)


def _preflight(client, origin="https://evil.example"):
    return client.options(
        "/doc",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "DELETE",
            "Access-Control-Request-Headers": "x-session-id",
        },
    )


class TestCrossOriginIsDeniedByDefault:
    def test_the_default_blocks_a_cross_origin_delete(self):
        response = _preflight(_client([]))
        assert response.headers.get("access-control-allow-origin") is None

    def test_the_old_wildcard_default_would_have_allowed_it(self):
        """Kept as evidence, not as a regression target.

        If this ever stops demonstrating the problem, the reasoning in the
        docstring above needs revisiting rather than quietly deleting.
        """
        response = _preflight(_client(["*"]))
        assert response.headers.get("access-control-allow-origin") == "*"

    def test_a_named_origin_does_not_admit_a_different_one(self):
        response = _preflight(_client(["https://app.example"]))
        assert response.headers.get("access-control-allow-origin") is None

    def test_a_named_origin_admits_itself(self):
        response = _preflight(_client(["https://app.example"]), origin="https://app.example")
        assert response.headers.get("access-control-allow-origin") == "https://app.example"

    def test_same_origin_requests_are_untouched(self):
        """The bundled configurator is served by this app, so it never involves
        CORS. Denying cross-origin must not break it."""
        assert _client([]).delete("/doc").status_code == 200

    def test_the_app_builds_a_deny_default(self, monkeypatch):
        monkeypatch.delenv("RAG_CORS_ORIGINS", raising=False)
        import inspect

        from lasagna.api.rest import app as app_module

        source = inspect.getsource(app_module.create_app)
        assert "else list(cors_origins or [])" in source, (
            "the CORS fallback must be an empty list, not a wildcard"
        )

    def test_a_wildcard_is_refused_when_auth_is_required(self, monkeypatch):
        """A wildcard origin lets any site drive authenticated requests from a
        browser, which defeats the point of turning auth on."""
        from lasagna.api.rest import app as app_module
        from lasagna.api.rest.auth_mode import REQUIRE_AUTH_ENV

        monkeypatch.setenv(REQUIRE_AUTH_ENV, "true")
        monkeypatch.setenv("RAG_JWT_SECRET", "a-secret-so-we-reach-the-cors-check")
        monkeypatch.setenv("RAG_CORS_ORIGINS", "*")
        with pytest.raises(ValueError, match="wildcard"):
            app_module.create_app()

    def test_a_missing_signing_secret_is_reported_before_the_cors_problem(self, monkeypatch):
        """Both are startup errors and both must be fixed, so the order is a
        choice: report the one that makes authentication impossible at all
        before the one that makes it circumventable from a browser.

        Stated as a test because the ordering is invisible in the source -- the
        two checks are 20 lines apart with nothing saying which comes first.
        """
        from lasagna.api.rest import app as app_module
        from lasagna.api.rest.auth_mode import REQUIRE_AUTH_ENV

        monkeypatch.setenv(REQUIRE_AUTH_ENV, "true")
        monkeypatch.delenv("RAG_JWT_SECRET", raising=False)
        monkeypatch.setenv("RAG_CORS_ORIGINS", "*")
        with pytest.raises(ValueError, match="RAG_JWT_SECRET"):
            app_module.create_app()


class TestTheDestructiveSurfaceIsRateLimited:
    def test_mcp_is_not_exempt(self):
        from lasagna.api.rest.app import _EXEMPT_PREFIXES

        assert "/mcp" not in _EXEMPT_PREFIXES, (
            "the embedded MCP surface includes bulk_delete_session and "
            "clear_session; exempting it leaves the least authenticated "
            "destructive routes unthrottled"
        )

    def test_the_configurator_is_not_exempt(self):
        from lasagna.api.rest.app import _EXEMPT_PREFIXES

        assert "/configurator" not in _EXEMPT_PREFIXES, (
            "the configurator exposes profile deletion and a reindex that "
            "deletes every document in the session"
        )

    def test_session_creation_is_still_not_exempt(self):
        """It never was, and the comment explaining why is the reasoning the
        two removals above follow."""
        from lasagna.api.rest.app import _EXEMPT_PREFIXES

        assert not any("/sessions" in p for p in _EXEMPT_PREFIXES)

    def test_health_probes_are_still_exempt(self):
        """Guards the guard: emptying the list entirely would also pass the
        assertions above, and would rate-limit kubelet."""
        from lasagna.api.rest.app import _EXEMPT_PREFIXES

        assert "/health" in _EXEMPT_PREFIXES
        assert "/api/v1/health" in _EXEMPT_PREFIXES


class TestSharedStateIsActuallyShared:
    """`configure_rate_limiter` and `configure_quota_manager` unconditionally
    built the in-memory implementations, and `RAGBackend._do_init` calls both on
    every startup — so they overwrote whatever `get_*()` had selected and the
    Redis variants were unreachable.

    The shipped k8s manifests set `RAG_CACHE_BACKEND: redis` and run two
    replicas behind an HPA, so each replica enforced its own private bucket:
    the effective rate limit was N times the configured one, and quota counters
    reset on every pod restart. ~250 lines of Lua token-bucket never executed.
    """

    def test_redis_is_selected_for_the_rate_limiter(self, monkeypatch):
        from lasagna.core import rate_limiter as rl

        monkeypatch.setattr(
            "lasagna.core.config.rag_config._config",
            {
                **__import__("lasagna.core.config", fromlist=["rag_config"]).rag_config._config,
                "cache_backend": "redis",
            },
            raising=False,
        )
        rl.configure_rate_limiter(rl.RateLimiterConfig(enabled=True))
        assert type(rl.get_rate_limiter()).__name__ == "RedisRateLimiter"

    def test_memory_is_still_the_default(self, monkeypatch):
        from lasagna.core import rate_limiter as rl

        rl.configure_rate_limiter(rl.RateLimiterConfig(enabled=True))
        assert type(rl.get_rate_limiter()).__name__ == "RateLimiter"

    def test_the_configure_functions_consult_the_cache_backend(self):
        """Read from source: the defect was that they did not look at all."""
        import inspect

        from lasagna.core import quota, rate_limiter

        for module, func in (
            (rate_limiter, "configure_rate_limiter"),
            (quota, "configure_quota_manager"),
        ):
            source = inspect.getsource(getattr(module, func))
            assert "cache_backend" in source, (
                f"{func} ignores the cache backend, so the Redis implementation "
                f"is unreachable in the served app"
            )
