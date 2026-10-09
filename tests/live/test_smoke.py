"""Phase 1: Smoke tests — verify the service is up and healthy.

Prerequisites:
    lasagna serve

This file used to assert that "all three services started by `lasagna serve`"
were reachable: the API, a registry on :8001 and a demo server on :7860. The
latter two are CUGA services that Lasagna does not have and never started, so
those tests skipped on a fixture that could not succeed. `lasagna serve` starts
one thing — the REST API, with the configurator mounted inside it.
"""

import requests


class TestServiceHealth:
    """Verify what `lasagna serve` actually starts."""

    def test_health_endpoint(self, api_server):
        """The versioned health route reports a healthy backend."""
        r = requests.get(f"{api_server}/api/v1/health", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") in ("healthy", "ok", True), f"Unexpected health: {data}"

    def test_liveness_probe(self, api_server):
        """`/api/v1/health/live` is what the container healthcheck polls, so it
        has to answer without touching the backend."""
        r = requests.get(f"{api_server}/api/v1/health/live", timeout=10)
        assert r.status_code == 200

    def test_root_endpoint(self, api_server):
        """Root returns service info."""
        r = requests.get(f"{api_server}/", timeout=10)
        assert r.status_code == 200

    def test_openapi_schema_is_served(self, api_server):
        """The SDK and the configurator are both generated against this."""
        r = requests.get(f"{api_server}/openapi.json", timeout=10)
        assert r.status_code == 200
        assert "paths" in r.json()


class TestTheServerIsConfiguredAsExpected:
    def test_cross_origin_is_denied_unless_configured(self, api_server):
        """The default changed from `*` to deny. A wildcard here would mean any
        page the operator visits can drive this API from their browser."""
        r = requests.options(
            f"{api_server}/api/v1/documents",
            headers={
                "Origin": "https://not-a-configured-origin.example",
                "Access-Control-Request-Method": "DELETE",
            },
            timeout=10,
        )
        assert r.headers.get("access-control-allow-origin") != "*"

    def test_an_asserted_session_id_is_refused_when_auth_is_on(self, api_server):
        """Only meaningful against a server started with RAG_REQUIRE_AUTH, so it
        reads the server's own report rather than assuming either mode."""
        health = requests.get(f"{api_server}/api/v1/health", timeout=10).json()
        if not health.get("require_auth"):
            import pytest

            pytest.skip("server is running without RAG_REQUIRE_AUTH")
        r = requests.get(
            f"{api_server}/api/v1/documents",
            headers={"X-Session-ID": "someone-elses-session"},
            timeout=10,
        )
        assert r.status_code == 401
