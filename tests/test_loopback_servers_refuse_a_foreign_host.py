"""With auth off, the REST API and the MCP server answer only to loopback names.

A web page the operator visits can point its own host name at 127.0.0.1 (DNS
rebinding) and then call the local server as if it were the page's own origin.
The REST API answered ``Host: attacker.example`` with 200. The compose MCP
service binds 0.0.0.0, where FastMCP turns its Host check off, and its safe
tool set includes ``archive_document``, ``patch_rag_config`` and
``reindex_document``.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from lasagna.mcp import transport_guard

#: An MCP message endpoint for a session that does not exist: past the Host
#: check the answer is 404, so a 421 can only come from the check.
_MESSAGES = "/messages/?session_id=0123456789abcdef0123456789abcdef"


@pytest.fixture
def auth_off(monkeypatch):
    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("RAG_ALLOWED_HOSTS", raising=False)


def _rest(base_url: str) -> TestClient:
    from lasagna.api.rest.app import create_app

    return TestClient(create_app(), base_url=base_url)


def test_rest_refuses_a_foreign_host_and_names_the_setting(auth_off):
    response = _rest("http://attacker.example:8000").get("/api/v1/documents")
    assert response.status_code == 421, response.text
    assert "RAG_ALLOWED_HOSTS" in response.json()["detail"]


@pytest.mark.parametrize(
    "base_url",
    ["http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000", "http://localhost"],
)
def test_rest_answers_its_loopback_names(auth_off, base_url):
    assert _rest(base_url).get("/").status_code == 200


def test_rest_answers_a_name_the_operator_allowed(auth_off, monkeypatch):
    monkeypatch.setenv("RAG_ALLOWED_HOSTS", "api:*, rag.internal")
    assert _rest("http://api:8000").get("/").status_code == 200
    assert _rest("http://rag.internal").get("/").status_code == 200
    # `name` is that name without a port; `name:*` is any port.
    assert _rest("http://rag.internal:8000").get("/").status_code == 421
    assert _rest("http://api.attacker.example:8000").get("/").status_code == 421


def test_probes_by_pod_address_are_answered(auth_off):
    """A kubelet probe or a Prometheus scrape names the pod IP in Host."""
    client = _rest("http://10.1.2.3:8000")
    assert client.get("/api/v1/health/live").status_code == 200
    assert client.get("/metrics").status_code != 421
    assert client.get("/api/v1/documents").status_code == 421


def test_with_auth_on_the_host_is_not_checked(monkeypatch):
    """Every request then needs a token a rebinding page does not have."""
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", "a-jwt-secret-for-tests-0123456789abcdef")
    assert _rest("http://rag.example.com").get("/").status_code == 200


@pytest.fixture
def run_mcp_main(monkeypatch):
    """Run ``lasagna-mcp`` in this process and return the server it built."""
    from lasagna.mcp import server as mcp_server

    for name in ("mcp", "_eager_init", "_owns_logging", "_pinned_level"):
        monkeypatch.setattr(mcp_server, name, getattr(mcp_server, name))
    monkeypatch.setattr(mcp_server, "_configure_rag_logging", lambda level: None)
    monkeypatch.setattr("lasagna._quiet.quiet_third_party_output", lambda: None)
    monkeypatch.setenv("RAG_SESSION_BACKEND", "memory")
    before = transport_guard.get_transport()

    built = {}
    real = mcp_server.create_mcp_server

    def build(**kwargs):
        server = built["server"] = real(**kwargs)
        server.run = lambda transport: built.setdefault("ran", transport)
        return server

    monkeypatch.setattr(mcp_server, "create_mcp_server", build)

    def run(*argv: str):
        monkeypatch.setattr("sys.argv", ["lasagna-mcp", *argv])
        mcp_server.main()
        assert built.get("ran") == "sse", "main() never served"
        return built["server"]

    yield run
    transport_guard.set_transport(before)


def test_the_mcp_server_on_every_interface_refuses_a_foreign_host(
    auth_off, monkeypatch, run_mcp_main
):
    # How docker-compose runs it: 0.0.0.0 inside the container, published on
    # loopback, and its own service name allowed.
    monkeypatch.setenv("RAG_BIND_IS_PROTECTED", "true")
    monkeypatch.setenv("RAG_ALLOWED_HOSTS", "mcp:*")
    server = run_mcp_main("--transport", "sse", "--host", "0.0.0.0", "--lazy-init")

    foreign = TestClient(server.sse_app(), base_url="http://attacker.example:8005")
    assert foreign.post(_MESSAGES, json={}).status_code == 421
    for base_url in ("http://127.0.0.1:8005", "http://mcp:8005"):
        client = TestClient(server.sse_app(), base_url=base_url)
        assert client.post(_MESSAGES, json={}).status_code == 404, base_url


def test_the_loopback_mcp_server_accepts_the_allowed_names(auth_off, monkeypatch, run_mcp_main):
    """On loopback FastMCP had its own list, which ignored RAG_ALLOWED_HOSTS."""
    monkeypatch.setenv("RAG_ALLOWED_HOSTS", "mcp:*")
    server = run_mcp_main("--transport", "sse", "--lazy-init")
    client = TestClient(server.sse_app(), base_url="http://mcp:8005")
    assert client.post(_MESSAGES, json={}).status_code == 404


def test_compose_lets_its_own_services_reach_each_other_by_name():
    """Other containers call the api and mcp by service name, not localhost."""
    import re

    yaml = pytest.importorskip("yaml")
    from tests._paths import DEPLOY_DIR

    compose = yaml.safe_load((DEPLOY_DIR / "docker-compose.yml").read_text())
    value = compose["x-env"]["RAG_ALLOWED_HOSTS"]
    default = re.fullmatch(r"\$\{RAG_ALLOWED_HOSTS:-(.*)\}", value).group(1)
    for service in ("api", "mcp"):
        assert service in compose["services"]
        assert f"{service}:*" in default.split(","), default
        assert compose["services"][service]["environment"]["RAG_ALLOWED_HOSTS"] == value


def test_the_mcp_refusal_names_the_setting_and_health_still_answers(auth_off):
    from lasagna.mcp import server as mcp_server

    server = mcp_server.create_mcp_server(
        host="0.0.0.0", port=8005, transport_security=transport_guard.http_transport_security()
    )
    client = TestClient(server.sse_app(), base_url="http://attacker.example:8005")
    response = client.post(_MESSAGES, json={})
    assert response.status_code == 421
    assert "RAG_ALLOWED_HOSTS" in response.text
    assert client.get("/health").status_code != 421
