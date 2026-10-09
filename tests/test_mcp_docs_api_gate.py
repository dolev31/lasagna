"""The /docs tester endpoint can invoke any tool, so it must be opt-in.

`POST /api/call` reaches every registered tool with arbitrary arguments and no
authentication — `delete_document`, `bulk_delete_session`, `reset_rag_config`
included. It was registered unconditionally while the MCP README recommended
`--host 0.0.0.0` as the *production* command, so following the documentation
published unauthenticated document deletion on every interface.

The endpoint exists only to power the "try it" button on the `/docs` page, so
requiring an explicit opt-in costs nothing real.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from lasagna.mcp.server import _docs_api_enabled


class TestItIsOffByDefault:
    def test_disabled_when_unset(self, monkeypatch):
        monkeypatch.delenv("LASAGNA_ENABLE_DOCS_API", raising=False)
        assert _docs_api_enabled() is False

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
    def test_disabled_for_anything_but_an_explicit_yes(self, monkeypatch, value):
        monkeypatch.setenv("LASAGNA_ENABLE_DOCS_API", value)
        assert _docs_api_enabled() is False, f"{value!r} should not enable it"

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
    def test_enabled_only_for_an_explicit_yes(self, monkeypatch, value):
        monkeypatch.setenv("LASAGNA_ENABLE_DOCS_API", value)
        assert _docs_api_enabled() is True


class TestTheUnsafeCombinationIsRefused:
    """A warning in a server's startup log is not read by whoever it affects."""

    def _run(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "lasagna.mcp", *args],
            capture_output=True,
            text=True,
            timeout=120,
        )

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10"])
    def test_public_bind_plus_docs_api_exits_nonzero(self, host):
        result = self._run("--transport", "sse", "--host", host, "--enable-docs-api")
        assert result.returncode != 0, "the server started with an exposed tool endpoint"
        assert "unauthenticated" in (result.stderr + result.stdout).lower()

    def test_localhost_plus_docs_api_is_allowed(self):
        """The legitimate use — a developer testing on their own machine."""
        result = self._run(
            "--transport", "sse", "--host", "127.0.0.1", "--enable-docs-api", "--help"
        )
        assert result.returncode == 0

    def test_the_flag_is_documented_in_help(self):
        result = self._run("--help")
        assert "--enable-docs-api" in result.stdout
        assert "unauthenticated" in result.stdout.lower()


class TestTheEndpointItselfRefuses:
    """The load-bearing check, and the one I nearly shipped without.

    Testing `_docs_api_enabled()` and the CLI proves the switch exists and that
    an unsafe invocation is rejected. Neither proves the *handler* honours the
    switch — I verified that by replacing the guard with `if False:` and
    watching all 19 other tests still pass. Only a real request over the wire
    catches it.
    """

    @pytest.fixture(scope="class")
    def server(self):
        import os
        import socket
        import time
        import urllib.error
        import urllib.request

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        env = dict(os.environ)
        env.pop("LASAGNA_ENABLE_DOCS_API", None)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "lasagna.mcp",
                "--transport",
                "sse",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--lazy-init",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        try:
            for _ in range(120):
                if proc.poll() is not None:
                    pytest.fail("server exited during startup")
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    break
                except (urllib.error.URLError, ConnectionError, OSError):
                    time.sleep(0.5)
            else:
                pytest.fail("server never became healthy")
            yield port
        finally:
            proc.terminate()
            proc.wait(timeout=30)

    def test_calling_a_tool_over_http_is_refused(self, server):
        """A real POST, which is what an attacker would send."""
        import json
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            f"http://127.0.0.1:{server}/api/call",
            data=json.dumps({"tool": "get_system_status", "arguments": {}}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                pytest.fail(
                    f"/api/call invoked a tool while disabled "
                    f"(HTTP {response.status}): {response.read()[:200]!r}"
                )
        except urllib.error.HTTPError as exc:
            assert exc.code == 404, f"expected 404 while disabled, got {exc.code}"
            body = exc.read().decode(errors="replace").lower()
            assert "disabled" in body and "authentication" in body, (
                f"the refusal should say why and how to enable it: {body[:200]}"
            )


class TestTheDocsNoLongerRecommendTheUnsafeCommand:
    def test_readme_does_not_present_a_public_bind_as_production(self):
        from pathlib import Path

        readme = Path(__file__).resolve().parents[1] / "src" / "lasagna" / "mcp" / "README.md"
        text = readme.read_text()
        for line in text.splitlines():
            if "0.0.0.0" in line and "production" in line.lower():
                pytest.fail(
                    f"README still presents a public bind as production without "
                    f"stating what must sit in front of it: {line.strip()}"
                )
