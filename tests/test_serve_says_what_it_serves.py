"""`lasagna serve` names what a request will run with, and little else.

Its start-up printed forty INFO lines, named the global default embedding model
(BAAI/bge-m3 at 1024 dimensions) while the default session searched with the
profile the user had chosen, warned that RAG_JWT_SECRET was unset on a loopback
server where that changes nothing, and opened a browser tab unasked. It now
logs at WARNING like every other command, prints the session's profile, model,
dimensions, vector store and data directory, and opens a browser only with
``--open``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests._paths import PACKAGE_DIR


def _env(home) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        {
            "HOME": str(home),
            "PYTHONPATH": str(PACKAGE_DIR.parent),
            "HF_HUB_OFFLINE": "1",
            "RAG_DB_DIR": str(home / "dbs"),
            "RAG_UPLOAD_DIR": str(home / "uploads"),
        }
    )
    return env


def _serve(home, *args: str) -> subprocess.CompletedProcess:
    """Run `lasagna serve` with uvicorn stubbed; the stub records what it got."""
    code = (
        "import sys, time, webbrowser, uvicorn\n"
        "webbrowser.open = lambda url: print('OPENED', url, flush=True)\n"
        "def run(app, **k):\n"
        "    print('UVICORN', k.get('log_level', 'info'), flush=True)  # uvicorn's default\n"
        "    __import__(app.split(':')[0])  # what uvicorn imports, and what warns\n"
        "    time.sleep(2.5)  # long enough for a browser thread to fire\n"
        "uvicorn.run = run\n"
        "sys.argv = ['lasagna', *sys.argv[1:]]\n"
        "from lasagna.core.session_config import get_knowledge_config_manager\n"
        "get_knowledge_config_manager().set_active_profile('speed', force=True)  # empty store\n"
        "from lasagna.cli import app\n"
        "app()\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        env=_env(home),
        timeout=300,
    )


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """`configure --profile speed`, then `serve`."""
    result = _serve(tmp_path_factory.mktemp("serve"), "serve", "--port", "8931")
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_the_banner_names_the_sessions_model_not_the_global_default(served):
    out = " ".join(served.stdout.split())
    assert "Profile: speed (session 'default')" in out
    assert "Embedding: all-MiniLM-L6-v2 (384 dimensions)" in out
    assert "bge-m3" not in out
    assert "Vector store: milvus" in out
    # RAG_DB_DIR and RAG_UPLOAD_DIR sit side by side in HOME here.
    assert "Data: ~ " in out, "the data directory is shown relative to home"


def test_it_logs_warnings_only_by_default(served):
    assert "UVICORN warning" in served.stdout
    assert "| INFO" not in served.stderr, served.stderr


def test_an_ephemeral_signing_secret_is_not_worth_a_warning_on_loopback(served):
    assert "RAG_JWT_SECRET" not in served.stderr, served.stderr


def test_no_browser_opens_unless_asked(served):
    assert "OPENED" not in served.stdout
    import typer

    from lasagna import cli

    serve = typer.main.get_command(cli.app).commands["serve"]
    option = next(p for p in serve.params if p.name == "open_browser")
    assert option.default is False and "--open" in option.opts


def test_verbose_reaches_the_server(monkeypatch):
    """`-v` on the root command was reset to INFO by `serve`."""
    from typer.testing import CliRunner

    from lasagna import cli

    ran = {}
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: ran.update(k))
    result = CliRunner().invoke(cli.app, ["-v", "serve", "--port", "8931"])
    assert result.exit_code == 0, result.output
    assert ran["log_level"] == "debug"
