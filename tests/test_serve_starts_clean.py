"""`lasagna serve` checks before it starts, and its banner says what is there.

A busy port printed the banner and then, 6 s later, uvicorn's "[Errno 48]
address already in use". Every start warned that APScheduler was not installed,
though a core install leaves it out on purpose. The banner advertised
/configurator, which answers 404 from a source checkout where the page is not
built, and described only session 'default' while the user's documents were in
another session. Stopping the server left "There appear to be 1 leaked
semaphore objects": tqdm's cross-process lock, created by the first bar.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lasagna import cli
from tests._paths import PACKAGE_DIR


def test_a_busy_port_is_refused_before_the_banner(monkeypatch):
    import uvicorn

    from lasagna.core import local_server

    monkeypatch.setattr(local_server, "local_store_held", lambda: False)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("the server started"))
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        r = CliRunner().invoke(cli.app, ["serve", "--port", str(port)], terminal_width=200)
    out = " ".join(r.output.split())
    assert r.exit_code == 1
    assert f"Port {port} on 127.0.0.1 is already in use" in out
    assert f"lasagna serve --port {port + 1}" in out
    assert "Lasagna API" not in out


def test_the_configurator_line_says_how_to_build_a_missing_page(monkeypatch, tmp_path):
    import lasagna.configurator as configurator

    monkeypatch.setattr(configurator, "FRONTEND_DIST", tmp_path / "missing")
    assert "npm run build" in cli._configurator_line("http://127.0.0.1:8000")
    monkeypatch.setattr(configurator, "FRONTEND_DIST", tmp_path)
    assert cli._configurator_line("http://h:1") == "Configurator: http://h:1/configurator"


def test_the_banner_names_the_sessions_with_documents(monkeypatch):
    from lasagna.core import model_downloads

    monkeypatch.setattr(model_downloads, "sessions_with_documents", lambda: {"my-corpus": 76})
    mgr = SimpleNamespace(
        get_chosen_config=lambda sid: {"embedding_model": "all-MiniLM-L6-v2"},
        get_active_profile=lambda session_id=None: "speed",
    )
    assert cli._sessions_with_documents_line(mgr, exclude="default") == [
        "Sessions with documents: my-corpus (76; speed, all-MiniLM-L6-v2)"
    ]


def test_a_progress_bar_creates_no_cross_process_lock(tmp_path):
    code = (
        "from lasagna.vector_backends._pymilvus import import_pymilvus\n"
        "import_pymilvus()\n"
        "from tqdm import tqdm\n"
        "for _ in tqdm(range(2), disable=True):\n"
        "    pass\n"
        "from tqdm.std import TqdmDefaultWriteLock\n"
        "print('LOCK', TqdmDefaultWriteLock.mp_lock)\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(PACKAGE_DIR.parent))
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert run.returncode == 0, run.stderr
    assert "LOCK None" in run.stdout, run.stdout
