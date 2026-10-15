"""A command run while `lasagna serve` holds the store says so in one sentence.

`lasagna status` with a server running printed milvus_lite's 25-line traceback
and then "Error: Failed to get status: The Milvus Lite store at /var/folders/
...", wrapped over five lines. The vector backend now raises
DataDirLockedError, the CLI recognizes it anywhere in the cause chain and prints
the fix with the path under ``~``, and the traceback is not printed at all.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from lasagna import cli
from tests._paths import PACKAGE_DIR


class DataDirLockedError(ConnectionError):
    """Stands in for the backend's class: the CLI matches on the name."""

    db_path = os.path.join(os.path.expanduser("~"), ".rag_knowledge", "dbs", "rag_milvus.db")


def test_the_cli_names_the_fix_wherever_the_lock_error_is_wrapped(monkeypatch):
    def held(session_id):
        try:
            raise DataDirLockedError("long message with an absolute path")
        except ConnectionError as exc:
            raise RuntimeError("Failed to initialize RAG Backend") from exc

    monkeypatch.setattr(cli, "_get_client", held)
    result = CliRunner().invoke(cli.app, ["status"])
    said = " ".join(result.output.split())
    assert result.exit_code == 1
    assert "Another process is using the local store at ~/.rag_knowledge/dbs/rag_milvus.db" in said
    assert "RAG_DB_DIR=<another directory>" in said
    assert "Failed to get status" not in said
    assert "absolute path" not in said


def test_other_failures_keep_their_own_message(monkeypatch):
    def broken(session_id):
        raise ValueError("embedding size 384 does not match 1024")

    monkeypatch.setattr(cli, "_get_client", broken)
    result = CliRunner().invoke(cli.app, ["status"])
    assert result.exit_code == 1
    assert "Failed to get status: embedding size 384 does not match 1024" in result.output


def test_a_real_held_store(tmp_path):
    pytest.importorskip("milvus_lite")
    home = tmp_path / "home"
    (home / "dbs").mkdir(parents=True)
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
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from pymilvus import MilvusClient\n"
            "MilvusClient(sys.argv[1])\n"
            "print('held', flush=True)\n"
            "time.sleep(120)\n",
            str(home / "dbs" / "rag_milvus.db"),
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.argv = ['lasagna', 'status']\nfrom lasagna.cli import app; app()",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
    finally:
        holder.kill()
        holder.wait(timeout=30)
    assert result.returncode == 1
    said = " ".join(result.stdout.split())
    assert "Error: Another process is using the local store at ~/dbs/rag_milvus.db" in said, said
    assert "Traceback" not in result.stdout + result.stderr, result.stderr
