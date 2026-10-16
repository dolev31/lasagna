"""An MCP client gets the tool list over stdio without waiting for torch.

``--session-backend`` defaulted to ``"persistent"``, so the flag always looked
given, every start reset the providers and called
``lasagna.api.rag_backend.reset_instance``, and importing that module loaded
torch and the embedding stack before the first message was read: about 7 s
from launch to ``tools/list``. The persistent default is now an environment
default, a reset happens only for a flag that was given, and it touches the
backend only if one was loaded.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time

from tests._paths import PACKAGE_DIR

HEAVY = ("torch", "transformers", "sentence_transformers", "pymilvus", "lasagna.api.rag_backend")


def _lines(stream, out: queue.Queue) -> None:
    for line in stream:
        out.put(line)


def _start(tmp_path, *flags: str):
    """`lasagna-mcp` over stdio; at exit it writes its session provider and the
    heavy modules it loaded."""
    report = tmp_path / "loaded.txt"
    code = (
        "import atexit, sys\n"
        f"sys.argv = ['lasagna-mcp', *{list(flags)!r}]\n"
        "def report():\n"
        f"    loaded = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "    from lasagna.providers import get_session_provider\n"
        "    sessions = type(get_session_provider()).__name__\n"
        f"    open({str(report)!r}, 'w').write(' '.join([sessions, *loaded]))\n"
        "atexit.register(report)\n"
        "from lasagna.mcp.server import main\n"
        "main()\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        {"HOME": str(tmp_path), "PYTHONPATH": str(PACKAGE_DIR.parent), "HF_HUB_OFFLINE": "1"}
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=open(tmp_path / "stderr.txt", "w"),
        env=env,
        text=True,
    )
    replies: queue.Queue = queue.Queue()
    threading.Thread(target=_lines, args=(proc.stdout, replies), daemon=True).start()
    return proc, replies, report


def _send(proc, message: dict) -> None:
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()


def _list_tools(tmp_path, *flags: str):
    started = time.monotonic()
    proc, replies, report = _start(tmp_path, *flags)
    try:
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
        )
        assert json.loads(replies.get(timeout=60))["id"] == 1
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = json.loads(replies.get(timeout=60))["result"]["tools"]
        seconds = time.monotonic() - started
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    sessions, *loaded = report.read_text().split() if report.exists() else ["<no report>"]
    return tools, seconds, sessions, loaded


def test_the_tool_list_comes_back_quickly_without_torch(tmp_path):
    tools, seconds, sessions, loaded = _list_tools(tmp_path)

    assert {t["name"] for t in tools} >= {"search", "upload_document"}
    assert loaded == [], f"loaded before any tool call: {loaded}"
    assert seconds < 2.0, f"tools/list took {seconds:.1f} s"
    # Profiles saved by `lasagna configure` still apply.
    assert sessions == "PersistentSessionProvider"


def test_a_given_flag_still_applies_without_loading_the_backend(tmp_path):
    tools, _seconds, sessions, loaded = _list_tools(
        tmp_path, "--session-backend", "memory", "--db-dir", str(tmp_path / "dbs")
    )

    assert tools
    assert loaded == [], f"loaded before any tool call: {loaded}"
    assert sessions == "InMemorySessionProvider"
