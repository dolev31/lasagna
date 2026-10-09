"""Imported as a library, Lasagna writes nothing to its host's stderr and
leaves the host's logging configuration alone.

It used to log at DEBUG for every user (raw query text included), and importing
the MCP server module rewired the stdlib root logger, because FastMCP calls
logging.basicConfig in its constructor. The CLIs and servers turn logs on.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

_PROBE = textwrap.dedent(
    """
    import logging
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    import lasagna
    from lasagna.core.config import rag_config
    _ = rag_config.default_top_k
    import lasagna.mcp.server  # noqa: F401
    after = (list(root.handlers), root.level)
    print("ROOT_UNCHANGED" if before == after else "ROOT_CHANGED")
    """
)


def test_importing_and_configuring_the_library_is_silent(tmp_path):
    env = {
        **os.environ,
        "RAG_DB_DIR": str(tmp_path / "dbs"),
        "RAG_UPLOAD_DIR": str(tmp_path / "up"),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ROOT_UNCHANGED")
    assert proc.stderr.strip() == "", f"library wrote to stderr:\n{proc.stderr[:2000]}"
