"""`lasagna-mcp --help` prints its options without loading the RAG stack.

The server module imported the REST backend at module scope, and importing any
part of `lasagna.api` built it, so asking for the options loaded torch, the
embedding models' code and pymilvus: 8 s for a usage line. The backend is now
imported on first use.
"""

from __future__ import annotations

import os
import subprocess
import sys

from tests._paths import PACKAGE_DIR

HEAVY = ("torch", "transformers", "pymilvus", "lasagna.api.rag_backend", "lasagna.api.rest.app")


def test_help_is_printed_without_the_backend(tmp_path):
    code = (
        "import sys, time\n"
        "start = time.perf_counter()\n"
        "sys.argv = ['lasagna-mcp', '--help']\n"
        "from lasagna.mcp.server import main\n"
        "try:\n"
        "    main()\n"
        "except SystemExit as exc:\n"
        "    assert not exc.code, exc.code\n"
        "print('SECONDS', time.perf_counter() - start)\n"
        f"print('LOADED', [m for m in {HEAVY!r} if m in sys.modules])\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update({"HOME": str(tmp_path), "PYTHONPATH": str(PACKAGE_DIR.parent)})
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert out.returncode == 0, out.stderr
    assert "--transport" in out.stdout
    assert "LOADED []" in out.stdout, out.stdout
    seconds = float(out.stdout.split("SECONDS ")[1].split()[0])
    assert seconds < 3.0, f"--help took {seconds:.1f} s"
