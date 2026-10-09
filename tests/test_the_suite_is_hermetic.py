"""The suite runs in a throwaway home directory with telemetry off.

A run used to leave ``nltk_data/`` (64 MB), ragas's user id and a matplotlib
cache in the developer's real home, and to post ragas usage events. These pin
the isolation ``tests/conftest.py`` sets up, so a refactor of it cannot
quietly undo it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def test_home_is_a_throwaway_directory():
    home = Path.home().resolve()
    assert home.is_relative_to(Path(tempfile.gettempdir()).resolve()), home
    assert Path(os.environ["RAG_DB_DIR"]).resolve().is_relative_to(home.parent)


def test_ragas_usage_events_are_off():
    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"


def test_the_model_cache_is_not_moved_into_the_throwaway_home():
    assert not Path(os.environ["HF_HOME"]).resolve().is_relative_to(Path.home().resolve())


_FORK_PROBE = """
import subprocess, sys
from lasagna.vector_backends.milvus_backend import MilvusBackend

backend = MilvusBackend(db_path=sys.argv[1])
backend.connect()
child = subprocess.run([sys.executable, "-c", "pass"], capture_output=True, text=True)
backend.disconnect()
print(repr(child.stderr))
"""


def test_a_child_process_inherits_no_grpc_fork_noise(tmp_path):
    """With a Milvus connection open, a child process's stderr stays empty.

    Run in a fresh interpreter with the suite's environment, which is how the
    noise appears: gRPC's fork handlers in the parent write to the child's
    stderr between fork and exec.
    """
    import subprocess
    import sys

    probe = subprocess.run(
        [sys.executable, "-c", _FORK_PROBE, str(tmp_path / "m.db")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    assert probe.stdout.strip().splitlines()[-1] == "''"
