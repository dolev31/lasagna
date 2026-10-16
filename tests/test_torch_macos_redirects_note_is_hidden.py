"""torch's macOS note "Redirects are currently not supported" is dropped, and only it.

transformers 5.20 imports torch's distributed checkpointing, which imports
torchelastic's output redirection; on macOS that module logs "NOTE: Redirects
are currently not supported in MacOs." as it loads. Users install from PyPI
without the constraints file, so they get the newest transformers, and the
README quickstart printed that line on every run. It says torchelastic cannot
redirect a worker's output on this OS, and Lasagna starts no torchelastic
worker. The pinned 5.19.0 does not import the module, so the probe logs the
note the way torch does on macOS, on any platform.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

_PROBE = textwrap.dedent(
    """
    import logging
    import lasagna  # noqa: F401
    try:
        from torch.distributed.elastic.multiprocessing import redirects
    except Exception:
        print("NO_TORCHELASTIC")
        raise SystemExit(0)
    redirects.IS_MACOS = True
    redirects.get_libc()
    redirects.logger.warning("another warning from the same module")
    logging.getLogger("torch.distributed.elastic.multiprocessing.api").warning(
        "a warning from a neighboring module"
    )
    """
)


def test_the_redirects_note_is_hidden_and_other_torch_warnings_are_not(tmp_path):
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "RAG_DB_DIR": str(tmp_path / "dbs"),
        "RAG_UPLOAD_DIR": str(tmp_path / "up"),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    if "NO_TORCHELASTIC" in proc.stdout:
        pytest.skip("this torch build has no torch.distributed.elastic")
    assert "Redirects are currently not supported" not in proc.stderr, proc.stderr[-2000:]
    assert "another warning from the same module" in proc.stderr, proc.stderr[-2000:]
    assert "a warning from a neighboring module" in proc.stderr, proc.stderr[-2000:]
