"""A plain ``pytest`` run keeps the Hugging Face hub offline.

CONTRIBUTING says to run ``pytest tests/``. Online, that downloaded about
2.3 GB of weights nobody asked for: the isolation tests built ColBERT
(colbertv2.0), monoT5 and the mxbai reranker from their real checkpoints, and
skipped them only when the hub was offline. CI never noticed, because its test
step sets HF_HUB_OFFLINE=1 itself. The suite now defaults to offline, and
``HF_HUB_OFFLINE=0`` is how a run that should download (CI's weights job) says so.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests._paths import PACKAGE_DIR

REPO = PACKAGE_DIR.parents[1]

#: Load the suite's conftest the way pytest does, before any test module, and
#: report what the hub client decided.
_PROBE = (
    "import tests.conftest, huggingface_hub.constants as c, transformers.utils.hub as t;"
    "print(c.HF_HUB_OFFLINE, t.is_offline_mode())"
)


def _hub_offline(**env):
    clean = {
        k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=REPO,
        env={**clean, **env},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.split()


def test_a_plain_run_is_offline():
    assert _hub_offline() == ["True", "True"]


@pytest.mark.parametrize("value", ["0", "false"])
def test_a_run_that_asks_to_download_stays_online(value):
    assert _hub_offline(HF_HUB_OFFLINE=value) == ["False", "False"]
