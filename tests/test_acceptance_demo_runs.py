"""The offline acceptance demo passes every step, from a clean HOME.

``examples/acceptance/offline_demo.py`` is the release's proof that the 0.1.0
features work together as a user meets them: synthesized questions validated
and swept over the baseline arms with a wrapped LangChain reranker, replayed
from the LLM cache, bundled, viewed and typeset, and the MCP server's safe
default. Each feature's own tests check it alone; nothing else checked that one
user can go through all of them in a row, so a change that broke the handoff
between two of them (a synthesized file the sweep refuses, a bundle the
leaderboard cannot group) would have passed the suite.

The demo loads all-MiniLM-L6-v2, so this runs where embedding weights are
allowed. With the hub offline and the model not cached it is skipped; online,
the model is fetched first and the demo itself still runs offline.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._paths import PACKAGE_DIR

DEMO = Path(__file__).resolve().parents[1] / "examples" / "acceptance" / "offline_demo.py"
CANNOT_RUN = 2


def _offline() -> bool:
    return any(
        os.environ.get(var, "").strip().lower() in {"1", "true", "yes", "on"}
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _demo(tmp_path: Path, *args: str, offline: bool = True) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_OFFLINE")}
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(PACKAGE_DIR.parent))
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    return subprocess.run(
        [sys.executable, str(DEMO), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.mark.needs_embedding_weights
def test_every_step_of_the_demo_passes(tmp_path):
    report = tmp_path / "report.json"
    args = ("--workdir", str(tmp_path / "demo"), "--report", str(report))
    run = _demo(tmp_path, *args)
    if run.returncode == CANNOT_RUN:
        if _offline():
            pytest.skip(json.loads(report.read_text())["reason"])
        fetched = _demo(tmp_path, "--download-models", offline=False)
        assert fetched.returncode == 0, fetched.stderr[-2000:]
        run = _demo(tmp_path, *args)

    log = tmp_path / "demo" / "demo.log"
    tail = log.read_text()[-3000:] if log.exists() else run.stderr[-3000:]
    assert run.returncode == 0, f"{run.stdout}\n--- end of demo.log ---\n{tail}"
    record = json.loads(report.read_text())
    assert record["outcome"] == "passed"
    assert [s["key"] for s in record["steps"]] == [key for key, _ in record["planned"]]
    assert len(record["planned"]) == 8 and all(s["ok"] for s in record["steps"])
    assert record["seconds"] < 300, f"the demo took {record['seconds']} s, over its 5 minutes"
    # One line per step, each with its time, and the working folder is cleaned up.
    lines = [ln for ln in run.stdout.splitlines() if ln.startswith("[")]
    assert [ln.split("]")[0] for ln in lines] == [f"[{i}/8" for i in range(1, 9)]
    assert all(": ok (" in ln and " s)." in ln for ln in lines)
    assert not (tmp_path / "demo").exists()
