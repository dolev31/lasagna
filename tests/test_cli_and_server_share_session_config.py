"""A profile set by one process reaches another that shares the store.

`lasagna configure --profile speed -s cliOnly` while `lasagna serve` ran left
the server answering ``active_profile: default`` for that session. Both used
the persistent store, but each process read it only when it started: a
session the server had not cached began from defaults, a cached one never
refreshed, and every write put back the writer's copy of *every* cached
session, including stale ones another process had since changed. A session
whose active profile was a saved custom one also came back as default after a
restart.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from lasagna.core.session_config import _session_state

REPO = Path(__file__).resolve().parents[1]


def _process(tmp_path):
    """A manager with its own cache, as another process has, on the shared store."""
    with patch(
        "lasagna.core.session_config.KnowledgeConfigManager._resolve_user_data_path",
        return_value=str(tmp_path),
    ):
        from lasagna.core.session_config import KnowledgeConfigManager

        return KnowledgeConfigManager()


@pytest.fixture
def sid():
    return f"shared-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def _no_recheck_delay(monkeypatch):
    monkeypatch.setattr(_session_state, "_RECHECK_SECONDS", 0.0)


def test_a_session_set_elsewhere_after_startup_is_seen(tmp_path, sid):
    server, cli = _process(tmp_path), _process(tmp_path)
    cli.set_active_profile("speed", session_id=sid, force=True)
    assert server.get_active_profile(sid) == "speed"


def test_a_cached_session_picks_up_a_change_made_elsewhere(tmp_path, sid):
    server, cli = _process(tmp_path), _process(tmp_path)
    assert server.get_active_profile(sid) == "default"
    cli.set_active_profile("speed", session_id=sid, force=True)
    cli.patch_config({"rerank_top_k": 13}, session_id=sid)
    assert server.get_active_profile(sid) == "speed"
    assert server.get_resolved_config(sid)["rerank_top_k"] == 13


def test_a_write_does_not_put_back_a_stale_copy_of_another_session(tmp_path, sid, monkeypatch):
    server, cli = _process(tmp_path), _process(tmp_path)
    other = f"{sid}-other"
    server.get_session_state(sid)  # cached as default
    cli.set_active_profile("speed", session_id=sid, force=True)
    monkeypatch.setattr(_session_state, "_RECHECK_SECONDS", 3600.0)  # stays stale
    server.patch_config({"rerank_top_k": 4}, session_id=other)
    assert _process(tmp_path).get_active_profile(sid) == "speed"


def test_a_saved_active_profile_survives_a_restart(tmp_path, sid):
    name = f"mine_{uuid.uuid4().hex[:6]}"
    first = _process(tmp_path)
    first.save_profile(name, {"rerank_top_k": 6}, session_id=sid)
    restarted = _process(tmp_path)
    assert restarted.get_active_profile(sid) == name
    assert restarted.get_resolved_config(sid)["rerank_top_k"] == 6
    first.delete_profile(name, session_id=sid)


# configure checks that the new profile's models load before it switches, so
# the speed profile's embedder must be cached (or downloadable: CI's weights job).
# The child inherits the suite's HF_HUB_OFFLINE, which is on unless a run opts in.
@pytest.mark.needs_embedding_weights("all-MiniLM-L6-v2")
def test_lasagna_configure_reaches_a_process_already_running(tmp_path, sid):
    server = _process(tmp_path)
    assert server.get_active_profile(sid) == "default"
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("LASAGNA_")},
        "PYTHONPATH": str(REPO / "src"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "lasagna", "configure", "--profile", "speed", "-s", sid, "--force"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-500:]
    assert server.get_active_profile(sid) == "speed"
