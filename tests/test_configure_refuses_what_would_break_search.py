"""`lasagna configure` must not record a profile that does not exist or breaks search.

`configure --profile bogus` printed "Applied profile: bogus". Switching to a
profile that embeds at another size than the stored documents succeeded, and
every later search failed with remedies that exist only in Python.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _cli(home: Path, *args: str, extra: str = "") -> subprocess.CompletedProcess:
    code = (
        "import sys\n"
        "from lasagna import cli\n"
        f"{extra}\n"
        "from typer.testing import CliRunner\n"
        "r = CliRunner().invoke(cli.app, sys.argv[1:])\n"
        "print(r.output); sys.exit(r.exit_code)\n"
    )
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))},
        "HOME": str(home),
        "PYTHONPATH": str(REPO / "src"),
        "RAG_DB_DIR": str(home / "dbs"),
        "HF_HUB_OFFLINE": "1",
    }
    return subprocess.run(
        [sys.executable, "-c", code, *args], capture_output=True, text=True, env=env, timeout=300
    )


def _active(home: Path) -> str:
    out = _cli(home, "configure")
    return next((ln for ln in out.stdout.splitlines() if "***" in ln), "")


def test_an_unknown_profile_is_refused_and_names_the_real_ones(tmp_path):
    result = _cli(tmp_path, "configure", "--profile", "bogus")
    assert result.returncode != 0
    assert "Unknown profile 'bogus'" in result.stdout
    assert "speed" in result.stdout


def test_a_profile_that_would_break_search_is_refused_and_the_old_one_kept(tmp_path):
    assert _cli(tmp_path, "configure", "--profile", "speed").returncode == 0
    stub = "cli._embedding_dim_mismatch = lambda sid: (384, 1024)"
    result = _cli(tmp_path, "configure", "--profile", "balanced", extra=stub)
    assert result.returncode != 0
    assert "384" in result.stdout and "1024" in result.stdout
    assert "speed" in _active(tmp_path)

    forced = _cli(tmp_path, "configure", "--profile", "balanced", "--force", extra=stub)
    assert forced.returncode == 0, forced.stdout
    assert "balanced" in _active(tmp_path)


def test_a_profile_switch_applies_after_a_client_copied_the_old_one(tmp_path, monkeypatch):
    # The CLI's client writes the active profile's values back as session
    # overrides, and overrides win: `configure --profile balanced` after speed
    # and an upload was recorded and search kept running speed.
    from unittest.mock import patch

    with patch(
        "lasagna.core.session_config.KnowledgeConfigManager._resolve_user_data_path",
        return_value=str(tmp_path),
    ):
        from lasagna.core.session_config import KnowledgeConfigManager

        mgr = KnowledgeConfigManager()
        sid = "switch"
        mgr.set_active_profile("speed", session_id=sid)
        speed = mgr.get_resolved_config(sid)
        state = mgr.get_session_state(sid)
        # What the client bridge leaves behind, plus one real customization.
        state.session_overrides = {
            "embedding_model": speed["embedding_model"],
            "embedding_dim": speed["embedding_dim"],
            "rerank_top_k": 17,
        }
        mgr.set_active_profile("balanced", session_id=sid)
        resolved = mgr.get_resolved_config(sid)
        assert resolved["embedding_dim"] != speed["embedding_dim"]
        assert resolved["rerank_top_k"] == 17
