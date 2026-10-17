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
    stub = (
        "import lasagna.api._profile_guard as g\n"
        "g.stored_embedding_dim = lambda collection, vector_backend=None: 384"
    )
    result = _cli(tmp_path, "configure", "--profile", "balanced", extra=stub)
    assert result.returncode != 0
    assert "384" in result.stdout and "1024" in result.stdout
    assert "--force" in result.stdout
    assert "speed" in _active(tmp_path)

    # A forced switch still checks that the new profile's models load; here they do.
    loads = "\nimport lasagna.core.model_downloads as m\nm.refusal = lambda *a, **k: None"
    forced = _cli(tmp_path, "configure", "--profile", "balanced", "--force", extra=stub + loads)
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
        # What the client bridge leaves behind, plus one real customization;
        # saved, as the bridge saves it (a switch reads the session as stored).
        mgr.patch_config(
            {
                "embedding_model": speed["embedding_model"],
                "embedding_dim": speed["embedding_dim"],
                "rerank_top_k": 17,
            },
            session_id=sid,
            trusted=True,
        )
        mgr.set_active_profile("balanced", session_id=sid)
        resolved = mgr.get_resolved_config(sid)
        assert resolved["embedding_dim"] != speed["embedding_dim"]
        assert resolved["rerank_top_k"] == 17


def test_an_empty_session_switches_while_another_process_holds_the_store(tmp_path):
    """The store being unreadable is "nothing to protect" only for a session with no documents.

    With `lasagna serve` holding the Milvus Lite lock, the guard's own connect
    failed, the failure read as "no documents", and `configure --profile
    standard` printed "Applied profile: standard" over a 384-dim store. The
    metadata database now says whether the session has documents, without the
    store: one that has none switches (the server's readiness check tells the
    user to run exactly this), and one that has some is refused
    (tests/test_config_writes_are_checked_before_they_apply.py).
    """
    assert _cli(tmp_path, "configure", "--profile", "speed").returncode == 0
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from pymilvus import MilvusClient\n"
            "MilvusClient(sys.argv[1])\n"
            "print('held', flush=True)\n"
            "time.sleep(120)\n",
            str(tmp_path / "dbs" / "rag_milvus.db"),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        result = _cli(tmp_path, "configure", "--profile", "standard")
    finally:
        holder.kill()
        holder.wait(timeout=30)
    assert result.returncode == 0, result.stdout
    assert "Applied profile: standard" in result.stdout
    assert "standard" in _active(tmp_path)
