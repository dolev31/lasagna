"""Two processes that change one session at once both keep their change.

``patch_config`` applied a patch to the process's copy of the session, up to a
second old, and wrote the whole row back with no lock across processes. Two api
workers patching different keys of one session at the same moment kept only
the second patch, and a change of ``rag_enabled`` or the focused documents put
back the overrides it had read. Every change now holds the session's lock in
the store and re-reads the session under it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Each child reads the session (so its copy is cached), waits until the other
# has too, then changes it; the one named second changes it a moment later,
# well inside the second its copy may stay cached.
_CHILD = textwrap.dedent(
    """
    import json, os, sys, time
    from lasagna.core.session_config import get_knowledge_config_manager

    change, me, other, second = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
    m = get_knowledge_config_manager()
    m.get_session_state("shared")
    open(me, "w").close()
    while not os.path.exists(other):
        time.sleep(0.01)
    if second:
        time.sleep(0.3)
    kind, value = change.split("=", 1)
    if kind == "rag_enabled":
        m.set_rag_enabled(value == "true", session_id="shared")
    elif kind == "focus":
        m.set_focused_document_ids([value], session_id="shared")
    else:
        m.patch_config({kind: int(value)}, session_id="shared")
    """
)


def _env(tmp_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        PYTHONPATH=str(REPO / "src"),
        RAG_DB_DIR=str(tmp_path / "dbs"),
        RAG_UPLOAD_DIR=str(tmp_path / "uploads"),
        HOME=str(tmp_path / "home"),
    )
    (tmp_path / "home").mkdir()
    return env


def _race(tmp_path: Path, first: str, second: str) -> dict:
    env = _env(tmp_path)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    procs = [
        subprocess.Popen([sys.executable, "-c", _CHILD, first, a, b, "0"], env=env),
        subprocess.Popen([sys.executable, "-c", _CHILD, second, b, a, "1"], env=env),
    ]
    for p in procs:
        assert p.wait(timeout=120) == 0
    read = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from lasagna.core.database import get_rag_database; "
            "print(json.dumps(get_rag_database().get_session('shared')))",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert read.returncode == 0, read.stderr[-2000:]
    import json

    return json.loads(read.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "first, second",
    [
        ("rerank_top_k=7", "default_top_k=9"),
        ("rerank_top_k=7", "rag_enabled=false"),
        ("rerank_top_k=7", "focus=doc_1"),
    ],
)
def test_both_changes_are_kept(tmp_path, first, second):
    row = _race(tmp_path, first, second)
    assert row["session_overrides"].get("rerank_top_k") == 7, row
    if second.startswith("default_top_k"):
        assert row["session_overrides"].get("default_top_k") == 9, row
    elif second.startswith("rag_enabled"):
        assert not row["rag_enabled"], row
    else:
        assert row["focused_document_ids"] == ["doc_1"], row
