"""A profile one process saves is kept when another process saves or deletes one.

Each process read the saved profiles once, at startup, and every save wrote
back the whole list it had read, after deleting every saved profile: process A
saved ``alpha``, process B (started before that) saved ``beta``, and only
``beta`` remained. Sessions on ``alpha`` then came back as ``default``. A
profile is now written and deleted one row at a time, and a process reads the
saved profiles again when a session names one it does not know.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Both children load the saved profiles first, then wait for each other; the
# one named second acts a moment after the first.
_CHILD = textwrap.dedent(
    """
    import os, sys, time
    from lasagna.core.session_config import get_knowledge_config_manager

    action, name, me, other, second = sys.argv[1:6]
    m = get_knowledge_config_manager()
    open(me, "w").close()
    while not os.path.exists(other):
        time.sleep(0.01)
    if second == "1":
        time.sleep(0.5)
    if action == "save":
        m.save_profile(name, {"rerank_top_k": 6}, session_id="s_" + name)
    else:
        m.delete_profile(name, session_id="s_" + name)
    """
)

_READ = textwrap.dedent(
    """
    import json
    from lasagna.core.database import get_rag_database
    from lasagna.core.session_config import get_knowledge_config_manager

    m = get_knowledge_config_manager()
    print(json.dumps({
        "saved": sorted(
            r["profile_name"]
            for r in get_rag_database().get_profiles("__anonymous__")
            if not r["is_builtin"]
        ),
        "alpha_session": m.get_active_profile("s_alpha"),
        "alpha_top_k": m.get_resolved_config("s_alpha").get("rerank_top_k"),
    }))
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
    (tmp_path / "home").mkdir(exist_ok=True)
    return env


def _run(env: dict, *args: str) -> str:
    done = subprocess.run(
        [sys.executable, "-c", *args], env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def _together(tmp_path: Path, env: dict, first: tuple, second: tuple) -> None:
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    for path in (a, b):
        if os.path.exists(path):
            os.remove(path)
    procs = [
        subprocess.Popen([sys.executable, "-c", _CHILD, *first, a, b, "0"], env=env),
        subprocess.Popen([sys.executable, "-c", _CHILD, *second, b, a, "1"], env=env),
    ]
    for p in procs:
        assert p.wait(timeout=120) == 0


def _state(env: dict) -> dict:
    return json.loads(_run(env, _READ).strip().splitlines()[-1])


def test_two_processes_saving_profiles_keep_both(tmp_path):
    env = _env(tmp_path)
    _together(tmp_path, env, ("save", "alpha"), ("save", "beta"))

    state = _state(env)
    assert state["saved"] == ["alpha", "beta"]
    assert state["alpha_session"] == "alpha"
    assert state["alpha_top_k"] == 6


def test_a_delete_in_one_process_removes_only_that_profile(tmp_path):
    env = _env(tmp_path)
    _together(tmp_path, env, ("save", "alpha"), ("save", "beta"))
    _together(tmp_path, env, ("save", "gamma"), ("delete", "beta"))

    assert _state(env)["saved"] == ["alpha", "gamma"]


def test_a_process_finds_a_profile_saved_after_it_started(tmp_path):
    """A session switched to a saved profile reads as on it in every process."""
    env = _env(tmp_path)
    script = textwrap.dedent(
        """
        import os, subprocess, sys
        from lasagna.core.session_config import _session_state, get_knowledge_config_manager

        _session_state._RECHECK_SECONDS = 0.0  # the session is read again on the next call
        running = get_knowledge_config_manager()
        running.get_session_state("s_alpha")
        subprocess.run([sys.executable, "-c", sys.argv[1]], check=True)
        print(running.get_active_profile("s_alpha"),
              running.get_resolved_config("s_alpha").get("rerank_top_k"))
        """
    )
    saver = (
        "from lasagna.core.session_config import get_knowledge_config_manager as g; "
        "g().save_profile('alpha', {'rerank_top_k': 6}, session_id='s_alpha')"
    )
    out = _run(env, script, saver)
    assert out.strip().splitlines()[-1] == "alpha 6", out
