"""An imported profile, or a new one made while activating it, reaches search everywhere.

A profile that was imported (``lasagna import``, ``POST /import``) or created by
``PUT /active`` with a config was held by the process that made it. The session
row only names the active profile, so every other process (and every later
request on PostgreSQL) did not find the name among the saved profiles and ran
``default``. Both now save the profile as the caller's before they activate it.
An import also refuses a built-in profile name given other settings, which
search would never have used.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_EXPORT = {
    "schema_version": 2,
    "active_profile": "mine",
    "profiles": [{"name": "mine", "description": "", "config": {"rerank_top_k": 9}}],
    "session_overrides": {},
}

_READ = (
    "import json; from lasagna.core.session_config import get_knowledge_config_manager as g; "
    "m = g(); print(json.dumps([m.get_active_profile('{s}'), m.get_chosen_config('{s}').get('rerank_top_k')]))"
)


def _env(tmp_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    (tmp_path / "home").mkdir(exist_ok=True)
    env.update(
        PYTHONPATH=os.pathsep.join([env.get("PYTHONPATH", ""), str(REPO / "src")]).lstrip(os.pathsep),
        RAG_DB_DIR=str(tmp_path / "dbs"),
        RAG_UPLOAD_DIR=str(tmp_path / "uploads"),
        HOME=str(tmp_path / "home"),
    )
    return env


def _run(env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True, timeout=180)


def _other_process_reads(env: dict, session: str) -> list:
    out = _run(env, "-c", _READ.format(s=session))
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_cli_import_is_seen_by_another_process(tmp_path):
    env = _env(tmp_path)
    export = tmp_path / "export.json"
    export.write_text(json.dumps(_EXPORT))
    done = _run(env, "-m", "lasagna.cli", "import", str(export), "-s", "s1")
    assert done.returncode == 0, done.stdout[-1000:] + done.stderr[-1000:]
    assert _other_process_reads(env, "s1") == ["mine", 9]


def test_a_new_profile_made_while_activating_it_is_seen_by_another_process(tmp_path):
    env = _env(tmp_path)
    made = _run(
        env,
        "-c",
        "from lasagna.core.session_config import get_knowledge_config_manager as g; "
        "g().set_active_profile('brandnew', config={'rerank_top_k': 4}, session_id='s3')",
    )
    assert made.returncode == 0, made.stderr[-2000:]
    assert _other_process_reads(env, "s3") == ["brandnew", 4]


def test_an_import_does_not_give_a_built_in_profile_other_settings(tmp_path):
    env = _env(tmp_path)
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "active_profile": "speed",
                "profiles": [{"name": "speed", "description": "", "config": {"rerank_top_k": 17}}],
            }
        )
    )
    done = _run(env, "-m", "lasagna.cli", "import", str(export), "-s", "s2")
    assert done.returncode != 0
    assert "built-in profile 'speed'" in done.stdout + done.stderr
    assert _other_process_reads(env, "s2")[1] != 17
