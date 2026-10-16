"""`lasagna status` lists documents without loading a model, and names the sessions that have them.

Without ``-s`` it said "No documents uploaded yet." while ``my-corpus`` held
three, and it took 7 s without a server: it built the whole backend (torch,
the embedding stack, Milvus Lite) to read rows from the metadata database.
"""

from __future__ import annotations

import os
import subprocess
import sys

from tests._paths import PACKAGE_DIR

STATUS = """
import sys
from lasagna.core.database import get_rag_database

db = get_rag_database()
for i in range(3):
    db.save_document(
        {"document_id": f"doc_{i}", "session_id": "my-corpus", "filename": f"f{i}.md",
         "file_size": 10, "file_type": "md", "status": "ready", "chunks_count": 2},
        f"/uploads/f{i}.md",
        collection_name="rag_documents_v2__all_minilm_l6_v2__384",
    )
db.rebuild_quota_counters()

from lasagna.cli import app

for argv in (["status"], ["status", "-s", "my-corpus"]):
    try:
        app(argv, standalone_mode=False)
    except SystemExit:
        pass
    print("=====")
heavy = [m for m in ("torch", "sentence_transformers", "pymilvus") if m in sys.modules]
print("heavy:", heavy)
"""


def test_status_names_the_sessions_with_documents_and_loads_no_model(tmp_path):
    (tmp_path / "status.py").write_text(STATUS)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        HOME=str(tmp_path / "home"),
        PYTHONPATH=str(PACKAGE_DIR.parent),
        HF_HUB_OFFLINE="1",
        COLUMNS="300",
        RAG_SESSION_BACKEND="persistent",
    )
    run = subprocess.run(
        [sys.executable, "status.py"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stderr[-3000:]
    default, mine, heavy = run.stdout.split("=====")
    assert "No documents in session 'default'. Sessions with documents: my-corpus (3)" in default
    assert "lasagna status -s my-corpus" in default
    assert "f0.md" in mine and "Documents: 3" in mine
    assert "heavy: []" in heavy, heavy
