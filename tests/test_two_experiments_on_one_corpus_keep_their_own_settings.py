"""Two experiment processes on one corpus each run with their own arm's settings.

The harness writes an arm's settings to its corpus session, named after the
corpus (``bench_corpus_<hash>``) so every run on it reuses its vectors. With the
session store shared by every process (the default since 0.1.0), two experiment
processes on one corpus shared that session's row: each read the other's arm
settings wherever search reads the session (the agentic loop, ingestion), and
the per-process lock around the harness's writes could not stop it. The
harness's sessions now keep their settings in their own process.

Each process here runs one arm through a reranker that records the session's
``rerank_top_k`` as search reads the session store, after both processes have
written their arm settings.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_RECORDER = textwrap.dedent(
    """
    import os, time
    from pathlib import Path

    import lasagna

    _waited = []


    @lasagna.reranker
    def recorder(query, text):
        from lasagna.core.config import rag_config
        from lasagna.knowledge_evaluation_sdk.experiment import _corpus_sessions

        out = Path(os.environ["RECORD_DIR"])
        arm, other = os.environ["ARM"], os.environ["OTHER_ARM"]
        if not _waited:
            # Both processes have written their arm's settings, and a cached
            # copy of the session has had time to be compared with the store.
            (out / f"{arm}.ready").touch()
            deadline = time.time() + 300
            while not (out / f"{other}.ready").exists() and time.time() < deadline:
                time.sleep(0.05)
            time.sleep(1.5)
            _waited.append(True)
        for info in list(_corpus_sessions.values()):
            seen = rag_config.with_session(info["session_id"]).rerank_top_k
            with open(out / f"{arm}.seen", "a") as f:
                f.write(f"{seen}\\n")
        return 0.0
    """
)


def _sweep(tmp_path: Path, arm: str, top_k: int) -> Path:
    path = tmp_path / f"{arm}.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            name: arm {arm}
            benchmarks: [{{name: smoke, split: test}}]
            tracks: [retrieval]
            pipeline:
              base_profile: speed
              config_overrides:
                - {{reranker: "recorder:recorder", rerank_top_k: {top_k}}}
            n_runs: 1
            mode: auto
            output_dir: {tmp_path / ("out-" + arm)}
            """
        )
    )
    return path


@pytest.mark.needs_embedding_weights("all-MiniLM-L6-v2")
def test_each_process_reads_the_settings_of_its_own_arm(tmp_path):
    (tmp_path / "recorder.py").write_text(_RECORDER)
    (tmp_path / "home").mkdir()
    base = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    base.update(
        PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(tmp_path)]),
        HOME=str(tmp_path / "home"),
        # One session store for both, as on one machine; a Milvus Lite file
        # each, which opens in one process at a time.
        RAG_DB_DIR=str(tmp_path / "dbs"),
        RAG_UPLOAD_DIR=str(tmp_path / "uploads"),
        RECORD_DIR=str(tmp_path),
    )
    procs = []
    for arm, other, top_k in (("a", "b", 3), ("b", "a", 7)):
        env = {
            **base,
            "ARM": arm,
            "OTHER_ARM": other,
            "LASAGNA_VECTOR_BACKEND_CONFIG": json.dumps(
                {"db_path": str(tmp_path / f"milvus-{arm}.db")}
            ),
        }
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "lasagna",
                    "benchmark",
                    "experiment",
                    str(_sweep(tmp_path, arm, top_k)),
                ],
                env=env,
                cwd=tmp_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    outputs = [p.communicate(timeout=900)[0] for p in procs]
    for p, out in zip(procs, outputs):
        assert p.returncode == 0, out[-3000:]

    seen = {arm: set((tmp_path / f"{arm}.seen").read_text().split()) for arm in ("a", "b")}
    assert seen == {"a": {"3"}, "b": {"7"}}, seen
