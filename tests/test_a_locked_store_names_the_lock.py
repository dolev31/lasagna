"""A second process on a Milvus Lite store is told the store is in use.

Running a benchmark while another held the store ended with "Failed to connect
to Milvus: Open local milvus failed". The cause, another process holding the
lock, appeared only in a traceback logged above it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("milvus_lite")


def test_a_store_held_by_another_process_names_the_lock(tmp_path):
    from lasagna.vector_backends.milvus_backend import MilvusBackend

    store = tmp_path / "rag_milvus.db"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from pymilvus import MilvusClient\n"
            "MilvusClient(sys.argv[1])\n"
            "print('held', flush=True)\n"
            "time.sleep(120)\n",
            str(store),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(ConnectionError) as refused:
            MilvusBackend(db_path=str(store)).connect()
    finally:
        holder.kill()
        holder.wait(timeout=30)
    message = str(refused.value)
    assert "in use by another process" in message
    assert "RAG_DB_DIR" in message
