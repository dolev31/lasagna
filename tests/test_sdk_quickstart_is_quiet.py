"""The README's Python quickstart writes nothing to stderr on the speed profile.

It printed gRPC's fork notices ("Other threads are currently calling into
gRPC, skipping fork() handlers", then one "FD from fork parent still in poll
list" line per descriptor, 121 lines in one run) and a "Loading weights" bar for
a model that was already cached. The fork was tqdm's: its first bar creates a
multiprocessing lock, which starts the resource tracker while Milvus Lite's
gRPC is running.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from tests._paths import PACKAGE_DIR

QUICKSTART = """
import asyncio
from lasagna.sdk import KnowledgeClient, config_for_speed

async def main():
    client = KnowledgeClient(session_id="quickstart", config=config_for_speed())
    doc = await client.upload("paper.md")
    await client.documents.wait_for_processing(doc.document_id)

    results = await client.search_documents("What did the ablation show?")
    for r in results.results:
        print(f"[{r.score:.2f}] {r.chunk.content[:120]}")

    await client.close()

asyncio.run(main())
"""


def _minilm_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
    return isinstance(hit, str)


@pytest.mark.skipif(not _minilm_cached(), reason="all-MiniLM-L6-v2 is not cached")
def test_the_quickstart_prints_its_results_and_nothing_else(tmp_path):
    (tmp_path / "paper.md").write_text(
        "# Ablation\n\nThe ablation showed that removing the reranker lowered NDCG@10 "
        "by 0.04 on SciFact.\n"
    )
    (tmp_path / "quickstart.py").write_text(QUICKSTART)
    # The library sets no environment variable for this, and neither may the test.
    quiet = ("GRPC_VERBOSITY", "GLOG_minloglevel", "TQDM_DELAY", "TQDM_DISABLE")
    # A new user's process: none of this suite's settings (its shared data
    # directory is held open by the test process), and no quieting variables.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in quiet and not k.startswith(("HF_HUB_DISABLE", "RAG_", "LASAGNA_"))
    }
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(PACKAGE_DIR.parent), HF_HUB_OFFLINE="1")
    run = subprocess.run(
        [sys.executable, "quickstart.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    assert "The ablation showed" in run.stdout
    grpc = [line for line in run.stderr.splitlines() if re.match(r"[IWEF]\d{4} ", line)]
    assert not grpc, grpc[:5]
    assert "it/s" not in run.stderr and "%|" not in run.stderr, run.stderr[-2000:]
