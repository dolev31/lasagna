"""The Colab notebook ``compare_your_reranker.ipynb`` as a plain script.

It runs the notebook's steps in order, in a working directory of its own, and
stops at the first one that fails. It is how the notebook is tested end to end
without Jupyter:

    pip install "lasagna-rag[eval]"
    python -c "import nltk; nltk.download('punkt_tab', quiet=True)"
    python examples/notebooks/compare_your_reranker.py [workdir]

The files it writes are the notebook's ``%%writefile`` cells, character for
character; ``tests/test_the_colab_notebook_runs.py`` checks the two agree. Every
``lasagna`` command runs with this interpreter (``python -m lasagna``), so the
script measures the installation it was started from. Retrieval only, so no
API key is needed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

#: Step 2: the technique, one decorated function.
MY_IDEAS = '''\
import re

import lasagna


@lasagna.reranker
def keyword_overlap(query, text):
    """Score a passage by the share of the query's words it contains."""
    words = set(re.findall(r"\\w+", query.lower()))
    return len(words & set(re.findall(r"\\w+", text.lower()))) / max(len(words), 1)
'''

#: Step 3: three arms, each one change away from the one before it.
SWEEP = """\
name: does keyword_overlap help
benchmarks:
  - name: beir/scifact
    n_samples: 100
tracks: [retrieval]
output_dir: results
n_runs: 1
pipeline:
  base_profile: speed
  config_overrides:
    - {name: cross_encoder, reranker: cross_encoder, chunker: fixed, chunk_size: 2000}
    - {name: keyword_overlap, reranker: "my_ideas:keyword_overlap", chunker: fixed, chunk_size: 2000}
    - {name: keyword_overlap_bm25, reranker: "my_ideas:keyword_overlap", retriever: bm25, chunker: fixed, chunk_size: 2000}
"""

#: Step 6: does the technique compose with hybrid search? A 2x2 factorial.
COMPOSE = """\
name: does keyword_overlap compose with hybrid search
benchmarks:
  - name: beir/scifact
    n_samples: 100
tracks: [retrieval]
output_dir: compose_results
pipeline:
  base_profile: speed
  factorial:
    factor_a: enable_hybrid_search
    factor_b: enable_reranking
    held_fixed: {reranker: "my_ideas:keyword_overlap", chunker: fixed, chunk_size: 2000}
"""

#: What each step must print for the run to count as a pass. An exit code of 0
#: alone would accept, for example, a sweep whose isolation check failed.
_EXPECT = {
    "validate": "Contract satisfied",
    "sweep": "Stage isolation: the configuration that reached each stage",
    "leaderboard": "keyword_overlap",
    # The verdict box's own words, whether 0, 1 or several pairs survive
    # ("No pair survives Holm correction at ...", "1 of 3 pairs survives
    # correction at ..."); "pairs survive correction" missed the one-pair case.
    "significance": "correction at family-wise alpha",
    "bundle": "run_id",
    "check bundle": "VALID",
    "factorial": "Stage isolation: the configuration that reached each stage",
    "interaction": "interaction",
}


def _lasagna(step: str, *args: str, log: Path | None = None) -> str:
    """Run one ``lasagna`` command, echo its output, and fail on a bad result."""
    print(f"\n$ lasagna {' '.join(args)}", flush=True)
    started = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, "-m", "lasagna", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = proc.wait()
    output = "".join(lines)
    if log is not None:
        log.write_text(output)
    print(f"[{step}: exit {code}, {time.monotonic() - started:.0f} s]", flush=True)
    if code != 0:
        raise SystemExit(f"step {step!r} failed with exit code {code}")
    if "Stage isolation FAILED" in output or _EXPECT[step] not in output:
        raise SystemExit(f"step {step!r} did not print {_EXPECT[step]!r}")
    return output


def main(workdir: Path) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"Working in {workdir}")
    os.chdir(workdir)

    # 2. Write the technique, and check it meets the reranker contract.
    Path("my_ideas.py").write_text(MY_IDEAS)
    _lasagna("validate", "component", "validate", "my_ideas:keyword_overlap")

    # 3. Measure it against a real control in a complete pipeline.
    Path("sweep.yaml").write_text(SWEEP)
    sweep_started = time.monotonic()
    _lasagna("sweep", "benchmark", "experiment", "sweep.yaml", log=Path("sweep.log"))
    sweep_seconds = time.monotonic() - sweep_started

    # 4. Read the result.
    _lasagna("leaderboard", "results", "leaderboard", "results/results.json", "--metric", "ndcg@10")
    _lasagna(
        "significance",
        "results",
        "significance",
        "results/results.json",
        "--metric",
        "ndcg@10",
        "--all",
    )

    # 5. The isolation line, then a bundle anyone can check.
    log = Path("sweep.log").read_text().splitlines()
    at = next(i for i, line in enumerate(log) if "Stage isolation" in line)
    print("\n".join(log[at : at + 2]))  # what `grep -A1 "Stage isolation"` shows
    _lasagna("bundle", "results", "bundle", "results/results.json", "-o", "run.json")
    _lasagna("check bundle", "results", "validate", "run.json")

    # 6. Does it compose with hybrid search?
    Path("compose.yaml").write_text(COMPOSE)
    _lasagna("factorial", "benchmark", "experiment", "compose.yaml")
    _lasagna(
        "interaction",
        "results",
        "interaction",
        "compose_results/results.json",
        "--metric",
        "ndcg@10",
    )

    total = time.monotonic() - started
    print(f"\nAll steps passed: sweep {sweep_seconds:.0f} s, everything {total:.0f} s.")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="lasagna-notebook-")
    main(Path(target).resolve())
