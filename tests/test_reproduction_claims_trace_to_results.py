"""The README's reproduction numbers come from examples/reproduction/results.csv.

The README said BM25 lands "within 0.008" of the BEIR paper. That compares
Lasagna's BM25 over one field with BEIR's Anserini run over separate title and
text fields; examples/reproduction/README.md gives the like-for-like gap,
0.015 on SciFact, against Anserini's single-field run. The README now says
both, and every figure in its table and its BM25 sentence is checked here
against the measured results.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
RESULTS = REPO / "examples" / "reproduction" / "results.csv"

#: The table's rows, by the method name results.csv uses.
_METHODS = {
    "BM25 (k1 0.9, b 0.4)": "bm25",
    "all-MiniLM-L6-v2": "dense_minilm",
    "bge-small-en-v1.5": "dense_bge_small",
    "BM25 + MiniLM-L6 cross-encoder": "bm25_ce",
}

#: Anserini's single-field ("flat") BM25, from the Pyserini BEIR regressions,
#: as examples/reproduction/README.md cites them.
_ANSERINI_FLAT = {"beir/scifact": 0.679, "beir/nfcorpus": 0.322}


def _measured():
    with RESULTS.open() as f:
        return {(r["method"], r["dataset"]): float(r["ndcg@10"]) for r in csv.DictReader(f)}


def _readme_section() -> str:
    text = README.read_text()
    return text[text.index("## It reproduces published numbers") : text.index("## Quickstart")]


def test_the_table_is_the_measured_results():
    measured = _measured()
    rows = [
        line
        for line in _readme_section().splitlines()
        if line.startswith("| ") and "---" not in line
    ]
    checked = 0
    for line in rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        method = _METHODS.get(cells[0])
        if method is None:
            continue
        scifact, nfcorpus = float(cells[2]), float(cells[4])
        assert scifact == round(measured[(method, "beir/scifact")], 3), line
        assert nfcorpus == round(measured[(method, "beir/nfcorpus")], 3), line
        checked += 1
    assert checked == len(_METHODS)


def test_the_bm25_sentence_states_the_like_for_like_gap():
    section = " ".join(_readme_section().split())
    measured = _measured()
    for dataset, flat in _ANSERINI_FLAT.items():
        gap = round(flat - round(measured[("bm25", dataset)], 3), 3)
        assert f"{gap:.3f}" in section, f"{dataset}: like-for-like gap {gap:.3f} not stated"
    assert "like for like" in section
    assert not re.search(r"within 0\.008 for BM25", section)
