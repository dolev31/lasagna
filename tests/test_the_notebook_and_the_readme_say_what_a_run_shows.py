"""The Colab notebook and the README comparison table say what a run really shows.

Three things the fourth audit found untrue:

- the notebook said the bundle's warnings "are recorded on purpose:
  `SINGLE_RUN` ... and `UNPINNED_DATASET`", while a retrieval-only run raises
  neither (nothing samples, and the dataset revision is recorded), so
  `results validate` printed only "VALID";
- the notebook's script expected "pairs survive correction", which the verdict
  does not print when exactly one pair survives ("1 of 3 pairs survives
  correction"), so a correct run failed the release check;
- the README's comparison table said "Published methods: 55 components" while
  20 of the 55 named no paper, `mmr` and `cross_encoder` among them. The two
  now cite Carbonell & Goldstein (1998) and Nogueira & Cho (2019), the ReAct
  loop cites Yao et al. (2022), and the table states the count that cite one.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from lasagna.core.registry import registry
from lasagna.knowledge_evaluation_sdk.bundle import _integrity
from lasagna.knowledge_evaluation_sdk.results import verdict_lines

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "examples" / "notebooks" / "compare_your_reranker.ipynb"


def _script():
    spec = importlib.util.spec_from_file_location("_nb_script", NOTEBOOK.with_suffix(".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(a: str, b: str, significant: bool) -> dict:
    return {
        "config_a": a,
        "config_b": b,
        "difference": 0.05,
        "p_value": 0.001 if significant else 0.5,
        "p_adjusted": 0.003 if significant else 0.9,
        "significant": significant,
        "effect_size": 0.8,
        "n_paired": 100,
    }


@pytest.mark.parametrize("survivors", [0, 1, 2])
def test_the_script_accepts_the_verdict_whatever_survives(survivors):
    rows = [_row("a", "b", survivors >= 1), _row("a", "c", survivors >= 2), _row("b", "c", False)]
    verdict = verdict_lines(rows, "ndcg@10", "beir/scifact", 0.05)
    text = " ".join(line.text for line in verdict.lines)
    assert _script()._EXPECT["significance"] in text, text


def test_a_retrieval_only_single_run_raises_neither_warning_the_notebook_named():
    """The revision comes from the catalogue the runner reads, not a made-up one.

    The row used to carry `"hf_revision": "abc123"`, so it passed while a real
    SciFact run recorded `hf_revision: ""` and raised UNPINNED_DATASET.
    """
    from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
    from lasagna.knowledge_evaluation_sdk.runner import _resolved_revision

    revision = _resolved_revision(benchmark_registry.get("beir/scifact"), "beir/scifact")
    assert revision, "beir/scifact, the README demo, records no dataset revision"
    rows = [
        {
            "benchmark_name": "beir/scifact",
            "config_name": arm,
            "resolved_mode": "live",
            "hf_revision": revision,
            "split": "test",
        }
        for arm in ("control", "mine")
    ]
    warnings = _integrity(rows, {"n_runs": 1, "nothing_samples": True})["warnings"]
    assert "SINGLE_RUN" not in warnings and "UNPINNED_DATASET" not in warnings


def test_the_notebook_says_this_run_prints_no_warnings():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    text = " ".join(" ".join("".join(c["source"]).split()) for c in cells)
    assert "Its warnings are recorded on purpose" not in text
    assert "prints VALID with no warnings" in text


#: A citation of a published method: an arXiv id, a DOI, or "Author et al. YEAR".
_CITATION = re.compile(r"^(arxiv:|doi:)|\bet al\. \d{4}|& \w+ \d{4}")


def _builtins():
    return [
        (stage, name)
        for stage, names in registry.list().items()
        for name in names
        if registry.is_builtin(stage, name)
    ]


def test_the_methods_named_in_the_audit_cite_their_papers():
    assert registry.provenance("reranker", "mmr")["paper"] == "doi:10.1145/290941.291025"
    assert registry.provenance("reranker", "cross_encoder")["paper"] == "arxiv:1901.04085"
    assert registry.provenance("agentic_pipeline", "legacy_react")["paper"] == "arxiv:2210.03629"


def test_the_readme_table_states_how_many_components_cite_a_paper():
    cited = [
        key
        for key in _builtins()
        if _CITATION.search(str(registry.provenance(*key).get("paper") or ""))
    ]
    row = next(
        line
        for line in (REPO / "README.md").read_text().splitlines()
        if "Published methods" in line
    )
    lasagna_cell = row.split("|")[2]
    assert f"{len(cited)} of {len(_builtins())} components" in lasagna_cell, lasagna_cell
