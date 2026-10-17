"""The composition efficiency is printed only when the ratio means something.

`results interaction` printed "composition efficiency: 77.0% (of the additive
prediction actually delivered)" for a run whose expected and observed effects
were both negative (-0.0081 and -0.0062) and whose interaction had p = 0.91:
a ratio of two effects that did not exist, read as a finding about how two
techniques compose. The ratio is reported when the interaction is significant
and the combination moved the metric the way the additive prediction did;
otherwise the report and the command say why it is left out.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from lasagna.cli import app
from tests.test_interaction import _results

CELLS = dict(neither="neither", only_a="hyde", only_b="reranking", both="hyde+reranking")


def _cells(eff_a, eff_b, interaction, noise=0.0, n=60, base=0.4):
    wobble = [((i * 37) % 11 - 5) / 100.0 for i in range(n)]
    jitter = [noise if i % 2 else -noise for i in range(n)]
    return {
        "neither": [base + w for w in wobble],
        "hyde": [base + eff_a + w for w in wobble],
        "reranking": [base + eff_b + w for w in wobble],
        "hyde+reranking": [
            base + eff_a + eff_b + interaction + w + j for w, j in zip(wobble, jitter)
        ],
    }


def _report(cells):
    return _results(cells).interaction_test("scifact", "ndcg@10", n_bootstrap=500, **CELLS)


def test_no_ratio_for_an_interaction_that_is_not_significant():
    report = _report(_cells(-0.004, -0.004, 0.002, noise=0.05))
    assert report["expected_if_additive"] < 0 and report["combined_effect"] < 0
    assert not report["significant"]
    assert report["composition_efficiency"] is None
    assert "not significant" in report["composition_efficiency_omitted"]


def test_no_ratio_when_the_combination_moved_the_other_way():
    report = _report(_cells(0.02, 0.02, -0.06))
    assert report["significant"]
    assert report["expected_if_additive"] > 0 > report["combined_effect"]
    assert report["composition_efficiency"] is None
    assert "other way" in report["composition_efficiency_omitted"]


def test_the_ratio_is_kept_where_it_reads_true():
    report = _report(_cells(0.04, 0.04, -0.04))
    assert report["composition_efficiency"] == pytest.approx(0.5)
    assert report["composition_efficiency_omitted"] is None


def test_the_command_says_why_the_ratio_is_left_out(tmp_path):
    path = tmp_path / "results.json"
    _results(_cells(-0.004, -0.004, 0.002, noise=0.05)).save(str(path))
    out = CliRunner().invoke(app, ["results", "interaction", str(path)], env={"COLUMNS": "200"})
    assert out.exit_code == 0, out.output
    words = " ".join(out.output.split())
    assert "composition efficiency: not reported" in words
    assert "not significant" in words
    assert "%" not in words.split("composition efficiency:")[1].split("additive")[0]
