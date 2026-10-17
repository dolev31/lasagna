"""A component's claimed number appears only beside a run of the benchmark it was claimed on.

Claims were keyed by metric alone: `@lasagna.reranker(claims={"ndcg@10": 0.42})`
and `lasagna benchmark run smoke --reranker my_ideas:keyword_overlap` printed
"claimed 0.4200 (Someone et al. 2026)" next to a 24-document smoke test, and
the LaTeX leaderboard's "Claimed" column did the same, as if a paper's SciFact
number and a smoke score measured the same thing. A claim now names its
benchmark (and split, if the paper gives one); the flat form is accepted only
with an explicit `benchmark` key, and refused without one, with the fix.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lasagna import components
from lasagna.cli import app
from lasagna.core.claims import claimed_value, normalize_claims
from lasagna.core.registry import registry
from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION


def _cleanup(*names):
    for name in names:
        registry._unregister("reranker", name)


class TestAClaimNamesItsBenchmark:
    def test_the_flat_form_without_a_benchmark_is_refused_with_the_fix(self):
        with pytest.raises(ValueError) as caught:

            @components.reranker("_claim_probe", claims={"ndcg@10": 0.42})
            def probe(query, text):
                return 0.0

        message = str(caught.value)
        assert "does not name the benchmark" in message
        assert 'claims={"beir/scifact": {"ndcg@10": 0.42}}' in message
        assert not registry.has("reranker", "_claim_probe")

    def test_the_flat_form_with_a_benchmark_key_is_accepted(self):
        assert normalize_claims({"benchmark": "beir/scifact", "ndcg@10": 0.42}) == {
            "beir/scifact": {"ndcg@10": 0.42}
        }

    def test_the_benchmark_keyed_form_round_trips_through_the_registry(self):
        try:

            @components.reranker(
                "_claim_probe", paper="Doe 2026", claims={"beir/scifact": {"ndcg@10": 0.456}}
            )
            def probe(query, text):
                return 0.0

            assert registry.provenance("reranker", "_claim_probe")["claims"] == {
                "beir/scifact": {"ndcg@10": 0.456}
            }
        finally:
            _cleanup("_claim_probe")

    @pytest.mark.parametrize(
        "claims",
        [{"beir/scifact": {"ndcg@10": "high"}}, {"beir/scifact": {}}, {"benchmark": ""}],
    )
    def test_a_malformed_claim_is_refused(self, claims):
        with pytest.raises(ValueError):
            normalize_claims(claims)


class TestAClaimMatchesItsBenchmarkAndSplit:
    CLAIMS = {"beir/scifact": {"ndcg@10": 0.456}, "beir/nfcorpus": {"ndcg@10": 0.3, "split": "dev"}}

    def test_only_the_named_benchmark_matches(self):
        assert claimed_value(self.CLAIMS, "BEIR/SciFact", "NDCG@10") == 0.456
        assert claimed_value(self.CLAIMS, "smoke", "ndcg@10") is None
        assert claimed_value(self.CLAIMS, "beir/scifact", "recall@10") is None

    def test_a_claim_that_names_a_split_matches_only_that_split(self):
        assert claimed_value(self.CLAIMS, "beir/nfcorpus", "ndcg@10", "dev") == 0.3
        assert claimed_value(self.CLAIMS, "beir/nfcorpus", "ndcg@10", "test") is None

    def test_a_flat_claim_recorded_by_an_older_build_matches_nothing(self):
        assert claimed_value({"ndcg@10": 0.42}, "smoke", "ndcg@10") is None


def _task(config: str, component: str, bench: str, score: float) -> dict:
    return {
        "task_key": f"{bench}|{config}|h|s|live|run_0",
        "run_index": 0,
        "benchmark_name": bench,
        "config_name": config,
        "split": "test",
        "metrics": {"ndcg@10": score},
        "per_sample_scores": [
            {"_sample_id": f"q{i}", "ndcg@10": score, "_retrieved_ids": [f"{config}-{i}"]}
            for i in range(10)
        ],
        "error": None,
        "config_overrides": {"reranker": component},
        "profile": "speed",
        "system": {"stages": {"reranker": {"component": component}}},
    }


@pytest.fixture
def two_benchmarks(tmp_path):
    """The same arm on smoke and on beir/scifact; the paper claims SciFact only."""
    claim = {
        "stage": "reranker",
        "paper": "Someone et al. 2026",
        "claims": {"beir/scifact": {"ndcg@10": 0.42}},
    }
    config = {
        "experiment_name": "claims",
        "n_runs": 1,
        "tracks": ["retrieval"],
        "metrics_version": METRICS_VERSION,
        "components": {"my_ideas:keyword_overlap": claim},
    }
    tasks = [
        _task("mine", "my_ideas:keyword_overlap", "smoke", 0.87),
        _task("mine", "my_ideas:keyword_overlap", "beir/scifact", 0.41),
    ]
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps({"experiment_name": "claims", "experiment_config": config, "results": tasks})
    )
    return path


def _section(text: str, heading: str) -> str:
    """The lines of the leaderboard under ``heading``, up to the next benchmark."""
    lines = text.splitlines()
    start = lines.index(heading)
    rest = lines[start + 1 :]
    end = next((i for i, line in enumerate(rest) if line and not line.startswith(" ")), len(rest))
    return "\n".join(rest[:end])


def test_the_text_leaderboard_shows_the_claim_on_scifact_only(two_benchmarks):
    out = CliRunner().invoke(app, ["results", "leaderboard", str(two_benchmarks)])
    assert out.exit_code == 0, out.output
    assert "claimed 0.4200 (Someone et al. 2026)" in _section(out.output, "beir/scifact")
    assert "claimed" not in _section(out.output, "smoke")


def test_the_latex_and_markdown_pages_show_the_claim_on_scifact_only(two_benchmarks, tmp_path):
    runner = CliRunner()
    latex = runner.invoke(app, ["results", "leaderboard", str(two_benchmarks), "--format", "latex"])
    assert latex.exit_code == 0, latex.output
    tables = latex.output.split(r"\begin{table}")
    smoke = next(t for t in tables if "smoke" in t and "scifact" not in t)
    scifact = next(t for t in tables if "scifact" in t)
    assert "Claimed" in scifact and "0.4200 (Someone et al. 2026)" in scifact
    assert "Claimed" not in smoke and "0.4200" not in smoke

    pages = tmp_path / "pages"
    md = runner.invoke(app, ["results", "leaderboard", str(two_benchmarks), "-o", str(pages)])
    assert md.exit_code == 0, md.output
    assert "0.4200 (Someone et al. 2026)" in (pages / "beir_scifact.md").read_text()
    assert "Claimed" not in (pages / "smoke.md").read_text()


def test_the_html_view_shows_the_claim_on_scifact_only(two_benchmarks, tmp_path):
    from lasagna.knowledge_evaluation_sdk.viewer import render_html

    page = render_html(two_benchmarks)
    smoke = page[page.index('id="bench-smoke"') :]
    scifact = page[page.index('id="bench-beir-scifact"') : page.index('id="bench-smoke"')]
    assert "0.4200 (Someone et al. 2026)" in scifact
    assert "Claimed" not in smoke
