"""`lasagna results view` and `results leaderboard --format latex`.

A run's answer was spread over four terminal commands (`leaderboard`,
`significance`, `per-query`, the isolation lines `benchmark experiment` prints)
and could not be handed to anyone as one thing. The page puts them in one
self-contained file, and must say exactly what the commands say: the verdict
comes from the same function as the CLI box, the ranking follows the same
rules as the text leaderboard. Arm names, questions and passages are free text
from the results file, so every one of them must arrive escaped.

The LaTeX leaderboard is the paper's version of the hub page: one booktabs
table per comparability group, so rows that may not be compared never share a
table, with every free-text cell escaped.
"""

from __future__ import annotations

import json
import re

import pytest

from lasagna.knowledge_evaluation_sdk.bundle import METRICS_VERSION
from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults, verdict_lines
from lasagna.knowledge_evaluation_sdk.stages import OTHER, STAGES
from lasagna.knowledge_evaluation_sdk.viewer import render_html

typer_testing = pytest.importorskip("typer.testing")

from lasagna.cli import app  # noqa: E402

runner = typer_testing.CliRunner()

N = 40


def _stages(reranker: str) -> dict:
    """A manifest identical across arms except the reranker stage they name."""
    manifest = {s: {"digest": "same", "component": None, "settings": {}} for s in (*STAGES, OTHER)}
    manifest["reranker"] = {"digest": f"d-{reranker}", "component": reranker, "settings": {}}
    return manifest


def _task(config, scores, *, bench="bench", trajectories=False, metric="ndcg@10"):
    samples = []
    for i, score in enumerate(scores):
        sample = {
            "_sample_id": f"q{i}",
            metric: score,
            # Each arm ranks its own second document, so no two arms return
            # identical lists (which the isolation check calls inert).
            "_retrieved_ids": [f"doc{i}", f"{config}-{i}"],
            "_expected_docs": [f"doc{i}"],
        }
        if trajectories:
            sample["_question"] = f"What is fact number {i}?"
            sample["_retrieved_contexts"] = [f"Passage {i} as {config} found it."]
            sample["_generated_answer"] = f"Answer {i}."
        samples.append(sample)
    return {
        "task_key": f"{bench}|{config}|h|s|live|run_0",
        "run_index": 0,
        "benchmark_name": bench,
        "config_name": config,
        "metrics": {metric: sum(scores) / len(scores), "latency_ms": 12.5},
        "per_sample_scores": samples,
        "error": None,
        "config_overrides": {"reranker": config},
        "profile": "speed",
        "system": {"stages": _stages(config)},
    }


def _results(tmp_path, tasks, name="viewer test"):
    config = {
        "experiment_name": name,
        "created_at": "2026-10-15T18:51:38.818843+00:00",
        "n_runs": 1,
        "tracks": ["retrieval"],
        "pipeline": {"base_profile": "speed"},
        "environment": {"lasagna_version": "0.1.0"},
        # A 0.1.0 run records the metric definitions it used; a bundle requires it.
        "metrics_version": METRICS_VERSION,
    }
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps({"experiment_name": name, "experiment_config": config, "results": tasks})
    )
    return path


@pytest.fixture
def two_arms(tmp_path):
    """`mine` clearly better than `control`, with query text and passages recorded."""
    return _results(
        tmp_path,
        [
            _task("control", [0.50 + 0.001 * i for i in range(N)], trajectories=True),
            _task("mine", [0.60 + 0.001 * i for i in range(N)], trajectories=True),
        ],
    )


def _text(page: str) -> str:
    """The page's visible words, tags removed and entities decoded."""
    import html

    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", page)).split())


class TestThePageSaysWhatTheCommandsSay:
    def test_every_arm_the_verdict_a_per_query_row_and_the_isolation_line(self, two_arms):
        page = render_html(two_arms)
        words = _text(page)
        assert "control" in words and "mine" in words
        assert "significant after Holm correction" in words
        assert "mine beats control" in words
        assert "ndcg@10 +0.1000 on bench" in words
        # A per-query row names the winner and the loser.
        assert re.search(r'<strong class="winner">mine</strong> over control', page)
        assert "What is fact number 7?" in words
        assert "Passage 7 as mine found it." in words
        assert (
            "Stage isolation: the configuration that reached each stage differs only in "
            "the stages the arms name (1 pair compared)."
        ) in words

    def test_the_verdict_is_the_one_the_cli_prints(self, two_arms):
        results = BenchmarkResults.load(str(two_arms))
        rows = results.pairwise_significance("ndcg@10", "bench")
        page = _text(render_html(two_arms))
        for line in verdict_lines(rows, "ndcg@10", "bench", 0.05).lines:
            assert " ".join(line.text.split()) in page

        cli = runner.invoke(app, ["results", "significance", str(two_arms)])
        assert cli.exit_code == 0, cli.output
        boxed = " ".join(cli.output.replace("│", " ").split())
        assert "mine beats control" in boxed and "mine beats control" in page

    def test_the_ranking_follows_the_text_leaderboard(self, tmp_path):
        # `partial` scored 30 of the 40 queries, so the text leaderboard lists
        # it unranked even though its mean is the highest.
        partial = _task("partial", [0.95] * 30)
        path = _results(
            tmp_path,
            [
                _task("low", [0.2 + 0.001 * i for i in range(N)]),
                _task("high", [0.7 + 0.001 * i for i in range(N)]),
                partial,
            ],
        )
        text = BenchmarkResults.load(str(path)).leaderboard("ndcg@10")
        assert re.search(r"1\. high .*\n\s+2\. low", text) and re.search(r"-\s+partial", text)

        page = render_html(path)
        board = page[page.index("<h3>Leaderboard</h3>") : page.index("<h3>Significance</h3>")]
        ranked = re.findall(r'<td class="num">(\d)</td><td class="arm">(\w+)</td>', board)
        assert ranked == [("1", "high"), ("2", "low")]
        assert "scored 30 of 40 queries, not ranked against the others" in _text(board)

    def test_lower_is_better_metrics_name_the_lower_side(self, tmp_path):
        path = _results(
            tmp_path,
            [
                _task("slow", [900.0 + i for i in range(N)], metric="latency_ms"),
                _task("fast", [300.0 + i for i in range(N)], metric="latency_ms"),
            ],
        )
        words = _text(render_html(path, metric="latency_ms"))
        assert "fast beats slow" in words
        assert "latency_ms (lower is better)" in words

    def test_the_diff_compares_the_arms_asked_for(self, tmp_path):
        path = _results(
            tmp_path,
            [
                _task("a", [0.1] * N),
                _task("b", [0.5] * N),
                _task("c", [0.9] * N),
            ],
        )
        default = _text(render_html(path))
        assert "c better on 40, b better on 0" in default  # the top two by default
        chosen = _text(render_html(path, diff=("a", "c")))
        assert "a better on 0, c better on 40" in chosen
        with pytest.raises(ValueError, match="No arm named 'zzz'.*They have: a, b, c"):
            render_html(path, diff=("a", "zzz"))

    def test_without_trajectories_it_shows_ids_and_says_how_to_record_more(self, tmp_path):
        path = _results(
            tmp_path,
            [_task("control", [0.5] * N), _task("mine", [0.4] * N)],
        )
        page = render_html(path)
        assert "save_trajectories: full" in page
        assert "Relevant: doc3" in _text(page)
        assert '<li class="hit" title="relevant">doc3 ✓</li>' in page


class TestABundleRendersTheSameTables:
    def test_bundle_and_results_file_agree(self, two_arms, tmp_path):
        bundle = BenchmarkResults.load(str(two_arms)).to_bundle(submitter={"name": "Ada"})
        bundle_path = tmp_path / "run.json"
        bundle_path.write_text(json.dumps(bundle, default=str))

        from_results = render_html(two_arms)
        from_bundle = render_html(bundle_path)

        def body(page):
            return page[page.index("</header>") : page.index("<footer>")]

        assert body(from_bundle) == body(from_results)
        header = _text(from_bundle[: from_bundle.index("</header>")])
        assert bundle["run_id"] in header and "Ada" in header
        assert f"Metrics version {bundle['metrics_version']}" in header


class TestUntrustedStringsAreEscaped:
    def test_an_arm_name_and_a_question_cannot_inject_markup(self, tmp_path):
        evil = '<script>alert("arm")</script>'
        task = _task(evil, [0.9] * N, trajectories=True)
        for sample in task["per_sample_scores"]:
            sample["_question"] = '<img src=x onerror="alert(1)">'
        path = _results(tmp_path, [_task("control", [0.1] * N), task], name="<b>exp</b>")
        page = render_html(path)
        assert "<script" not in page
        assert "<img" not in page
        assert "&lt;script&gt;alert(&quot;arm&quot;)&lt;/script&gt;" in page
        assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in page
        assert "<title>&lt;b&gt;exp&lt;/b&gt;</title>" in page

    def test_the_page_loads_nothing(self, two_arms):
        page = render_html(two_arms)
        assert "default-src 'none'" in page
        assert not re.search(r"<(script|link|img|iframe)\b", page)
        assert not re.search(r"(src|href)=[\"']?(https?:)?//", page)


def test_a_thousand_queries_stay_a_reasonable_page(tmp_path):
    n = 1000
    path = _results(
        tmp_path,
        [
            _task("control", [((i * 37) % 100) / 100 for i in range(n)], trajectories=True),
            _task("mine", [((i * 53) % 100) / 100 for i in range(n)], trajectories=True),
        ],
    )
    page = render_html(path)
    assert len(page.encode()) < 3_000_000
    # The table is collapsed until asked for.
    table = page.index('<table class="per-query">')
    assert page.rfind("<details>", 0, table) > page.rfind("</details>", 0, table)


class TestTheCommand:
    def test_it_writes_the_page(self, two_arms, tmp_path):
        out = tmp_path / "out" / "view.html"
        result = runner.invoke(app, ["results", "view", str(two_arms), "-o", str(out)])
        assert result.exit_code == 0, result.output
        assert "Wrote" in result.output
        assert "mine beats control" in _text(out.read_text())

    def test_a_folder_holding_results_json_is_accepted(self, two_arms, tmp_path):
        out = tmp_path / "view.html"
        result = runner.invoke(app, ["results", "view", str(two_arms.parent), "-o", str(out)])
        assert result.exit_code == 0, result.output

    def test_an_unknown_metric_names_the_recorded_ones(self, two_arms, tmp_path):
        result = runner.invoke(
            app, ["results", "view", str(two_arms), "-m", "recal@10", "-o", str(tmp_path / "v")]
        )
        assert result.exit_code == 1
        words = " ".join(result.output.split())
        assert "No metric 'recal@10'" in words and "latency_ms, ndcg@10" in words

    def test_a_file_that_is_not_results_is_refused_in_one_sentence(self, tmp_path):
        bad = tmp_path / "other.json"
        bad.write_text('{"hello": 1}')
        result = runner.invoke(app, ["results", "view", str(bad), "-o", str(tmp_path / "v")])
        assert result.exit_code == 1
        assert "neither a results.json nor a bundle" in " ".join(result.output.split())


# ---------------------------------------------------------------------------
# LaTeX leaderboard
# ---------------------------------------------------------------------------


def _bundle(submitter, key, config, values, metric="ndcg@10", benchmark="beir/scifact"):
    return {
        "run_id": f"{submitter}-{key}-{config}",
        "submitter": {"name": submitter},
        "configs": [
            {
                "name": config,
                "comparability_key": key,
                "held_fixed": {
                    "split": "test",
                    "n_samples": 100,
                    "embedder_model": "all-MiniLM-L6-v2",
                    "fixed_stages": {"retriever": "r1"},
                },
                "varied_dimensions": ["reranker"],
            }
        ],
        "results": [
            {
                "benchmark_name": benchmark,
                "config_name": config,
                "metrics": {metric: v},
                "run_index": i,
            }
            for i, v in enumerate(values)
        ],
    }


@pytest.fixture
def bundles(tmp_path):
    folder = tmp_path / "runs"
    folder.mkdir()
    for i, b in enumerate(
        [
            _bundle("lab_a", "key_one", "reranker=cross_encoder", (0.66, 0.67)),
            _bundle("lab_b", "key_one", "reranker=my_ideas:keyword_overlap", (0.57,)),
            _bundle("lab_c", "key_two", "bm25 & {friends} ~50% #1 $5 ^2 \\x", (0.40,)),
        ]
    ):
        (folder / f"b{i}.json").write_text(json.dumps(b))
    return folder


class TestTheLatexLeaderboard:
    def test_one_tabular_per_comparability_group_with_cells_escaped(self, bundles):
        from lasagna.knowledge_evaluation_sdk.leaderboard import grouped_rows, render_latex

        latex = render_latex(grouped_rows(bundles, "ndcg@10"), "ndcg@10")
        assert latex.count("\\begin{tabular}") == 2 == latex.count("\\end{tabular}")
        assert latex.count("\\toprule") == 2
        assert "reranker=my\\_ideas:keyword\\_overlap" in latex
        assert "my_ideas" not in latex
        assert "ndcg@10" in latex and "beir/scifact" in latex
        assert (
            "bm25 \\& \\{friends\\} \\textasciitilde{}50\\% \\#1 \\$5 "
            "\\textasciicircum{}2 \\textbackslash{}x"
        ) in latex
        # Best first and bold within its own table only; spread when runs > 1.
        first = latex.index("reranker=cross")
        assert first < latex.index("reranker=my")
        assert "$\\mathbf{0.6650 \\pm 0.0071}$" in latex
        assert latex.count("\\mathbf") == 1

    def test_the_command_prints_latex_or_writes_files(self, bundles, tmp_path):
        printed = runner.invoke(
            app,
            ["results", "leaderboard", str(bundles), "--format", "latex"],
            env={"COLUMNS": "60"},
        )
        assert printed.exit_code == 0, printed.output
        assert printed.output.startswith("% Generated by `lasagna results leaderboard")
        # Unwrapped at a narrow terminal: a wrapped `%` comment would turn its
        # second half into text in the paper.
        lines = printed.output.splitlines()
        assert lines[0] == (
            "% Generated by `lasagna results leaderboard --format latex`, ranked by ndcg@10 "
            "(higher is better). Needs \\usepackage{booktabs}."
        )
        assert lines[1].startswith("% Rows in one table") and lines[1].endswith("compared.")
        assert printed.output.count("\\begin{tabular}") == 2

        out = tmp_path / "pages"
        written = runner.invoke(
            app,
            ["results", "leaderboard", str(bundles), "--format", "latex", "-o", str(out)],
        )
        assert written.exit_code == 0, written.output
        assert (out / "beir_scifact.tex").read_text().count("\\begin{tabular}") == 2

        refused = runner.invoke(app, ["results", "leaderboard", str(bundles), "--format", "pdf"])
        assert refused.exit_code == 1
        assert "Use --format markdown or --format latex" in " ".join(refused.output.split())


def test_a_lower_is_better_leaderboard_ranks_the_smallest_first(tmp_path):
    """`group` meant to sort latency smallest first and read the metric off a
    row key nothing wrote, so every hub table put the slowest system on top."""
    from lasagna.knowledge_evaluation_sdk.leaderboard import collect_rows, group

    rows = collect_rows(
        [
            _bundle("a", "k", "slow", (900.0,), metric="latency_ms"),
            _bundle("b", "k", "fast", (300.0,), metric="latency_ms"),
        ],
        "latency_ms",
    )
    assert [r["config"] for r in group(rows)["beir/scifact"]["k"]] == ["fast", "slow"]
