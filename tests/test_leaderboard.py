"""A leaderboard must never put incomparable numbers in the same table.

That is the single property worth testing here, because it is the one that
makes the difference between a leaderboard and a list of everyone's best
number — and it is invisible in the output when it goes wrong. A table that
silently mixes a 100-sample run with a 200-sample one still renders fine.

The verified badge gets the same scrutiny: it is a reproduction claim, so
self-confirmation and single-run rows must not be able to earn it.
"""

from __future__ import annotations

import json
import re

import pytest

from lasagna.knowledge_evaluation_sdk.leaderboard import (
    build,
    collect_rows,
    group,
    load_bundles,
    mark_verified,
    render_markdown,
)
from tests._bundles import sealed


def _bundle(submitter, key, config="baseline", values=(0.5,), varied=(), benchmark="smoke"):
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
                    "embedder_model": "bge-m3",
                    # Recorded, so rows group by key across bundles; see
                    # test_comparability_key_covers_every_fixed_stage.py for
                    # rows without it.
                    "fixed_stages": {"retriever": "r1"},
                },
                "varied_dimensions": list(varied),
            }
        ],
        "results": [
            {
                "benchmark_name": benchmark,
                "config_name": config,
                "metrics": {"ndcg@10": v},
                "run_index": i,
            }
            for i, v in enumerate(values)
        ],
    }


class TestGroupingIsByComparability:
    def test_different_keys_never_share_a_table(self):
        rows = collect_rows([_bundle("a", "key1"), _bundle("b", "key2")], "ndcg@10")
        grouped = group(rows)
        assert set(grouped["smoke"]) == {"key1", "key2"}
        assert all(len(v) == 1 for v in grouped["smoke"].values())

    def test_same_key_shares_a_table(self):
        rows = collect_rows([_bundle("a", "k"), _bundle("b", "k", config="other")], "ndcg@10")
        assert len(group(rows)["smoke"]["k"]) == 2

    def test_rows_are_ranked_within_a_group(self):
        rows = collect_rows(
            [_bundle("a", "k", values=(0.3,)), _bundle("b", "k", config="c2", values=(0.8,))],
            "ndcg@10",
        )
        assert [r["mean"] for r in group(rows)["smoke"]["k"]] == [0.8, 0.3]

    def test_benchmarks_are_kept_apart(self):
        rows = collect_rows(
            [_bundle("a", "k"), _bundle("b", "k", benchmark="beir/scifact")], "ndcg@10"
        )
        assert set(group(rows)) == {"smoke", "beir/scifact"}


class TestAggregationAcrossRuns:
    def test_runs_collapse_to_mean_and_spread(self):
        rows = collect_rows([_bundle("a", "k", values=(0.4, 0.5, 0.6))], "ndcg@10")
        assert rows[0]["n_runs"] == 3
        assert rows[0]["mean"] == pytest.approx(0.5)
        assert rows[0]["std"] > 0

    def test_failed_tasks_are_excluded(self):
        """A failed run contributes no number; averaging it in as 0 would be a lie."""
        b = _bundle("a", "k", values=(0.5, 0.5))
        b["results"][1]["error"] = "boom"
        rows = collect_rows([b], "ndcg@10")
        assert rows[0]["n_runs"] == 1

    def test_a_missing_metric_produces_no_row(self):
        rows = collect_rows([_bundle("a", "k")], "recall@100")
        assert rows == []


class TestVerifiedMeansReproduced:
    def test_a_different_submitter_inside_the_interval_verifies(self):
        rows = collect_rows(
            [
                _bundle("lab-a", "k", values=(0.40, 0.50, 0.60)),
                _bundle("lab-b", "k", config="c2", values=(0.49, 0.50, 0.51)),
            ],
            "ndcg@10",
        )
        mark_verified(rows)
        assert all(r["verified"] for r in rows)

    def test_self_confirmation_does_not_count(self):
        """Otherwise the badge is earned by publishing twice."""
        rows = collect_rows(
            [
                _bundle("lab-a", "k", values=(0.40, 0.50, 0.60)),
                _bundle("lab-a", "k", config="c2", values=(0.49, 0.50, 0.51)),
            ],
            "ndcg@10",
        )
        mark_verified(rows)
        assert not any(r["verified"] for r in rows)

    def test_a_number_outside_the_interval_does_not_verify(self):
        rows = collect_rows(
            [
                _bundle("lab-a", "k", values=(0.49, 0.50, 0.51)),
                _bundle("lab-b", "k", config="c2", values=(0.90, 0.91, 0.92)),
            ],
            "ndcg@10",
        )
        mark_verified(rows)
        assert not any(r["verified"] for r in rows)

    def test_a_single_run_can_never_be_verified(self):
        """It reports no spread, so there is no interval to land inside."""
        rows = collect_rows(
            [
                _bundle("lab-a", "k", values=(0.5,)),
                _bundle("lab-b", "k", config="c2", values=(0.5,)),
            ],
            "ndcg@10",
        )
        mark_verified(rows)
        assert not any(r["verified"] for r in rows)

    def test_an_exact_reproduction_of_a_deterministic_run_verifies(self):
        """Two submitters with bit-identical scifact results (0.6684 ± 0.0000,
        two runs each) got no verified row: the rule required std > 0, so a
        deterministic retrieval pipeline could never be verified."""
        rows = collect_rows(
            [
                _bundle("lab-a", "k", values=(0.6684, 0.6684)),
                _bundle("lab-b", "k", config="c2", values=(0.6684, 0.6684)),
            ],
            "ndcg@10",
        )
        mark_verified(rows)
        assert all(r["verified"] for r in rows)

    def test_a_deterministic_run_is_not_verified_by_a_different_number(self):
        rows = collect_rows(
            [
                _bundle("lab-a", "k", values=(0.6684, 0.6684)),
                _bundle("lab-b", "k", config="c2", values=(0.6685, 0.6685)),
            ],
            "ndcg@10",
        )
        mark_verified(rows)
        assert not any(r["verified"] for r in rows)

    def test_a_different_configuration_does_not_verify(self):
        """Same number, different held-fixed config, is not a reproduction."""
        rows = collect_rows(
            [
                _bundle("lab-a", "k1", values=(0.49, 0.50, 0.51)),
                _bundle("lab-b", "k2", config="c2", values=(0.50, 0.50, 0.50)),
            ],
            "ndcg@10",
        )
        mark_verified(rows)
        assert not any(r["verified"] for r in rows)


class TestRendering:
    def test_each_group_gets_its_own_table(self):
        rows = collect_rows([_bundle("a", "key1"), _bundle("b", "key2")], "ndcg@10")
        mark_verified(rows)
        md = render_markdown(group(rows), "ndcg@10", "smoke")
        assert md.count("| ndcg@10 | Varied |") == 2, "groups must not share a table"

    def test_the_trust_model_is_on_the_page(self):
        """A reader must not have to find the README to know what this is."""
        rows = collect_rows([_bundle("a", "k")], "ndcg@10")
        mark_verified(rows)
        md = render_markdown(group(rows), "ndcg@10", "smoke")
        assert "not independently reproduced" in md
        assert "Do not edit by hand" in md

    def test_a_baseline_is_labelled(self):
        rows = collect_rows([_bundle("a", "k")], "ndcg@10")
        mark_verified(rows)
        assert "_baseline_" in render_markdown(group(rows), "ndcg@10", "smoke")

    def test_the_varied_stage_is_shown(self):
        rows = collect_rows([_bundle("a", "k", varied=["reranker"])], "ndcg@10")
        mark_verified(rows)
        assert "`reranker`" in render_markdown(group(rows), "ndcg@10", "smoke")


class TestEndToEnd:
    def test_builds_from_a_directory(self, tmp_path):
        (tmp_path / "2026" / "lab-a").mkdir(parents=True)
        (tmp_path / "2026" / "lab-a" / "run.json").write_text(
            json.dumps(sealed(_bundle("lab-a", "k", values=(0.4, 0.5, 0.6))))
        )
        (tmp_path / "2026" / "lab-b.json").write_text(
            json.dumps(sealed(_bundle("lab-b", "k", config="c2", values=(0.49, 0.5, 0.51))))
        )

        out = tmp_path / "leaderboards"
        index = build(tmp_path, metric="ndcg@10", output=out)

        assert index["n_rows"] == 2
        assert index["benchmarks"]["smoke"]["n_groups"] == 1
        assert index["benchmarks"]["smoke"]["n_verified"] == 2
        assert (out / "smoke.md").is_file()
        assert json.loads((out / "index.json").read_text())["metric"] == "ndcg@10"

    def test_a_malformed_bundle_does_not_stop_the_rest(self, tmp_path):
        (tmp_path / "good.json").write_text(json.dumps(sealed(_bundle("a", "k"))))
        (tmp_path / "broken.json").write_text("{not json")
        assert len(load_bundles(tmp_path)) == 1

    def test_benchmark_names_with_slashes_become_one_file(self, tmp_path):
        (tmp_path / "r.json").write_text(
            json.dumps(sealed(_bundle("a", "k", benchmark="beir/scifact")))
        )
        out = tmp_path / "lb"
        build(tmp_path, output=out)
        assert (out / "beir_scifact.md").is_file()


class TestGeneratedPagesArePublishable:
    """The page is committed to a public repo, so it must carry no local paths.

    The first version linked bundles by absolute path, which both broke every
    link once the page was committed and published the generating machine's
    home directory into a leaderboard.
    """

    def test_no_absolute_path_reaches_the_page(self, tmp_path):
        runs = tmp_path / "runs" / "2026" / "lab-a"
        runs.mkdir(parents=True)
        (runs / "run.json").write_text(json.dumps(sealed(_bundle("lab-a", "k"))))

        out = tmp_path / "leaderboards"
        build(tmp_path / "runs", output=out)
        page = (out / "smoke.md").read_text()

        assert str(tmp_path) not in page
        assert "/Users/" not in page and "/home/" not in page

    def test_links_resolve_from_the_page(self, tmp_path):
        """`leaderboards/smoke.md` -> `../runs/...`, or every link 404s."""
        runs = tmp_path / "runs" / "2026" / "lab-a"
        runs.mkdir(parents=True)
        (runs / "run.json").write_text(json.dumps(sealed(_bundle("lab-a", "k"))))

        out = tmp_path / "leaderboards"
        build(tmp_path / "runs", output=out)

        targets = re.findall(r"\]\(([^)]+\.json)\)", (out / "smoke.md").read_text())
        assert targets, "no bundle links on the page"
        for target in targets:
            assert (out / target).resolve().is_file(), f"broken link: {target}"
