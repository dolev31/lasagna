"""Holm-corrected comparisons must report the adjusted p, not only the raw one.

`lasagna results significance` printed the raw p under a "Holm-Bonferroni"
header: on SciFact it showed 0.0001 where the adjusted value is 0.0003.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults


def test_adjusted_p_is_the_step_down_holm_value(monkeypatch):
    raw = {("a", "b"): 0.01, ("a", "c"): 0.04, ("b", "c"): 0.03}
    br = BenchmarkResults(task_results=[])
    monkeypatch.setattr(BenchmarkResults, "configs", property(lambda self: ["a", "b", "c"]))
    br._by_bench_config = {("bench", c): [object()] for c in ("a", "b", "c")}
    monkeypatch.setattr(
        br,
        "significance_test",
        lambda ca, cb, metric, bench, **_: {
            "p_value": raw[(ca, cb)],
            "effect_size": 0.0,
            "difference": 0.0,
        },
    )
    rows = {(r["config_a"], r["config_b"]): r for r in br.pairwise_significance("m", "bench")}
    # Sorted p: 0.01, 0.03, 0.04 -> 3*0.01, max(0.03, 2*0.03), max(0.06, 1*0.04)
    assert rows[("a", "b")]["p_adjusted"] == pytest.approx(0.03)
    assert rows[("b", "c")]["p_adjusted"] == pytest.approx(0.06)
    assert rows[("a", "c")]["p_adjusted"] == pytest.approx(0.06)
    # The decision agrees with the adjusted p at alpha=0.05.
    assert all(r["significant"] == (r["p_adjusted"] <= 0.05) for r in rows.values())


def _grid(n_configs, n_queries, step=0.5):
    """Configs that each differ from every other by at least `step` per query."""
    from lasagna.knowledge_evaluation_sdk.runner import TaskResult

    return BenchmarkResults(
        [
            TaskResult(
                task_key=f"c{c:02d}",
                benchmark_name="bench",
                config_name=f"c{c:02d}",
                run_index=0,
                per_sample_scores=[
                    {"_sample_id": f"q{i}", "m": c * step + (i % 5) * 0.01}
                    for i in range(n_queries)
                ],
            )
            for c in range(n_configs)
        ]
    )


def test_a_large_family_can_still_reject():
    """33 configs make 528 pairs, so the first Holm threshold is 0.05/528, below
    the 1/10,001 floor of a 10,000-draw sampled p-value: every pair differed by
    at least 0.5 on every query and none was significant."""
    rows = _grid(33, 31).pairwise_significance("m", "bench")
    assert len(rows) == 528
    assert all(r["significant"] for r in rows)


def test_an_untestable_pair_is_reported_and_kept_in_the_family():
    from lasagna.knowledge_evaluation_sdk.runner import TaskResult

    tasks = list(_grid(3, 12)._results)
    tasks.append(
        TaskResult(
            task_key="empty",
            benchmark_name="bench",
            config_name="empty",
            run_index=0,
            per_sample_scores=[{"_sample_id": "q0", "other_metric": 1.0}],
        )
    )
    rows = BenchmarkResults(tasks).pairwise_significance("m", "bench")
    untestable = [r for r in rows if r["p_value"] is None]
    assert len(rows) == 6 and len(untestable) == 3
    assert all("empty" in (r["config_a"], r["config_b"]) for r in untestable)
    assert all(r["untestable"] and not r["significant"] for r in untestable)
    # Holm over 6 hypotheses, not 3: the smallest adjusted p is 6 times the raw.
    tested = sorted((r for r in rows if r["p_value"] is not None), key=lambda r: r["p_value"])
    assert tested[0]["p_adjusted"] == pytest.approx(min(1.0, 6 * tested[0]["p_value"]))


def test_each_row_says_how_many_queries_it_paired():
    rows = _grid(2, 12).pairwise_significance("m", "bench")
    assert rows[0]["n_paired"] == 12
    assert rows[0]["n_dropped_a"] == rows[0]["n_dropped_b"] == 0
