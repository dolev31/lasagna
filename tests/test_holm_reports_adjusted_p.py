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
        lambda ca, cb, metric, bench: {
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
