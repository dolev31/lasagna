"""A paired test must compare a query against itself.

`significance_test` subtracts two arrays element-wise, so position *i* of each
has to be the same query. It used to build those arrays by filtering each config
independently and then checking only that the lengths matched — so if the metric
was missing on different samples in each config, the arrays could come out the
same length while comparing query 2 against query 3. The length guard passed and
the p-value was silently meaningless.

Worse, the arrays came from whichever run finished first: `_by_bench_config` was
built in `asyncio.as_completed` order, so two identical invocations could select
different runs and report different p-values.

The correct join already existed in this file, in
`domain_stratified_significance_test`. These tests pin it for both.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk.results import BenchmarkResults
from lasagna.knowledge_evaluation_sdk.runner import TaskResult


def _task(config: str, run_index: int, scores: list[dict], benchmark: str = "b") -> TaskResult:
    return TaskResult(
        task_key=f"{benchmark}|{config}|h|s|live|run_{run_index}",
        benchmark_name=benchmark,
        config_name=config,
        run_index=run_index,
        metrics={"ndcg@10": sum(s.get("ndcg@10", 0) for s in scores) / max(len(scores), 1)},
        per_sample_scores=scores,
    )


def _s(sid: str, **metrics) -> dict:
    return {"_sample_id": sid, **metrics}


class TestPairingIsByIdentityNotPosition:
    def test_misaligned_samples_are_joined_not_zipped(self):
        """Equal lengths, different queries — the exact shape the length guard
        could not see."""
        a = _task(
            "A",
            0,
            [
                _s("q1", **{"ndcg@10": 1.0}),
                _s("q2", **{"ndcg@10": 1.0}),
                _s("q3", **{"ndcg@10": 0.0}),
            ],
        )
        # B is missing q2 but has q4: same length, different membership.
        b = _task(
            "B",
            0,
            [
                _s("q1", **{"ndcg@10": 1.0}),
                _s("q3", **{"ndcg@10": 0.0}),
                _s("q4", **{"ndcg@10": 1.0}),
            ],
        )
        res = BenchmarkResults([a, b])

        ids, sa, sb, da, db = res._paired_scores("b", "A", "B", "ndcg@10")
        assert ids == ["q1", "q3"], "must pair only the queries present in both"
        assert sa == [1.0, 0.0] and sb == [1.0, 0.0]
        assert da == 1 and db == 1, "the dropped sample on each side must be reported"

    def test_paired_arrays_are_never_mismatched_queries(self):
        """Property form: every paired position refers to one sample id."""
        a = _task("A", 0, [_s(f"q{i}", **{"m": float(i)}) for i in range(10)])
        b = _task("B", 0, [_s(f"q{i}", **{"m": float(i) * 2}) for i in (0, 2, 4, 6, 8)])
        res = BenchmarkResults([a, b])

        ids, sa, sb, _, _ = res._paired_scores("b", "A", "B", "m")
        for sid, va, vb in zip(ids, sa, sb):
            i = int(sid[1:])
            assert va == float(i) and vb == float(i) * 2

    def test_metric_missing_on_one_side_is_excluded_from_both(self):
        a = _task("A", 0, [_s("q1", m=1.0), _s("q2", m=1.0)])
        b = _task("B", 0, [_s("q1", m=0.0), _s("q2", other=5.0)])
        res = BenchmarkResults([a, b])

        ids, sa, sb, _, _ = res._paired_scores("b", "A", "B", "m")
        assert ids == ["q1"] and sa == [1.0] and sb == [0.0]

    def test_no_shared_samples_raises_rather_than_comparing_nothing(self):
        a = _task("A", 0, [_s("q1", m=1.0)])
        b = _task("B", 0, [_s("z9", m=0.0)])
        res = BenchmarkResults([a, b])

        with pytest.raises(ValueError, match="same queries"):
            res.significance_test("A", "B", "m", benchmark="b")


class TestRunSelectionIsDeterministic:
    def test_arrival_order_does_not_change_the_numbers(self):
        """Tasks arrive in `asyncio.as_completed` order. Whichever run lands
        first must not decide which numbers get tested.

        This used to assert that run 0 won. It now asserts the stronger
        property that made run selection moot: every run is averaged, so there
        is no "winning" run to depend on arrival order. 1.0 and 0.0 average to
        0.5 — under the old behaviour this read 1.0 or 0.0 depending on which
        task the event loop returned first.
        """
        a0 = _task("A", 0, [_s("q1", m=1.0)])
        a1 = _task("A", 1, [_s("q1", m=0.0)])
        b0 = _task("B", 0, [_s("q1", m=0.5)])

        forward = BenchmarkResults([a0, a1, b0])
        reversed_arrival = BenchmarkResults([a1, a0, b0])

        _, fa, _, _, _ = forward._paired_scores("b", "A", "B", "m")
        _, ra, _, _, _ = reversed_arrival._paired_scores("b", "A", "B", "m")
        assert fa == ra == [0.5], "both runs must contribute, in either arrival order"

    def test_significance_is_stable_across_arrival_order(self):
        pytest.importorskip("numpy")
        scores_a = [_s(f"q{i}", m=float(i % 2)) for i in range(40)]
        scores_b = [_s(f"q{i}", m=float((i + 1) % 2)) for i in range(40)]
        a0, a1 = _task("A", 0, scores_a), _task("A", 1, scores_b)
        b0 = _task("B", 0, scores_b)

        p1 = BenchmarkResults([a0, a1, b0]).significance_test(
            "A", "B", "m", benchmark="b", n_bootstrap=200
        )["p_value"]
        p2 = BenchmarkResults([a1, b0, a0]).significance_test(
            "A", "B", "m", benchmark="b", n_bootstrap=200
        )["p_value"]
        assert p1 == p2, "identical data in a different arrival order gave a different p-value"

    def test_pairing_diagnostics_are_reported(self):
        pytest.importorskip("numpy")
        a = _task("A", 0, [_s(f"q{i}", m=float(i % 2)) for i in range(20)])
        b = _task("B", 0, [_s(f"q{i}", m=float(i % 2)) for i in range(15)])
        out = BenchmarkResults([a, b]).significance_test(
            "A", "B", "m", benchmark="b", n_bootstrap=100
        )
        assert out["n_paired"] == 15
        assert out["n_dropped_a"] == 5
        assert out["bootstrap_seed"] == 42
