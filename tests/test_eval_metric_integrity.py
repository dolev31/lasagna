"""The evaluation SDK must not report a number it cannot stand behind.

This subsystem is the instrument that produces every research figure, so its
failure mode is worse than a crash: a wrong number that looks fine. Two ways
that happened, both fixed and pinned here.
"""

import pytest
from loguru import logger

from lasagna.knowledge_evaluation_sdk.ragas_evaluator import _extract_score

NAN = float("nan")


@pytest.fixture
def captured_logs():
    """Collect WARNING-and-above records emitted during a test."""
    records = []
    sink = logger.add(lambda m: records.append(str(m)), level="WARNING")
    yield records
    logger.remove(sink)


class TestNaNJudgementsAreAccountedFor:
    """RAGAS returns NaN for judgements that failed.

    Dropping them silently and averaging the survivors changes the denominator
    without telling anyone: a faithfulness of 0.85 over 60 usable samples of
    100 is a different claim from 0.85 over 100, and only one of them is what
    the report says.
    """

    def test_mean_is_over_valid_samples(self):
        assert _extract_score({"faithfulness": [1.0, 0.5, NAN, NAN]}, "faithfulness") == 0.75

    def test_dropped_samples_are_reported(self, captured_logs):
        _extract_score({"faithfulness": [1.0, 0.5, NAN, NAN]}, "faithfulness")
        joined = "".join(captured_logs)
        assert "2 of 4" in joined, f"drop count not reported: {joined}"
        assert "excluded" in joined

    def test_clean_input_stays_quiet(self, captured_logs):
        _extract_score({"faithfulness": [1.0, 0.0]}, "faithfulness")
        assert not captured_logs, f"unexpected noise on clean input: {captured_logs}"

    def test_total_failure_is_not_reported_as_a_zero_score(self, captured_logs):
        """All judgements failing must not look like the model scoring zero.

        This is the exact shape of a prior incident: a completely failed
        evaluation surfaced as faithfulness=0.0 and was read as a real result.
        """
        score = _extract_score({"faithfulness": [NAN, NAN, NAN]}, "faithfulness")
        joined = "".join(captured_logs)

        # Still a number, because these scores are JSON-serialised and NaN is
        # not valid JSON — but it must be impossible to miss in the log.
        assert score == 0.0
        assert "NOT a real" in joined, (
            f"a fully failed metric must say so explicitly; got: {joined}"
        )

    def test_scalar_results_pass_through(self):
        assert _extract_score({"faithfulness": 0.9}, "faithfulness") == 0.9

    def test_unknown_metric_raises(self):
        """Absent is not zero — callers must be able to tell the difference."""
        with pytest.raises(KeyError):
            _extract_score({"faithfulness": [1.0]}, "no_such_metric")

    def test_alias_keys_resolve(self):
        """RAGAS renames metrics between versions; aliases must still resolve."""
        assert (
            _extract_score({"llm_context_precision_with_reference": [0.8]}, "context_precision")
            == 0.8
        )
