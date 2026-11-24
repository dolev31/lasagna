"""Test that Lasagna can extract scores from RAGAS result keys."""

import pytest
from ragas.metrics import FactualCorrectness, Faithfulness, NoiseSensitivity

from lasagna.knowledge_evaluation_sdk.ragas_evaluator import _extract_score


def test_extract_score_for_factual_correctness_mode_f1():
    """RAGAS returns factual correctness as 'factual_correctness(mode=f1)'.
    Must be able to read this score when asked for 'factual_correctness'.
    """
    # 1 Real RAGAS metric object
    metric = FactualCorrectness()
    # Its real key is: 'factual_correctness(mode=f1)'
    real_ragas_key = f"{metric.name}(mode={metric.mode})"

    # 2 Simulate RAGAS returning a score of 0.85 under that key
    ragas_output = {real_ragas_key: [0.85]}

    # 3 Lasagna asks for 'factual_correctness' and should get 0.85
    score = _extract_score(ragas_output, "factual_correctness")
    assert score == 0.85


def test_extract_score_for_noise_sensitivity_mode_relevant():
    """RAGAS returns noise sensitivity as 'noise_sensitivity(mode=relevant)'.
     Must be able to read this score when asked for 'noise_sensitivity'.
    """
    metric = NoiseSensitivity()
    # Its real key is: 'noise_sensitivity(mode=relevant)'
    real_ragas_key = f"{metric.name}(mode={metric.mode})"

    ragas_output = {real_ragas_key: [0.30]}

    score = _extract_score(ragas_output, "noise_sensitivity")
    assert score == 0.30


def test_extract_score_for_metric_without_mode():
    """Standard metrics like Faithfulness don't have a mode.
    They should continue working as normal.
    """
    metric = Faithfulness()
    real_ragas_key = metric.name  # 'faithfulness'

    ragas_output = {real_ragas_key: [0.95]}

    score = _extract_score(ragas_output, "faithfulness")
    assert score == 0.95

def test_dataframe_columns_resolve_for_mode_metrics():
    """Verify that DataFrame columns with mode suffixes map correctly to metric names."""
    from lasagna.knowledge_evaluation_sdk.ragas_evaluator import _find_ragas_key

    columns = [
        "user_input",
        "response",
        "factual_correctness(mode=f1)",
        "noise_sensitivity(mode=relevant)",
        "faithfulness",
    ]

    assert _find_ragas_key("factual_correctness", columns) == "factual_correctness(mode=f1)"
    assert _find_ragas_key("noise_sensitivity", columns) == "noise_sensitivity(mode=relevant)"
    assert _find_ragas_key("faithfulness", columns) == "faithfulness"