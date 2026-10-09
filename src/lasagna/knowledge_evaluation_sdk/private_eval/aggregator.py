"""Cross-track objective aggregation.

Combines IR metrics (retrieval track) and generation metrics (answer quality)
into a single composite score for comparing RAG configurations across both
evaluation tracks. Supports optional cost-aware scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Default weight profiles.
#
# These names must match what the harness actually emits, and they did not. The
# IR weights named "map" and "mrr" while `per_sample_scores` carries "map@10"
# and "mrr@10", so half the IR weight resolved to nothing on every run. The
# generation weights named "judge_score", "factual_accuracy" and "completeness",
# none of which exist anywhere in this package — `ragas_evaluator` emits
# "faithfulness", "answer_relevancy", "context_precision" and "context_recall" —
# so `gen_score` was always 0.0 and the composite was silently halved.
#
# `test_default_weights_name_real_metrics` in the test suite pins these against
# the metric names the harness emits, so this cannot drift again.
_DEFAULT_IR_WEIGHTS = {"ndcg@10": 0.5, "map@10": 0.3, "mrr@10": 0.2}
_DEFAULT_GEN_WEIGHTS = {
    "faithfulness": 0.4,
    "answer_relevancy": 0.3,
    "context_precision": 0.3,
}
_DEFAULT_TRACK_WEIGHTS = {"ir": 0.5, "gen": 0.5}


@dataclass
class ConfigResult:
    """Aggregated results for a single config."""

    config_name: str
    ir_metrics: Dict[str, float] = field(default_factory=dict)
    gen_metrics: Dict[str, float] = field(default_factory=dict)
    cost_metrics: Dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    ir_score: float = 0.0
    gen_score: float = 0.0
    tracks_evaluated: List[str] = field(default_factory=list)


class ObjectiveAggregator:
    """Aggregates IR and generation metrics into a single composite score."""

    def __init__(
        self,
        ir_weights: Optional[Dict[str, float]] = None,
        gen_weights: Optional[Dict[str, float]] = None,
        track_weights: Optional[Dict[str, float]] = None,
    ):
        self._ir_weights = ir_weights or _DEFAULT_IR_WEIGHTS
        self._gen_weights = gen_weights or _DEFAULT_GEN_WEIGHTS
        self._track_weights = track_weights or _DEFAULT_TRACK_WEIGHTS

    def aggregate(
        self,
        ir_metrics: Dict[str, float],
        gen_metrics: Dict[str, float],
    ) -> float:
        """Compute weighted normalized composite score across both tracks.

        Returns a score in [0, 1] where higher is better.
        """
        ir_score = self._weighted_score(ir_metrics, self._ir_weights)
        gen_score = self._weighted_score(gen_metrics, self._gen_weights)

        has_ir = bool(ir_metrics)
        has_gen = bool(gen_metrics)

        if has_ir and has_gen:
            return self._track_weights["ir"] * ir_score + self._track_weights["gen"] * gen_score
        elif has_ir:
            return ir_score
        elif has_gen:
            return gen_score
        return 0.0

    def aggregate_cost_aware(
        self,
        ir_metrics: Dict[str, float],
        gen_metrics: Dict[str, float],
        cost_metrics: Dict[str, float],
        latency_budget_ms: float = 500.0,
        token_budget: float = 4000.0,
    ) -> float:
        """Composite score with cost-awareness, priced in tokens.

        Configs exceeding the latency or token budget get a 50% penalty.

        The budget used to be in dollars, read from `estimated_cost_usd` -- a
        key nothing ever wrote. It was therefore always 0.0, always under
        budget, and the penalty never fired for any config in any run. Tokens
        are measured, so this constraint now does something.
        """
        base_score = self.aggregate(ir_metrics, gen_metrics)

        penalty = 1.0
        p95 = cost_metrics.get("p95_latency_ms", 0.0)
        if p95 > latency_budget_ms:
            penalty *= 0.5

        tokens_per_query = cost_metrics.get("input_tokens", 0.0) + cost_metrics.get(
            "output_tokens", 0.0
        )
        if tokens_per_query > token_budget:
            penalty *= 0.5

        return base_score * penalty

    def rank_configs(self, results: List[ConfigResult]) -> List[ConfigResult]:
        """Rank configs by composite score (descending)."""
        for r in results:
            r.ir_score = self._weighted_score(r.ir_metrics, self._ir_weights)
            r.gen_score = self._weighted_score(r.gen_metrics, self._gen_weights)
            r.composite_score = self.aggregate(r.ir_metrics, r.gen_metrics)
        return sorted(results, key=lambda r: r.composite_score, reverse=True)

    def _weighted_score(self, metrics: Dict[str, float], weights: Dict[str, float]) -> float:
        """Compute weighted average of available metrics."""
        if not metrics:
            return 0.0

        total_weight = 0.0
        total_score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                total_score += weight * metrics[metric]
                total_weight += weight

        if total_weight == 0:
            return 0.0
        return total_score / total_weight

    def to_markdown(self, results: List[ConfigResult]) -> str:
        """Generate a markdown comparison table."""
        ranked = self.rank_configs(results)
        lines = [
            "## Config Ranking (Composite Score)",
            "",
            "| Rank | Config | Composite | IR Score | Gen Score | Tracks |",
            "|------|--------|-----------|----------|-----------|--------|",
        ]
        for i, r in enumerate(ranked, 1):
            tracks = ", ".join(r.tracks_evaluated) if r.tracks_evaluated else "?"
            lines.append(
                f"| {i} | {r.config_name} | {r.composite_score:.4f} | "
                f"{r.ir_score:.4f} | {r.gen_score:.4f} | {tracks} |"
            )
        return "\n".join(lines)
