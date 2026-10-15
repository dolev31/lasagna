"""Cost / latency / quality tradeoff analyzer.

Analyzes Pareto frontiers, marginal cost of features, efficiency metrics,
and generates tier recommendations from experiment results.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# There was a price table here, in USD per million tokens, and every figure
# derived from it was fiction. The token counts it multiplied were always zero
# -- read from a metadata key nothing ever wrote -- so `quality_per_dollar` came
# out as `inf` for every config, and the cost budget below never excluded
# anything. It is not restored with corrected prices: rates change, they differ
# by tier, region and commitment, and a rate frozen into a published report is
# indistinguishable from a measurement to whoever reads it next year.
#
# The cost axis is tokens. They are what was observed, they are comparable
# across time and provider, and a reader who wants dollars multiplies by the
# rate they actually pay.

# Latency budget defaults
SPEED_TIER_P95_MS = 200.0
BALANCED_TIER_P95_MS = 500.0
#: Token budget for the balanced tier, per query. 0 disables the constraint.
BALANCED_TIER_TOKENS = 4000.0


@dataclass
class ConfigEfficiency:
    """Efficiency metrics for a single config."""

    config_name: str
    quality: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_tokens_per_query: float = 0.0
    quality_per_1k_tokens: float = 0.0
    quality_per_second: float = 0.0
    total_llm_calls: int = 0
    on_pareto: bool = False


@dataclass
class MarginalAnalysis:
    """Marginal analysis comparing two configs."""

    baseline: str
    candidate: str
    delta_quality: float = 0.0
    delta_quality_pct: float = 0.0
    delta_latency_ms: float = 0.0
    delta_latency_pct: float = 0.0
    delta_tokens: float = 0.0
    delta_tokens_pct: float = 0.0
    quality_gain_per_ms: float = 0.0
    quality_gain_per_1k_tokens: float = 0.0
    verdict: str = ""


@dataclass
class TierRecommendation:
    """Recommended config for a deployment tier."""

    tier: str
    config_name: str
    quality: float
    p95_latency_ms: float
    tokens_per_query: float
    rationale: str


class TradeoffAnalyzer:
    """Analyzes quality vs cost/latency tradeoffs across configs."""

    def __init__(self) -> None:
        pass

    def efficiency_table(
        self,
        config_results: Dict[str, Dict[str, Any]],
        quality_metric: str = "ndcg@10",
    ) -> List[ConfigEfficiency]:
        """Build efficiency metrics for each config.

        Args:
            config_results: {config_name: {"metrics": {...}, "per_sample_scores": [...]}}
            quality_metric: Which metric to use as the quality signal.

        Returns:
            List of ConfigEfficiency sorted by quality descending.
        """
        efficiencies = []
        for config_name, data in config_results.items():
            metrics = data.get("metrics", {})
            per_sample = data.get("per_sample_scores", [])

            quality = metrics.get(quality_metric, 0.0)

            # Per QUERY, not per row. A live end-to-end run emits two rows for
            # the same `_sample_id` -- one for retrieval, one for generation --
            # each carrying its own slice of the cost. Verified across six
            # committed e2e result files: 99 rows for 56 queries, 43 of them
            # duplicated. Averaging over rows therefore reported roughly half
            # the true per-query latency and token count, in the table whose
            # entire purpose is the cost side of a quality/cost trade-off.
            #
            # Sum within a query, then average across queries.
            latencies = list(_per_query_totals(per_sample, ("latency_ms",)).values())
            tokens = list(_per_query_totals(per_sample, ("input_tokens", "output_tokens")).values())

            avg_latency = statistics.mean(latencies) if latencies else 0.0
            p50 = _percentile(latencies, 50) if latencies else 0.0
            p95 = _percentile(latencies, 95) if latencies else 0.0
            p99 = _percentile(latencies, 99) if latencies else 0.0

            avg_tokens = statistics.mean(tokens) if tokens else 0.0

            llm_calls = sum(s.get("llm_calls", 0) for s in per_sample)

            # 0.0, not inf. A config that spent no tokens has no measured
            # quality-per-token, and `inf` sorts it to the top of the efficiency
            # table -- which is how a config with no cost data came to be
            # recommended as the most efficient one.
            qp1k = quality / (avg_tokens / 1000.0) if avg_tokens > 0 else 0.0
            qps = quality / (avg_latency / 1000) if avg_latency > 0 else 0.0

            efficiencies.append(
                ConfigEfficiency(
                    config_name=config_name,
                    quality=quality,
                    avg_latency_ms=avg_latency,
                    p50_latency_ms=p50,
                    p95_latency_ms=p95,
                    p99_latency_ms=p99,
                    avg_tokens_per_query=avg_tokens,
                    quality_per_1k_tokens=qp1k,
                    quality_per_second=qps,
                    total_llm_calls=llm_calls,
                )
            )

        # Mark Pareto frontier
        frontier = self._compute_pareto(efficiencies, "quality", "avg_latency_ms")
        for e in efficiencies:
            e.on_pareto = e.config_name in frontier

        return sorted(efficiencies, key=lambda e: e.quality, reverse=True)

    def pareto_frontier(
        self,
        efficiencies: List[ConfigEfficiency],
        quality_attr: str = "quality",
        cost_attr: str = "avg_latency_ms",
    ) -> List[str]:
        """Return config names on the Pareto frontier.

        A config is on the frontier if no other config has both
        strictly better quality AND strictly lower cost.
        """
        return self._compute_pareto(efficiencies, quality_attr, cost_attr)

    def marginal_analysis(
        self,
        baseline: ConfigEfficiency,
        candidate: ConfigEfficiency,
    ) -> MarginalAnalysis:
        """Compare two configs to understand the marginal cost of quality."""
        dq = candidate.quality - baseline.quality
        dl = candidate.avg_latency_ms - baseline.avg_latency_ms
        dt = candidate.avg_tokens_per_query - baseline.avg_tokens_per_query

        dq_pct = (dq / baseline.quality * 100) if baseline.quality > 0 else 0.0
        dl_pct = (dl / baseline.avg_latency_ms * 100) if baseline.avg_latency_ms > 0 else 0.0
        dt_pct = (
            (dt / baseline.avg_tokens_per_query * 100) if baseline.avg_tokens_per_query > 0 else 0.0
        )

        qpm = dq / dl if dl > 0 else 0.0
        qp1k = dq / (dt / 1000.0) if dt > 0 else 0.0

        # Generate verdict
        if dq <= 0:
            verdict = "Not recommended: no quality improvement."
        elif dt <= 0:
            verdict = "Strictly better: higher quality on fewer tokens."
        elif dq_pct > 10 and dt_pct < 100:
            verdict = "Recommended: significant quality gain at moderate token increase."
        elif dq_pct > 5 and dt_pct > 200:
            verdict = f"Quality tier only: +{dq_pct:.1f}% quality but {dt_pct:.0f}% more tokens."
        else:
            verdict = f"Marginal: +{dq_pct:.1f}% quality for +{dt_pct:.0f}% tokens."

        return MarginalAnalysis(
            baseline=baseline.config_name,
            candidate=candidate.config_name,
            delta_quality=dq,
            delta_quality_pct=dq_pct,
            delta_latency_ms=dl,
            delta_latency_pct=dl_pct,
            delta_tokens=dt,
            delta_tokens_pct=dt_pct,
            quality_gain_per_ms=qpm,
            quality_gain_per_1k_tokens=qp1k,
            verdict=verdict,
        )

    def tier_recommendations(
        self,
        efficiencies: List[ConfigEfficiency],
        speed_p95_ms: float = SPEED_TIER_P95_MS,
        balanced_p95_ms: float = BALANCED_TIER_P95_MS,
        balanced_tokens: float = BALANCED_TIER_TOKENS,
    ) -> List[TierRecommendation]:
        """Recommend best config for each deployment tier."""
        recommendations = []

        # Speed tier: best quality where p95 < budget
        speed_candidates = [e for e in efficiencies if e.p95_latency_ms <= speed_p95_ms]
        if speed_candidates:
            best = max(speed_candidates, key=lambda e: e.quality)
            recommendations.append(
                TierRecommendation(
                    tier="speed",
                    config_name=best.config_name,
                    quality=best.quality,
                    p95_latency_ms=best.p95_latency_ms,
                    tokens_per_query=best.avg_tokens_per_query,
                    rationale=f"Best quality ({best.quality:.4f}) under {speed_p95_ms}ms p95 latency",
                )
            )

        # Balanced tier: best quality under both latency and cost budgets
        balanced_candidates = [
            e
            for e in efficiencies
            if e.p95_latency_ms <= balanced_p95_ms
            and (e.avg_tokens_per_query <= balanced_tokens or balanced_tokens <= 0)
        ]
        if balanced_candidates:
            best = max(balanced_candidates, key=lambda e: e.quality)
            recommendations.append(
                TierRecommendation(
                    tier="balanced",
                    config_name=best.config_name,
                    quality=best.quality,
                    p95_latency_ms=best.p95_latency_ms,
                    tokens_per_query=best.avg_tokens_per_query,
                    rationale=(
                        f"Best quality ({best.quality:.4f}) under "
                        f"{balanced_p95_ms}ms p95 and {balanced_tokens:.0f} tokens/query"
                    ),
                )
            )

        # Quality tier: best quality regardless of cost
        if efficiencies:
            best = max(efficiencies, key=lambda e: e.quality)
            recommendations.append(
                TierRecommendation(
                    tier="quality",
                    config_name=best.config_name,
                    quality=best.quality,
                    p95_latency_ms=best.p95_latency_ms,
                    tokens_per_query=best.avg_tokens_per_query,
                    rationale=f"Highest quality ({best.quality:.4f}) regardless of spend",
                )
            )

        return recommendations

    def to_markdown(
        self,
        efficiencies: List[ConfigEfficiency],
        recommendations: Optional[List[TierRecommendation]] = None,
    ) -> str:
        """Generate a full tradeoff report in markdown."""
        lines = ["## Tradeoff Analysis", ""]

        # Efficiency table
        lines.append("### Efficiency Table")
        lines.append("")
        lines.append(
            "| Config | Quality | Avg Latency | P95 Latency | Tokens/Q | "
            "Quality/1k tokens | Pareto |"
        )
        lines.append("|--------|---------|-------------|-------------|----------|-----|--------|")
        for e in efficiencies:
            pareto = "Y" if e.on_pareto else ""
            lines.append(
                f"| {e.config_name} | {e.quality:.4f} | "
                f"{e.avg_latency_ms:.0f}ms | {e.p95_latency_ms:.0f}ms | "
                f"{e.avg_tokens_per_query:.0f} | "
                f"{e.quality_per_1k_tokens:.4f} | {pareto} |"
            )

        # Pareto frontier
        frontier = [e for e in efficiencies if e.on_pareto]
        if frontier:
            lines.extend(["", "### Pareto Frontier (quality vs latency)", ""])
            # Latency, not tokens. `efficiency_table` computes the frontier with
            # `_compute_pareto(efficiencies, "quality", "avg_latency_ms")`, and
            # this line said "uses fewer tokens" -- a stray edit from the
            # dollars-to-tokens conversion. It named an axis the flag was not
            # computed on, so the section listed configs that are strictly
            # dominated on the criterion it stated. Call `pareto_frontier(...,
            # cost_attr="avg_tokens_per_query")` for a token frontier.
            lines.append("Configs where no other is both higher quality AND faster:")
            for i, e in enumerate(sorted(frontier, key=lambda x: x.avg_latency_ms), 1):
                lines.append(
                    f"{i}. **{e.config_name}**: quality={e.quality:.4f}, "
                    f"latency={e.avg_latency_ms:.0f}ms, "
                    f"tokens={e.avg_tokens_per_query:.0f}"
                )

        # Tier recommendations
        if recommendations:
            lines.extend(["", "### Tier Recommendations", ""])
            for r in recommendations:
                lines.append(
                    f"- **{r.tier.upper()}**: {r.config_name} ("
                    f"quality={r.quality:.4f}, p95={r.p95_latency_ms:.0f}ms, "
                    f"tokens={r.tokens_per_query:.0f})"
                )
                lines.append(f"  {r.rationale}")

        return "\n".join(lines)

    def _compute_pareto(
        self,
        efficiencies: List[ConfigEfficiency],
        quality_attr: str,
        cost_attr: str,
    ) -> List[str]:
        """Compute Pareto frontier (maximize quality, minimize cost)."""
        frontier = []
        for e in efficiencies:
            q = getattr(e, quality_attr, 0.0)
            c = getattr(e, cost_attr, 0.0)
            dominated = False
            for other in efficiencies:
                oq = getattr(other, quality_attr, 0.0)
                oc = getattr(other, cost_attr, 0.0)
                if oq > q and oc < c:
                    dominated = True
                    break
            if not dominated:
                frontier.append(e.config_name)
        return frontier


def _per_query_totals(per_sample: List[Dict[str, Any]], fields: tuple) -> Dict[Any, float]:
    """Sum ``fields`` within each query, keyed by ``_sample_id``.

    A live end-to-end run emits one row per (query, phase): retrieval and
    generation each carry their own ``latency_ms``. Treating rows as queries
    halves every cost figure, so group first.

    Rows with no ``_sample_id`` cannot be grouped and are kept as individual
    queries -- that is the offline-retrieval shape, where one row IS one query.
    """
    totals: Dict[Any, float] = {}
    for index, sample in enumerate(per_sample):
        if not any(f in sample for f in fields):
            continue
        # Fall back to a per-row key so ungrouped data behaves as it did.
        key = sample.get("_sample_id") or f"__row_{index}"
        totals[key] = totals.get(key, 0.0) + sum(sample.get(f, 0) or 0 for f in fields)
    return totals


def _percentile(data: List[float], pct: float) -> float:
    """Compute percentile without numpy."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    d = k - f
    return sorted_data[f] * (1 - d) + sorted_data[c] * d
