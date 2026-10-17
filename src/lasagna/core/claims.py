"""The numbers a component's paper claims, keyed by the benchmark they were measured on.

A claim is only comparable with a measurement on the same benchmark (and the
same split, when the paper names one). Claims keyed by metric alone,
``claims={"ndcg@10": 0.42}``, were printed as "claimed 0.4200" beside a
24-document smoke test, and in the LaTeX "Claimed" column of any run. So a
claim names its benchmark::

    claims={"beir/scifact": {"ndcg@10": 0.456}}
    claims={"beir/scifact": {"ndcg@10": 0.456, "split": "test"}}
    claims={"benchmark": "beir/scifact", "ndcg@10": 0.456}   # the flat form

and a results table shows it only on a run of that benchmark.

Its metric names are checked at registration: ``"ndgc@10"`` registered and its
Claimed column never appeared. A metric the harness does not record is an
error; a benchmark no registry has is a warning, since a plugin may add it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional

_EXAMPLE = 'claims={"beir/scifact": {"ndcg@10": 0.456}}'

#: Ranked-list metrics the harness records at a cut-off; a claim names one
#: with its k, as ``ndcg@10`` (``ir_metrics.corpus_metrics``).
RANKED_METRICS = (
    "ndcg",
    "precision",
    "recall",
    "f1",
    "mrr",
    "map",
    "hit_rate",
    "r_precision",
    "complete_recall",
)

#: Answer metrics the harness records with no cut-off: the judge-free ones and
#: the judged ones (``ragas_evaluator.METRIC_MAP``).
ANSWER_METRICS = (
    "exact_match",
    "token_f1",
    "rouge_l",
    "contains_answer",
    "answer_similarity",
    "abstention_rate",
    "citation_recall",
    "citation_precision",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "factual_correctness",
    "noise_sensitivity",
    "context_relevance",
)


def _metric_problem(benchmark: str, name: str) -> Optional[str]:
    """Why ``name`` is not a metric a run records, or None when it is one."""
    from ._text import did_you_mean

    lower = name.lower()
    ranked = re.fullmatch(r"([a-z][a-z0-9_]*)@([1-9][0-9]*)", lower)
    if lower in ANSWER_METRICS or (ranked and ranked.group(1) in RANKED_METRICS):
        return None
    k = ranked.group(2) if ranked else "10"
    known = [f"{m}@{k}" for m in RANKED_METRICS] + list(ANSWER_METRICS)
    return (
        f"claims for {benchmark!r}: no metric named {name!r}{did_you_mean(lower, known)}, so "
        f"its Claimed column would never appear. A claim names a metric runs record: "
        f"{', '.join(f'{m}@k' for m in RANKED_METRICS)} (as in ndcg@10), or "
        f"{', '.join(ANSWER_METRICS)}."
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _metrics(benchmark: str, values: Mapping[str, Any]) -> Dict[str, Any]:
    """One benchmark's claims, checked: metric names to numbers, and an optional split."""
    out: Dict[str, Any] = {}
    for key, value in values.items():
        if key == "split":
            if not isinstance(value, str) or not value:
                raise ValueError(f"claims for {benchmark!r}: split must be a name, got {value!r}.")
        elif not _is_number(value):
            raise ValueError(
                f"claims for {benchmark!r}: {key!r} must be a number, got {value!r}. "
                f"For example {_EXAMPLE}."
            )
        else:
            problem = _metric_problem(benchmark, str(key))
            if problem:
                raise ValueError(problem)
        out[str(key)] = value
    if not any(key != "split" for key in out):
        raise ValueError(f"claims for {benchmark!r} name no metric. For example {_EXAMPLE}.")
    return out


def normalize_claims(claims: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Dict[str, Any]]]:
    """``{benchmark: {metric: value, ["split": name]}}`` from what a component declared.

    Accepts that form, or a flat dict with an explicit ``"benchmark"`` key (and
    optionally ``"split"``). Returns None for no claims.

    Raises:
        ValueError: The claims name no benchmark, or a value is not a number.
            The message shows the form to use.
    """
    if not claims:
        return None
    if not isinstance(claims, Mapping):
        raise ValueError(f"claims must be a dict, got {type(claims).__name__}. {_EXAMPLE}.")
    if "benchmark" in claims:
        benchmark = claims["benchmark"]
        if not isinstance(benchmark, str) or not benchmark:
            raise ValueError(f"claims: benchmark must be a benchmark name, got {benchmark!r}.")
        rest = {k: v for k, v in claims.items() if k != "benchmark"}
        return {benchmark: _metrics(benchmark, rest)}
    if all(isinstance(value, Mapping) for value in claims.values()):
        return {str(bench): _metrics(str(bench), values) for bench, values in claims.items()}
    metric, value = next(
        ((k, v) for k, v in claims.items() if not isinstance(v, Mapping) and k != "split"),
        next(iter(claims.items())),
    )
    raise ValueError(
        f"claims={dict(claims)!r} does not name the benchmark the paper measured "
        f"{metric!r} on, so a results table would print it beside a run of any "
        f'benchmark. Name it: claims={{"beir/scifact": {{"{metric}": {value}}}}}, or add '
        f'"benchmark": "beir/scifact" to the dict, with the benchmark the paper used.'
    )


def unregistered_benchmarks(claims: Optional[Mapping[str, Any]]) -> List[str]:
    """A warning for each benchmark ``claims`` names that no benchmark registry has.

    A warning, not an error: a plugin may register the benchmark later. Not
    checked while the benchmark registry is still loading its plugins (a
    plugin that registers a component as it loads), which would wait on itself.
    """
    if not claims:
        return []
    try:
        from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry as reg
    except Exception:  # noqa: BLE001 - the evaluation package is optional here
        return []
    if not reg._builtins_registered and reg._lock.locked():
        return []
    known = reg.names()
    from ._text import did_you_mean

    return [
        f"claims name the benchmark {name!r}, which no registry has"
        f"{did_you_mean(name, known)}: the claim shows only beside a run of a benchmark by "
        f"that name. `lasagna benchmark list` lists them; ignore this if a plugin registers it."
        for name in claims
        if name not in known
    ]


def claimed_value(
    claims: Optional[Mapping[str, Any]], benchmark: str, metric: str, split: str = ""
) -> Optional[float]:
    """The number ``claims`` gives for ``metric`` on ``benchmark``, or None.

    Benchmark and metric names compare case-insensitively. A claim that names
    a split matches only a run of that split. Claims recorded in the old flat
    form, which name no benchmark, match nothing.
    """
    try:
        by_benchmark = normalize_claims(claims)
    except ValueError:
        return None
    for name, values in (by_benchmark or {}).items():
        if name.lower() != str(benchmark).lower():
            continue
        if values.get("split") and values["split"] != split:
            continue
        for key, value in values.items():
            if key != "split" and key.lower() == metric.lower():
                return float(value)
    return None
