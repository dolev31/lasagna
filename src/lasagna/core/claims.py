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
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

_EXAMPLE = 'claims={"beir/scifact": {"ndcg@10": 0.456}}'


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
    metric, value = next((k, v) for k, v in claims.items() if not isinstance(v, Mapping))
    raise ValueError(
        f"claims={dict(claims)!r} does not name the benchmark the paper measured "
        f"{metric!r} on, so a results table would print it beside a run of any "
        f'benchmark. Name it: claims={{"beir/scifact": {{"{metric}": {value}}}}}, or add '
        f'"benchmark": "beir/scifact" to the dict, with the benchmark the paper used.'
    )


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
