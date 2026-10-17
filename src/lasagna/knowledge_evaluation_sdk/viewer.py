"""One run as one static HTML page: ``lasagna results view``.

The terminal commands each answer one question (``leaderboard``,
``significance``, ``per-query``). A reviewer, a co-author or a later you wants
all of them at once, and wants to pass the answer around as a file. This
renders a ``results.json`` or a bundle as a single page with:

* a header: experiment, date, package and metrics version, run id;
* the stage isolation check, run-wide, in the words ``benchmark experiment``
  prints;
* per benchmark: the leaderboard for the chosen metric (ranked by the same
  rules as ``BenchmarkResults.leaderboard``), the Holm-corrected significance
  table with the verdict ``results significance`` prints, every metric per arm,
  and a per-query diff between two arms (the top two by the chosen metric
  unless ``diff`` names them), with what each arm retrieved where the results
  recorded it.

The page is self-contained and works offline: inline CSS, no script, no font,
no image, and a content security policy that forbids loading anything, so it
can be attached to an email or a review. Every string from the results (arm
names, questions, passages) is escaped. The per-query table is collapsed, and
passages are cut to their first few hundred characters, so 1,000 queries stay
a page a browser opens without effort.

Only the standard library renders it: jinja2 is an ``[eval]`` extra and this
works on a core install.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .results import (
    BenchmarkResults,
    Verdict,
    _aggregate,
    _mean_by_sample,
    format_effect_size,
    format_p,
    lower_is_better,
    verdict_lines,
)
from .run_provenance import claim_text
from .runner import TaskResult

#: The metric shown when none is asked for: the first of these the run recorded.
_PREFERRED_METRICS = (
    "ndcg@10",
    "answer_correctness",
    "faithfulness",
    "token_f1",
    "exact_match",
)
#: Passages shown per arm and query, and how much of each.
_CONTEXTS_SHOWN = 3
_CONTEXT_CHARS = 300
#: Retrieved document ids shown per arm and query.
_IDS_SHOWN = 10
#: Above this many pairs the significance table starts collapsed.
_PAIRS_OPEN = 15
#: The family-wise level of the significance table, as `results significance`.
_ALPHA = 0.05

Source = Union[BenchmarkResults, Mapping[str, Any], str, Path]


def _e(value: Any) -> str:
    """Escape anything for HTML text or a quoted attribute."""
    return html.escape("" if value is None else str(value), quote=True)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class _Run:
    """The results, the raw rows the isolation check reads, and what a bundle adds."""

    def __init__(self, results: BenchmarkResults, rows: List[Dict[str, Any]], **meta: Any):
        self.results = results
        self.rows = rows
        self.meta = meta


def _is_bundle(data: Mapping[str, Any]) -> bool:
    return "configs" in data and "results" in data and "experiment" in data


def _load(source: Source) -> _Run:
    if isinstance(source, BenchmarkResults):
        rows = [r.to_dict() for r in source._results]
        return _Run(source, rows)
    name = ""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir() and (path / "results.json").is_file():
            path = path / "results.json"
        name = path.name
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = source
    if not isinstance(data, Mapping) or not isinstance(data.get("results"), list):
        raise ValueError(
            f"{name or 'This file'} is neither a results.json nor a bundle: it has no "
            f"'results' list. Pass the results.json a run wrote, or a bundle from "
            f"`lasagna results bundle`."
        )
    rows = [dict(r) for r in data["results"]]
    tasks = [TaskResult.from_dict(r) for r in rows]
    if _is_bundle(data):
        experiment = dict(data.get("experiment") or {})
        results = BenchmarkResults(tasks, experiment.get("experiment_name", ""), experiment)
        return _Run(
            results,
            rows,
            file=name,
            kind="bundle",
            run_id=data.get("run_id"),
            metrics_version=data.get("metrics_version"),
            schema_version=data.get("schema_version"),
            submitter=data.get("submitter") or {},
            integrity=data.get("integrity") or {},
        )
    config = dict(data.get("experiment_config") or {})
    results = BenchmarkResults(tasks, data.get("experiment_name", ""), config)
    return _Run(
        results,
        rows,
        file=name,
        kind="results",
        metrics_version=data.get("metrics_version") or config.get("metrics_version"),
        package_version=data.get("package_version") or config.get("package_version"),
    )


def _default_metric(results: BenchmarkResults) -> str:
    names = results.get_metric_names()
    for preferred in _PREFERRED_METRICS:
        if preferred in names:
            return preferred
    quality = [m for m in names if not lower_is_better(m) and not m.endswith("_per_query")]
    if quality or names:
        return (quality or names)[0]
    raise ValueError("These results record no metrics, so there is nothing to show.")


# ---------------------------------------------------------------------------
# What each section shows, as data
# ---------------------------------------------------------------------------


def _ranking(results: BenchmarkResults, bench: str, metric: str) -> List[Dict[str, Any]]:
    """The leaderboard for one benchmark, by `BenchmarkResults.leaderboard`'s rules.

    Ranked among arms that scored the same queries; an arm that scored a
    subset, failed every run, recorded no value or never ran is listed
    unranked with the reason.
    """
    entries = []
    for config in results.configs:
        tasks = results._by_bench_config.get((bench, config), [])
        ok = [t for t in tasks if not t.error and metric in t.metrics]
        failed = sum(1 for t in tasks if t.error)
        entries.append((config, tasks, ok, failed, set(_mean_by_sample(ok, metric))))
    universe = set().union(*(e[4] for e in entries)) if entries else set()

    ranked, unranked = [], []
    for config, tasks, ok, failed, queries in entries:
        row: Dict[str, Any] = {"config": config, "rank": None, "mean": None, "note": ""}
        if not tasks:
            row["note"] = "not run"
            unranked.append(row)
            continue
        if not ok:
            reason = "failed" if failed else f"no {metric} recorded"
            row["note"] = f"{reason} ({len(tasks)} runs)"
            unranked.append(row)
            continue
        agg = _aggregate([t.metrics[metric] for t in ok])
        row.update(mean=agg.mean, std=agg.std, n_runs=agg.n_runs, n_queries=len(queries))
        row["claim"] = results.claim_for(config, metric, bench)
        notes = [f"{failed} of {len(tasks)} runs failed"] if failed else []
        if queries and queries != universe:
            notes.append(
                f"scored {len(queries)} of {len(universe)} queries, not ranked against the others"
            )
            unranked.append(row)
        else:
            ranked.append(row)
        row["note"] = "; ".join(notes)
    ranked.sort(key=lambda r: r["mean"], reverse=not lower_is_better(metric))
    for i, row in enumerate(ranked, 1):
        row["rank"] = i
    return ranked + unranked


def _diff_arms(
    ranking: List[Dict[str, Any]], diff: Optional[Tuple[str, str]]
) -> Optional[Tuple[str, str]]:
    if diff:
        return diff
    scored = [r for r in ranking if r["mean"] is not None]
    if len(scored) < 2:
        return None
    return scored[0]["config"], scored[1]["config"]


def _winner(a: float, b: float, metric: str) -> int:
    """1 when A is better, -1 when B is, 0 for a tie."""
    if math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12):
        return 0
    better_a = a < b if lower_is_better(metric) else a > b
    return 1 if better_a else -1


# ---------------------------------------------------------------------------
# HTML pieces
# ---------------------------------------------------------------------------


def _num(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return _e(format_effect_size(value))
    return f"{value:.{digits}f}"


def _cell(mean: float, std: float, n_runs: int) -> str:
    spread = f' <span class="spread">± {std:.4f}</span>' if n_runs > 1 else ""
    return f"{mean:.4f}{spread}"


def _table(head: Sequence[str], rows: Iterable[str], cls: str = "") -> str:
    """A table; a heading that starts with ``#`` heads a right-aligned number column."""
    th = "".join(
        f'<th class="num">{h[1:]}</th>' if h.startswith("#") else f"<th>{h}</th>" for h in head
    )
    klass = f' class="{cls}"' if cls else ""
    return (
        f'<div class="scroll"><table{klass}><thead><tr>{th}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _verdict_html(verdict: Verdict) -> str:
    def span(role: str, text: str) -> str:
        if role == "winner":
            return f'<strong class="winner">{_e(text)}</strong>'
        if role == "command":
            return f"<code>{_e(text)}</code>"
        return _e(text)

    lines = "".join(
        f"<p>{''.join(span(role, text) for role, text in line.spans) or '&nbsp;'}</p>"
        for line in verdict.lines
    )
    return (
        f'<div class="verdict {_e(verdict.style)}" role="status">'
        f'<div class="verdict-title">{_e(verdict.title)}</div>{lines}</div>'
    )


def _header(run: _Run, metric: str, benchmarks: List[str]) -> str:
    results, meta = run.results, run.meta
    config = results.experiment_config or {}
    env = config.get("environment") or {}
    title = results.experiment_name or config.get("experiment_name") or "Lasagna results"
    submitter = meta.get("submitter") or {}
    who = ", ".join(v for v in (submitter.get("name"), submitter.get("affiliation")) if v)
    created = config.get("created_at")
    facts = [
        ("Date", _date(created)),
        ("Package", meta.get("package_version") or env.get("lasagna_version")),
        ("Metrics version", meta.get("metrics_version")),
        ("Run id", meta.get("run_id")),
        ("Submitter", who),
        ("Source", meta.get("file")),
        ("Arms", len(results.configs)),
        ("Benchmarks", ", ".join(benchmarks)),
        ("Runs per arm", config.get("n_runs")),
        ("Tracks", ", ".join(config.get("tracks") or [])),
        ("Base profile", (config.get("pipeline") or {}).get("base_profile")),
        ("Ranked by", f"{metric} ({'lower' if lower_is_better(metric) else 'higher'} is better)"),
    ]
    failed = len(results.failed_tasks)
    if failed:
        facts.append(("Failed tasks", f"{failed} of {results.total_tasks}"))
    # The recorded timestamp, to the microsecond, on hover over the short form.
    titles = {"Date": f' title="{_e(created)}"'} if created not in (None, "") else {}
    dl = "".join(
        f"<div><dt>{_e(k)}</dt><dd{titles.get(k, '')}>{_e(v)}</dd></div>"
        for k, v in facts
        if v not in (None, "")
    )
    integrity = meta.get("integrity") or {}
    flags = ""
    blocking, warnings = integrity.get("blocking") or [], integrity.get("warnings") or []
    if blocking or warnings:
        chips = "".join(f'<span class="chip bad">{_e(c)}</span>' for c in blocking) + "".join(
            f'<span class="chip warn">{_e(c)}</span>' for c in warnings
        )
        flags = f'<p class="flags">Bundle integrity: {chips}</p>'
    nav = ""
    if len(benchmarks) > 1:
        links = "".join(f'<a href="#{_anchor(b)}">{_e(b)}</a>' for b in benchmarks)
        nav = f'<nav aria-label="Benchmarks">{links}</nav>'
    return (
        f'<header><p class="eyebrow">Lasagna results</p><h1>{_e(title)}</h1>'
        f'<dl class="facts">{dl}</dl>{flags}{nav}</header>'
    )


def _date(value: Any) -> Any:
    """``2026-10-15T18:51:38.8+00:00`` as ``2026-10-15 18:51 UTC``; anything else as is."""
    from datetime import datetime, timedelta

    try:
        when = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return value
    offset = when.utcoffset()
    zone = "" if offset is None else " UTC" if offset == timedelta(0) else when.strftime(" %z")
    return when.strftime("%Y-%m-%d %H:%M") + zone


def _anchor(bench: str) -> str:
    return "bench-" + "".join(ch if ch.isalnum() else "-" for ch in bench)


def _isolation(run: _Run) -> str:
    """The isolation check, in the words `lasagna benchmark experiment` prints."""
    from .stages import compared_pairs, isolation_report

    rows = run.rows
    report = isolation_report(rows)
    pairs = compared_pairs(rows)
    parts = []
    if report["undeclared"]:
        items = "".join(f"<li>{_e(line)}</li>" for line in report["undeclared"])
        parts.append(
            '<div class="note bad"><p>Stage isolation FAILED: the arms differ outside the '
            "stages they name, so the comparison is not attributable.</p>"
            f"<ul>{items}</ul></div>"
        )
    elif pairs and not report["inert"]:
        noun = "pair" if pairs == 1 else "pairs"
        parts.append(
            '<div class="note good"><p>Stage isolation: the configuration that reached each '
            f"stage differs only in the stages the arms name ({pairs} {noun} compared).</p></div>"
        )
    elif not pairs and len(rows) > 1:
        parts.append(
            '<div class="note warn"><p>Stage isolation not checked: fewer than two arms '
            "finished on a benchmark, so there was nothing to compare.</p></div>"
        )
    for line in report["unverified"]:
        parts.append(f'<div class="note warn"><p>Stage isolation unverified: {_e(line)}</p></div>')
    for line in report["inert"]:
        parts.append(f'<div class="note warn"><p>Inert arm: {_e(line)}</p></div>')
    if not parts:
        parts.append('<p class="muted">One arm only, so there was nothing to compare.</p>')
    return f'<section id="isolation"><h2>Stage isolation</h2>{"".join(parts)}</section>'


def _leaderboard(ranking: List[Dict[str, Any]], metric: str) -> str:
    # The number a component's paper claims, beside the measured one.
    show_claim = any(r.get("claim") for r in ranking)
    rows = []
    for r in ranking:
        rank = str(r["rank"]) if r["rank"] else "–"
        value = _cell(r["mean"], r["std"], r["n_runs"]) if r["mean"] is not None else "-"
        runs = str(r.get("n_runs", "")) if r["mean"] is not None else ""
        queries = str(r.get("n_queries") or "") if r["mean"] is not None else ""
        klass = ' class="unranked"' if not r["rank"] else ""
        claim = f"<td>{_e(claim_text(r.get('claim')))}</td>" if show_claim else ""
        rows.append(
            f'<tr{klass}><td class="num">{rank}</td><td class="arm">{_e(r["config"])}</td>'
            f'<td class="num">{value}</td>{claim}<td class="num">{runs}</td>'
            f'<td class="num">{queries}</td><td>{_e(r["note"])}</td></tr>'
        )
    head = ["#Rank", "Arm", "#" + _e(metric), *(["Claimed"] if show_claim else [])]
    head += ["#Runs", "#Queries", "Note"]
    return (
        "<h3>Leaderboard</h3>"
        '<p class="muted">Ranked over arms that scored the same queries.</p>'
        + _table(head, rows, "leaderboard")
    )


def _metric_table(results: BenchmarkResults, bench: str, metric: str) -> str:
    arms = [c for c in results.configs if results._by_bench_config.get((bench, c))]
    names = sorted(
        {m for c in arms for t in results._by_bench_config[(bench, c)] for m in t.metrics},
        key=lambda m: (m != metric, lower_is_better(m), m),
    )
    rows = []
    for name in names:
        by_arm = results.aggregate_by_benchmark(name).get(bench, {})
        values = [m.mean for m in by_arm.values()]
        # Marked only where the arms differ: a row of equal values has no best.
        best = (min if lower_is_better(name) else max)(values) if len(set(values)) > 1 else None
        cells = []
        for arm in arms:
            agg = by_arm.get(arm)
            if agg is None:
                cells.append('<td class="num muted">-</td>')
                continue
            klass = "num best" if best is not None and agg.mean == best else "num"
            cells.append(f'<td class="{klass}">{_cell(agg.mean, agg.std, agg.n_runs)}</td>')
        rows.append(f'<tr><th scope="row">{_e(name)}</th>{"".join(cells)}</tr>')
    head = ["Metric"] + [f'#<span class="arm">{_e(a)}</span>' for a in arms]
    return (
        "<h3>Every metric, per arm</h3>"
        '<p class="muted">Mean over runs; the best value in each row is marked.</p>'
        + _table(head, rows, "metrics")
    )


def _significance(results: BenchmarkResults, bench: str, metric: str) -> str:
    title = "<h3>Significance</h3>"
    try:
        rows = results.pairwise_significance(metric, bench, correction="holm", alpha=_ALPHA)
    except ValueError as exc:
        return f'{title}<p class="muted">{_e(exc)}</p>'
    if not rows:
        return (
            f'{title}<p class="muted">Significance needs per-query scores from at least two '
            "arms on this benchmark.</p>"
        )
    body = []
    for r in rows:
        pair = f'<span class="arm">{_e(r["config_a"])}</span> vs <span class="arm">{_e(r["config_b"])}</span>'
        if r.get("p_value") is None:
            body.append(
                f'<tr><td class="pair">{pair}</td><td colspan="5" class="muted">could not be tested, counted '
                f"as p = 1: {_e(r.get('untestable', ''))}</td><td></td></tr>"
            )
            continue
        mark = '<span class="chip good">significant</span>' if r.get("significant") else "-"
        body.append(
            f'<tr><td class="pair">{pair}</td><td class="num">{r["difference"]:+.4f}</td>'
            f'<td class="num">{_e(format_effect_size(r.get("effect_size")))}</td>'
            f'<td class="num">{_e(format_p(r["p_value"]))}</td>'
            f'<td class="num">{_e(format_p(r.get("p_adjusted", r["p_value"])))}</td>'
            f'<td class="num">{_e(r.get("n_paired", ""))}</td><td>{mark}</td></tr>'
        )
    n = len(rows)
    table = _table(["Pair (A vs B)", "#Diff (A − B)", "#d_z", "#p", "#p (Holm)", "#n", ""], body)
    opened = " open" if n <= _PAIRS_OPEN else ""
    return (
        f"{title}{_verdict_html(verdict_lines(rows, metric, bench, _ALPHA))}"
        f"<details{opened}><summary>All {n} {'pair' if n == 1 else 'pairs'}, "
        f"paired sign-flip test, Holm-corrected at family-wise alpha={_ALPHA}</summary>"
        f"{table}</details>"
    )


def _clip(text: Any, limit: int = _CONTEXT_CHARS) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _evidence(row: Optional[Mapping[str, Any]]) -> str:
    """What one arm retrieved and answered for one query, as far as it was recorded."""
    if not row:
        return '<p class="muted">Not scored.</p>'
    parts = []
    answer = row.get("_generated_answer")
    if answer:
        parts.append(f'<p class="answer"><b>Answer:</b> {_e(_clip(answer))}</p>')
    contexts = row.get("_retrieved_contexts")
    if isinstance(contexts, list) and contexts:
        items = "".join(f"<li>{_e(_clip(c))}</li>" for c in contexts[:_CONTEXTS_SHOWN])
        parts.append(f'<ol class="passages">{items}</ol>')
    ids = row.get("_retrieved_ids")
    if isinstance(ids, list) and ids:
        expected = {str(d) for d in row.get("_expected_docs") or []}
        shown = []
        for doc in ids[:_IDS_SHOWN]:
            hit = str(doc) in expected
            mark = ' class="hit" title="relevant"' if hit else ""
            shown.append(f"<li{mark}>{_e(doc)}{' ✓' if hit else ''}</li>")
        parts.append(f'<ol class="ids">{"".join(shown)}</ol>')
    return "".join(parts) or '<p class="muted">Nothing recorded.</p>'


def _recorded_note(rows: Iterable[Mapping[str, Any]]) -> str:
    rows = list(rows)
    if any(isinstance(r.get("_retrieved_contexts"), list) for r in rows):
        return ""
    if any(r.get("_question") for r in rows):
        return (
            '<p class="note warn">Passages were not recorded. Add '
            "<code>save_trajectories: full</code> to each arm's entry under "
            "<code>pipeline.config_overrides</code> and run again to see the top five "
            "passages per query here.</p>"
        )
    return (
        '<p class="note warn">Query text and passages were not recorded: a retrieval run '
        "records each query's id and the document ids it retrieved. To see the question, "
        "the answer and the top passages, run the generation track "
        "(<code>tracks: [retrieval, generation]</code>) with "
        "<code>save_trajectories: full</code> in each arm's entry under "
        "<code>pipeline.config_overrides</code>.</p>"
    )


def _per_query(results: BenchmarkResults, bench: str, metric: str, a: str, b: str) -> str:
    title = "<h3>Per query</h3>"
    rows = results.per_query(bench)
    by_arm: Dict[str, Dict[str, Dict[str, Any]]] = {a: {}, b: {}}
    for row in rows:
        if row["config"] in by_arm:
            by_arm[row["config"]][str(row["sample_id"])] = row
    for arm in (a, b):
        if not by_arm[arm]:
            return f'{title}<p class="muted">{_e(arm)} scored no queries on {_e(bench)}.</p>'

    counts = {"a": 0, "b": 0, "tie": 0, "only": 0}
    decided, ties, partial = [], [], []
    for sid in sorted(set(by_arm[a]) | set(by_arm[b])):
        ra, rb = by_arm[a].get(sid), by_arm[b].get(sid)
        va = ra.get(metric) if ra else None
        vb = rb.get(metric) if rb else None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            w = _winner(float(va), float(vb), metric)
            key = {1: "a", -1: "b", 0: "tie"}[w]
            counts[key] += 1
            (ties if w == 0 else decided).append((sid, ra, rb, va, vb, w))
        else:
            counts["only"] += 1
            partial.append((sid, ra, rb, va, vb, None))
    decided.sort(key=lambda r: (-abs(r[3] - r[4]), r[0]))

    body = []
    for sid, ra, rb, va, vb, w in decided + ties + partial:
        question = (ra or {}).get("_question") or (rb or {}).get("_question") or ""
        if w is None:
            scored = a if va is not None else b
            result = f'only <span class="arm">{_e(scored)}</span> scored it'
        elif w == 0:
            result = '<span class="muted">tie</span>'
        else:
            win, lose = (a, b) if w == 1 else (b, a)
            result = f'<strong class="winner">{_e(win)}</strong> over {_e(lose)}'
        delta = f"{va - vb:+.4f}" if w is not None else "-"
        query = f'<code class="qid">{_e(sid)}</code>'
        if question:
            query += f' <span class="question">{_e(_clip(question))}</span>'
        expected = (ra or {}).get("_expected_docs") or (rb or {}).get("_expected_docs")
        relevant = (
            f'<p class="relevant">Relevant: {_e(", ".join(str(d) for d in expected))}</p>'
            if expected
            else ""
        )
        body.append(
            f'<tr class="query"><td>{query}</td><td class="num">{_num(va)}</td>'
            f'<td class="num">{_num(vb)}</td><td class="num">{delta}</td>'
            f"<td>{result}</td></tr>"
            f'<tr class="evidence"><td colspan="5"><details><summary>What each arm '
            f"retrieved</summary>{relevant}"
            f'<div class="sides"><div><h4 class="arm">{_e(a)}</h4>{_evidence(ra)}</div>'
            f'<div><h4 class="arm">{_e(b)}</h4>{_evidence(rb)}</div></div></details></td></tr>'
        )

    total = sum(counts.values())
    summary = (
        f'<span class="arm">{_e(a)}</span> better on {counts["a"]}, '
        f'<span class="arm">{_e(b)}</span> better on {counts["b"]}, '
        f"tied on {counts['tie']}"
        + (f", scored by one arm only on {counts['only']}" if counts["only"] else "")
        + f" of {total} queries"
    )
    head = [
        "Query",
        f'#<span class="arm">{_e(a)}</span>',
        f'#<span class="arm">{_e(b)}</span>',
        "#Diff",
        "Better",
    ]
    note = _recorded_note([*by_arm[a].values(), *by_arm[b].values()])
    return (
        f"{title}"
        f'<p>{_e(metric)}, <span class="arm">{_e(a)}</span> against '
        f'<span class="arm">{_e(b)}</span>; largest differences first. '
        f"Choose two other arms with <code>--diff A B</code>.</p>"
        f"{note}<details><summary>{summary}</summary>{_table(head, body, 'per-query')}</details>"
    )


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

_CSS = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1b1f24; --muted: #59636e; --line: #d5dbe1; --soft: #f3f5f7;
  --accent: #1f5bd1; --good: #116329; --good-bg: #e6f4ea; --warn: #7d5300; --warn-bg: #fff4d6;
  --bad: #b3261e; --bad-bg: #fde8e6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1216; --fg: #e5e8ec; --muted: #9ba5b0; --line: #2c333b; --soft: #181c22;
    --accent: #85a8ff; --good: #5fd07e; --good-bg: #10281a; --warn: #f0c35a; --warn-bg: #2c2410;
    --bad: #ff8a80; --bad-bg: #2f1513;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 16px 80px; }
h1 { font-size: 1.75rem; line-height: 1.2; margin: 0 0 16px; overflow-wrap: anywhere; }
h2 { font-size: 1.3rem; margin: 40px 0 12px; padding-top: 16px; border-top: 1px solid var(--line);
  overflow-wrap: anywhere; }
h3 { font-size: 1.05rem; margin: 28px 0 8px; }
h4 { font-size: .9rem; margin: 0 0 6px; }
p { margin: 6px 0; }
code { font: .88em/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--soft); padding: 1px 4px; border-radius: 4px; overflow-wrap: anywhere; }
.eyebrow { text-transform: uppercase; letter-spacing: .08em; font-size: .75rem; color: var(--muted);
  margin: 0 0 4px; }
.muted { color: var(--muted); }
.facts { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px 24px;
  margin: 0; }
.facts dt { font-size: .75rem; color: var(--muted); }
.facts dd { margin: 0; overflow-wrap: anywhere; }
nav { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-top: 16px; }
nav a, a { color: var(--accent); }
.scroll { overflow-x: auto; margin: 8px 0; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; vertical-align: top; padding: 6px 10px; border-bottom: 1px solid var(--line); }
thead th { font-weight: 600; color: var(--muted); border-bottom: 2px solid var(--line); white-space: nowrap; }
tbody th { font-weight: 500; white-space: nowrap; }
.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.spread { color: var(--muted); }
.arm { overflow-wrap: anywhere; }
td.arm, td.pair { min-width: 10em; }
td.best { font-weight: 700; }
td.best::after { content: " ★"; color: var(--accent); font-weight: 400; }
tr.unranked td { color: var(--muted); }
.chip { display: inline-block; font-size: .78rem; padding: 0 8px; border-radius: 999px; margin: 0 4px 2px 0; }
.chip.good { background: var(--good-bg); color: var(--good); }
.chip.warn { background: var(--warn-bg); color: var(--warn); }
.chip.bad { background: var(--bad-bg); color: var(--bad); }
.note { border-left: 4px solid var(--line); padding: 8px 12px; margin: 8px 0; background: var(--soft);
  border-radius: 0 6px 6px 0; }
.note.good { border-color: var(--good); background: var(--good-bg); }
.note.warn { border-color: var(--warn); background: var(--warn-bg); }
.note.bad { border-color: var(--bad); background: var(--bad-bg); }
.note ul { margin: 6px 0 0; padding-left: 20px; }
.verdict { border: 2px solid var(--warn); border-radius: 8px; padding: 10px 14px; margin: 8px 0 12px;
  max-width: 820px; }
.verdict.significant { border-color: var(--good); }
.verdict-title { font-weight: 700; margin-bottom: 4px; color: var(--warn); }
.verdict.significant .verdict-title { color: var(--good); }
.verdict p { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.winner { color: var(--good); }
details { margin: 8px 0; }
summary { cursor: pointer; color: var(--accent); }
.per-query tr.query td { border-bottom: none; padding-bottom: 0; }
.per-query tr.query td:first-child { min-width: 18em; }
.per-query tr.evidence td { padding-top: 0; }
.per-query tr.evidence details { margin: 2px 0 4px; }
.question { display: block; margin-top: 2px; }
.qid { font-size: .8em; }
.evidence summary { font-size: .82rem; }
.relevant { font-size: .82rem; color: var(--muted); }
.sides { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px;
  margin-top: 6px; }
.passages, .ids { margin: 4px 0; padding-left: 22px; font-size: .85rem; }
.passages li { margin-bottom: 4px; overflow-wrap: anywhere; }
.ids li.hit { color: var(--good); font-weight: 600; }
.answer { font-size: .85rem; overflow-wrap: anywhere; }
footer { margin-top: 48px; color: var(--muted); font-size: .8rem; }
"""


def render_html(
    results: Source,
    metric: Optional[str] = None,
    benchmark: Optional[str] = None,
    diff: Optional[Tuple[str, str]] = None,
) -> str:
    """Render one run as a self-contained HTML page.

    Args:
        results: A :class:`BenchmarkResults`, the path of a ``results.json``
            (or the folder holding one) or a bundle, or either file parsed.
        metric: The metric to rank, test and diff by; ``ndcg@10`` when
            recorded, otherwise the first answer metric the run has.
        benchmark: Show one benchmark only.
        diff: The two arms the per-query diff compares; the top two by
            ``metric`` on each benchmark when omitted.

    Raises:
        ValueError: The file is neither, or names a metric, benchmark or arm
            the run does not have. The message lists what it does have.
    """
    run = _load(results)
    res = run.results
    if not res.total_tasks:
        raise ValueError("These results hold no tasks, so there is nothing to show.")
    benchmarks = res.benchmarks
    if benchmark:
        benchmarks = [res.resolve_benchmark(benchmark)]
    metric = metric or _default_metric(res)
    res.require_metric(metric)
    if diff:
        diff = (str(diff[0]), str(diff[1]))
        unknown = [arm for arm in diff if arm not in res.configs]
        if unknown:
            raise ValueError(
                f"No arm named {unknown[0]!r} in these results. They have: "
                f"{', '.join(res.configs)}."
            )
        if diff[0] == diff[1]:
            raise ValueError("--diff needs two different arms.")

    sections = []
    for bench in benchmarks:
        ranking = _ranking(res, bench, metric)
        arms = _diff_arms(ranking, diff)
        per_query = (
            _per_query(res, bench, metric, *arms)
            if arms
            else '<h3>Per query</h3><p class="muted">One arm only, so there is nothing to compare.</p>'
        )
        sections.append(
            f'<section id="{_anchor(bench)}"><h2>{_e(bench)}</h2>'
            f"{_leaderboard(ranking, metric)}{_significance(res, bench, metric)}"
            f"{_metric_table(res, bench, metric)}{per_query}</section>"
        )

    title = res.experiment_name or "Lasagna results"
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        # Nothing on this page loads anything or runs anything.
        '<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body><main>"
        f"{_header(run, metric, benchmarks)}{_isolation(run)}{''.join(sections)}"
        "<footer>Generated by <code>lasagna results view</code>. Numbers are the run's "
        "own; the page recomputes the leaderboard, the significance tests and the "
        "isolation check from them.</footer>"
        "</main></body></html>\n"
    )
