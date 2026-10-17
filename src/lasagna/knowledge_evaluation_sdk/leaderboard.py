"""Rendering validated bundles as a leaderboard, grouped by what was held fixed.

A single table of everyone's best number is the usual way these become
misleading: the rows behind it used different embedding models, different sample
counts, different splits. Averaging or ranking across that compares nothing.

So rows are grouped by ``(benchmark, comparability_key)``. Everything inside a
group held the same configuration fixed, which is what makes the numbers
commensurable — by construction rather than by hope. A group is a table; the
configuration it held fixed is its header.

A row whose held-fixed block records no stage digests (``fixed_stages``: a
bundle from before metrics version 1.4, or a task that recorded no stage
manifest) cannot show what its retriever, reranker or query enhancer held
fixed, so an equal key proves nothing about them. Such rows share a table only
with rows from the same bundle, and are never verified.

**Reproduction by collision.** A row is marked verified when a *different*
submitter published a bundle with the same comparability key whose mean lands
inside this row's reported interval. No coordination and no extra compute: it
falls out of data the hub already has. The interval is mean ± std across runs,
which is what the bundle reports — deliberately not a bootstrap CI, because the
runs are few and pretending otherwise would overstate the precision. When the
runs agree exactly, the interval is the mean itself.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .bundle import InvalidBundleError, config_entry, invalid_mark, validation_problems
from .run_provenance import claim_text

#: A row from a single run reports no spread, so it can never be confirmed or
#: contradicted by a collision. It is shown, and never marked verified, unless
#: its run recorded ``nothing_samples``: then a second run would repeat it, and
#: its interval is the mean itself.
_MIN_RUNS_FOR_INTERVAL = 2

#: The interval of a row whose runs agree exactly. Such a pipeline is
#: deterministic, and another lab's number reproduces it by equalling it; the
#: tolerance only absorbs floating-point summation order.
_EXACT = 1e-9


def _is_bundle(data: Any) -> bool:
    return isinstance(data, dict) and "comparability_key" not in data and "configs" in data


def _checked(data: Dict[str, Any], path: str) -> Dict[str, Any]:
    """``data`` with what ``results validate`` finds wrong with it, and its path.

    Checked before anything is added: the run id hashes every other field.
    """
    data["_problems"] = validation_problems(data)
    data["_path"] = path
    return data


def _one_file(path: Path) -> Dict[str, Any]:
    """A bundle, or a run's ``results.json`` bundled in memory.

    Bundled the way ``lasagna results bundle`` would, so its rows carry the
    same comparability keys as the file that command writes. A file named on
    its own is not skipped when it cannot be read: there is nothing else to
    show, so the error reaches the caller. A bundle carries the problems
    ``results validate`` finds in it; a results.json is not validated.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not _is_bundle(data):
        if not (isinstance(data, dict) and isinstance(data.get("results"), list)):
            raise ValueError(f"{path.name} is neither a bundle nor a results.json from a run.")
        from .results import BenchmarkResults

        data = BenchmarkResults.load(str(path)).to_bundle()
        # A run's own file, bundled here: nothing validated it, and a link to
        # it would be a path on this machine.
        data["_source"] = "results"
        data["_path"] = ""
        return data
    return _checked(data, path.name)


def load_bundles(directory: str | Path) -> List[Dict[str, Any]]:
    """Read every bundle under a directory, newest path order last, or the one
    file ``directory`` names (a bundle, or a run's ``results.json``).

    Unreadable files are skipped rather than fatal: one malformed submission
    should not stop the leaderboard for everyone else. The validator is what
    keeps malformed files out; this is defence in depth.

    Each bundle carries ``_problems``, what ``results validate`` finds wrong
    with it; :func:`checked_bundles` leaves the invalid ones out.

    Paths are recorded **relative to the scanned directory**. An absolute path
    would be both broken as a link once the page is committed and a small leak
    of whoever generated it -- the leaderboard is published, and no bundle field
    is allowed to carry a local filesystem layout.
    """
    root = Path(directory)
    if root.is_file():
        return [_one_file(root)]
    bundles = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if _is_bundle(data):
            bundles.append(_checked(data, path.relative_to(root).as_posix()))
    return bundles


def _stages_recorded(row: Dict[str, Any]) -> bool:
    """Whether the row's key covers what each stage held fixed."""
    stages = (row.get("held_fixed") or {}).get("fixed_stages")
    return isinstance(stages, dict) and bool(stages)


def _group_key(row: Dict[str, Any]) -> str:
    """The comparability key, scoped to the bundle when stages were not recorded."""
    key = row["comparability_key"]
    return key if _stages_recorded(row) else f"{key}@{row.get('run_id', '')}"


def _submitter(bundle: Dict[str, Any]) -> str:
    who = bundle.get("submitter") or {}
    return who.get("name") or "anonymous"


def _aggregate(values: List[float]) -> Tuple[float, float]:
    """Mean and standard deviation across runs.

    std is 0.0 for a single run — reported as such rather than hidden, since
    "no spread measured" and "no spread exists" are different claims and only
    the run count distinguishes them.
    """
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def collect_rows(bundles: Iterable[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    """One row per (bundle, config), aggregated across that config's runs.

    A row whose component claims a number for ``metric`` on the row's
    benchmark (``claims=`` on its decorator) carries it as ``claim``, with the
    paper.
    """
    from .run_provenance import claimed

    rows: List[Dict[str, Any]] = []
    for bundle in bundles:
        components = (bundle.get("experiment") or {}).get("components")
        by_config: Dict[Tuple[str, str], List[float]] = {}
        systems: Dict[Tuple[str, str], Any] = {}
        splits: Dict[Tuple[str, str], str] = {}
        for result in bundle.get("results", []):
            if result.get("error"):
                continue
            value = (result.get("metrics") or {}).get(metric)
            if value is None:
                continue
            slot = (result.get("benchmark_name", ""), result.get("config_name", ""))
            by_config.setdefault(slot, []).append(float(value))
            systems.setdefault(slot, result.get("system"))
            splits.setdefault(slot, result.get("split") or "")

        for (benchmark, config_name), values in by_config.items():
            cfg = config_entry(bundle, benchmark, config_name)
            mean, std = _aggregate(values)
            rows.append(
                {
                    "benchmark": benchmark,
                    "config": config_name,
                    "submitter": _submitter(bundle),
                    "comparability_key": cfg.get("comparability_key", ""),
                    "held_fixed": cfg.get("held_fixed", {}),
                    "varied_dimensions": cfg.get("varied_dimensions", []),
                    "varied_levels": cfg.get("varied_levels", {}) or {},
                    "mean": mean,
                    "std": std,
                    "n_runs": len(values),
                    "nothing_samples": bool(
                        (bundle.get("experiment") or {}).get("nothing_samples")
                    ),
                    "run_id": bundle.get("run_id", ""),
                    "path": bundle.get("_path", ""),
                    "source": bundle.get("_source", "bundle"),
                    # Set only when asked to show a bundle validation rejects.
                    "invalid": invalid_mark(bundle.get("_problems") or []),
                    # What `group` ranks by: without it every metric sorted
                    # highest first, latency and token counts included.
                    "metric": metric,
                    "claim": claimed(
                        components,
                        systems.get((benchmark, config_name)),
                        metric,
                        benchmark,
                        splits.get((benchmark, config_name), ""),
                    ),
                }
            )
    return rows


def mark_verified(rows: List[Dict[str, Any]]) -> None:
    """Set ``verified`` on rows another submitter independently landed inside.

    Requires a *different* submitter: a lab confirming its own number is not a
    reproduction, and allowing it would make the badge meaningless the moment
    anyone published twice.

    Requires the same ``varied_levels`` too, and that is not redundant with the
    comparability key. The key deliberately excludes the dimension under test
    (see `bundle`'s module docstring), so every arm of a sweep shares one key by
    construction — which meant a baseline row could be stamped "verified"
    because another lab's *reranker* arm happened to land inside its ±1σ. Two
    different systems agreeing on a number is not a reproduction of either.

    `hub/README.md` publishes this badge as "a real reproduction signal rather
    than a badge", and lists the restrictions it relies on; this is the third
    one, which the code did not implement.

    A row from an invalid bundle (shown with ``allow_invalid``) neither earns
    the badge nor confirms another row.
    """
    for row in rows:
        row["verified"] = False
        exact = row["n_runs"] < _MIN_RUNS_FOR_INTERVAL and row.get("nothing_samples")
        if (row["n_runs"] < _MIN_RUNS_FOR_INTERVAL and not exact) or not _stages_recorded(row):
            continue
        if row.get("invalid"):
            continue
        # std == 0 used to disqualify the row, so a deterministic pipeline
        # could never be verified, even by a bit-identical reproduction.
        spread = max(row["std"], _EXACT)
        low, high = row["mean"] - spread, row["mean"] + spread
        row["verified"] = any(
            other is not row
            and other["comparability_key"] == row["comparability_key"]
            and other["comparability_key"]
            and _stages_recorded(other)
            and not other.get("invalid")
            and other["submitter"] != row["submitter"]
            and (other.get("varied_levels") or {}) == (row.get("varied_levels") or {})
            and low <= other["mean"] <= high
            for other in rows
        )


def group(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Nest rows as ``{benchmark: {group_key: [row, ...]}}``, best first.

    The group key is the comparability key, suffixed with ``@run_id`` for rows
    that recorded no stage digests, so those never share a table with another
    bundle's rows.
    """
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in rows:
        out.setdefault(row["benchmark"], {}).setdefault(_group_key(row), []).append(row)
    from .results import lower_is_better

    for keys in out.values():
        for group_rows in keys.values():
            # Latency and tokens rank smallest first; a published table sorted
            # them highest first and put the slowest system on top.
            metric = str(group_rows[0].get("metric", "")) if group_rows else ""
            group_rows.sort(key=lambda r: r["mean"], reverse=not lower_is_better(metric))
    return out


_HEADER_FIELDS = ("split", "n_samples", "embedder_model", "chunker", "retriever_top_k")


def render_markdown(
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]],
    metric: str,
    benchmark: str,
    link_prefix: str = "",
) -> str:
    """Render one benchmark's groups as markdown.

    ``link_prefix`` is the path from the rendered page back to the bundle
    directory. Pages are usually written to ``leaderboards/`` while bundles live
    in ``runs/``, so without it every link on the page is broken.
    """
    groups = grouped.get(benchmark, {})
    from_results = any(r.get("source") == "results" for rows in groups.values() for r in rows)
    trust = (
        f"Ranked by `{metric}`. **From a run's results.json: self-reported, not validated "
        "and not independently reproduced.** To publish it, bundle it (`lasagna results "
        "bundle`) and check the bundle (`lasagna results validate`)."
        if from_results
        else f"Ranked by `{metric}`. **Self-reported, not independently reproduced.** Every "
        "row links to its bundle; `lasagna results validate` recomputes a bundle's "
        "comparability keys, run id and aggregates."
    )
    marks = sorted({r["invalid"] for rows in groups.values() for r in rows if r.get("invalid")})
    lines = [f"# {benchmark}", ""]
    if marks:
        lines += [
            f"**{'; '.join(marks)}.** Rows marked INVALID come from bundles that "
            "`lasagna results validate` rejects: their numbers are not results.",
            "",
        ]
    lines += [
        trust,
        "",
        "Each table is one comparability group: those rows held the same "
        "configuration fixed, so they may be compared with each other. Rows in "
        "different tables may not.",
        "",
    ]

    for key in sorted(groups, key=lambda k: -groups[k][0]["mean"]):
        rows = groups[key]
        held = rows[0]["held_fixed"]
        described = ", ".join(
            f"{f}={held.get(f)}" for f in _HEADER_FIELDS if held.get(f) is not None
        )
        scoped = (
            []
            if _stages_recorded(rows[0])
            else [
                "_Stage settings not recorded, so these rows are compared only "
                "with rows from the same bundle._",
                "",
            ]
        )
        # The paper's own number beside the measured one, when a row's
        # component claims one for this metric.
        show_claim = any(r.get("claim") for r in rows)
        lines += [
            f"## Held fixed: `{rows[0]['comparability_key']}`",
            "",
            f"{described or '_configuration not recorded_'}",
            "",
            *scoped,
            f"| {metric} |"
            + (" Claimed |" if show_claim else "")
            + " Varied | Submitter | Runs | Verified | Bundle |",
            "|---|" + ("---|" if show_claim else "") + "---|---|---|---|---|",
        ]
        for row in rows:
            # Levels when they are recorded, names otherwise. A 2x2's four
            # cells all vary the same two flags, so names alone render them
            # identically and the design becomes unreadable in the one table
            # that exists to display it.
            levels = row.get("varied_levels") or {}
            if levels:
                varied = ", ".join(f"`{k}={levels[k]}`" for k in sorted(levels))
            else:
                varied = ", ".join(f"`{d}`" for d in row["varied_dimensions"]) or "_baseline_"
            spread = f" ± {row['std']:.4f}" if row["n_runs"] > 1 else ""
            if row.get("invalid"):
                spread += f" **{row['invalid']}**"
            verified = "yes" if row.get("verified") else ""
            claim = f" {claim_text(row.get('claim'))} |" if show_claim else ""
            run = f"`{row['run_id'][:12]}`"
            bundle = f"[{run}]({link_prefix}{row['path']})" if row.get("path") else run
            lines.append(
                f"| {row['mean']:.4f}{spread} |{claim} {varied} | {row['submitter']} | "
                f"{row['n_runs']} | {verified} | {bundle} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "**Verified** means a different submitter published a bundle with the "
        "same comparability key whose mean fell inside this row's mean ± std, "
        "or equalled the mean when its runs showed no spread. It is a "
        "reproduction signal, not an endorsement. A single-run row can never be "
        "verified, because one run cannot show whether the number varies.",
        "",
        "_Generated by `lasagna results leaderboard`. Do not edit by hand._",
    ]
    return "\n".join(lines) + "\n"


def render_latex(
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]],
    metric: str,
    benchmark: Optional[str] = None,
) -> str:
    """Render comparability groups as LaTeX: one booktabs ``tabular`` per group.

    The same grouping as the markdown page, for a paper: a table per
    (benchmark, comparability key), captioned with what its rows held fixed,
    so rows that may not be compared never share a table. The best mean in a
    table is bold (smallest first for lower-is-better metrics). Needs
    ``\\usepackage{booktabs}``. ``benchmark`` limits the output to one.
    """
    from .results import _escape_latex as esc
    from .results import lower_is_better

    names = [benchmark] if benchmark else sorted(grouped)
    better = "lower" if lower_is_better(metric) else "higher"
    out = [
        f"% Generated by `lasagna results leaderboard --format latex`, ranked by {metric} "
        f"({better} is better). Needs \\usepackage{{booktabs}}.",
        "% Rows in one table held the same configuration fixed; rows in different "
        "tables may not be compared.",
    ]
    for name in names:
        groups = grouped.get(name, {})
        # Best group first, in the metric's own direction.
        sign = 1 if lower_is_better(metric) else -1
        for key in sorted(groups, key=lambda k: sign * groups[k][0]["mean"]):
            rows = groups[key]
            held = rows[0]["held_fixed"]
            described = ", ".join(
                f"{f}={held.get(f)}" for f in _HEADER_FIELDS if held.get(f) is not None
            )
            scoped = (
                ""
                if _stages_recorded(rows[0])
                else " Stage settings were not recorded, so these rows are compared only "
                "with rows from the same bundle."
            )
            # In the caption and beside each number, so neither a cropped
            # table nor a copied row loses it.
            marks = sorted({r["invalid"] for r in rows if r.get("invalid")})
            warning = "".join(f"\\textbf{{{esc(m)}.}} " for m in marks)
            caption = (
                f"{warning}{esc(name)}: {esc(metric)}. Held fixed: "
                f"{esc(described) if described else 'configuration not recorded'}"
                f" (comparability key \\texttt{{{esc(rows[0]['comparability_key'])}}}).{esc(scoped)}"
            )
            best = rows[0]["mean"]

            def _varied(row) -> str:
                levels = row.get("varied_levels") or {}
                if levels:
                    return ", ".join(f"{k}={levels[k]}" for k in sorted(levels))
                return ", ".join(row["varied_dimensions"]) or "baseline"

            # Columns that say something for these rows: for one local run,
            # "Varied" repeated "Config", every submitter was "anonymous" and
            # "Verified" was empty.
            show_varied = any(_varied(r) != r["config"] for r in rows)
            show_submitter = any(r["submitter"] != "anonymous" for r in rows)
            show_verified = any(r.get("verified") for r in rows)
            # The number the component's paper claims, beside the measured one.
            show_claim = any(r.get("claim") for r in rows)
            columns = ["Config"] + ["Varied"] * show_varied + ["Submitter"] * show_submitter
            columns += ["Runs", esc(metric)] + ["Claimed"] * show_claim
            columns += ["Verified"] * show_verified
            spec = "l" * (1 + show_varied + show_submitter) + "rr" + "l" * show_claim
            spec += "l" * show_verified
            out += [
                "",
                "\\begin{table}[t]",
                "\\centering",
                f"\\caption{{{caption}}}",
                f"\\begin{{tabular}}{{{spec}}}",
                "\\toprule",
                " & ".join(columns) + " \\\\",
                "\\midrule",
            ]
            for row in rows:
                value = f"{row['mean']:.4f}"
                if row["n_runs"] > 1:
                    value = f"{value} \\pm {row['std']:.4f}"
                if len(rows) > 1 and row["mean"] == best:
                    value = f"\\mathbf{{{value}}}"
                cells = [esc(row["config"])]
                cells += [esc(_varied(row))] * show_varied
                cells += [esc(row["submitter"])] * show_submitter
                mark = f" \\textbf{{{esc(row['invalid'])}}}" if row.get("invalid") else ""
                cells += [str(row["n_runs"]), f"${value}${mark}"]
                cells += [esc(claim_text(row.get("claim")))] * show_claim
                cells += ["yes" if row.get("verified") else ""] * show_verified
                out.append(" & ".join(cells) + " \\\\")
            out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


#: The formats `build` can write, by file extension.
FORMATS = {"markdown": ".md", "latex": ".tex"}


def _bundles_recording(directory: str | Path, metric: str) -> List[Dict[str, Any]]:
    """The bundles under ``directory``, refusing a ``metric`` none of them records.

    A typo used to list every row as "no ndgc@10 recorded" and succeed.
    """
    from .results import require_metric

    bundles = load_bundles(directory)
    if bundles:
        recorded = {
            name
            for bundle in bundles
            for result in bundle.get("results", [])
            for name in (result.get("metrics") or {})
        }
        where = Path(directory).name if Path(directory).is_file() else "these bundles"
        require_metric(metric, recorded, where)
    return bundles


def checked_bundles(
    directory: str | Path, metric: str = "ndcg@10", *, allow_invalid: bool = False
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The bundles to rank under ``directory``, and the invalid ones left out.

    Invalid is what ``results validate`` calls INVALID (``validation_problems``):
    an edited bundle was ranked, printed in LaTeX and listed on the page next
    to its original run id. One file named on its own raises
    :class:`~.bundle.InvalidBundleError`. In a directory an invalid bundle is
    left out and returned, so one bad submission does not stop the
    leaderboard for everyone else. ``allow_invalid`` keeps them; their rows
    then carry the watermark and are never verified. A results.json is not a
    bundle and is not validated.
    """
    bundles = _bundles_recording(directory, metric)
    if allow_invalid:
        return bundles, []
    invalid = [b for b in bundles if b.get("_problems")]
    if invalid and Path(directory).is_file():
        raise InvalidBundleError(Path(directory).name, invalid[0]["_problems"])
    return [b for b in bundles if not b.get("_problems")], invalid


def group_bundles(bundles: Iterable[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    """``bundles`` as ranked comparability groups, verified rows marked."""
    rows = collect_rows(bundles, metric)
    mark_verified(rows)
    return group(rows)


def grouped_rows(
    directory: str | Path, metric: str = "ndcg@10", *, allow_invalid: bool = False
) -> Dict[str, Any]:
    """Read the valid bundles under ``directory`` (or the one file it names)
    as ranked comparability groups (see :func:`checked_bundles`)."""
    bundles, _ = checked_bundles(directory, metric, allow_invalid=allow_invalid)
    return group_bundles(bundles, metric)


def build(
    directory: str | Path,
    metric: str = "ndcg@10",
    output: Optional[str | Path] = None,
    fmt: str = "markdown",
    *,
    allow_invalid: bool = False,
) -> Dict[str, Any]:
    """Read bundles (a directory, one bundle or one ``results.json``), group
    them, and optionally write one page per benchmark (``fmt`` is
    ``"markdown"`` or ``"latex"``) plus an index.

    Invalid bundles are handled as :func:`checked_bundles` says; the index
    lists them under ``invalid``, with whether they were shown."""
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt!r}. Use 'markdown' or 'latex'.")
    bundles, left_out = checked_bundles(directory, metric, allow_invalid=allow_invalid)
    rows = collect_rows(bundles, metric)
    mark_verified(rows)
    grouped = group(rows)

    index = {
        "metric": metric,
        "n_bundles": len({r["run_id"] for r in rows}),
        "n_rows": len(rows),
        "benchmarks": {
            name: {
                "n_groups": len(keys),
                "n_rows": sum(len(v) for v in keys.values()),
                "n_verified": sum(1 for v in keys.values() for r in v if r.get("verified")),
            }
            for name, keys in sorted(grouped.items())
        },
        "invalid": [
            {
                "path": b.get("_path", ""),
                "watermark": invalid_mark(b["_problems"]),
                "problems": [p.strip() for p in b["_problems"]],
                "shown": shown,
            }
            for b, shown in [*((b, True) for b in bundles), *((b, False) for b in left_out)]
            if b.get("_problems")
        ],
    }

    if output is not None:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        # Relative, so the committed page links correctly and carries no trace
        # of the machine that generated it. A single file is linked from its
        # folder, as a bundle in a scanned directory is.
        root = Path(directory)
        root = root.parent if root.is_file() else root
        prefix = os.path.relpath(root.resolve(), out.resolve())
        prefix = "" if prefix == "." else f"{PurePosixPath(*Path(prefix).parts)}/"
        for name in grouped:
            safe = name.replace("/", "_")
            page = (
                render_latex(grouped, metric, name)
                if fmt == "latex"
                else render_markdown(grouped, metric, name, link_prefix=prefix)
            )
            (out / f"{safe}{FORMATS[fmt]}").write_text(page, encoding="utf-8")
        (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    return index
