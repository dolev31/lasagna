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

#: A row from a single run reports no spread, so it can never be confirmed or
#: contradicted by a collision. It is shown, and never marked verified.
_MIN_RUNS_FOR_INTERVAL = 2

#: The interval of a row whose runs agree exactly. Such a pipeline is
#: deterministic, and another lab's number reproduces it by equalling it; the
#: tolerance only absorbs floating-point summation order.
_EXACT = 1e-9


def load_bundles(directory: str | Path) -> List[Dict[str, Any]]:
    """Read every bundle under a directory, newest path order last.

    Unreadable files are skipped rather than fatal: one malformed submission
    should not stop the leaderboard for everyone else. The validator is what
    keeps malformed files out; this is defence in depth.

    Paths are recorded **relative to the scanned directory**. An absolute path
    would be both broken as a link once the page is committed and a small leak
    of whoever generated it -- the leaderboard is published, and no bundle field
    is allowed to carry a local filesystem layout.
    """
    root = Path(directory)
    bundles = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "comparability_key" not in data and "configs" in data:
            data["_path"] = path.relative_to(root).as_posix()
            bundles.append(data)
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
    """One row per (bundle, config), aggregated across that config's runs."""
    rows: List[Dict[str, Any]] = []
    for bundle in bundles:
        keys = {c["name"]: c for c in bundle.get("configs", [])}
        by_config: Dict[Tuple[str, str], List[float]] = {}
        for result in bundle.get("results", []):
            if result.get("error"):
                continue
            value = (result.get("metrics") or {}).get(metric)
            if value is None:
                continue
            slot = (result.get("benchmark_name", ""), result.get("config_name", ""))
            by_config.setdefault(slot, []).append(float(value))

        for (benchmark, config_name), values in by_config.items():
            cfg = keys.get(config_name, {})
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
                    "run_id": bundle.get("run_id", ""),
                    "path": bundle.get("_path", ""),
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
    """
    for row in rows:
        row["verified"] = False
        if row["n_runs"] < _MIN_RUNS_FOR_INTERVAL or not _stages_recorded(row):
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
    for keys in out.values():
        for group_rows in keys.values():
            group_rows.sort(key=lambda r: r["mean"], reverse=True)
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
    lines = [
        f"# {benchmark}",
        "",
        f"Ranked by `{metric}`. **Self-reported and machine-validated, not "
        "independently reproduced** — every row links to its bundle.",
        "",
        "Each table is one comparability group: those rows held the same "
        "configuration fixed, so they may be compared with each other. Rows in "
        "different tables may not.",
        "",
    ]

    groups = grouped.get(benchmark, {})
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
        lines += [
            f"## Held fixed: `{rows[0]['comparability_key']}`",
            "",
            f"{described or '_configuration not recorded_'}",
            "",
            *scoped,
            f"| {metric} | Varied | Submitter | Runs | Verified | Bundle |",
            "|---|---|---|---|---|---|",
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
            verified = "yes" if row.get("verified") else ""
            lines.append(
                f"| {row['mean']:.4f}{spread} | {varied} | {row['submitter']} | "
                f"{row['n_runs']} | {verified} | "
                f"[`{row['run_id'][:12]}`]({link_prefix}{row['path']}) |"
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


def build(
    directory: str | Path,
    metric: str = "ndcg@10",
    output: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Read bundles, group them, and optionally write markdown plus an index."""
    rows = collect_rows(load_bundles(directory), metric)
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
    }

    if output is not None:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        # Relative, so the committed page links correctly and carries no trace
        # of the machine that generated it.
        prefix = os.path.relpath(Path(directory).resolve(), out.resolve())
        prefix = "" if prefix == "." else f"{PurePosixPath(*Path(prefix).parts)}/"
        for name in grouped:
            safe = name.replace("/", "_")
            (out / f"{safe}.md").write_text(
                render_markdown(grouped, metric, name, link_prefix=prefix), encoding="utf-8"
            )
        (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    return index
