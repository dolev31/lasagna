"""Render a gate run as markdown, and diff it against the previous run.

The regression section is the reason reports are kept rather than printed: a
30% latency slide or a silently disabled phase is invisible in a single run and
obvious across two.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .contract import Metric, RunResult, Status, redact_local_paths

_ICON = {
    Status.PASSED: "PASS",
    Status.FAILED: "FAIL",
    Status.SKIPPED: "SKIP",
    Status.WARNED: "WARN",
}


def render(run: RunResult, previous: Optional[dict] = None) -> str:
    out: List[str] = []
    a = out.append

    verdict = "GO" if run.go else "NO-GO"
    a(f"# Lasagna release gate — {verdict}")
    a("")
    a(f"_{run.started_at} · {run.duration_s:.0f}s_")
    a("")

    if not run.go:
        a(f"**{len(run.blockers)} critical check(s) failed. This build must not ship.**")
        a("")
        for c in run.blockers:
            a(f"- **{c.name}** — {c.detail or 'failed'}")
            if c.repro:
                a(f"  - reproduce: `{c.repro}`")
        a("")

    if run.unvalidated:
        a(
            "> **Unvalidated.** These phases could not run, so nothing is known "
            "about them. This is not the same as passing."
        )
        a("")
        for p in run.unvalidated:
            a(f"> - `{p.key}` — {p.skipped_reason}")
        a("")

    # -- environment ---------------------------------------------------------
    a("## Environment")
    a("")
    a("| | |")
    a("|---|---|")
    for k, v in run.environment.items():
        a(f"| {k} | {v if v not in (None, '', []) else '—'} |")
    a("")

    # -- phases --------------------------------------------------------------
    a("## Phases")
    a("")
    a("| Phase | Result | Checks | Time |")
    a("|---|---|---|---|")
    for p in run.phases:
        passed = sum(1 for c in p.checks if c.status is Status.PASSED)
        a(f"| {p.title} | {_ICON[p.status]} | {passed}/{len(p.checks)} | {p.duration_s:.0f}s |")
    a("")

    for p in run.phases:
        a(f"### {p.title}")
        a("")
        if p.skipped_reason:
            a(f"Skipped — {p.skipped_reason}")
            a("")
            continue

        failures = [c for c in p.checks if c.status is Status.FAILED]
        if failures:
            a("**Failures**")
            a("")
            for c in failures:
                tag = "blocks release" if c.blocks_release else "warning"
                a(f"- `{c.name}` ({tag}) — {c.detail or 'failed'}")
                if c.repro:
                    a(f"  - reproduce: `{c.repro}`")
            a("")

        skips = [c for c in p.checks if c.status is Status.SKIPPED]
        if skips:
            a("**Not run**")
            a("")
            for c in skips:
                a(f"- `{c.name}` — {c.detail}")
            a("")

        if p.metrics:
            a("| Metric | Value | |")
            a("|---|---:|---|")
            for m in p.metrics:
                a(f"| {m.name} | {_fmt(m.value)} | {m.unit}{' · ' + m.note if m.note else ''} |")
            a("")

    # -- regression ----------------------------------------------------------
    if previous:
        a("## Change since last run")
        a("")
        deltas = _diff(run, previous)
        if deltas:
            a("| Metric | Previous | Now | Change |")
            a("|---|---:|---:|---|")
            for row in deltas:
                a(f"| {row['name']} | {_fmt(row['old'])} | {_fmt(row['new'])} | {row['verdict']} |")
        else:
            a("No comparable metrics in the previous run.")
        a("")

        gone = _phases_that_stopped_running(run, previous)
        if gone:
            a(
                "**Phases that ran last time and did not run now:** "
                + ", ".join(f"`{k}`" for k in gone)
                + ". A phase quietly ceasing to run looks like success."
            )
            a("")

    # Redact here rather than at the call site. `results.json` was scrubbed
    # because `RunResult.to_dict()` redacts, but the markdown was rendered from
    # the raw run and shipped the developer's home directory inside every
    # `reproduce:` command -- and the report is the artefact people paste into
    # issues. Making the renderer responsible means there is no way to emit an
    # unredacted one.
    return redact_local_paths("\n".join(out) + "\n")


def _fmt(v: float) -> str:
    # Counts are metrics too: `leftover_files` rendering as "0.0000 entries"
    # reads like a measurement precise to four places rather than a tally.
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v):,}"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    # Never round a nonzero measurement down to a flat "0". A report claiming
    # something took no time at all is worse than one saying 1.2e-05.
    return s if s not in ("", "-", "0", "-0") else f"{v:.2g}"


def _diff(run: RunResult, previous: dict) -> List[Dict]:
    """Compare metrics by (phase, name), flagging moves beyond 10%."""
    old: Dict[tuple, Metric] = {}
    for p in previous.get("phases", []):
        for m in p.get("metrics", []):
            old[(p["key"], m["name"])] = m

    rows = []
    for p in run.phases:
        for m in p.metrics:
            prev = old.get((p.key, m.name))
            if not prev or not prev.get("value"):
                continue
            o, n = float(prev["value"]), float(m.value)
            if o == 0:
                continue
            pct = (n - o) / abs(o) * 100
            if abs(pct) < 10:
                verdict = f"{pct:+.0f}% (noise)"
            else:
                improved = (n < o) if m.lower_is_better else (n > o)
                verdict = f"{pct:+.0f}% {'better' if improved else 'WORSE'}"
            rows.append({"name": f"{p.key}·{m.name}", "old": o, "new": n, "verdict": verdict})
    return rows


def _phases_that_stopped_running(run: RunResult, previous: dict) -> List[str]:
    now_ran = {p.key for p in run.phases if p.status is not Status.SKIPPED}
    before_ran = {p["key"] for p in previous.get("phases", []) if p.get("status") != "skipped"}
    return sorted(before_ran - now_ran)


def load_previous(reports_dir: Path, exclude: Path) -> Optional[dict]:
    """Most recent results.json under ``reports_dir``, ignoring this run."""
    candidates = sorted(
        (p for p in reports_dir.glob("*/results.json") if p.parent != exclude),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        try:
            return json.loads(c.read_text())
        except Exception:
            continue
    return None
