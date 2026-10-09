"""Lasagna end-to-end release gate.

Run this before tagging a release. It installs the package the way a stranger
would, exercises it outside-in, and writes a dated report with a GO/NO-GO
verdict.

    python -m e2e.run                     # everything available on this machine
    python -m e2e.run --phase p1 p2       # a subset
    python -m e2e.run --quick             # CI subset: no Docker, no services
    python -m e2e.run --list              # what would run, and why

Exit status is 0 for GO and 1 for NO-GO, so it can gate a release script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

from .contract import PhaseResult, RunResult, capture_environment
from .report import load_previous, render

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Phases that need no Docker and no services — the CI subset. These cover the
# defects most likely to reach a user: a broken install, a broken quickstart, a
# broken core journey.
QUICK_PHASES = ("p1", "p2", "p3", "p4")


def _phases() -> Dict[str, Callable[[], PhaseResult]]:
    """Import phases lazily so one broken phase cannot stop the whole gate."""
    from .phases import (
        p1_install,
        p2_cold_start,
        p3_docs,
        p4_journey,
        p8_quality,
        p9_lifecycle,
    )

    return {
        "p1": p1_install.run,
        "p2": p2_cold_start.run,
        "p3": p3_docs.run,
        "p4": p4_journey.run,
        "p8": p8_quality.run,
        "p9": p9_lifecycle.run,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lasagna end-to-end release gate")
    ap.add_argument("--phase", nargs="+", metavar="KEY", help="run only these phases")
    ap.add_argument("--quick", action="store_true", help=f"CI subset: {', '.join(QUICK_PHASES)}")
    ap.add_argument("--list", action="store_true", help="list phases and exit")
    ap.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    args = ap.parse_args(argv)

    available = _phases()

    if args.list:
        print("Phases:")
        for key, fn in available.items():
            doc = (fn.__module__.split(".")[-1]).replace("_", " ")
            print(f"  {key:<4} {doc}")
        print(f"\nQuick subset: {', '.join(QUICK_PHASES)}")
        return 0

    selected = list(available)
    if args.quick:
        selected = [k for k in selected if k in QUICK_PHASES]
    if args.phase:
        unknown = set(args.phase) - set(available)
        if unknown:
            print(f"unknown phase(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        selected = [k for k in selected if k in args.phase]

    started = datetime.now(timezone.utc)
    run = RunResult(
        started_at=started.strftime("%Y-%m-%d %H:%M:%S UTC"),
        environment=capture_environment(),
    )

    print(f"Lasagna release gate — {len(selected)} phase(s)\n")
    t0 = time.monotonic()

    for key in selected:
        print(f"  {key} ...", end=" ", flush=True)
        p_t0 = time.monotonic()
        try:
            result = available[key]()
        except Exception as exc:
            # A phase that raises is a failed phase, not a failed run. The
            # remaining phases still carry information worth having.
            result = PhaseResult(key=key, title=f"{key} (crashed)")
            result.check(
                f"{key} completed",
                False,
                detail=f"{type(exc).__name__}: {exc}",
                repro=f"python -m e2e.run --phase {key}",
            )
            result.checks[-1].detail += "\n" + traceback.format_exc(limit=3)
        result.duration_s = time.monotonic() - p_t0
        run.phases.append(result)

        status = result.status
        note = f" — {result.skipped_reason}" if result.skipped_reason else ""
        print(f"{status.value}{note} ({result.duration_s:.0f}s)")

    run.duration_s = time.monotonic() - t0

    # -- write the report ----------------------------------------------------
    out_dir = args.reports_dir / started.strftime("%Y-%m-%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = load_previous(args.reports_dir, exclude=out_dir)

    (out_dir / "results.json").write_text(json.dumps(run.to_dict(), indent=2))
    (out_dir / "report.md").write_text(render(run, previous))

    print()
    print(f"verdict: {'GO' if run.go else 'NO-GO'}")
    if run.blockers:
        print(f"{len(run.blockers)} critical failure(s):")
        for c in run.blockers:
            print(f"  - {c.name}: {c.detail.splitlines()[0] if c.detail else 'failed'}")
    if run.unvalidated:
        print(f"unvalidated (not passed): {', '.join(p.key for p in run.unvalidated)}")
    print(f"report: {out_dir / 'report.md'}")

    return 0 if run.go else 1


if __name__ == "__main__":
    raise SystemExit(main())
