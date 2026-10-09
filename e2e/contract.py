"""Result contract shared by every end-to-end phase.

The report is assembled from these objects, never scraped from logs. That is
deliberate: a release gate whose output has to be parsed out of stdout cannot
be trended across releases, and cannot say anything useful about a run that
crashed halfway.

Two rules the phases must honour:

1. **Record metrics even on failure.** A phase that fails after measuring
   install time should still report the install time. "It broke and here is
   nothing" is the output this system exists to replace.

2. **Never report an unrun check as passed.** A phase that cannot run — no
   Docker, no API key — returns SKIPPED with a reason. The report renders that
   as *unvalidated*, which is a different thing from *fine*.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class Status(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNED = "warned"


class Severity(str, Enum):
    """What a failure means for the release.

    CRITICAL blocks. WARN records and lets the release proceed — used for
    performance thresholds, which a busy laptop can miss without the package
    being broken.
    """

    CRITICAL = "critical"
    WARN = "warn"


@dataclass
class Check:
    """One assertion a human would recognise as a thing that must be true."""

    name: str
    status: Status
    severity: Severity = Severity.CRITICAL
    detail: str = ""
    #: Exact shell command that reproduces this check in isolation. A failure
    #: a reader cannot reproduce is a rumour.
    repro: str = ""
    duration_s: float = 0.0

    @property
    def blocks_release(self) -> bool:
        return self.status is Status.FAILED and self.severity is Severity.CRITICAL


@dataclass
class Metric:
    """A measured number, with its unit, so the report can be trended."""

    name: str
    value: float
    unit: str
    #: Lower is better for latency, higher for throughput. Used for regression
    #: direction when diffing two runs.
    lower_is_better: bool = True
    note: str = ""


@dataclass
class PhaseResult:
    key: str
    title: str
    checks: List[Check] = field(default_factory=list)
    metrics: List[Metric] = field(default_factory=list)
    #: Set when the phase could not run at all (missing Docker, missing key).
    skipped_reason: str = ""
    duration_s: float = 0.0

    # -- construction helpers -------------------------------------------------

    def check(
        self,
        name: str,
        ok: bool,
        *,
        severity: Severity = Severity.CRITICAL,
        detail: str = "",
        repro: str = "",
        duration_s: float = 0.0,
    ) -> bool:
        self.checks.append(
            Check(
                name=name,
                status=Status.PASSED if ok else Status.FAILED,
                severity=severity,
                detail=detail,
                repro=repro,
                duration_s=duration_s,
            )
        )
        return ok

    def skip(self, name: str, reason: str, *, severity: Severity = Severity.CRITICAL) -> None:
        self.checks.append(
            Check(name=name, status=Status.SKIPPED, severity=severity, detail=reason)
        )

    def measure(
        self,
        name: str,
        value: float,
        unit: str,
        *,
        lower_is_better: bool = True,
        note: str = "",
    ) -> None:
        self.metrics.append(
            Metric(name=name, value=value, unit=unit, lower_is_better=lower_is_better, note=note)
        )

    # -- verdict --------------------------------------------------------------

    @property
    def status(self) -> Status:
        if self.skipped_reason:
            return Status.SKIPPED
        if not self.checks:
            # A phase that ran and asserted nothing has verified nothing.
            # Reporting that as PASSED is the exact failure this design exists
            # to prevent — it turns "nobody checked" into "checked and fine".
            # A phase with legitimately nothing to do must say so via
            # skipped_reason.
            return Status.FAILED
        if any(c.status is Status.FAILED and c.severity is Severity.CRITICAL for c in self.checks):
            return Status.FAILED
        if any(c.status is Status.FAILED for c in self.checks):
            return Status.WARNED
        if all(c.status is Status.SKIPPED for c in self.checks):
            return Status.SKIPPED
        return Status.PASSED

    @property
    def blockers(self) -> List[Check]:
        blocking = [c for c in self.checks if c.blocks_release]
        if not self.checks and not self.skipped_reason:
            # Surface the empty-phase defect as a blocker in its own right,
            # otherwise a phase that asserts nothing contributes no blockers and
            # the run sails through.
            blocking.append(
                Check(
                    name=f"{self.key}: phase asserted nothing",
                    status=Status.FAILED,
                    severity=Severity.CRITICAL,
                    detail=(
                        "the phase ran but produced no checks, so it verified "
                        "nothing. If it genuinely had nothing to do, it must set "
                        "skipped_reason instead."
                    ),
                    repro=f"python -m e2e.run --phase {self.key}",
                )
            )
        return blocking

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class RunResult:
    """A whole gate run: environment, phases, verdict."""

    started_at: str
    environment: Dict[str, Any]
    phases: List[PhaseResult] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def blockers(self) -> List[Check]:
        found = [c for p in self.phases for c in p.blockers]
        if not self.phases:
            # A run with no phases verified nothing. Saying GO would be the
            # single most dangerous output this tool could produce, and it is
            # easy to reach by accident: a --phase filter that matches nothing.
            found.append(
                Check(
                    name="gate ran at least one phase",
                    status=Status.FAILED,
                    severity=Severity.CRITICAL,
                    detail="no phases ran, so nothing was verified",
                    repro="python -m e2e.run --list",
                )
            )
        return found

    @property
    def go(self) -> bool:
        return not self.blockers

    @property
    def unvalidated(self) -> List[PhaseResult]:
        """Phases that could not run. Reported as unvalidated, never as passed."""
        return [p for p in self.phases if p.status is Status.SKIPPED]

    def to_dict(self) -> Dict[str, Any]:
        return redact_local_paths(
            {
                "started_at": self.started_at,
                "verdict": "GO" if self.go else "NO-GO",
                "duration_s": round(self.duration_s, 1),
                "environment": self.environment,
                "blockers": [asdict(c) for c in self.blockers],
                "unvalidated": [p.key for p in self.unvalidated],
                "phases": [p.to_dict() for p in self.phases],
            }
        )


# ---------------------------------------------------------------------------
# Path redaction
# ---------------------------------------------------------------------------


def redact_local_paths(obj: Any) -> Any:
    """Replace machine-local absolute paths anywhere in a report.

    Reports are committed, so whatever they contain is published. Subprocess
    output, tracebacks and repro commands all carry absolute paths, and those
    disclose the developer's home directory, their local layout, and the names
    of unrelated projects that happen to sit alongside this one — an earlier
    run committed 287 references to the internal monorepo this package was
    extracted from, purely because a temp path ran through a venv there.

    Ordering matters: the repo root is rewritten before the home directory, so
    a path inside the checkout becomes ``<repo>/...`` rather than ``~/...``.
    Longest-first for the same reason.
    """
    import os
    import re

    repo = str(Path(__file__).resolve().parents[1])
    home = os.path.expanduser("~")
    tmp = tempfile.gettempdir().rstrip("/")

    # The gate's own interpreter is frequently some unrelated project's
    # virtualenv. Its full path is no use to anyone else reproducing a check,
    # and it names a directory that has nothing to do with this package.
    subs = [
        (p, tag)
        for p, tag in (
            (sys.executable, "python"),
            (repo, "<repo>"),
            (home, "<home>"),
            (tmp, "<tmp>"),
        )
        if p
    ]
    subs.sort(key=lambda kv: len(kv[0]), reverse=True)

    # /var/folders/XX/<token>/T is the macOS per-user temp root; the token
    # identifies the user even when TMPDIR is unset. Match exactly that much,
    # not the whole path -- the part after it names what the phase was doing
    # and is worth keeping in the report.
    patterns = [
        (re.compile(r"(?:/private)?/var/folders/[^/\s]+/[^/\s]+/[A-Za-z]+"), "<tmp>"),
    ]

    # Subprocess output is truncated for the report, which routinely cuts a
    # path mid-string ("...lders/xg/<token>/T/..."). The prefix the pattern
    # above anchors on is then gone, so the identifying token survives. Redact
    # the opaque components on their own as well, which does not depend on any
    # surrounding text being intact.
    subs += [(part, "<tmp-id>") for part in tmp.strip("/").split("/") if len(part) >= 12]
    subs.sort(key=lambda kv: len(kv[0]), reverse=True)

    def scrub(s: str) -> str:
        for path, tag in subs:
            s = s.replace(path, tag)
        for rx, tag in patterns:
            s = rx.sub(tag, s)
        return s

    if isinstance(obj, str):
        return scrub(obj)
    if isinstance(obj, dict):
        return {k: redact_local_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_local_paths(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def capture_environment() -> Dict[str, Any]:
    """Record what the numbers in this report were produced on.

    A p95 latency without the machine it was measured on is not comparable to
    anything, so trending across releases needs this.
    """
    env: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
    }
    try:
        import os

        env["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover
        pass
    try:
        import psutil  # type: ignore

        env["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        env["ram_gb"] = None

    env["docker"] = _probe(["docker", "version", "--format", "{{.Server.Version}}"])
    env["git_commit"] = _probe(["git", "rev-parse", "--short", "HEAD"])
    env["git_dirty"] = bool(_probe(["git", "status", "--porcelain"]))

    # Which optional capabilities are available decides which phases can run at
    # all, so record it rather than inferring later from a skip reason.
    #
    # Includes the gitignored .env, because phases read it: a report that said
    # "no keys present" while the quality phase had just made real LLM calls
    # would be contradicting itself. Names only — never values.
    import os
    from pathlib import Path

    candidates = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "WATSONX_API_KEY",
        "WATSONX_APIKEY",
        "OPENROUTER_API_KEY",
    )
    present = {k for k in candidates if os.environ.get(k)}

    dotenv = Path(__file__).resolve().parents[1] / ".env"
    if dotenv.exists():
        try:
            for line in dotenv.read_text().splitlines():
                name, _, value = line.strip().partition("=")
                if name.strip() in candidates and value.strip():
                    present.add(name.strip())
        except Exception:
            pass

    env["llm_keys_present"] = sorted(present)
    return env


def _probe(cmd: List[str]) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or None
    except Exception:
        return None


class timed:
    """Context manager returning elapsed seconds via ``.elapsed``."""

    def __enter__(self) -> "timed":
        self._t0 = time.monotonic()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.monotonic() - self._t0
