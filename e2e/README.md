# End-to-end release gate

The last thing run before tagging a release.

Everything in `tests/` is tested from *inside* the repo — importing `lasagna`
from `src/` with the working tree on `sys.path`. That answers "does the code
work", not the question a release actually turns on:

> Does a stranger, on their own machine, `pip install lasagna-rag`, follow the
> README, and get a working RAG comparison?

This gate answers that one. It installs the package the way a user would,
exercises it outside-in, and writes a dated report with metrics, failures, and
a GO/NO-GO verdict.

## Running it

```bash
python -m e2e.run              # everything this machine can run
python -m e2e.run --quick      # CI subset: no Docker, no services
python -m e2e.run --phase p4   # one phase
python -m e2e.run --list       # what would run
```

Exit status is `0` for GO and `1` for NO-GO, so it can gate a release script:

```bash
python -m e2e.run && git tag v0.1.0
```

## Phases

| | Phase | Needs | Blocks release on |
|---|---|---|---|
| `p1` | Install | — | wheel builds, core install and import work |
| `p2` | Cold start | — | import purity, the quickstart's CLI commands, the first search finishing and finding the document |
| `p3` | Documentation | — | any documented example that errors |
| `p4` | Core journey | — | register → compare → swap → run |
| `p8` | Quality | LLM API key | — (warns; records *unvalidated* without a key) |
| `p9` | Lifecycle | — | import broken after upgrade, or still working after uninstall |

**There is no `p5`, `p6` or `p7`.** This table used to list a backend matrix, an
interfaces phase and a stress phase, each with a "blocks release on" column
describing a guarantee — conformance on any backend, an advertised interface
off-contract, tenant isolation and unbounded memory. None of them exists:
`run.py` registers six phases and `e2e/phases/` holds six files, so
`--phase p5` is rejected as unknown. A gate document that advertises checks
nobody runs is worse than a shorter one, because a reader takes the guarantee at
face value. The numbering keeps its gap rather than being renumbered, so that
older reports stay readable.

`p1`–`p4` need nothing but a Python interpreter, which is why they are the CI
subset: they catch the defects most likely to reach a user — a broken install,
a broken quickstart, a broken core journey.

`p9` also needs no Docker, but it installs the package three times over
(previous version, upgrade, reinstall) and takes a couple of minutes, so it
stays out of the per-PR subset. It is the only phase that exercises what
happens on a user's *second* day rather than their first: an upgrade that
merges into the package directory instead of replacing it, or an uninstall that
leaves a shadow copy importable. Its upgrade-hygiene check works by injecting a
probe module into a synthetic older release and asserting the probe is
importable *before* the upgrade — otherwise its absence afterwards would prove
nothing.

## Two rules the phases follow

**Metrics are recorded even when a phase fails.** A phase that fails after
measuring install time still reports the install time. "It broke and here is
nothing" is the output this system exists to replace.

**An unrun check is never reported as passed.** A phase that cannot run — no
Docker, no API key — is `SKIPPED` with a reason, and the report renders it as
*unvalidated*. That is a different thing from *fine*. It matters most for `p8`:
without an API key, the entire generation half of a RAG framework goes
unexercised, and a report that quietly said "passed" would be lying about the
thing most worth knowing.

## Reports

Written to `e2e/reports/<timestamp>/` as `report.md` and `results.json`.

They are committed deliberately. A single run cannot tell you that p95 latency
slid 30% or that a phase quietly stopped running; two runs can. The report's
last section diffs against the previous one and flags both.

## Severity

`CRITICAL` failures block the release. `WARN` failures are recorded and let it
proceed — used for performance thresholds, because a busy laptop missing a p95
target does not mean the package is broken, and a gate that cries wolf gets
switched off.

Correctness is never `WARN`.
