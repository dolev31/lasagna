# Governance

Short, because the project is small and pretending otherwise would be worse
than saying so.

## Who decides

Lasagna is currently maintained by a single maintainer, who has final say on
what merges. That is not a permanent structure; it is a description of where
the project is. If a second person is reviewing regularly and their judgment
is good, they get commit rights, and this file changes.

Deciding "what merges" does not mean deciding what is true. Anything that
touches a number is settled by evidence, not by the maintainer.

## Decisions about correctness

**A number is not settled by seniority.** If you can show that a metric,
statistic, or benchmark adapter is wrong, that is the end of the argument
regardless of who wrote it. The bar is a failing test or a reproducible
demonstration, not an opinion.

This has already happened several times to code the maintainer wrote, and the
[CHANGELOG](CHANGELOG.md) records the results that changed because of it.

## Result disputes

Open a [result dispute](.github/ISSUE_TEMPLATE/result_dispute.yml). Almost all
of them resolve without a judgment call: run

```bash
lasagna results compare theirs.json yours.json
```

and the field-level diff usually shows the two runs held different things fixed.
That is what the comparability key is for, and it is a better answer than
anybody's ruling.

When a genuine defect is found in a published result, the row is marked disputed
with a link to the issue. **Nothing is deleted.** A leaderboard that quietly
removes numbers it no longer likes is not a record of anything.

## Breaking changes

Pre-1.0, breaking changes ship when they make the library more correct, with the
reasoning in the [CHANGELOG](CHANGELOG.md). Three things get a deprecation period
rather than a straight break, because they invalidate other people's work:

- **Component ABCs**, which would break installed third-party plugins.
- **Metric definitions**, which change what published numbers mean. These bump
  `METRICS_VERSION`, so old results drop out of comparison instead of being
  silently mixed with new ones.
- **The REST API under `/api/v1` and the MCP tool schemas**, which clients and
  agent configs are written against. A route, parameter, response field or tool
  is deprecated for at least one minor release (marked in its description and
  the CHANGELOG) before it is removed or changed incompatibly; additions need no
  deprecation. `tests/snapshots/openapi.json` and `tests/snapshots/mcp_tools.json`
  record both surfaces, so a change to either is a reviewed diff.

## Scope

Lasagna is a RAG evaluation framework: swap one stage, hold the rest fixed, get
an attributable difference. Proposals are weighed against whether they make that
easier or more trustworthy.

The project deliberately does **not** run evaluation as a hosted service. See
[hub/README.md](hub/README.md) for why, and for what is offered instead.

## Conduct

[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) applies everywhere, including code
review. Enforcement is the maintainer's responsibility.
