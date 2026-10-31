# Working on Lasagna

Instructions for a coding agent with this repository open. Short on purpose:
this is what changes your behavior, not a second README.

Detailed task guides ship as skills. Load them with:

```bash
lasagna skills install        # writes .claude/skills/
lasagna skills list
```

`lasagna-component` (build a component), `lasagna-eval` (measure a change),
`lasagna-rag` (use retrieval), `lasagna-support` (diagnose and report).

## What this project is

A RAG framework whose point is **attribution**: swap one of eight pipeline
stages, hold the rest fixed, and the measured difference belongs to the swap.
Everything below follows from protecting that property.

## Rules that override normal instincts

**A number without provenance is not a result.** If you change something that
could move a metric, say what was held fixed. `n_samples`, split and `n_runs`
belong next to any figure you report.

**Never state a measurement you did not take.** No estimated benchmark scores,
no "should improve recall by ~5%". Run it or say you did not.

**Components must not mutate their inputs.** Deep-copy first. The pipeline may
hand the same objects to several stages, and in-place edits make results depend
on execution order, which silently destroys attribution.

**A test that passes without the fix is not a test.** After fixing a bug, revert
the fix and confirm the test fails. Several defects in this repo shipped behind
tests that never exercised the broken path.

**Documentation is checked by tests.** `tests/test_documented_claims.py`,
`tests/test_skills.py` and `tests/test_pypi_readme.py` assert that counts,
command names and links are real. Changing a CLI command name or a tool count
means updating the docs in the same commit; the suite will tell you.

## Layout

| Path | What lives there |
|---|---|
| `src/lasagna/core/` | registry, ABCs, config, provenance, diagnostics |
| `src/lasagna/api/` | search orchestrator, REST surface |
| `src/lasagna/retrieval/` | retrievers, hybrid search, reranking, query enhancement |
| `src/lasagna/processing/` | document processors and chunkers |
| `src/lasagna/generation/` | answer generators (the `generator` stage) |
| `src/lasagna/agentic/` | multi-step retrieval: the structured graph (CRAG, quality gates) and the published loops (Self-Ask, IRCoT, FLARE, Adaptive-RAG, Self-RAG) |
| `src/lasagna/vector_backends/` | 6 backends behind one adapter interface |
| `src/lasagna/knowledge_evaluation_sdk/` | benchmarks, metrics, bundles, leaderboard |
| `src/lasagna/mcp/` | MCP server, 29 tools |
| `e2e/` | release gate: phases, typed contract, GO/NO-GO |
| `hub/` | shared results hub: schema, validator, trust model |

## Commands

```bash
pytest tests/ --ignore=tests/live --ignore=tests/stress -q   # the suite CI runs
ruff check . && ruff format .                                # both must be clean
lasagna doctor                                               # what is wrong locally
python -m e2e.run                                            # full gate (slow: builds a wheel)
```

## Traps specific to this codebase

**Two registries exist.** `lasagna.core.registry` is the public plugin registry
(`@registry.register("reranker", ...)`, entry points). `lasagna.retrieval.registry`
is an older internal one. New work uses the core registry.

**Public components need adapting into the pipeline.** The orchestrator calls
rerankers with an internal signature; `adapt_reranker_for_pipeline` bridges it.
A component that passes `lasagna component validate` is not thereby proven to
work in the pipeline: the validator exercises it in isolation.

**Milvus caps `limit + offset` at 16384.** Never paginate a full scan with
`query()`. Use `backend.iter_all()`, which uses a native cursor.

**Never hard-code a vendor-routed model name** (`Azure/...`, `watsonx/...`) as a
default. Use `get_default_model_name()`. A test enforces this.

**First run downloads about 3.2 GB** (BGE-M3 and its reranker). Not a hang.
`lasagna configure --profile speed` uses a 92 MB model instead.

**Retrieval with fixed models is deterministic.** A retrieval-only sweep needs
`n_runs: 1`; repeat runs only when an LLM stage is in the loop. Configuration
(which setting wins, where data lives) is in `docs/configuration.md`.

## Before you commit

Read `CONTRIBUTING.md`. Explain *why* in the commit message; the diff already
shows what.

## Before you open a pull request

The bar is in `CONTRIBUTING.md` under "Quality bar", and it is the same for an
agent as for a person. Check the description and the diff the way CI will:

```bash
python scripts/check_pr_body.py --file body.md --diff origin/main
```

`tests/test_pr_body_check.py` shows what passes and what fails. In particular:

- Fill in every section of `.github/PULL_REQUEST_TEMPLATE.md`. Under "Agent use",
  "Written with:" names you, truthfully. Leave "Read in full by:" for the person
  who reads every line of the diff; never fill it in for them.
- Never tick a checklist box for something you did not do or run.
- Run the change in the production environment, the Docker Compose stack in
  `src/lasagna/deploy` built from the branch, and paste the `docker compose`
  commands and their output under "How it was tested" (CONTRIBUTING.md,
  "Tested in the production environment"). If your sandbox has no Docker
  daemon, tell the person you work for; a GitHub Codespace runs Docker. Never
  report a run that did not happen in the stack as one that did, and never ask
  a maintainer to make the run for you: it is the contributor's to make.
- A test that fails without the change. A line in `CHANGELOG.md` under
  Unreleased for anything a user sees.
- Every `noqa`, `type: ignore`, `pragma: no cover` and skip gives its reason;
  every TODO links its issue. An `except` that only passes says why.
- "Evidence" is a result you ran (`lasagna results significance`), or "No number
  can move:" with the reason. Never a number you did not measure.
