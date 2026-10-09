# The results hub

Everything needed to stand up a shared leaderboard for Lasagna results. It is
deliberately a **separate repository with no server, no compute and no
secrets** — a submission is a pull request adding one JSON file.

```
<you>/lasagna-results/       (the repo you create)
├── runs/<year>/<submitter>/<slug>-<hash>.json
├── schema/run-1.0.json                 <- copy hub/run-1.0.schema.json here
├── leaderboards/<benchmark>.md         generated on merge, never hand-edited
├── index.json                          generated on merge
└── .github/workflows/
    ├── validate.yml                    <- copy hub/validate.yml here
    └── publish.yml                     <- copy hub/publish.yml here
```

## Why a repo and not a service

A hosted evaluation service has to run other people's components on our
hardware: queueing, sandboxing, cost, abuse, and an on-call rota. None of that
makes a number more trustworthy. What makes it trustworthy is knowing what
produced it — which is what the bundle carries, and which a text file in git
preserves perfectly.

## The validator is the product

The leaderboard is a rendering of validated bundles. The validation is the part
that gives a row meaning:

```bash
pip install "lasagna-rag[eval]"
lasagna results validate runs/2026/some-lab/my-reranker-a1b2c3.json
```

It checks the schema, requires complete provenance, and **recomputes both the
comparability key and the run id from the recorded fields**. A bundle whose
numbers were edited by hand no longer matches its own id; one whose
`held_fixed` was edited to fake comparability no longer matches its key. That
catches mistakes and naive edits; it is not a signature, since anyone who edits a number can recompute the id.

Reviewers read provenance. They never rerun anything.

## Leaderboards group by comparability

```bash
lasagna results leaderboard runs --metric ndcg@10 --output leaderboards
```

Rows are grouped by `(benchmark, comparability_key)`. Everything inside a group
held the same things fixed, so the numbers are commensurable by construction;
each group renders as its own table with that configuration in the header.
A row that records no stage settings (`fixed_stages`) shares a table only with rows from its own bundle.

This is the difference between a leaderboard and a list. A single table of
everyone's best number, with different embedding models and sample counts
behind it, is the usual way these become misleading.

## Trust model, stated plainly

**Self-reported and machine-validated, not independently reproduced.** Every row
links to its bundle. Nobody is claiming to have rerun anyone's experiment.

### Reproduction by collision

A row is marked `verified` when a **different** submitter publishes a bundle
with the same comparability key whose mean lands inside the row's mean ± std (or equals the mean, when its runs show no spread).
No coordination, no extra compute, no trusted third party — it falls out of data
the hub already has, and it is a real reproduction signal rather than a badge.

Two deliberate restrictions: a submitter cannot verify their own row, or the
badge is earned by publishing twice; and a single-run row can never be verified,
because it reports no spread for anything to land inside.

## Setting it up

1. Create the repository.
2. Copy `run-1.0.schema.json` to `schema/run-1.0.json`, and `validate.yml` and
   `publish.yml` into `.github/workflows/`. Regenerate the schema with
   `lasagna results schema` rather than editing it — it is built from the same
   constants the bundler uses, so a hand-edited copy can require a different set
   of fields from the one the comparability key is computed over.
3. Replace the placeholder org in this repo's `[project.urls]` and in
   `CITATION.cff` with the real one.
4. Publish the first bundle yourself, through the PR flow, before inviting
   anyone. A submission process nobody has walked end to end will have a step
   that does not work.
