# Contributing a component

How a researcher gets from an idea to a published, checkable result — and what
makes each step auditable rather than just easy.

The design goal is narrow: **it should be hard to produce a number you cannot
defend.** Everything below follows from that.

---

## The path, end to end

Five steps. Each is one command, and each is a checkpoint that can fail.

| # | Step | Command | What it rules out |
|---|---|---|---|
| 1 | Write it | — | — |
| 2 | Check the contract | `lasagna component validate mod:Class` | it does not satisfy the ABC |
| 3 | Check it runs | `lasagna benchmark run smoke` | it breaks inside the pipeline |
| 4 | Check it helps | `lasagna benchmark experiment sweep.yaml` | the gain is noise, or was never measured |
| 5 | Publish | `lasagna results bundle` → `validate` | nobody can tell what produced the number |

The ordering is the design. Every step catches a failure the next one would
otherwise disguise as a result.

---

## 1. Write it

One method, for six of the seven stages.

```python
import copy
from lasagna.core.base_components import BaseReranker
from lasagna.core.registry import registry


@registry.register("reranker", "my_reranker",
                   paper="Someone et al. 2026", claims={"ndcg@10": 0.42})
class MyReranker(BaseReranker):
    def rerank(self, query, documents, top_k=5, **kwargs):
        out = [copy.deepcopy(d) for d in documents]
        out.sort(key=lambda d: self.score(query, d), reverse=True)
        return out[:top_k]
```

`paper` and `claims` are optional and worth filling in: they come back out of
the registry, so a results table can show the claimed number beside the one you
reproduced. That comparison is the single most useful thing a reader of your
result can see, and nothing else in the ecosystem records it.

**Deep-copy first.** The pipeline may hand the same objects to several stages.
A component that edits them in place makes results depend on the order stages
ran in — which destroys exactly the attribution this framework exists to
provide.

---

## 2. Check the contract

```bash
lasagna component validate my_package.reranker:MyReranker
```

Registration and conformance are different things, and the difference is
load-bearing. A component can register successfully and still not satisfy its
ABC; it is then listed with `implements_contract: False`, and comparing against
it measures the interface difference along with the algorithm.

```python
[r["name"] for r in registry.list_rerankers() if r["implements_contract"]]
```

That list is the set of *fair* baselines. Use it.

### What validation does not prove

It exercises the component in isolation. That is deliberate — it is the fastest
possible check — but it means a green result does not mean the pipeline can
call your component. It could not, until recently: the orchestrator invoked
rerankers with an internal signature and a documented `BaseReranker` raised
`TypeError` the first time it ran for real, four green checks into the process.

Which is why step 3 exists and is not optional.

---

## 3. Check it runs

```bash
lasagna benchmark run smoke
```

24 documents, 12 queries, held inside the package. No dataset download, no API
key. It runs the real pipeline end to end, so a green run means your component
is genuinely wired in.

It is deliberately easy. The score means "this works", never "this is good".

---

## 4. Check it helps

This is the step the whole exercise is for, and the one where numbers most often
go wrong.

```yaml
# sweep.yaml
name: does my reranker help
benchmarks:
  - name: beir/scifact
    n_samples: 200
pipeline:
  rerankers: [cross_encoder, my_reranker]   # a control and your component
n_runs: 3
```

```bash
lasagna benchmark experiment sweep.yaml
```

Three things this enforces that a hand-rolled comparison does not:

**A control arm.** Without one you cannot distinguish "my component reordered
things" from "my component never ran". The two look identical from outside.

**Everything else held fixed.** Same chunker, same embedder, same top-k, same
split, same sample. The difference is attributable because nothing else moved.

**More than one run.** Retrieval metrics move between runs. `n_runs: 1` cannot
separate a real gap from noise, and per-sample scores are carried through and
paired by sample identity so significance testing is actually possible.

> **Component overrides only apply in live mode.** `generation_only` never
> constructs the pipeline, so passing a component there is a hard error rather
> than a plausible number scored against gold context. `mode: auto` picks
> correctly and logs which it chose and why.

---

## 5. Publish

```bash
lasagna results bundle results.json -o run.json
lasagna results validate run.json --strict
```

The bundle carries library versions, hardware, the resolved sweep, the corpus
fingerprint, the sample-id digest, and a **comparability key** — a hash over
everything held fixed, deliberately *excluding* the dimension under test.

That exclusion is the whole design. A key covering everything would make every
row incomparable with every other; a key covering nothing would make all rows
comparable regardless of what they measured.

Submit to [the results hub](../hub/README.md) as a pull request adding one file.

---

## What makes this auditable

Four properties, each mechanical rather than social.

**The number cannot be edited.** `run_id` hashes the bundle contents and the
comparability key hashes the held-fixed block. `lasagna results validate`
recomputes both, so a hand-edited metric no longer matches its own bundle, and a
`held_fixed` edited to fake comparability no longer matches its key. A reviewer
can accept a submission by reading provenance rather than rerunning it.

**Incomparable results cannot be silently ranked together.** Leaderboards group
by `(benchmark, comparability_key)`. Rows in one table held the same
configuration fixed; rows in different tables are never ranked against each
other. A single table of everyone's best number, with different embedding models
and sample counts behind it, is the usual way these mislead.

**Metric drift is caught.** `metrics_version` is inside the key, so redefining
`ndcg@10` drops old numbers out of comparison instead of quietly mixing them
with new ones. Nothing else catches this.

**Reproduction is a signal, not a badge.** A row is marked `verified` only when
a *different* submitter publishes a bundle with the same key whose mean lands
inside the reported interval. You cannot verify your own row, and a single-run
row can never be verified because it reports no spread to land inside.

### What it is not

**Self-reported and machine-validated, not independently reproduced.** Nobody is
claiming to have rerun anyone's experiment. Stating that plainly is what makes
the rest of the guarantees worth anything.

---

## Shipping it as a package

Copy [`examples/lasagna-plugin-template/`](../examples/lasagna-plugin-template/)
— a real installable package registering one reranker and one benchmark through
entry points, with tests. CI installs and runs it, so it cannot rot into a
snippet that no longer works.

```toml
[project.entry-points."lasagna_components"]
reranker.my_reranker = "my_pkg:MyReranker"

[project.entry-points.lasagna_benchmarks]
my_dataset = "my_pkg:MyBenchmarkAdapter"
```

```bash
lasagna plugins list
```

Run it. Discovery logs failures and continues, so a broken entry point does not
raise — your component is simply never used, and the results quietly come from
the built-in default. `lasagna plugins list` and `lasagna doctor` both surface
the failure with its exception.

---

## Known friction

Stated rather than hidden, because you will hit these.

- **`agentic_pipeline` is registrable but `lasagna component validate` rejects
  it.** The type is real and documented; the validator has no case for it yet.
- **Only BEIR benchmarks support live mode.** 18 of 31 cannot measure retrieval.
  `lasagna benchmark list` marks which can; the rest score generation against
  gold context.
- **`lasagna component validate` does not check for input mutation**, despite
  that being the property most likely to silently corrupt a measurement. Copy
  defensively; the plugin template's tests cover it.
- **`judge_model` is `None` in the comparability key for retrieval-only runs.**
  That is accurate rather than a gap — there is no judge — but it means two runs
  scored by *different* judges group together if neither recorded one. Pin a
  dated judge snapshot whenever you report a generation metric.

---

## Getting help

`lasagna doctor` first — it answers most questions before they are asked. Then
[a bug report](../.github/ISSUE_TEMPLATE/bug_report.yml), or
[a component contribution](../.github/ISSUE_TEMPLATE/component_contribution.yml)
if you want yours listed. If a published number does not reproduce, open a
[result dispute](../.github/ISSUE_TEMPLATE/result_dispute.yml) — run
`lasagna results compare` first, since the field-level diff usually settles it.

Agents working in a repo that uses Lasagna can run `lasagna skills install` to
get the `lasagna-component` and `lasagna-support` skills, which encode this
document as executable steps.
