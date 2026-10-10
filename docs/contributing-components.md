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
| 1 | Write it | one decorator on a function | — |
| 2 | Check the contract | `lasagna component validate my_ideas:keyword_overlap` | it does not satisfy its stage's contract (types, `top_k`, empty input, no mutation of its input) |
| 3 | Check it runs | `lasagna benchmark run smoke --reranker my_ideas:keyword_overlap ...` | it breaks inside the pipeline |
| 4 | Check it helps | `lasagna benchmark experiment sweep.yaml` | the gain is noise, or was never measured |
| 5 | Publish | `lasagna results bundle` → `validate` | nobody can tell what produced the number |

The ordering is the design. Every step catches a failure the next one would
otherwise disguise as a result.

---

## 1. Write it

One decorator on a plain function. Put this in `my_ideas.py`:

```python
import lasagna


@lasagna.reranker("keyword_overlap", paper="Someone et al. 2026", claims={"ndcg@10": 0.42})
def keyword_overlap(query, text):
    """Count the query words that appear in the passage."""
    return len(set(query.lower().split()) & set(text.lower().split()))
```

The decorator wraps the function in the stage's interface (`BaseReranker`) and
registers it. The wrapper hands the function copies of what it receives, so it
cannot change what another stage sees. Each stage takes the natural signature:

| Stage | Function |
|---|---|
| reranker | `f(query, text) -> float`, or `f(query, documents, top_k) -> documents` |
| query_enhancer | `f(query) -> list[str]` |
| chunker | `f(text) -> list[str]` |
| embedder | `f(texts) -> one vector per text` |
| retriever | `f(query, top_k) -> list[str]` or documents |
| document_processor | `f(file_path) -> str` |
| generator | `f(query, contexts) -> str` (the answer) |

The agentic pipeline is the one stage with no one-line form: a loop is a class
(below).

`paper` and `claims` are optional and worth filling in: they come back out of
the registry, so a results table can show the claimed number beside the one you
reproduced. That comparison is the single most useful thing a reader of your
result can see, and nothing else in the ecosystem records it.

A technique with state or a model is a class instead: subclass the stage's
interface (`BaseReranker` and so on), register it with
`@registry.register("reranker", "name")`, and deep-copy its inputs before
changing them.

---

## 2. Check the contract

```bash
lasagna component validate my_ideas:keyword_overlap
```

It imports from the working directory, as a sweep does. A function registered
with `@lasagna.<stage>` is checked as the wrapper a sweep would run; a plain,
undecorated function needs `--stage <stage>`; a class is named the same way,
`my_package.reranker:MyReranker`.

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
lasagna benchmark run smoke --reranker my_ideas:keyword_overlap --profile speed --tracks retrieval
```

24 documents, 12 queries, held inside the package. No dataset download, no API
key. It runs the real pipeline end to end with your component in its stage, so a
green run means your component is genuinely wired in. `my_ideas:keyword_overlap`
is an import path: the CLI imports it from the working directory, so the
function needs no package. Installed plugins are named by their registered name.
`--query-enhancer` and `--generator` name a component of those stages the same
way.

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
tracks: [retrieval]                                 # no LLM or RAGAS needed
pipeline:
  base_profile: speed                               # no LLM key, small models
  rerankers: [cross_encoder, "my_ideas:keyword_overlap"]   # a control and yours
n_runs: 3
```

```bash
lasagna benchmark experiment sweep.yaml
lasagna results significance "<output>/results.json" -m ndcg@10 --all
```

Every listed built-in implements its stage's interface, so `cross_encoder` is a
like-for-like control. Naming a reranker turns reranking on for that arm, even
on `speed`. Before ingesting anything the run checks that each named component
passes its stage's contract, and when it finishes it confirms that the
configuration reaching each stage differed only at the reranker (or says which
other stage moved), and flags your arm as inert if its ranked lists never moved.
The second
command prints each pair with the difference, the effect size and both the raw
and the Holm-adjusted p:

```
ndcg@10 on beir/scifact — 1 pairs, Holm-Bonferroni

A                      B            diff    d_z       p  p (Holm)
reranker=cross_encoder  reranker=my_ideas:keyword_overlap  ...
```

Three things this enforces that a hand-rolled comparison does not:

**A control arm.** Without one you cannot distinguish "my component reordered
things" from "my component never ran". The two look identical from outside.

**Everything else held fixed.** Same chunker, same embedder, same top-k, same
split, same sample. The difference is attributable because nothing else moved.

**More than one run.** Retrieval metrics move between runs. `n_runs: 1` cannot
separate a real gap from noise, and per-sample scores are carried through and
paired by sample identity so significance testing is actually possible.

Other stages sweep the same way. `chunkers:`, `embedders:`, `query_enhancers:`
and `generators:` take lists like `rerankers:`. A chunker or embedder arm gets a
corpus of its own, chunked or embedded by its component (an embedder also gets
its own collection), so arms never share an index. A retriever or an agentic
pipeline is named in a `config_overrides` entry, such as
`{name: bm25, retriever: bm25}`, with `dense` as the retriever control. Document
processors do not sweep, since benchmark corpora are loaded as text.
[The guide, section 8](guide.md#sweep-the-other-stages) has a retriever sweep
and a generator sweep.

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

**A naive edit is caught.** `run_id` hashes the bundle contents and the
comparability key hashes the held-fixed block. `lasagna results validate`
recomputes both, so a hand-edited metric no longer matches its own bundle, and a
`held_fixed` edited to fake comparability no longer matches its key. It is not a
signature: anyone who edits a number can recompute the id, so rows stay self-reported.

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
- **Not every benchmark has a corpus of its own.** BEIR benchmarks run live
  against their own corpus. SQuAD v2, HotpotQA, MuSiQue and RAGBench also run
  live, against a corpus pooled from the samples' gold passages: an easier task
  than the original, stamped `pooled_gold` (see
  [the guide, section 8](guide.md#8-measure-it-on-a-benchmark)). The rest score
  generation against gold context only.
- **`lasagna component validate` checks input mutation for rerankers (the
  documents), embedders (the list of texts) and generators (the list of
  contexts).** The other stages are handed strings or a file path, which cannot
  be changed in place. A class component that keeps state should still copy what
  it is handed before changing it.
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
