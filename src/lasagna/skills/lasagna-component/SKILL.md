---
name: lasagna-component
description: Use when implementing, validating, benchmarking or publishing a Lasagna pipeline component: a reranker, chunker, embedder, query enhancer, retriever, document processor, agentic pipeline or generator. Covers the contract, the conformance check, and how to prove the component actually helps.
---

# Build a Lasagna component

Eight stages are pluggable. Implement one abstract base class and the pipeline
will use your component in place of the built-in, with everything else held
fixed, which is what makes a measured difference attributable to your change.

| Stage | Base class |
|---|---|
| Reranker | `BaseReranker` |
| Chunker | `BaseChunker` |
| Embedder | `BaseEmbedder` |
| Query enhancer | `BaseQueryEnhancer` |
| Retriever | `BaseRetriever` |
| Document processor | `BaseDocumentProcessor` |
| Agentic pipeline | `BaseAgenticPipeline` |
| Generator | `BaseGenerator` |

Every stage but the agentic pipeline also takes a plain function with one
decorator (`@lasagna.reranker`, `@lasagna.generator` and so on), and a sweep can
name that function as `module:function`.

## 1. Implement

```python
import copy
from lasagna.core.base_components import BaseReranker
from lasagna.core.registry import registry


@registry.register("reranker", "my_reranker", paper="Someone et al. 2026",
                   claims={"ndcg@10": 0.42})
class MyReranker(BaseReranker):
    def rerank(self, query, documents, top_k=5, **kwargs):
        out = [copy.deepcopy(d) for d in documents]   # never mutate the input
        out.sort(key=lambda d: my_score(query, d), reverse=True)
        return out[:top_k]
```

Two things that are easy to get wrong and hard to notice:

- **Deep-copy before touching anything.** The pipeline may hand the same objects
  to more than one component. A stage that mutates its input makes results
  depend on the order stages ran in.
- **Return at most `top_k`.** The caller decides how many results it wants.

`paper` and `claims` are optional but carried through to the listing API, so a
results table can show the claimed number beside the reproduced one.

## 2. Validate the contract

```bash
lasagna component validate my_package.module:MyReranker
```

This is the difference between "registered" and "interchangeable". Registration
alone does not mean the class satisfies its ABC: a non-conforming component is
listed with `implements_contract: False`, which is how you find out **before**
your results depend on it.

## 3. Prove it helps

Conforming is not the same as helping. Measure against a baseline with
everything else fixed:

```bash
lasagna benchmark run smoke --reranker my_reranker --profile speed
lasagna benchmark experiment sweep.yaml        # the comparison, from a sweep file
```

```yaml
# sweep.yaml
name: does my reranker help
benchmarks: [{name: beir/scifact, n_samples: 200}]
tracks: [retrieval]
pipeline:
  base_profile: speed
  rerankers: [cross_encoder, my_reranker]       # a control and yours
n_runs: 1
```

Always include a baseline arm. Without it you cannot distinguish "my component
reordered things" from "my component never ran"; the two look identical from
outside.

`n_runs: 1` is enough here: retrieval with fixed models returns the same lists
on every run, and the significance test pairs the arms query by query. Repeat
runs when an LLM stage is in the loop (an LLM reranker, HyDE, an agentic
pipeline, answer generation), with `n_runs: 3` or more.

**Component overrides only apply in live mode.** `generation_only` never
constructs the pipeline, so passing a component there is a hard error rather
than a plausible number scored from gold context.

## 4. Publish so others can check

```bash
lasagna results bundle results.json -o run.json
lasagna results validate run.json
```

The bundle carries versions, hardware, resolved sweep, corpus fingerprint and a
comparability key. `validate` recomputes the key and the run id, so an edited
number no longer matches its own bundle.

`--strict` also fails on the warnings, which do not block a local run:
`SINGLE_RUN` (one run reports no spread, so the results hub can never mark the
row verified), `UNRESOLVED_JUDGE_ALIAS` (a floating judge alias points at
different weights over time) and `UNPINNED_DATASET`, among others.

## 5. Ship it as a package

Copy `examples/lasagna-plugin-template/`: an installable package registering
one reranker and one benchmark through entry points, with tests.

```toml
[project.entry-points."lasagna_components"]
reranker.my_reranker = "my_pkg:MyReranker"

[project.entry-points.lasagna_benchmarks]
my_dataset = "my_pkg:MyBenchmarkAdapter"
```

Verify discovery actually found it: a broken entry point is logged and skipped,
so the symptom is your component silently not being used:

```bash
lasagna plugins list
```

## Rules

- Run `lasagna component validate` before benchmarking. Benchmarking a
  non-conforming component produces numbers that mean nothing.
- Never compare across different `n_samples`, splits, or embedding models.
  `lasagna results compare` will tell you when two runs are not comparable, and
  which field differs.
- Check the model license before publishing a number. Several strong rerankers
  are non-commercial; see `docs/model-licenses.md`.
