# Write a component

A component fills one stage of the pipeline. Lasagna has eight stages, and you
can write a component for each one. The path has five steps, and each one is a
checkpoint that can fail:

| # | Step | Command |
|---|---|---|
| 1 | Write it | one decorator on a function |
| 2 | Check the contract | `lasagna component validate my_ideas:keyword_overlap` |
| 3 | Check it runs | `lasagna benchmark run smoke --reranker my_ideas:keyword_overlap --profile speed` |
| 4 | Check it helps | `lasagna benchmark experiment sweep.yaml` |
| 5 | Publish | `lasagna results bundle`, then `validate` |

[Contributing a component](../contributing-components.md) explains each step and
what it rules out. This page is the short version for writing the code.

## A function per stage

Most techniques fit in one function. The decorator `@lasagna.<stage>` wraps the
function in the stage's interface and registers it. The wrapper hands your
function copies of what it receives, so it cannot change what another stage
sees.

| Stage | Decorator | Function |
|---|---|---|
| reranker | `@lasagna.reranker` | `f(query, text) -> float`, or `f(query, documents, top_k) -> documents` |
| query enhancer | `@lasagna.query_enhancer` | `f(query) -> list[str]` |
| chunker | `@lasagna.chunker` | `f(text) -> list[str]` |
| embedder | `@lasagna.embedder` | `f(texts) -> one vector per text` |
| retriever | `@lasagna.retriever` | `f(query, top_k) -> list[str]` or documents |
| document processor | `@lasagna.document_processor` | `f(file_path) -> str` |
| generator | `@lasagna.generator` | `f(query, contexts) -> str` |

```python
import lasagna

@lasagna.reranker(
    "keyword_overlap", paper="Someone et al. 2026", claims={"beir/scifact": {"ndcg@10": 0.42}}
)
def keyword_overlap(query, text):
    """Count the query words that appear in the passage."""
    return len(set(query.lower().split()) & set(text.lower().split()))
```

The agentic pipeline is the one stage with no one-line form, because a loop is a
class.

## A class when you have state

A technique with state or a model is a class. Subclass the stage's interface
(`BaseReranker`, `BaseAgenticPipeline` and so on from
`lasagna.core.base_components`), register it with
`@registry.register("reranker", "name")`, and copy your inputs before you change
them.

Registration and conformance are different things. A class can register and
still not satisfy its interface. It is then listed with
`implements_contract: False`, and comparing against it measures the interface
difference along with the algorithm. This lists the fair baselines:

```python
from lasagna.core.registry import registry

[r["name"] for r in registry.list_rerankers() if r["implements_contract"]]
```

## Record the paper and the claim

`paper` and `claims` are optional, and they are worth filling in. They come back
out of the registry, and a run records them. On a run of the benchmark a claim
names, the leaderboards and `lasagna results view` show the claimed number beside
the one you reproduced.

A claim is keyed by benchmark, `{"beir/scifact": {"ndcg@10": 0.42}}`, with
`"split": "test"` inside if the paper names a split. Three rules apply:

- A claim that names no benchmark is refused, because it would print beside a run
  of any benchmark.
- A bad metric name is refused when you register.
- An unknown benchmark is a warning that names the closest one.

## Check the contract

```bash
lasagna component validate my_ideas:keyword_overlap
```

The command imports from the working directory, as a sweep does. It checks the
return types, `top_k`, empty input and that the component does not change its
input. A plain function with no decorator needs `--stage <stage>`. A class is
named `my_package.reranker:MyReranker`.

A green result does not mean the pipeline can call your component, because the
check runs the component alone. Step 3, `lasagna benchmark run smoke`, runs it
inside the pipeline.

## Share it as a package

To publish a component as an installable package, start from the
[plugin template](https://github.com/dolev31/lasagna/tree/main/examples/lasagna-plugin-template).
`lasagna plugins list` shows installed plugins and names any that failed to load.
For a new kind of component, not just a new implementation, see
[Extending](https://github.com/dolev31/lasagna/blob/main/src/lasagna/sdk/docs/extending.md).

## Next

[Compare a component](compare-a-component.md) turns the component into a result.
