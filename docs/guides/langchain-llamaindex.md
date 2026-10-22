# Use LangChain or LlamaIndex components

`lasagna.from_langchain` and `lasagna.from_llamaindex` register an object you
already have. Lasagna then compares it like any built-in component.

| You have | It becomes |
|---|---|
| A LangChain document compressor, or a LlamaIndex node postprocessor | A reranker |
| Embeddings | An embedder |
| A retriever | A retriever |

The listing describes the entry as `wrapped: langchain.<Class>`.

```python
import lasagna
from langchain_core.embeddings import DeterministicFakeEmbedding
from lasagna.core.registry import registry

name = lasagna.from_langchain(DeterministicFakeEmbedding(size=64))
print(name)  # langchain.DeterministicFakeEmbedding
entry = next(e for e in registry.list_embedders() if e["name"] == name)
print(entry["description"])  # wrapped: langchain.DeterministicFakeEmbedding
```

## Name it in a sweep

From Python, name the returned name like any other component.

A sweep file run with `lasagna benchmark experiment` names the object where you
defined it instead. With `cohere = CohereRerank(top_n=10)` in `my_rerankers.py`,
the arm `"my_rerankers:cohere"` is wrapped the same way:

```yaml
name: Cohere against the cross-encoder
benchmarks:
  - name: beir/scifact
    n_samples: 200
tracks: [retrieval]
pipeline:
  base_profile: speed
  rerankers: [cross_encoder, "my_rerankers:cohere"]
n_runs: 1
```

## Three things to know

- **A wrapped retriever brings its own index.** It searches whatever the object
  was built over, not the corpus Lasagna ingested. For a benchmark, its index
  must hold that benchmark's corpus, with each corpus id in the metadata
  (`doc_id`, or the key you pass as `id_key=`). A document with no id gets a hash
  of its text, which matches no relevance judgment.
- **The object's own limits apply.** A compressor's `top_n` or a retriever's `k`
  caps how many documents come back. Set it at least as high as the pipeline's
  depth, or the arm is scored on shorter lists than the others.
- **LlamaIndex is an extra.** Install it with `pip install "lasagna-rag[llamaindex]"`.
  LangChain needs nothing more.

## Next

- [Compare a component](compare-a-component.md) runs the sweep.
- [Write a component](write-a-component.md) covers components that are not wrapped.
