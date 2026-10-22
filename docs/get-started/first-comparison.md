# Your first comparison

You will write a reranker in six lines, check it, and test it against a
cross-encoder inside a full pipeline. The first part needs no model and no
corpus. The sweep needs no API key.

## 1. Write the technique

Put this in `my_ideas.py`. The decorator wraps your function in the reranker
interface and registers it. `paper` and `claims` are optional. They are
recorded with every run that uses the component.

```python
# my_ideas.py
import lasagna

@lasagna.reranker(paper="Doe et al. 2026", claims={"beir/scifact": {"ndcg@10": 0.456}})
def keyword_overlap(query, text):  # your technique: score one passage
    return len(set(query.lower().split()) & set(text.lower().split()))

if __name__ == "__main__":  # try it on three passages
    from lasagna.core.base_components import ScoredDocument
    from lasagna.core.registry import registry

    candidates = [
        ScoredDocument(content="short", score=0.9, document_id="d1", chunk_id="c1"),
        ScoredDocument(content="a considerably longer passage of text", score=0.5,
                       document_id="d2", chunk_id="c2"),
        ScoredDocument(content="middling length here", score=0.7,
                       document_id="d3", chunk_id="c3"),
    ]

    mine = registry.get("reranker", "keyword_overlap").rerank("length of a passage", candidates)
    entry = next(e for e in registry.list_rerankers() if e["name"] == "keyword_overlap")
    print("retrieval order:", [d.document_id for d in candidates])
    print("my reranker:    ", [d.document_id for d in mine])
    print("claims recorded:", entry["claims"])
    print("interchangeable:", entry["implements_contract"])
```

```bash
python my_ideas.py
```

```text
retrieval order: ['d1', 'd2', 'd3']
my reranker:     ['d2', 'd3', 'd1']
claims recorded: {'beir/scifact': {'ndcg@10': 0.456}}
interchangeable: True
```

`interchangeable: True` means the component satisfies its stage's interface, so
a comparison against a built-in measures your idea and not an interface
difference.

## 2. Test it inside a full pipeline

Save this as `sweep.yaml`. One arm is a built-in cross-encoder, the control.
The other is yours. Every other stage stays fixed.

```yaml
name: does keyword_overlap help
benchmarks: [{name: smoke}]
tracks: [retrieval]
pipeline:
  base_profile: speed
  rerankers: [cross_encoder, "my_ideas:keyword_overlap"]
n_runs: 1
```

```bash
lasagna benchmark experiment sweep.yaml
```

The first run downloads 183 MB once (a 92 MB embedder and a 92 MB
cross-encoder). Look at the failed tasks before the scores. A task that failed
is listed there and not in the table.

## 3. Ask whether the difference is real

```bash
lasagna results significance "<output>/results.json" -m ndcg@10 --all
```

Replace `<output>` with the folder the run printed. The command tests every pair
of arms with a paired sign-flip test, corrects the p-values with Holm, and ends
with a box that says what the table supports. For a real run on 100 SciFact
queries, see [step 4 of Contributing a component](../contributing-components.md#4-check-it-helps).
To switch from the 12-query `smoke` benchmark to a real one, name
`beir/scifact` and set `n_samples`.

## Where to go next

- [Write a component](../guides/write-a-component.md) covers all eight stages.
- [Compare a component](../guides/compare-a-component.md) covers sweeps for
  every stage and how to read the result.
- [Your own data](../own-data.md) runs the same comparison on your documents.
