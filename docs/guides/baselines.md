# Baselines

"Does retrieval help at all?" and "would a long context window do as well?" are
answered by arms in a sweep, not by a separate tool.

```yaml
name: is retrieval worth it
benchmarks:
  - name: hotpotqa
    n_samples: 100
tracks: [generation]
pipeline:
  base_profile: speed
  config_overrides:
    - {name: rag}                                # the pipeline as configured
    - {name: no_retrieval, no_rag: true}         # answers from the question alone
    - {name: long_context, generator: long_context}
n_runs: 3
```

This sweep needs an LLM API key and the `eval` extra.

## The three arms

- **`rag`** is the pipeline as configured.
- **`no_rag: true`** retrieves nothing. The arm's retrieval track is skipped and
  it answers with the `closed_book` generator, which sees the question alone.
  The arm is scored on generation only, so a sweep with `tracks: [retrieval]`
  refuses it before anything runs.
- **`long_context`** reads all ten retrieved passages in rank order, where the
  default reads five, and the judge sees the same ten.

`closed_book` is also a generator on its own. Name it under `generators:` if you
want it as an arm without the `no_rag` flag.

## Arms with no passages

An arm whose generator reads no passages is not scored on faithfulness or the
context metrics, because those would be meaningless with no context. The report
and the leaderboard say so and point at exact match, token F1 and factual
correctness instead.

The stage isolation check files a `no_rag` arm under the retriever stage, so a
run that varies only that flag passes the check.

## Next

[Compare a component](compare-a-component.md) covers the rest of the sweep file.
