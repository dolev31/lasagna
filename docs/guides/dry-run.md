# Dry run and cost estimates

Check what a sweep will do before it spends time or tokens.

## Dry run

```bash
lasagna benchmark experiment sweep.yaml --dry-run              # with a 3-sample probe
lasagna benchmark experiment sweep.yaml --dry-run --no-probe   # nothing runs at all
```

A dry run prints the arms, the tasks, the models the run loads and whether each
is already downloaded, and a token estimate. Unless you pass `--no-probe`, it
runs three samples per benchmark and extrapolates tokens and time to the whole
sweep. Every number is labeled an estimate, and nothing is written under the
results folder.

For the plugin template's sweep it printed:

```text
Arms (2): reranker=cross_encoder, reranker=template_length
Tasks (2): 1 benchmark x 2 arms x 1 run
  smoke  split=test  12 samples
Models:
  embedder           all-MiniLM-L6-v2  cached
  reranker           cross-encoder/ms-marco-MiniLM-L-6-v2  cached
  reranker           component template_length
Tokens: no generation track, so no answer is generated or judged; an arm still calls an LLM if its pipeline does (HyDE, an agentic pipeline, an LLM reranker).
Probe of smoke (estimate): arm 'reranker=cross_encoder' on 3 samples took 2.8 s including setup (loading models, ingesting), 0.12 s per query, 0 LLM calls
  Extrapolated to 24 queries (estimate, every arm assumed to spend what this one did): about 3 s of query time plus setup, 0 system tokens and 0 judge tokens
```

## Dollars, only if you ask

A run records tokens, which are observed. Add `prices:` to the sweep file for an
estimate in dollars:

```yaml
prices: builtin          # the dated table in the package (builtin-2026-10)
# or your own rates, in USD per million tokens, which win over the table:
# prices: {gpt-6-luna: {input_per_1m: 0.10, output_per_1m: 0.50}}
```

Each task whose tokens all have a rate gets `cost_usd_estimate`, and the run
metadata records the table and the rates used. A rate that holds only for
shorter prompts gives no estimate for a task whose calls average longer. With no
`prices:`, a run reports tokens only.

## Next

The [LLM response cache](llm-cache.md) keeps a second run from paying again.
