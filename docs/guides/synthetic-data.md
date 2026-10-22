# Make a development set from your documents

Use this when you have documents but no questions. Lasagna asks an LLM to write
questions from your files, checks each one, and saves them in the
[JSONL format](../own-data.md#jsonl) that `benchmark run` and `benchmark experiment` read.

!!! warning "For development, not for claims"
    Scores on a synthetic set read high. See
    [Synthetic sets are for development, not for claims](../own-data.md#synthetic-sets-are-for-development-not-for-claims).

## Plan first

```bash
lasagna benchmark synthesize ./docs --n 200 --out my_bench.jsonl \
    --mix factoid=0.6,multihop=0.3,unanswerable=0.1 --seed 42 --dry-run
```

`--dry-run` prints the plan (documents, passages, questions of each type, LLM
calls) and a token estimate. It calls no LLM and writes nothing.

## Generate

Drop `--dry-run` and set an API key such as `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`. Use `MODEL_NAME` or `--model` to choose the model.

The command reads every supported file under the folder (`.md` and `.txt` as
plain text, PDF, DOCX and HTML through the same extractor as an upload). It cuts
each file into 200-word passages with the `fixed` chunker, then asks for three
kinds of question:

| Type | What it asks |
|---|---|
| `factoid` | A short fact found in one passage. |
| `multihop` | A fact from each of two passages in two different documents. |
| `unanswerable` | A question on the subject of a passage that the passage does not answer. |

A second call checks each question. Rows that fail are dropped and counted, so
the file can hold fewer than `--n` rows. The summary says how many and why.

The same folder, seed and model give the same plan and the same prompts.

## Score one question type

Each row records how it was made, including `question_type`. To score one type,
filter on it in the benchmark entry of a sweep:

```yaml
name: Synthetic multihop
benchmarks:
  - path: my_bench.jsonl
    filters: {question_type: multihop}
pipeline:
  base_profile: speed
  rerankers: [cross_encoder, mmr]
n_runs: 1
```

## From Python

```python
from lasagna.knowledge_evaluation_sdk.synthesize import synthesize, write_jsonl

rows = synthesize("docs", n=200, mix="factoid=0.6,multihop=0.3,unanswerable=0.1", seed=42)
write_jsonl(rows, "my_bench.jsonl")
```

## Next

- [Your own data](../own-data.md) covers the formats and what each can measure.
- [Compare a component](compare-a-component.md) runs a sweep over the new file.
