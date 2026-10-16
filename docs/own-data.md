# Your own data

Run a sweep on your own questions with no adapter to write: a JSONL or CSV file,
or a folder in the BEIR layout. You can also have an LLM write a development set
from your documents (see [Synthetic questions](#synthetic-questions-from-your-documents)).

```bash
lasagna benchmark validate-data my_data.jsonl              # check it, see what it supports
lasagna benchmark run my_data.jsonl --profile speed            # retrieval, keyless
```

In a sweep, name the file with `path:` instead of `name:`. A relative path is
read from the sweep file's folder:

```yaml
benchmarks:
  - {path: data/my_data.jsonl, n_samples: 50}
  - {path: data/support_tickets.csv, name: tickets}   # optional display name
  - {path: data/my_beir}                              # a BEIR folder
```

## Formats

### JSONL

One JSON object per line. Only `question` is required.

| Field | Type | Meaning |
|---|---|---|
| `question` | string | The question. |
| `answer` or `answers` | string, or list of strings | The reference answer. With a list, exact match and token F1 accept any of them; the first is the reference the judge sees. |
| `contexts` or `context` | list of strings, or one string | The passages that answer the question. |
| `id` | string or number | The question's id. Defaults to `row<line number>`. Must be unique. |
| `expected_docs` | list of strings | Names for the passages in `contexts`, in the same order: `contexts[i]` is the text of `expected_docs[i]`. Use it when several rows cite the same document by name. |
| `metadata` | object | Anything else. `metadata.unanswerable: true` marks a question with no answer; a row with it has no `answer` (or `"answer": ""`). Other keys can be used as `filters:` on the benchmark entry. |

```jsonl
{"id": "q1", "question": "What does the soda add to soda-lime glass?", "answer": "It lowers the melting point of the silica.", "contexts": ["Soda-lime glass is made by melting silica sand with sodium carbonate and calcium oxide. The soda lowers the melting point of the silica from about 1700 to roughly 1000 degrees Celsius."]}
{"id": "q2", "question": "Who introduced the solidus?", "answers": ["Constantine the Great", "Constantine I"], "context": "The solidus was a gold coin introduced by Constantine the Great in roughly 312 AD."}
{"id": "q3", "question": "What did the solidus cost to mint?", "contexts": ["The solidus was a gold coin introduced by Constantine the Great in roughly 312 AD."], "metadata": {"unanswerable": true}}
```

Any other field is an error that names the field, so a typo such as `anwser` is
caught instead of silently leaving every row without an answer. Put fields of
your own under `metadata`.

### CSV

A header row is required. The columns are `question` (required), `answer`,
`contexts` and `id`. Separate several passages in one `contexts` cell with `|||`.
CSV cannot mark a question unanswerable; use JSONL for that.

```csv
id,question,answer,contexts
q1,Who introduced the solidus?,Constantine the Great,The solidus was a gold coin introduced by Constantine the Great in roughly 312 AD.
q2,What replaced the debased solidus?,The hyperpyron,The hyperpyron replaced the debased solidus after the monetary reform of Alexios I Komnenos in 1092.|||Debasement of the solidus began in earnest under Nikephoros II.
```

### BEIR folder

The layout of BEIR's own downloads:

```
my_beir/
  corpus.jsonl      {"_id": "d1", "title": "...", "text": "..."}
  queries.jsonl     {"_id": "q1", "text": "..."}
  qrels/test.tsv    query-id <tab> corpus-id <tab> score   (header optional)
```

Each `qrels/<split>.tsv` is a split, chosen with `split:`. A score above 0 marks a
document relevant; graded scores feed NDCG. A judgment that names a document or
query missing from the folder is an error, with the count and the first few ids.

## What each format can measure

| Data | Retrieval | Generation |
|---|---|---|
| BEIR folder | yes, against its corpus | no (no answers) |
| JSONL or CSV with `contexts` and answers | yes, against every row's contexts pooled into one corpus | yes |
| JSONL or CSV with answers and no `contexts` | no (nothing to retrieve) | closed-book only, with the `closed_book` generator |

Retrieval on a file with `contexts` searches a pool of every row's passages:
each question has to find its own among all the others. That pool holds only
gold passages, with no hard negatives, so it is an easier search than a real
corpus of the same size. Results are labeled `corpus_provenance: pooled_gold`
and never share a leaderboard group with results on a real corpus. A passage
that several rows share is one document, relevant to each of them. The
passages of an unanswerable row join the pool but are relevant to nothing.

A run that asks for a track the data cannot support is refused before any model
loads, with the reason and the fix.

## Names, checkpoints and comparisons

A file registers as `local/<file name>@<12 hex digits>`, the start of the
sha256 of its bytes (for a BEIR folder, of the files read from it), or as
`<name>@<12 hex digits>` when the entry gives `name:`. The full digest is
recorded as the dataset revision. Because the hash is in the benchmark name:

- an edited file is a different benchmark, so `--resume` never restores a result
  computed on the old version and the old corpus is never reused;
- two files with different content never share a leaderboard group or a
  comparability key, even under the same file name;
- the same bytes under the same name are the same benchmark, on any machine.

`lasagna benchmark validate-data` prints the name a file will run under.

## Synthetic questions from your documents

```bash
lasagna benchmark synthesize ./docs --n 200 --out my_bench.jsonl \
    --mix factoid=0.6,multihop=0.3,unanswerable=0.1 --seed 42 --dry-run
```

This reads every supported file under `./docs` (`.md` and `.txt` as plain text,
PDF, DOCX and HTML through the same extractor as an upload), cuts each into
200-word passages with the `fixed` chunker, and asks an LLM for three kinds of
question:

- `factoid`: answered by a short fact in one passage;
- `multihop`: needs a fact from each of two passages in two different
  documents (the second passage is the one sharing the most words with the
  first);
- `unanswerable`: on the subject of a passage, but not answered by it.

A second call checks each question. An answer must be supported by its source
passages; an unanswerable question must not be answered by its passage or by the
three passages sharing the most words with it. Rows that fail are dropped and
counted, so the file can hold fewer than `--n` rows; the summary says how many
and why.

`--dry-run` prints the plan (documents, passages, questions of each type, LLM
calls) and a token estimate, calls no LLM and writes nothing. Without
`--dry-run` an LLM is needed: set an API key variable such as `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` (without one, the error lists every variable read), and
`MODEL_NAME` or `--model` to choose the model. Calls ask for temperature 0
(a model that takes no temperature is sent none; see
[Configuration](configuration.md#temperature-and-repeatable-runs)). The same
folder, seed and model give the same plan and the same prompts.

The output is the JSONL format above. Each row records how it was made:

```json
{"id": "synth-0001", "question": "...", "answer": "...", "contexts": ["..."],
 "metadata": {"question_type": "factoid", "source_doc_ids": ["guide.md"],
              "prompt_id": "factoid-v1", "prompt_sha256": "...", "model": "...",
              "seed": 42, "synthetic": true, "unanswerable": false}}
```

`filters: {question_type: multihop}` on the benchmark entry scores one type.

From Python:

```python
from lasagna.knowledge_evaluation_sdk.synthesize import synthesize, write_jsonl

rows = synthesize("docs", n=200, mix="factoid=0.6,multihop=0.3,unanswerable=0.1", seed=42)
write_jsonl(rows, "my_bench.jsonl")
```

### Synthetic sets are for development, not for claims

A question an LLM wrote from a passage tends to reuse that passage's words and
the model's own phrasing. A retriever finds such questions easier than the ones
your users ask, and a pipeline built on the same family of models finds them
easier still, so scores on a synthetic set read high, and favor the systems
most like the model that wrote it. The answerability check is made by that same
model and inherits its mistakes.

Use a synthetic set to compare configurations on your own documents while you
build. Report results on human-written questions: a public benchmark, or a set
your users or annotators wrote.
