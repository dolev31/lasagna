# Your own data

Run a sweep on your own questions with no adapter to write: a JSONL or CSV file,
or a folder in the BEIR layout.

```bash
lasagna benchmark validate-data my_data.jsonl              # check it, see what it supports
lasagna benchmark run my_data.jsonl --profile speed --tracks retrieval
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
