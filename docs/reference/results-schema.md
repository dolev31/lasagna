# Results schema

A results bundle is one JSON file that carries a run and its provenance. You make
one with `lasagna results bundle`, and check it with `lasagna results validate`.
The schema is generated from the same constants the bundler uses, so a hub that
validates against it cannot require different fields from the ones Lasagna writes.

Print it, or write it to a file:

```bash
lasagna results schema
lasagna results schema -o run-schema.json
```

The bundle schema version is 1.0. Readers accept 1.0 up to, not including, 2.0.

## Top-level fields

| Field | Required | What it holds |
|---|---|---|
| `schema_version` | yes | The schema version, `"1.0"`. |
| `metrics_version` | yes | The metric definitions. Numbers under different values are not comparable. |
| `run_id` | yes | A hash of the bundle contents. Validation recomputes it, so an edited number no longer matches. |
| `submitter` | no | An object with `name` and `affiliation`. |
| `experiment` | yes | The environment and run-level facts. See below. |
| `configs` | yes | One entry per benchmark and arm. See below. |
| `results` | yes | One entry per benchmark and arm with its metrics and per-query scores. |
| `integrity` | yes | `warnings` and `blocking`, both lists of strings. |

## `experiment`

| Field | What it holds |
|---|---|
| `environment` | Required: `lasagna_version`, `python` and `platform`. It has no paths, host names or credential names, because bundles are published. |
| `components` | The paper and claimed metrics of each component the run named, keyed by the name the run used. |
| `nothing_samples` | True when only retrieval ran and no task called an LLM. One run is then the whole answer, and `SINGLE_RUN` is not raised. |
| `determinism` | One of `seeded_per_task`, `seeded_global_concurrent` or `seeded`. Never "reproducible", because LLM judges and GPU kernels defeat that claim. |

## `configs`

Each entry describes one arm on one benchmark.

| Field | What it holds |
|---|---|
| `name` | The arm's name. |
| `benchmark` | The benchmark the entry's key covers. |
| `comparability_key` | A hash of `held_fixed`. Validation recomputes it. |
| `held_fixed` | The configuration held fixed. See below. |
| `varied_dimensions` | What this arm changed: component slots such as `reranker`, or config flags. Excluded from the key. |
| `varied_levels` | The value each varied dimension took, so the cells of a factorial are distinguishable. |

### `held_fixed`

These fields feed the comparability key:

`metrics_version`, `benchmark`, `split`, `n_samples`, `sampling_seed`,
`sample_id_digest`, `corpus_fingerprint`, `corpus_provenance`, `resolved_mode`,
`embedder_model`, `embedder_dim`, `chunker`, `retriever_top_k`, `vector_backend`,
`judge_model`, `bm25_stemming`, `fixed_stages`, `similarity_model` and
`corpus_ingest_gap`.

`corpus_provenance` defaults to `native`. A field a metrics version did not have
yet is not required from older bundles. See
[comparability and provenance](../concepts/comparability.md).

## `results`

Each entry has `benchmark_name`, `config_name` and `metrics`, and usually
`per_sample_scores`: one row per query. A retrieval row carries `_retrieved_ids`,
the ranked list the metrics were computed on, and `_retrieved` with the document
id, score and rank of each result. The `rank` is authoritative. Scores can tie, so
sorting by score does not rebuild the ranking.

## Integrity messages

`blocking` problems make `lasagna results validate` fail. `warnings` do not, unless
you pass `--strict`. The codes you will meet include `SINGLE_RUN`,
`UNRESOLVED_JUDGE_ALIAS` and `PARTIAL_CORPUS`. See
[read and publish results](../guides/results.md).
