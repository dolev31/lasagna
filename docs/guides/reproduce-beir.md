# Reproduce published BEIR numbers

Lasagna's built-in retrievers and reranker reproduce published numbers on BEIR
SciFact and NFCorpus. Both dense models land within 0.002 NDCG@10 of their MTEB
results. BM25 followed by the MiniLM cross-encoder lands within 0.004 of the BEIR
paper. No method was tuned. BM25 uses the k1 and b that BEIR used, and the dense
models and the cross-encoder use their published checkpoints unchanged.

## Results

NDCG@10 on the BEIR test splits: SciFact (300 queries, 5,183 documents) and
NFCorpus (323 queries, 3,633 documents). The difference is Lasagna minus
published.

| Method | Dataset | Published | Source | Lasagna | Difference |
|---|---|---|---|---|---|
| BM25 (k1 0.9, b 0.4) | SciFact | 0.665 | BEIR, Table 2 | 0.664 | -0.001 |
| BM25 (k1 0.9, b 0.4) | NFCorpus | 0.325 | BEIR, Table 2 | 0.317 | -0.008 |
| all-MiniLM-L6-v2 | SciFact | 0.645 | MTEB results | 0.645 | 0.000 |
| all-MiniLM-L6-v2 | NFCorpus | 0.316 | MTEB results | 0.317 | +0.001 |
| bge-small-en-v1.5 | SciFact | 0.713 | MTEB results | 0.713 | 0.000 |
| bge-small-en-v1.5 | NFCorpus | 0.343 | MTEB results | 0.344 | +0.001 |
| BM25 + MiniLM-L6 cross-encoder | SciFact | 0.688 | BEIR, Table 2 | 0.685 | -0.003 |
| BM25 + MiniLM-L6 cross-encoder | NFCorpus | 0.350 | BEIR, Table 2 | 0.351 | +0.001 |

The BM25 rows sit slightly below the BEIR paper because the paper indexed the
title and the text as separate fields, and Lasagna indexes one field. Against the
single-field Anserini runs, BM25 is 0.015 (SciFact) and 0.005 (NFCorpus) below.
The
[reproduction README](https://github.com/dolev31/lasagna/blob/main/examples/reproduction/README.md)
has the Recall@100 tables, the arms with no published target, and the notes on
each difference.

[results.csv](https://github.com/dolev31/lasagna/blob/main/examples/reproduction/results.csv)
has every measured number to four decimals. It also records the query count and
where the numbers come from: the run date, the package and metrics versions, and
the commit.

## Rerun it

From a clone of the repository:

```bash
pip install "lasagna-rag[eval]"
python -m nltk.downloader punkt_tab
lasagna benchmark experiment examples/reproduction/beir_tier1.yaml
```

To compare your run with these numbers, write its own `results.csv` and diff the
measured columns (the first seven). No output means they match:

```bash
python examples/reproduction/results_csv.py "<the Output folder the run printed>" > mine.csv
diff <(cut -d, -f1-7 examples/reproduction/results.csv) <(cut -d, -f1-7 mine.csv)
```

The sweep needs no LLM and no API key. The first run downloads the two datasets
(about 15 MB) and three models (about 300 MB). The sweep is 28 tasks, 14 arms on
each dataset. On an Apple M1 Max it took 31 minutes with the models and datasets
already downloaded, and 37 minutes on a first run.

Every arm is deterministic, so the sweep runs once (`n_runs: 1`). Three runs, each
from an empty store, gave identical results.

## What the sweep holds fixed

Every arm starts from the `speed` profile. An arm then names one first stage
(`retriever: bm25`, `dense` or `hybrid_rrf`) and, in the cross-encoder arm, one
reranker. Nothing else differs between arms.

- **Whole documents.** BEIR scores documents, and the harness retrieves chunks.
  The sweep makes each document exactly one chunk, so no document is cut.
- **Truncation at the model's limit.** all-MiniLM-L6-v2 reads the first 256
  tokens of a document. bge-small-en-v1.5 and the cross-encoder read the first
  512, as in the published setups.
- **Query instruction.** bge-small-en-v1.5 queries get the instruction from its
  model card, which MTEB also applies. all-MiniLM-L6-v2 takes none.
- **Rerank depth.** The cross-encoder reorders the BM25 top 100.
- **Scoring.** NDCG@10 uses linear gain, as `trec_eval` does. Rescoring every
  ranked list with `pytrec_eval`, the evaluator BEIR uses, gave the same numbers to
  four decimals.

## Sources

- **BEIR**: Thakur, Reimers, Rücklé, Srivastava and Gurevych, "BEIR: A
  Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval
  Models", NeurIPS 2021 Datasets and Benchmarks Track,
  [arXiv:2104.08663v4](https://arxiv.org/abs/2104.08663v4).
- **MTEB results**: [embeddings-benchmark/results](https://github.com/embeddings-benchmark/results).
- **Anserini and Pyserini**: the
  [Pyserini BEIR regressions](https://castorini.github.io/pyserini/2cr/beir.html).
