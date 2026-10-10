# Reproducing published BEIR numbers

Lasagna's built-in retrievers and reranker reproduce published numbers on BEIR
SciFact and NFCorpus. Both dense models land within 0.002 NDCG@10 of their MTEB
results, BM25 followed by the MiniLM cross-encoder within 0.004 of the BEIR
paper, and BM25 within 0.008. Every test query of both datasets is scored
against the full corpus. No method was tuned: BM25 uses the k1 and b that BEIR
used, and the dense models and the cross-encoder use their published
checkpoints unchanged.

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

Recall@100, from the same sweep:

| Method | Dataset | Published | Source | Lasagna | Difference |
|---|---|---|---|---|---|
| BM25 (k1 0.9, b 0.4) | SciFact | 0.908 | BEIR, Table 9 | 0.906 | -0.002 |
| BM25 (k1 0.9, b 0.4) | NFCorpus | 0.250 | BEIR, Table 9 | 0.243 | -0.007 |
| all-MiniLM-L6-v2 | SciFact | 0.925 | MTEB results | 0.925 | 0.000 |
| all-MiniLM-L6-v2 | NFCorpus | 0.312 | MTEB results | 0.312 | 0.000 |
| bge-small-en-v1.5 | SciFact | 0.945 | MTEB results | 0.942 | -0.003 |
| bge-small-en-v1.5 | NFCorpus | 0.312 | MTEB results | 0.311 | -0.001 |

Arms with no published target, from the same sweep:

| Method | SciFact NDCG@10 | NFCorpus NDCG@10 | SciFact Recall@100 | NFCorpus Recall@100 |
|---|---|---|---|---|
| BM25 with Lasagna's defaults (k1 1.2, b 0.75) | 0.676 | 0.317 | 0.913 | 0.243 |
| Hybrid RRF: BM25 + all-MiniLM-L6-v2 | 0.708 | 0.353 | 0.955 | 0.323 |
| Hybrid RRF: BM25 + bge-small-en-v1.5 | 0.721 | 0.364 | 0.968 | 0.316 |

[results.csv](results.csv) has every measured number to four decimals, with the
query count, the run date and the package commit.

## Rerun it

From a clone of the repository:

```bash
pip install "lasagna-rag[eval]"
python -m nltk.downloader punkt_tab
lasagna benchmark experiment examples/reproduction/beir_tier1.yaml
```

It needs no LLM and no API key. The first run downloads the two datasets (about
15 MB) and three models (about 300 MB). The sweep is 28 tasks, 14 arms on each
dataset, and took 33 minutes on an Apple M1 Max (10 cores, 64 GB, embeddings and
the cross-encoder on MPS), with a peak memory of 2.6 GB. Results land in
`~/.lasagna/research/results/BEIR_Tier_1_reproduction_<timestamp>/`.

Every arm is deterministic, so the sweep runs once (`n_runs: 1`). Three runs,
each from an empty store, gave identical results: every metric and every ranked
list of all 28 tasks.

## What the sweep holds fixed

Every arm starts from the `speed` profile, which turns off hybrid fusion,
reranking, MMR, HyDE, query enhancement and agentic retrieval. An arm then names
one first stage (`retriever: bm25`, `dense` or `hybrid_rrf`) and, in the
cross-encoder arm, one reranker. Nothing else differs between arms.

- **Whole documents.** BEIR scores documents, and the harness retrieves chunks.
  The sweep makes each document exactly one chunk: fixed-size chunking with a
  100,000-character chunk, the structure-aware and hierarchical chunkers off
  (they would otherwise take precedence over `chunking_strategy`), and a
  storage cap of 20,000 characters, above the longest document (10,128).
  Counting the stored rows after the run gave 5,183 chunks for the 5,183
  SciFact documents and 3,633 for the 3,633 NFCorpus documents, in both
  embedding collections, with no document cut.
- **Truncation at the model's limit.** all-MiniLM-L6-v2 reads the first 256
  tokens of a document, and bge-small-en-v1.5 and the cross-encoder the first
  512, as in the published setups.
- **Query instruction.** bge-small-en-v1.5 queries get the instruction from its
  model card, "Represent this sentence for searching relevant passages: ", which
  MTEB also applies. all-MiniLM-L6-v2 takes none.
- **Rerank depth.** The cross-encoder reorders the BM25 top 100. On every query
  of both datasets its top 100 holds the same documents as BM25's, so all 100
  candidates reached it.
- **Scoring.** NDCG@10 uses linear gain, as trec_eval does, so NFCorpus's graded
  judgments (1 and 2) count as BEIR counts them. Rescoring every ranked list of
  the run with `pytrec_eval`, the evaluator BEIR uses, gave the same numbers to
  four decimals.

## Notes on the differences

**BM25.** The camera-ready BEIR paper ran Anserini with Lucene's defaults (k1
0.9, b 0.4) and indexed the title and the text as separate fields. The sweep uses
the same k1 and b over one field holding the title and the text, which is the
"flat" condition of the Anserini and Pyserini BEIR regressions. Those report
0.679 (SciFact) and 0.322 (NFCorpus) for it, so Lasagna sits 0.015 and 0.005
below the closest like-for-like Lucene number. Scoring the same corpus with
rank_bm25 directly, outside the harness, gives Lasagna's numbers to four
decimals, so the rest of the difference lies in the BM25 variant rather than in
the harness: rank_bm25's Okapi IDF differs from Lucene's, no stopwords are
removed, and NLTK's Porter stemmer stands in for Lucene's analyzer.

**Short BM25 lists on NFCorpus.** 75 of the 323 NFCorpus queries are one or two
rare words ("amnesia", "Alli", "airport scanners") that fewer than ten documents
contain. BM25 returns only documents that share a term with the query, as Lucene
does, so those lists are shorter than ten and are scored as BEIR scores them.
The run logs a warning that says so.

**The cross-encoder** reranks Lasagna's own BM25 top 100, not BEIR's, so its
difference includes the first stage's: Lasagna's BM25 recall@100 is 0.002
(SciFact) and 0.007 (NFCorpus) below BEIR's.

**Which BEIR numbers.** The first BEIR preprints (arXiv 2104.08663v1 and v2,
April 2021) ran BM25 on Elasticsearch and reported 0.620 (SciFact) and 0.297
(NFCorpus). The NeurIPS 2021 version (v3 and v4) switched to Anserini, and its
numbers are the ones in the tables. We compare against the peer-reviewed
version.

**Two MTEB sources for bge-small-en-v1.5.** The MTEB results repository and the
model card agree on NDCG@10 (SciFact 0.7127 and 0.7128, NFCorpus 0.3426 and
0.3431) and differ slightly on SciFact recall@100 (0.945 and 0.942). The tables
cite the results repository.

## What building this caught

Building this sweep found two problems in the harness. Both are fixed, with
regression tests, in the code that produced the numbers above. First, the
evaluation ingest cut every chunk to `max_content_length` (4,000 characters by
default) before embedding and indexing it, and the setting was filed as
operational, so a sweep could not raise it and nothing logged the cut. Asked for
whole documents, the harness indexed only the first 4,000 characters of the 20
SciFact and 12 NFCorpus documents that are longer. The setting is now an
ingestion setting that an arm can change, and a cut is logged. Second, the
warning about short NFCorpus lists advised raising `retrieval_chunk_overfetch`,
which cannot help when BM25 has run out of matching documents; it now names the
cause.

## Sources

- **BEIR**: Thakur, Reimers, Rücklé, Srivastava and Gurevych, "BEIR: A
  Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval
  Models", NeurIPS 2021 Datasets and Benchmarks Track,
  [arXiv:2104.08663v4](https://arxiv.org/abs/2104.08663v4). Table 2 (NDCG@10),
  columns BM25 and BM25+CE; Table 9 (Recall@100), column BM25. Section 4 gives
  the setup: Anserini BM25 with k1 0.9 and b 0.4, and BM25+CE reranking the BM25
  top 100 with the 6-layer MiniLM cross-encoder that Table 6 links as
  `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **MTEB results**: [embeddings-benchmark/results](https://github.com/embeddings-benchmark/results),
  `results/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a/`
  (mteb 1.12.75) and
  `results/BAAI__bge-small-en-v1.5/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/`
  (mteb 1.20.0), files `SciFact.json` and `NFCorpus.json`, test split,
  `ndcg_at_10` (the main score) and `recall_at_100`. The
  [bge-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
  lists its own MTEB scores in its metadata.
- **Anserini and Pyserini**: the [Pyserini BEIR regressions](https://castorini.github.io/pyserini/2cr/beir.html),
  columns BM25 Flat and BM25 MF. The flat index concatenates title and text into
  one field; the multifield one is BEIR's setup and gives the same numbers as
  BEIR's Tables 2 and 9.

## Data and environment

- Datasets, as the BEIR adapter loads them (unpinned): `mteb/scifact` at
  cf10ab68 for SciFact, and `BeIR/nfcorpus` at b5026a0e with
  `BeIR/nfcorpus-qrels` at a451b3b2 for NFCorpus. MTEB scored `mteb/scifact` at
  0228b52c and `mteb/nfcorpus` at ec0fa4fe. The NFCorpus test judgments in
  `mteb/nfcorpus` and `BeIR/nfcorpus-qrels` are identical (12,334 for 323
  queries).
- Models: `sentence-transformers/all-MiniLM-L6-v2` at 1110a243 (MTEB scored
  8b3219a9), `BAAI/bge-small-en-v1.5` at 5c38ec7c (the revision MTEB scored),
  and `cross-encoder/ms-marco-MiniLM-L-6-v2` at 233902d2.
- Run from a source checkout at the commit in `results.csv`, with Python
  3.12.12, torch 2.14.1, sentence-transformers 6.1.0, transformers 5.19.0,
  rank-bm25 0.2.2 and milvus-lite 3.2.1 (the versions in
  `constraints/0.1.0.txt`), on macOS 26.5.
