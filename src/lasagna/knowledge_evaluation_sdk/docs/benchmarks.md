# Benchmark Catalogue

31 public benchmarks registered in the evaluation SDK, organized by category. Five more
( and the four ) are internal harness fixtures, so the registry
reports 36 in total.

## Regenerating This Table

If adapters are added or modified, regenerate the master table from the registry:

```bash
PYTHONPATH=src python3 -c "
from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
for name, info in sorted(benchmark_registry.list_benchmarks().items()):
    print(f'| {name} | {info.category.value} | {info.eval_mode.value} | {info.description} |')
"
```

---

## Master Table

| Name | Category | Eval Mode | Description |
|------|----------|-----------|-------------|
| `bamboogle` | multi_hop | generation | Bamboogle -- 125 multi-hop questions that Google cannot answer directly |
| `beir/arguana` | single_hop | retrieval | ArguAna -- argument retrieval (1.4K queries, 8.7K docs) |
| `beir/climate-fever` | single_hop | retrieval | Climate-FEVER -- climate claim verification (1.5K queries) |
| `beir/dbpedia-entity` | single_hop | retrieval | DBPedia-Entity -- entity retrieval (400 queries, 4.6M docs) |
| `beir/fever` | single_hop | retrieval | FEVER (BEIR) -- fact verification retrieval (6.7K queries) |
| `beir/fiqa` | single_hop | retrieval | FiQA -- financial opinion QA (6.6K queries, 57K docs) |
| `beir/hotpotqa` | single_hop | retrieval | HotpotQA (BEIR) -- multi-hop reasoning (7.4K queries, 5.2M docs) |
| `beir/msmarco` | single_hop | retrieval | MS MARCO (BEIR split) -- passage retrieval (6.9K dev queries) |
| `beir/nfcorpus` | single_hop | retrieval | NF Corpus -- biomedical IR benchmark (3.6K queries, 170K docs) |
| `beir/nq` | single_hop | retrieval | Natural Questions (BEIR) -- Google search questions (3.5K queries) |
| `beir/quora` | single_hop | retrieval | Quora Duplicate Questions -- duplicate detection (10K queries) |
| `beir/scidocs` | single_hop | retrieval | SciDocs -- scientific document recommendations (1K queries) |
| `beir/scifact` | single_hop | retrieval | SciFact -- scientific claim verification (1.1K claims, 5K docs) |
| `beir/trec-covid` | single_hop | retrieval | TREC-COVID -- COVID-19 related retrieval (50 queries, 171K docs) |
| `fever` | fact_verification | generation | FEVER -- Fact Extraction and VERification (185K claims) |
| `hotpotqa` | multi_hop | end_to_end | HotpotQA -- multi-hop question answering with supporting facts |
| `kilt/nq` | single_hop | generation | KILT Natural Questions -- knowledge-intensive QA |
| `kilt/triviaqa` | single_hop | generation | KILT TriviaQA -- trivia questions as knowledge-intensive task |
| `ms_marco` | passage_ranking | generation | MS MARCO -- passage ranking (8.8M passages) |
| `multihop_rag` | multi_hop | end_to_end | MultiHop-RAG -- multi-hop RAG benchmark from news articles |
| `musique` | multi_hop | generation | MuSiQue -- multi-hop questions requiring 2-4 composition steps |
| `natural_questions` | single_hop | generation | Natural Questions -- Google search questions with Wikipedia answers |
| `ragbench/hotpotqa` | single_hop | end_to_end | RAGBench (hotpotqa) -- TRACe-style RAG evaluation |
| `ragbench/msmarco` | single_hop | end_to_end | RAGBench (msmarco) -- TRACe-style RAG evaluation |
| `ragbench/pubmedqa` | single_hop | end_to_end | RAGBench (pubmedqa) -- TRACe-style RAG evaluation |
| `ragbench/squad` | single_hop | end_to_end | RAGBench (squad) -- TRACe-style RAG evaluation |
| `ragbench/techqa` | single_hop | end_to_end | RAGBench (techqa) -- TRACe-style RAG evaluation |
| `squad_v2` | single_hop | generation | SQuAD v2 -- reading comprehension with unanswerable questions |
| `triviaqa` | single_hop | generation | TriviaQA -- trivia question answering (95K QA pairs) |
| `wiki_multihop` | multi_hop | generation | 2WikiMultiHopQA -- multi-hop reasoning over two Wikipedia articles |

---

## Benchmarks by Category

### Single-Hop QA (7 standalone benchmarks)

| Name | Eval Mode | Description | Queries | Paper |
|------|-----------|-------------|---------|-------|
| `squad_v2` | generation | Reading comprehension with unanswerable questions | 11,873 | arxiv:1806.03822 |
| `natural_questions` | generation | Google search questions with Wikipedia answers | 7,830 | arxiv:1901.08634 |
| `triviaqa` | generation | Trivia question answering (95K QA pairs) | 95,000 | arxiv:1705.03551 |
| `popqa` | generation | Entity-centric QA across popularity tiers | 14,267 | arxiv:2212.10511 |
| `beir/nfcorpus` | retrieval | Biomedical IR benchmark | 3,600 | arxiv:2104.08663 |
| `beir/scifact` | retrieval | Scientific claim verification | 1,100 | arxiv:2004.14974 |
| `beir/fiqa` | retrieval | Financial opinion QA | 6,600 | -- |

### Multi-Hop Reasoning (5 benchmarks)

| Name | Eval Mode | Description | Queries | Paper |
|------|-----------|-------------|---------|-------|
| `hotpotqa` | end_to_end | Multi-hop QA with supporting facts | 112,779 | arxiv:1809.09600 |
| `musique` | generation | 2-4 step compositional questions | 25,000 | arxiv:2108.00573 |
| `wiki_multihop` | generation | 2WikiMultiHopQA -- reasoning over two Wikipedia articles | 192,606 | arxiv:2011.01060 |
| `multihop_rag` | end_to_end | MultiHop-RAG from news articles | 2,556 | arxiv:2401.15391 |
| `bamboogle` | generation | 125 curated multi-hop questions Google cannot answer directly | 125 | arxiv:2304.09848 |

### Passage Ranking (2 benchmarks)

| Name | Eval Mode | Description | Queries | Docs |
|------|-----------|-------------|---------|------|
| `ms_marco` | generation | MS MARCO passage ranking | 808,731 | 8.8M |
| `beir/msmarco` | retrieval | MS MARCO (BEIR split) -- dev queries | 6,900 | 8.8M |

### Fact Verification (3 benchmarks)

| Name | Eval Mode | Description | Queries | Paper |
|------|-----------|-------------|---------|-------|
| `fever` | generation | FEVER -- Fact Extraction and VERification | 185,445 | arxiv:1803.05355 |
| `beir/fever` | retrieval | FEVER (BEIR) -- retrieval split | 6,700 | -- |
| `beir/climate-fever` | retrieval | Climate-FEVER -- climate claim verification | 1,500 | -- |

### Full BEIR Suite (13 retrieval benchmarks)

All BEIR benchmarks use `eval_mode=retrieval` and `category=single_hop`. They ship with qrels
for computing NDCG@10, Recall@K, and MAP.

| Name | Description | Queries | Documents |
|------|-------------|---------|-----------|
| `beir/nfcorpus` | NF Corpus -- biomedical IR | 3,600 | 170K |
| `beir/scifact` | SciFact -- scientific claims | 1,100 | 5K |
| `beir/fiqa` | FiQA -- financial opinion QA | 6,600 | 57K |
| `beir/hotpotqa` | HotpotQA -- multi-hop reasoning | 7,400 | 5.2M |
| `beir/arguana` | ArguAna -- argument retrieval | 1,400 | 8.7K |
| `beir/msmarco` | MS MARCO -- passage retrieval | 6,900 | 8.8M |
| `beir/trec-covid` | TREC-COVID -- COVID-19 retrieval | 50 | 171K |
| `beir/nq` | Natural Questions | 3,500 | 2.68M |
| `beir/quora` | Quora Duplicate Questions | 10,000 | 523K |
| `beir/dbpedia-entity` | DBPedia entity retrieval | 400 | 4.6M |
| `beir/fever` | FEVER -- fact verification retrieval | 6,700 | 5.4M |
| `beir/climate-fever` | Climate-FEVER -- climate claims | 1,500 | 5.4M |
| `beir/scidocs` | SciDocs -- scientific document recs | 1,000 | 25K |

### End-to-End RAG -- RAGBench (5 benchmarks)

RAGBench provides TRACe-style evaluation (retrieval + generation) across multiple domains.
All use `eval_mode=end_to_end` and `category=single_hop`.

| Name | Domain | Paper |
|------|--------|-------|
| `ragbench/hotpotqa` | Multi-hop QA | arxiv:2407.11005 |
| `ragbench/squad` | Reading comprehension | arxiv:2407.11005 |
| `ragbench/msmarco` | Passage ranking | arxiv:2407.11005 |
| `ragbench/pubmedqa` | Biomedical QA | arxiv:2407.11005 |
| `ragbench/techqa` | Technical QA | arxiv:2407.11005 |

### Knowledge-Intensive -- KILT (2 benchmarks)

KILT wraps existing QA datasets as knowledge-intensive tasks with a shared Wikipedia corpus.
Both use `eval_mode=generation` and `category=single_hop`.

| Name | Description | Paper |
|------|-------------|-------|
| `kilt/nq` | KILT Natural Questions -- knowledge-intensive QA | arxiv:2009.02252 |
| `kilt/triviaqa` | KILT TriviaQA -- trivia as knowledge-intensive task | arxiv:2009.02252 |

---

## Evaluation Modes

The `eval_mode` field on each `BenchmarkInfo` tells the evaluation runner what to measure.

**retrieval** -- IR metrics only. The benchmark ships qrels (query-document relevance
judgments). The pipeline retrieves documents and scores are computed against those judgments.
Metrics: NDCG@10, Recall@K, MAP, MRR. No LLM generation step. Used by all BEIR benchmarks.

**generation** -- Answer quality only. The benchmark provides questions and gold answers.
The pipeline generates answers and scores are computed against gold references. Metrics:
Exact Match, F1, faithfulness. No retrieval evaluation. Used by SQuAD, NQ, TriviaQA, FEVER,
MuSiQue, Bamboogle, wiki_multihop, MS MARCO, KILT.

**end_to_end** -- Both retrieval and generation. The benchmark provides qrels and gold
answers. The pipeline retrieves, generates, and both steps are scored. Metrics: all of the
above. Used by HotpotQA (full), MultiHop-RAG, and all RAGBench variants.

### Offline vs. Live

Orthogonal to `eval_mode`, each benchmark run can execute in one of two modes:

- **offline** (default): Score against gold labels without a vector database. The benchmark
  provides inline context or pre-retrieved passages. Fast, reproducible, no infrastructure
  needed. Suitable for CI and quick iteration.

- **live**: Ingest the benchmark corpus into a vector database, run the full search pipeline,
  and compute IR metrics against qrels. Tests real end-to-end behavior including embedding,
  indexing, retrieval, and reranking. Requires a running vector backend (Milvus, Weaviate, or
  pgvector). Use for integration testing and production readiness validation.

### Pooled-gold corpora

Eight benchmarks ship each question's supporting passages inline and no corpus:
`squad_v2`, `hotpotqa`, `musique`, and the five `ragbench/*` domains. `auto` runs them
live against a corpus pooled from those passages across the whole sample, so each query
must find its own among every other query's.

**These numbers are not comparable with a native-corpus run.** HotpotQA and MuSiQue drop
their distractor paragraphs at conversion, so the pool contains no hard negatives — only
other questions' answers — and recall reads higher than it would against a real corpus of
the same size. Results carry `corpus_provenance: pooled_gold`, which is part of the
comparability key, so a pooled row can never join a BEIR row in a leaderboard group. For
the same questions against a real corpus, use `beir/hotpotqa`.

Ten benchmarks carry no context at all (`natural_questions`, `triviaqa`, `fever`,
`popqa`, `bamboogle`, `wiki_multihop`, `multihop_rag`, `ms_marco`, and the two
`kilt/*`) and stay generation-only. There is nothing to pool.

```bash
# Offline (default)
python -m lasagna benchmark run beir/nfcorpus --samples 100

# Live (requires vector backend)
python -m lasagna benchmark run beir/nfcorpus --samples 100 --mode live
```

---

## Adding Custom Benchmarks

### 1. Implement the adapter

Subclass `BaseBenchmarkAdapter` and implement `info()` and `load()`:

```python
from lasagna.knowledge_evaluation_sdk.benchmarks import (
    BaseBenchmarkAdapter,
    BenchmarkInfo,
    BenchmarkSample,
    BenchmarkCategory,
    BenchmarkEvalMode,
    benchmark_registry,
)

@benchmark_registry.register("my_dataset")
class MyBenchmarkAdapter(BaseBenchmarkAdapter):
    def info(self) -> BenchmarkInfo:
        return BenchmarkInfo(
            name="my_dataset",
            description="My custom benchmark -- 500 domain-specific questions",
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=BenchmarkEvalMode.GENERATION,
            num_queries=500,
            license="MIT",
            tags=["custom", "domain_specific"],
        )

    def load(self, split="test", n_samples=None, seed=42,
             cache_dir=None, filters=None):
        # Load your data from any source: HuggingFace, local files, API, etc.
        data = _load_my_data(split)

        samples = [
            BenchmarkSample(
                query_id=str(i),
                question=row["question"],
                ground_truth=row["answer"],
            )
            for i, row in enumerate(data)
        ]

        if n_samples and n_samples < len(samples):
            import random
            rng = random.Random(seed)
            samples = rng.sample(samples, n_samples)

        return samples

    def load_corpus(self, cache_dir=None, max_docs=100_000):
        """Optional: return {doc_id: text} for live-mode ingestion."""
        return _load_my_corpus(max_docs=max_docs)

    def available_splits(self):
        return ["train", "test"]

    def available_filters(self):
        return {"difficulty": ["easy", "hard"]}
```

### 2. Register via entry points (for pip-installable packages)

Add a `lasagna_benchmarks` entry-point group to your `pyproject.toml`:

```toml
[project.entry-points.lasagna_benchmarks]
my_dataset = "my_package.benchmarks:MyBenchmarkAdapter"
```

The registry discovers entry points automatically on first access. No import or manual
registration needed -- install the package and the benchmark appears in `benchmark_registry.list_benchmarks()`.

### 3. Verify registration

```bash
PYTHONPATH=src python3 -c "
from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry
assert 'my_dataset' in benchmark_registry
info = benchmark_registry.get('my_dataset').info()
print(f'{info.name}: {info.description} ({info.category.value}, {info.eval_mode.value})')
"
```
