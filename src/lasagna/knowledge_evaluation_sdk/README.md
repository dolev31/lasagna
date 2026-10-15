# Knowledge Evaluation SDK

## Overview

The Knowledge Evaluation SDK provides three complementary evaluation workflows for the Lasagna RAG knowledge system:

1. **QA testing with LLM-as-Judge** -- Upload a knowledge document, run queries from a ground-truth JSONL file through the RAG pipeline, and score each response on a 1--5 scale using an LLM judge. Produces per-case scores for relevance, completeness, and factual accuracy alongside aggregate pass rates and latency statistics.

2. **IR benchmark evaluation** -- Evaluate retrieval quality against 31 standard academic benchmarks (BEIR, SQuAD, HotpotQA, TriviaQA, MS MARCO, Natural Questions, PopQA, MuSiQue, Bamboogle, KILT, FEVER, and more) via a unified benchmark adapter registry. Computes NDCG@k, MAP, MRR, Precision, Recall, F1, Hit Rate, R-precision and complete (all-gold) recall using pure-Python IR metric implementations. Answers are scored too: exact match, token F1, ROUGE-L, answer containment, answer similarity, ALCE citation recall and precision, abstention, and the judged RAGAS metrics. [docs/metrics.md](docs/metrics.md) defines each one.

3. **Ablation / grid-search research** -- Sweep pipeline components across variant lists: rerankers, query enhancers, retrievers, agentic pipelines and generators as components; chunkers and embedders as components too, each arm on a corpus it chunked or embedded itself; and any setting, with every chunking and embedding setting re-ingesting per arm. Document processors do not sweep, since benchmark corpora are loaded as text. Sweeps collect per-sample scores and run paired sign-flip significance tests with t intervals and Holm--Bonferroni correction. Export results to Markdown tables, LaTeX (publication-ready), pandas DataFrames, and matplotlib plots (radar, heatmap).

---

## Quick Start

### Use Case 1: QA evaluation with LLM-as-Judge

```python
from lasagna.knowledge_evaluation_sdk import (
    KnowledgeEvaluator, EvalConfig,
)

evaluator = KnowledgeEvaluator(
    config=EvalConfig(
        document_path="./data/knowledge_base.docx",
        ground_truth_path="./data/ground_truth.jsonl",
    )
)
metrics = await evaluator.run()
print(metrics.summary())
```

### Use Case 2: IR benchmark evaluation

```python
from lasagna.knowledge_evaluation_sdk import (
    Experiment, ExperimentConfig, run_benchmark,
)

# One-liner
results = await run_benchmark("beir/nfcorpus", n_samples=100, mode="live")
print(results.to_markdown())

# Or from a YAML config file
exp = Experiment.from_file("beir_sweep.yaml")
results = await exp.run()
```

### Use Case 3: Ablation study

```python
from lasagna.knowledge_evaluation_sdk import sweep
from lasagna.core.pipeline import KnowledgePipeline

pipeline = KnowledgePipeline.from_profile("standard")
results = sweep(
    pipeline=pipeline,
    component="reranker",
    variants=["mxbai-rerank-large", "bge-reranker-v2", "ms-marco-MiniLM"],
    dataset=my_dataset,
    metrics=["faithfulness", "context_precision"],
    evaluator=my_evaluator,
)
print(results.to_markdown())
sig = results.significance_test("mxbai-rerank-large", "bge-reranker-v2", "faithfulness")
print(f"p={sig['p_value']:.4f}, significant={sig['significant']}")
```

---

## Module Structure

| File | Purpose |
|---|---|
| `__init__.py` | Public API with lazy-loading for optional heavy dependencies |
| `__main__.py` | `python -m ...` entry point (delegates to `cli.main()`) |
| `models.py` | Data models: `EvalCase`, `EvalResult`, `EvalMetrics`, `EvalConfig`, `JudgeScore`, `ScoreLevel` |
| `evaluator.py` | `KnowledgeEvaluator` -- main orchestrator for document upload, RAG query, and LLM-as-Judge scoring |
| `judge.py` | `LLMJudge` -- structured 1--5 scoring with relevance/completeness/factual accuracy sub-scores |
| `experiment.py` | `Experiment` / `ExperimentConfig` -- YAML/TOML config loader, Cartesian product task generation, execution orchestrator |
| `runner.py` | `BenchmarkRunner` / `BenchmarkTask` / `TaskResult` -- parallel execution engine with semaphore, session isolation, and checkpoint resumption |
| `ir_metrics.py` | Pure-Python IR metrics: `ndcg_at_k`, `precision_at_k`, `recall_at_k`, `mrr`, `map_score`, `f1_at_k`, `hit_rate_at_k`, `r_precision`, `complete_recall_at_k`, `corpus_metrics` |
| `answer_metrics.py` | Answer metrics without a judge: exact match, token F1, ROUGE-L, answer containment, abstention, answer similarity, ALCE citation precision and recall |
| `stages.py` | Which stage each setting belongs to, the per-task stage manifest, and the isolation check across arms |
| `datasets.py` | HuggingFace dataset loaders with local caching, format converters for 14 benchmark formats |
| `ablation.py` | `sweep()` (single-component), `grid_search()` (multi-component Cartesian), `AblationResults` with significance testing |
| `results.py` | `BenchmarkResults` -- multi-dimensional aggregation (benchmark x config x run), leaderboard, pairwise significance, export |
| `reporting.py` | `ReportGenerator` -- Markdown reports, LaTeX tables with significance markers, radar plots, heatmaps |
| `checkpoint.py` | `CheckpointManager` -- atomic JSON checkpointing for experiment resumption |
| `ragas_evaluator.py` | `RAGASEvaluator` / `DetailedEvalResult` -- optional RAGAS integration (faithfulness, context precision/recall, factual correctness) |
| (see `lasagna/integrations/cuga_agent.py`) | Optional CUGA agent adapter, kept outside this package so the evaluator has no agent-framework coupling |
| `cli.py` | argparse CLI: `--document`, `--ground-truth`, `--judge-platform`, `--verbose`, `--full-agent` |
| `benchmarks/` | Benchmark adapter framework |
| `benchmarks/_base.py` | `BaseBenchmarkAdapter` ABC, `BenchmarkSample`, `BenchmarkInfo`, `BenchmarkCategory`, `BenchmarkEvalMode` |
| `benchmarks/_registry.py` | `BenchmarkRegistry` singleton -- auto-discovers and registers adapters |
| `benchmarks/adapters/` | 14 concrete adapters: BEIR, SQuAD, Natural Questions, MS MARCO, TriviaQA, HotpotQA, PopQA, MuSiQue, 2WikiMultiHop, MultiHop-RAG, Bamboogle, RAGBench, KILT, FEVER |
| `data/` | Static data files (knowledge documents, ground-truth JSONL) |
| `research/` | Research experiment configs and scripts |

---

## Architecture

The SDK has two primary execution paths:

### Path A: Evaluator (document-centric QA testing)

```
EvalConfig --> KnowledgeEvaluator
                  |
                  +--> KnowledgeClient.upload(document)
                  +--> KnowledgeClient.get_context(query)  [or full agent invoke]
                  +--> LLM answering model (RAG context + question --> answer)
                  +--> LLMJudge.evaluate(question, expected, response, context)
                  +--> EvalMetrics.from_results(results)
                  +--> Save: JSONL results, JSON metrics, text report
```

`KnowledgeEvaluator` loads cases from a ground-truth JSONL file, uploads a knowledge document via `KnowledgeClient`, retrieves RAG context for each query, generates answers through an LLM, and scores responses with `LLMJudge`. Supports both RAG-only mode (standalone) and full-agent mode (via `lasagna.integrations.cuga_agent`).

### Path B: Experiment (benchmark-centric IR/pipeline evaluation)

```
ExperimentConfig --> Experiment
                       |
                       +--> generate_tasks()  [benchmarks x pipeline_configs x runs]
                       +--> BenchmarkRunner.run_tasks()
                       |       +--> benchmark_registry.get(name).load(samples)
                       |       +--> eval_fn(task, samples)  [offline or live mode]
                       |       +--> CheckpointManager.save(task_key, result)
                       +--> BenchmarkResults
                       |       +--> aggregate_by_benchmark() / aggregate_by_pipeline()
                       |       +--> significance_test() / pairwise_significance()
                       |       +--> leaderboard() / to_markdown() / to_latex()
                       +--> ReportGenerator.save_all(output_dir)
```

`Experiment` loads config from YAML/TOML, generates the Cartesian product of (benchmarks x pipeline configurations x run indices), and delegates execution to `BenchmarkRunner`. Each task loads samples from the benchmark registry, runs either offline (gold-label scoring) or live (corpus ingestion + real retrieval + IR metrics) evaluation, and checkpoints results atomically. `BenchmarkResults` aggregates across dimensions with mean/std, produces leaderboards, and supports paired sign-flip significance testing.

---

## External Dependencies

### Required

| Package | Usage |
|---|---|
| `loguru` | Structured logging throughout the SDK |
| `pydantic` | `JudgeScore` structured output model |
| `langchain-core` | `BaseChatModel`, `ChatPromptTemplate`, `HumanMessage`, `SystemMessage` |

### Optional

| Package | Usage | Install |
|---|---|---|
| `ragas` | RAGAS evaluation metrics (faithfulness, context precision/recall) | `pip install lasagna-rag[eval]` |
| `datasets` | HuggingFace benchmark dataset loading | `pip install datasets` |
| `pandas` | `AblationResults.to_dataframe()`, `BenchmarkResults.to_dataframe()` | `pip install lasagna-rag[eval]` |
| `matplotlib` | `AblationResults.plot()`, `ReportGenerator.radar_plot()`, `ReportGenerator.heatmap()` | `pip install lasagna-rag[eval]` |
| `numpy` | Sign-flip significance tests, vectorized metric computation | `pip install lasagna-rag[eval]` |
| `seaborn` | Enhanced heatmap rendering (falls back to matplotlib if absent) | `pip install seaborn` |
| `pyyaml` | YAML experiment config loading | `pip install pyyaml` |

All optional dependencies are lazy-imported at call sites. The core evaluation path (evaluator + judge + IR metrics) works without any optional packages installed.

---

## CLI

### QA evaluation

```bash
# Basic usage
python -m lasagna.knowledge_evaluation_sdk \
    -d ./data/knowledge_base.docx \
    -g ./data/ground_truth.jsonl

# Custom judge model and verbose output
python -m lasagna.knowledge_evaluation_sdk \
    -d ./data/knowledge_base.docx \
    -g ./data/ground_truth.jsonl \
    -o ./my_results \
    --judge-platform openai \
    --judge-model gpt-6-luna \
    --verbose

# Full agent mode (requires cuga.sdk)
python -m lasagna.knowledge_evaluation_sdk \
    -d ./data/knowledge_base.docx \
    -g ./data/ground_truth.jsonl \
    --full-agent
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `-d`, `--document` | (required) | Path to knowledge document (PDF, DOCX, TXT) |
| `-g`, `--ground-truth` | (required) | Path to JSONL ground truth file |
| `-o`, `--output` | `./eval_results` | Output directory |
| `--top-k` | `5` | Number of chunks to retrieve |
| `--no-reranking` | `false` | Disable reranking |
| `--use-hyde` | `false` | Enable HyDE query expansion |
| `--judge-platform` | `openai` | LLM judge platform (openai, azure, groq, watsonx, google-genai, openrouter) |
| `--judge-model` | platform default | Judge model name |
| `--max-concurrent` | `3` | Maximum concurrent evaluations |
| `-v`, `--verbose` | `false` | Verbose logging |
| `--full-agent` | `false` | Use full SDK agent invoke flow |

Exit codes: `0` = pass rate >= 80%, `1` = pass rate < 80%, `2` = evaluation error.

---

## Links

- [Research experiments](./research/README.md) -- experiment configs, reproduction scripts, and research notes.
- [SDK Documentation](../sdk/README.md) -- Knowledge SDK overview and documentation index.
