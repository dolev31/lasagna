# RAG Benchmark Research

Comprehensive benchmark suite for RAG pipeline evaluation and research.

## Quickstart

```python
from lasagna.sdk import run_benchmark

# One-liner evaluation
results = await run_benchmark("beir/nfcorpus", n_samples=100)
print(results.to_markdown())
```

## Running Experiments

### CLI

```bash
# List available benchmarks
python -m lasagna benchmark list

# Verify the install: 24 documents, no download, ~30 seconds
python -m lasagna benchmark run smoke

# Run a real benchmark (downloads the corpus on first use)
python -m lasagna benchmark run beir/nfcorpus --samples 100 --runs 3

# Run a full experiment from config
python -m lasagna benchmark experiment experiments/beir_sweep.yaml

# Resume an interrupted experiment
python -m lasagna benchmark experiment experiments/beir_sweep.yaml --resume

# Compare results
python -m lasagna benchmark compare results_a.json results_b.json --format latex
```

### Python API

```python
from lasagna.knowledge_evaluation_sdk import Experiment, ExperimentConfig
from lasagna.knowledge_evaluation_sdk.experiment import BenchmarkSpec, PipelineSweepSpec

# From YAML/TOML config
exp = Experiment.from_file("experiments/beir_sweep.yaml")
results = await exp.run()

# Programmatic
exp = Experiment(config=ExperimentConfig(
    name="My Experiment",
    benchmarks=[
        BenchmarkSpec(name="beir/nfcorpus", n_samples=200),
        BenchmarkSpec(name="beir/scifact", n_samples=200),
    ],
    pipeline=PipelineSweepSpec(
        rerankers=["cross_encoder", "combined", "ensemble"],
    ),
    n_runs=3,
    mode="live",
))
results = await exp.run()

# Export
results.save("results.json")
print(results.to_markdown())
print(results.to_latex())

# Statistical analysis
sig = results.significance_test("config_a", "config_b", "ndcg@10", "beir/nfcorpus")
print(f"p={sig['p_value']:.4f}, significant={sig['significant']}")
```

## Available Benchmarks

### Single-hop QA
- `beir/nfcorpus` — Biomedical IR (3.6K queries)
- `beir/scifact` — Scientific claim verification
- `beir/fiqa` — Financial QA
- `squad_v2` — Reading comprehension
- `natural_questions` — Google search questions
- `triviaqa` — Trivia QA (95K pairs)
- `popqa` — Entity-centric QA across popularity tiers (14K pairs)

### Multi-hop Reasoning
- `hotpotqa` — Multi-hop QA with supporting facts
- `musique` — 2-4 step compositional questions
- `wiki_multihop` — 2WikiMultiHopQA
- `multihop_rag` — MultiHop-RAG from news
- `bamboogle` — 125 curated multi-hop questions

### Passage Ranking
- `ms_marco` — MS MARCO (8.8M passages)
- `beir/msmarco` — BEIR split

### Fact Verification
- `fever` — FEVER (185K claims)
- `beir/fever` — BEIR retrieval split
- `beir/climate-fever` — Climate claims

### Full BEIR Suite
`beir/nfcorpus`, `beir/scifact`, `beir/fiqa`, `beir/hotpotqa`,
`beir/arguana`, `beir/msmarco`, `beir/trec-covid`, `beir/nq`,
`beir/quora`, `beir/dbpedia-entity`, `beir/fever`, `beir/climate-fever`,
`beir/scidocs`

### End-to-End RAG
- `ragbench/hotpotqa` — TRACe-style evaluation
- `ragbench/squad`, `ragbench/msmarco`, `ragbench/pubmedqa`, `ragbench/techqa`

### Knowledge-Intensive
- `kilt/nq`, `kilt/triviaqa`

## Adding a Custom Benchmark

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
class MyBenchmark(BaseBenchmarkAdapter):
    def info(self):
        return BenchmarkInfo(
            name="my_dataset",
            description="My custom benchmark",
            category=BenchmarkCategory.SINGLE_HOP,
            eval_mode=BenchmarkEvalMode.GENERATION,
        )

    def load(self, split="test", n_samples=None, seed=42, **kwargs):
        # Load your data and return BenchmarkSample list
        return [
            BenchmarkSample(
                query_id="1",
                question="What is X?",
                ground_truth="X is Y",
            ),
        ]
```

For distribution via pip, use the `lasagna_benchmarks` entry-point group:

```toml
# pyproject.toml
[project.entry-points.lasagna_benchmarks]
my_dataset = "my_package.benchmarks:MyBenchmark"
```

## Evaluation Modes

- **offline** (default): Score against gold labels. No vector DB needed. Fast.
- **live**: Ingest benchmark corpus into vector DB, search via pipeline,
  compare retrieved docs to gold qrels. Tests real end-to-end behavior.

## YAML Config Reference

```yaml
name: "Experiment Name"
description: "Optional description"

benchmarks:
  - name: beir/nfcorpus     # Required
    split: test              # Default: test
    n_samples: 200           # Optional limit
    filters: {}              # Adapter-specific filters

pipeline:
  base_profile: standard    # Base config profile
  rerankers: []              # Component variants to sweep
  embedders: []
  chunkers: []
  query_enhancers: []
  config_overrides: []       # Explicit override dicts

mode: auto                   # auto | live | generation_only
                             #   auto: live when the benchmark ships a corpus,
                             #         generation_only when it does not
                             #   (`offline` is a deprecated alias for
                             #    generation_only -- it never meant offline)
n_runs: 1                    # Runs per (benchmark x config)
n_samples: null              # Global sample limit
seed: 42
max_concurrency: 4
output_dir: ""               # Default: ~/.lasagna/research/results/
output_formats:              # markdown, json, latex
  - markdown
  - json
```

## Output Structure

```
~/.lasagna/research/results/{experiment}_{timestamp}/
├── results.json          # Machine-readable (all raw data)
├── report.md             # Human-readable summary
├── table.tex             # LaTeX table for papers
├── .checkpoints/         # Resume data
└── figures/              # Plots (radar, heatmap)
```

## Interpreting Results

- **ndcg@10**: Normalized Discounted Cumulative Gain — primary IR metric
- **precision@k**: Fraction of top-k results that are relevant
- **recall@k**: Fraction of relevant docs found in top-k
- **mrr**: Mean Reciprocal Rank — position of first relevant result
- **map**: Mean Average Precision
- **faithfulness**: LLM-judged answer fidelity to context
- **context_precision**: Relevance of retrieved context

Statistical tests use paired bootstrap resampling with Holm-Bonferroni
correction for multiple comparisons (>2 configs).
