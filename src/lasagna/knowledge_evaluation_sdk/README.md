# knowledge_evaluation_sdk: benchmarks, sweeps, results

This package measures pipelines. It loads benchmark samples through adapters, runs every
(benchmark, arm, run) task through the real pipeline, scores retrieval and answers, and turns
the per-sample scores into result files, significance tests, bundles and leaderboards. The
`lasagna benchmark` and `lasagna results` commands in `cli.py` (package root) are thin wrappers
over it.

Running a benchmark or a sweep: [docs/guide.md](../../../docs/guide.md#8-measure-it-on-a-benchmark).
Your own data: [docs/own-data.md](../../../docs/own-data.md). Metric definitions:
[docs/metrics.md](docs/metrics.md). Benchmark catalogue: [docs/benchmarks.md](docs/benchmarks.md).
Example sweep files: [research/](research/README.md).

## Two paths

**Experiment path (benchmarks and sweeps).** `experiment.py`:
`ExperimentConfig` (parsed from YAML or TOML by `Experiment.from_file`, unknown keys refused)
-> `Experiment.generate_tasks()` (benchmarks x arms x runs) -> `BenchmarkRunner.run_tasks()`
(`runner.py`: bounded concurrency, a session id per task from its config hash,
`CheckpointManager` for `--resume`)
-> the eval function for the task's mode -> `BenchmarkResults` (`results.py`) ->
`results.json`, `report.md`, `table.tex` (`_save_outputs`). Default output is
`~/.lasagna/research/results/<name>_<timestamp>/` (`_results_root`).

The eval functions: `_eval_live` ingests the benchmark corpus once per corpus key (shared
across arms that do not change ingestion), searches through the pipeline, and scores
retrieval and generated answers; `_eval_offline` answers from the gold context and scores the
answers, with no retrieval and no vector store. `resolve_mode` maps `auto` to `live` when the adapter ships a corpus and to
`generation_only` otherwise (`offline` is an alias).

**Evaluator path (one document, LLM judge).** `evaluator.py`: `KnowledgeEvaluator` uploads a
document through `lasagna.sdk.KnowledgeClient`, answers each ground-truth question from the
retrieved context, and scores it with `LLMJudge` (`judge.py`, 1 to 5 for relevance,
completeness and factual accuracy). With `use_full_agent=True` it goes through
`lasagna.integrations.cuga_agent` instead, which keeps the agent framework out of this package.
`python -m lasagna.knowledge_evaluation_sdk --help` lists its CLI flags; the exit code is 0 at
a pass rate of 80% or more, 1 below it, 2 on error.

## Module map

| File | Responsibility |
| ---- | -------------- |
| `__init__.py` | Lazy public API (`Experiment`, `run_benchmark`, `BenchmarkResults`, `sweep`, `grid_search`, the IR metrics, the adapter types, ...) |
| `experiment.py` | Config parsing and checks, arm generation, task eval functions, corpus ingestion cache, `run_benchmark` |
| `runner.py` | `BenchmarkTask`, `TaskResult`, `BenchmarkRunner` |
| `_fast_ingest.py` | Corpus ingestion for evaluation: chunks in memory, batches embeddings across documents, inserts in bulk, skipping the per-upload path |
| `stages.py` | Which stage each setting belongs to, per-task stage manifests, `isolation_report` (undeclared and inert changes) |
| `ir_metrics.py`, `answer_metrics.py` | Retrieval metrics and judge-free answer metrics |
| `ragas_evaluator.py` | `RAGASEvaluator`: the judged RAGAS metrics (`eval` extra) |
| `judge.py`, `models.py`, `evaluator.py`, `cli.py`, `__main__.py` | The evaluator path |
| `results.py` | `BenchmarkResults`: aggregation, paired sign-flip tests with Holm correction, leaderboard, Markdown and LaTeX |
| `bundle.py`, `run_provenance.py`, `leaderboard.py`, `viewer.py` | Results bundles and their comparability key (`METRICS_VERSION`), run provenance, leaderboard rendering, the static HTML view (`lasagna results view`) |
| `ablation.py` | `sweep`, `grid_search`, `AblationResults` over `KnowledgePipeline` (Python API, not used by the CLI) |
| `cost.py`, `seeding.py`, `checkpoint.py`, `_helpers.py` | Dry-run cost estimates, `seed_everything`, atomic task checkpoints, dependency-free helpers for tests |
| `synthesize.py` | `lasagna benchmark synthesize`: LLM-written question sets from a folder of documents |
| `datasets.py`, `reporting.py` | Hugging Face loaders for `KnowledgePipeline.run_and_evaluate`; `ReportGenerator` (Markdown, LaTeX, plots) |
| `benchmarks/` | `_base.py` (`BaseBenchmarkAdapter`, `BenchmarkSample`, `BenchmarkInfo`, ...), `_registry.py` (`benchmark_registry`, entry-point group `lasagna_benchmarks`), `adapters/` (one file per benchmark family, plus `smoke` and `local_files`) |
| `private_eval/` | Builder for a two-track (retrieval, generation) evaluation set assembled from public benchmarks |
| `research/` | Shipped example sweep files and the generated `DATASETS.md` |
| `docs/` | `metrics.md`, `benchmarks.md`, `architecture.md` (shipped in the wheel) |

## Design notes

**Strict stages.** `_default_eval_fn` runs `_eval_live` inside `core.strict.strict()`, so a
stage that cannot run (no LLM for expansion, an agentic search that errors) fails the task
instead of scoring the baseline under the arm's name.

**Isolation is checked, not assumed.** Every task records its stage manifest; `stages.py`
compares arms and flags a stage that changed without being named, or a named change that never
reached the pipeline. `lasagna results validate` refuses a bundle with undeclared changes.

**Comparability.** `bundle.py` decides which rows may be compared: everything except the arm
must be held fixed. `METRICS_VERSION` is part of that key; bump it, with a comment saying what
changed, when a metric definition changes, so old and new numbers are not compared.

**Seeds and caches.** `seeding.seed_everything` seeds every source a task can reach. The LLM
response cache (`llm_cache:` in a sweep, `lasagna.utils.llm_cache`) opens one scope per task,
seeded with `seed + run_index`, so repeated runs of an arm keep their own responses.

**Adding a benchmark.** Subclass `BaseBenchmarkAdapter` in `benchmarks/adapters/`, register it,
then regenerate the dataset terms with
`lasagna benchmark licenses -o src/lasagna/knowledge_evaluation_sdk/research/DATASETS.md`;
`tests/test_dataset_licenses.py` fails when a registered benchmark is missing from it.
