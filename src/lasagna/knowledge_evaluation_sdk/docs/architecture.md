# knowledge_evaluation_sdk: Architecture

This document describes the internal architecture of the
`knowledge_evaluation_sdk` package: its two evaluation paths, module
dependency structure, benchmark adapter pattern, checkpoint lifecycle,
and integration seams with the parent knowledge package.

---

## 1. Data Flow Diagrams

### Path A: QA Evaluation (evaluator.py)

Single-document ground-truth evaluation using LLM-as-a-judge scoring.

```
Ground Truth JSONL -----> EvalCase (models.py)
                              |
                              v
Document path ---------> KnowledgeEvaluator
                              |
                   +----------+----------+
                   |                     |
            use_full_agent=False   use_full_agent=True
                   |                     |
          KnowledgeClient()     integrations.cuga_agent
          (standalone RAG)       .create_agent()
                   |                     |
                   +----------+----------+
                              |
                   Upload document (once)
                              |
                   For each EvalCase:
                   |   1. RAG search  --> rag_context (str)
                   |   2. LLM answer generation
                   |      (SystemMessage + HumanMessage via
                   |       get_llm_provider().get_chat_model)
                   |   3. LLMJudge.evaluate()
                   |      question + expected + response + context
                   |          --> JudgeScore (1-5, sub-metrics)
                   |
                   v
            EvalMetrics.from_results(List[EvalResult])
                   |
                   +---> eval_results_{ts}.jsonl   (per-case detail)
                   +---> eval_metrics_{ts}.json    (aggregate)
                   +---> eval_report_{ts}.txt      (human-readable)
```

Key classes:

| Class              | File          | Role                                         |
|--------------------|---------------|----------------------------------------------|
| KnowledgeEvaluator | evaluator.py  | Orchestrator: init, upload, eval loop, save   |
| LLMJudge           | judge.py      | Wraps an LLM with structured-output scoring   |
| EvalCase           | models.py     | Single ground-truth row (utterance + expected) |
| EvalResult         | models.py     | One case's output + judge score + latency      |
| EvalMetrics        | models.py     | Aggregate pass rate, mean score, latency stats |

### Path B: Benchmark Experiment (experiment.py)

Multi-benchmark, multi-config sweep with checkpointing and statistics.

```
Config YAML/TOML -----> ExperimentConfig
                              |
                              v
                         Experiment
                              |
                   generate_tasks()
                   = Cartesian product of:
                     benchmarks  x  pipeline_configs  x  n_runs
                              |
                              v
                   BenchmarkRunner (runner.py)
                   [asyncio.Semaphore(max_concurrency)]
                              |
            +-----------------+-----------------+
            |                                   |
     CheckpointManager                    For each task:
     is_completed(task_key)?              |
        yes --> restore & skip            |  1. benchmark_registry.get(name)
        no  --> run eval_fn               |     --> BaseBenchmarkAdapter.load()
                                          |     --> List[BenchmarkSample]
                                          |
                                          |  2. eval_fn(task, samples) --> TaskResult
                                          |     +-- _eval_offline: KnowledgePipeline
                                          |     |   .from_profile() + .run_and_evaluate()
                                          |     +-- _eval_live: KnowledgeClient.upload()
                                          |         + KnowledgePipeline.query()
                                          |         + ir_metrics.corpus_metrics()
                                          |
                                          |  3. checkpoint.save(task_key, result)
                                          |
            +-----------------+-----------------+
                              |
                              v
                      BenchmarkResults (results.py)
                              |
                   +----------+----------+----------+
                   |          |          |          |
              aggregate  significance  export    compare
              by_bench   paired_boot   markdown  two runs
              by_config  + Holm corr.  latex
                                       json
                                       DataFrame
```

Key task-key composition (for checkpointing and session isolation):

```
task_key = "{benchmark_name}|{config_name}|{config_hash}|{sampling_hash}|{mode}|run_{i}"
             ^                ^             ^              ^               ^       ^
             beir/nfcorpus    reranker=X    sha256(profile sha256(split,  auto     0
                                            +overrides)    n_samples,
                                            [:12]          seed,filters)
                                                           [:12]
```

When the experiment sets a `resume_fingerprint` (from `run_provenance`), the key gains a
`|env_{fingerprint}` suffix (a hash of the metric definitions, the package version, the judge and
the settings), so a resumed run never restores results computed under any other of those.

---

## 2. Dependency Graph

### Module coupling tiers

```
+======================================================================+
|                     TIER 0: Pure Python (+ loguru)                   |
|                     No external or parent-package deps               |
|----------------------------------------------------------------------|
|  ir_metrics.py        Standard IR metrics (NDCG, MAP, MRR, P/R/F1)  |
|  checkpoint.py        Atomic JSON checkpoints (json + tempfile)      |
|  runner.py            BenchmarkRunner + BenchmarkTask + TaskResult   |
|  results.py           BenchmarkResults aggregation + statistics      |
|                       (numpy only for significance_test)             |
|  benchmarks/_base.py  BaseBenchmarkAdapter ABC + BenchmarkSample     |
|  benchmarks/_registry.py  Thread-safe singleton registry             |
+======================================================================+
          |
          | imports
          v
+======================================================================+
|              TIER 1: Coupled to knowledge package                   |
|----------------------------------------------------------------------|
|  evaluator.py  --> providers.get_llm_provider                        |
|                    providers.get_config_provider                      |
|                    sdk.KnowledgeClient                               |
|  judge.py      --> providers.get_llm_provider                        |
|  experiment.py --> core.pipeline.KnowledgePipeline                   |
|                    sdk.client.KnowledgeClient (live mode)            |
|  ablation.py   --> core.pipeline.KnowledgePipeline (via duck-typing) |
+======================================================================+
          |
          | lazy / optional imports
          v
+======================================================================+
|              TIER 2: Optional external packages                     |
|----------------------------------------------------------------------|
|  integrations/cuga_agent.py --> cuga.sdk.CugaAgent (lazy import)  |
|  ragas_evaluator.py      --> ragas (lazy _import_ragas())             |
|  reporting.py            --> matplotlib, seaborn (optional)           |
|  datasets.py             --> huggingface datasets (lazy)              |
|  results.py              --> pandas (to_dataframe only)               |
|  ablation.py             --> matplotlib (plot only), pandas (df only) |
|  benchmarks/adapters/*   --> datasets (lazy, per-adapter)             |
+======================================================================+
```

### Boundary between eval SDK and parent knowledge package

```
knowledge_evaluation_sdk/          knowledge/ (parent)
 +-----------------------+          +----------------------+
 |  evaluator.py --------+--------> | providers/           |
 |  judge.py     --------+--------> |   get_llm_provider   |
 |  experiment.py -------+--------> | core/pipeline.py     |
 |                       |          |   KnowledgePipeline   |
 |  integrations.cuga_agent -+- - -> | (cuga.sdk, outside)  |
 |       (lazy import)   |          +----------------------+
 +-----------------------+
```

All coupling from the eval SDK into the parent package flows through
three narrow interfaces:

1. `providers.get_llm_provider()` / `providers.get_config_provider()`
2. `core.pipeline.KnowledgePipeline`
3. `sdk.client.KnowledgeClient`

The `lasagna/integrations/cuga_agent.py` module is the sole touch-point for the
top-level `cuga.sdk` package. It uses lazy imports so the rest of the
SDK has zero static coupling to the parent project.

---

## 3. Benchmark Adapter Pattern

### Class hierarchy

```
BaseBenchmarkAdapter  (ABC, benchmarks/_base.py)
  |
  |-- info() -> BenchmarkInfo             [abstract]
  |-- load(split, n_samples, ...) -> List[BenchmarkSample]  [abstract]
  |-- load_corpus() -> Optional[Dict[str,str]]  [default: None]
  |-- available_splits() -> List[str]            [default: ["test"]]
  |-- available_filters() -> Dict[str,List[str]] [default: {}]
```

### Concrete adapters (benchmarks/adapters/)

Each adapter file exports a `register(registry)` function that
instantiates the adapter and calls `registry.register_adapter(name, inst)`.

Built-in adapters:

```
beir.py               beir/nfcorpus, beir/scifact, beir/fiqa, ...
squad.py              squad_v2
natural_questions.py  natural_questions
ms_marco.py           ms_marco
triviaqa.py           triviaqa
hotpotqa_multihop.py  hotpotqa
musique.py            musique
wiki_multihop.py      wiki_multihop
multihop_rag.py       multihop_rag
bamboogle.py          bamboogle
ragbench.py           ragbench
kilt.py               kilt/nq, kilt/triviaqa
fever.py              fever
```

### Discovery chain (BenchmarkRegistry)

```
BenchmarkRegistry (thread-safe singleton)
    |
    |-- First access triggers _ensure_builtins():
    |     1. _register_builtins()
    |        --> adapters/__init__.py :: register_all(registry)
    |            --> imports each adapter module
    |            --> calls adapter.register(registry)
    |
    |     2. _discover_entry_points()
    |        --> importlib.metadata.entry_points()
    |            group = "lasagna_benchmarks"
    |        --> for each entry point:
    |            ep.load() --> adapter class or instance
    |            registry.register_adapter(ep.name, adapter)
    |
    |-- Public API:
    |     .get(name) -> BaseBenchmarkAdapter
    |     .list_benchmarks() -> Dict[str, BenchmarkInfo]
    |     .names() -> List[str]
    |     .register(name) -> decorator
```

Third-party packages can register custom benchmark adapters via
`pyproject.toml` entry points:

```toml
[project.entry-points.lasagna_benchmarks]
my_bench = "my_package.adapters:MyBenchmarkAdapter"
```

---

## 4. Checkpoint/Resume Lifecycle

### Directory layout

```
{output_dir}/
  .checkpoints/
    beir_nfcorpus_config_a_config_hash_..._run_0_{sha256[:16]}.json
    beir_nfcorpus_config_b_config_hash_..._run_0_{sha256[:16]}.json
    ...
```

### File naming

Each checkpoint file name is derived from the task key:

```python
safe = task_key.replace("/", "_").replace("|", "_")[:80]
h    = sha256(task_key)[:16]
path = .checkpoints / f"{safe}_{h}.json"
```

The SHA-256 suffix guarantees uniqueness even when the safe-name
prefix truncates.

### Atomic write pattern

```
1. Open tempfile in .checkpoints/ dir (suffix=".tmp", delete=False)
2. json.dump({"task_key": ..., "completed_at": ..., "result": ...})
3. Close tempfile
4. os.rename(tmp_path, final_path)      <-- atomic on same filesystem
```

If the process crashes between steps 2 and 4, only the .tmp file
remains. It is never read by `completed_keys()` (which globs *.json),
so incomplete checkpoints are invisible.

### Resume flow

```
Experiment.run()
    |
    v
BenchmarkRunner.run_tasks(tasks, eval_fn, checkpoint)
    |
    For each task in tasks:
    |   if checkpoint.is_completed(task.task_key):
    |       result = checkpoint.load(task.task_key)
    |       --> TaskResult.from_dict(result)
    |       --> append to results, skip execution
    |   else:
    |       --> run eval_fn(task, samples)
    |       --> checkpoint.save(task.task_key, result.to_dict())
    |
    v
    List[TaskResult]  (mix of restored + freshly computed)
```

### Task key stability

The task key includes:

- `benchmark_name`: which dataset
- `config_name`: human-readable pipeline config label
- `config_hash`: SHA-256 of `{profile, overrides}` (12 hex chars)
- `sampling_hash`: SHA-256 of `{split, n_samples, seed, filters}` (12 hex chars)
- `mode`: the requested mode, `"auto"`, `"live"` or `"generation_only"` (`"offline"` is an alias of the last); the mode it resolved to is recorded separately, in `resolved_mode`
- `run_index`: run number for variance estimation
- `env_<fingerprint>`, when set: the `resume_fingerprint` of the run's settings

Changing any sampling parameter (split, n_samples, seed, filters)
produces a different `sampling_hash`, which produces a different
task key, which means the old checkpoint is not matched. This
prevents `--resume` from silently restoring stale results when
parameters change.

### Clearing checkpoints

```python
checkpoint.clear()  # Deletes all *.json in .checkpoints/
```

---

## 5. Integration Seams

The eval SDK touches the parent knowledge package at five key points.

### 5.1 KnowledgePipeline.from_profile(profile_name)

Creates a fully-configured pipeline from a named profile (e.g.,
"standard", "fast", "research"). Used by `_eval_offline()` and
`_eval_live()` in `experiment.py`.

```python
pipeline = KnowledgePipeline.from_profile(task.profile)
```

The profile loads default component selections and config values from
the session config system.

### 5.2 pipeline.replace(component_type, component_name)

Hot-swaps a pipeline component at runtime. Used by
`_apply_overrides_to_pipeline()` in `experiment.py` and by `sweep()`
/ `grid_search()` in `ablation.py`.

```python
pipeline.replace("reranker", "mxbai-rerank-large")
```

Valid component types: `reranker`, `embedder`, `chunker`, `retriever`,
`document_processor`, `query_enhancer`.

Components are resolved via `core.registry` which maps string names to
registered class implementations.

### 5.3 pipeline.query(question, top_k, session_id)

Executes the full search pipeline (embedding, retrieval, reranking)
and returns ranked `SearchResult` objects. Used by `_eval_live()` to
compute IR metrics against a real vector store.

```python
results = pipeline.query(question, top_k=10, session_id=session_id)
```

This is the preferred search entry point for experiments because it
forwards component overrides (reranker, embedder, query_enhancer) to
the `SearchOrchestrator`.

### 5.4 pipeline.run_and_evaluate(dataset, ...)

Runs retrieval through the pipeline for each question in the dataset,
then optionally scores answers using a provided evaluator (e.g.,
`RAGASEvaluator`). Returns a `PipelineResult` with `.metrics` and
`.per_sample` attributes. Used by `_eval_offline()` in `experiment.py`
and by `sweep()` / `grid_search()` in `ablation.py`.

```python
result = pipeline.run_and_evaluate(
    dataset=eval_cases,
    session_id=session_id,
    evaluator=ragas_evaluator,
)
```

### 5.5 KnowledgeClient.upload(path, document_id=...)

Ingests a document into the vector store. Used by `_eval_live()` to
materialize benchmark corpus files before querying, and by
`KnowledgeEvaluator` to upload the evaluation document.

```python
async with KnowledgeClient(session_id=sid) as client:
    await client.upload(str(doc_path), document_id=valid_id)
```

The `document_id` parameter enables deterministic ID assignment for
corpus documents, which is essential for mapping retrieval results
back to benchmark qrels for IR metric computation.
