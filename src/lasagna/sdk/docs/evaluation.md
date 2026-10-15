# Evaluation

## Evaluation

### Basic Evaluation

```python
from lasagna.sdk.evaluation import (
    KnowledgeEvaluator,
    EvalConfig,
)

# Create evaluator
evaluator = KnowledgeEvaluator(client)

# Prepare test cases
test_cases = [
    {
        "query": "What is the Q3 revenue?",
        "expected_docs": ["financial_report.pdf"],
    },
    {
        "query": "Who is the CEO?",
        "expected_docs": ["annual_report.pdf"],
    },
]

# Run evaluation
results = await evaluator.evaluate(test_cases, top_k=5)

print(f"Precision: {results.precision:.2%}")
print(f"Recall: {results.recall:.2%}")
print(f"F1 Score: {results.f1:.2%}")
print(f"MRR: {results.mrr:.2f}")
print(f"Avg Latency: {results.latency_ms:.0f}ms")
```

### LLM-as-Judge Evaluation

```python
# With expected answers
test_cases = [
    {
        "query": "What is the quarterly revenue?",
        "expected_answer": "Revenue was $10M in Q3 2024",
    },
]

results = await evaluator.evaluate_with_judge(test_cases)
```

### Custom Evaluation Config

```python
config = EvalConfig(
    judge_model="gpt-6-luna",
    num_samples=100,
    metrics=["precision", "recall", "f1", "mrr", "ndcg"],
    verbose=True,
)

evaluator = KnowledgeEvaluator(client, config=config)
```

---

## Evaluation and Benchmarking

The SDK includes a built-in evaluation framework for measuring RAG retrieval quality using LLM-as-a-judge scoring, RAGAS-compatible metrics, and ablation study support.

### KnowledgeEvaluator

The main evaluator orchestrates end-to-end evaluation: uploading documents, running queries, scoring responses, and generating reports.

```python
from lasagna.knowledge_evaluation_sdk import (
    KnowledgeEvaluator,
    EvalConfig,
)

# Configure evaluation
config = EvalConfig(
    document_path="./knowledge_base/quarterly_report.pdf",
    ground_truth_path="./benchmarks/ground_truth.jsonl",
    judge_model_platform="openai",
    judge_model_name="gpt-6-luna",
    output_dir="./eval_results",
    verbose=True,
    max_concurrent=3,
)

# Run evaluation
evaluator = KnowledgeEvaluator(config)
metrics = await evaluator.run()

print(f"Average Score: {metrics.avg_score:.2f}/5")
print(f"Pass Rate: {metrics.pass_rate:.1%}")
print(f"RAG Usage Rate: {metrics.rag_usage_rate:.1%}")
print(f"Avg Latency: {metrics.avg_latency_ms:.0f}ms")
print(f"Avg RAG Latency: {metrics.avg_rag_latency_ms:.0f}ms")

# Cleanup
await evaluator.cleanup()
```

### Ground Truth Format

Ground truth files use JSONL format with one evaluation case per line:

```json
{"utterance": "What was Q3 revenue?", "expected_response": "Q3 revenue was $10M, up 15% YoY."}
{"utterance": "Who is the CEO?", "expected_response": "The CEO is Jane Smith, appointed in 2022."}
```

### LLM-as-a-Judge Scoring

The evaluator uses an LLM judge to score responses on a 1-5 scale:

| Score | Level | Description |
|-------|-------|-------------|
| 5 | Excellent | Complete, accurate answer with all key details |
| 4 | Good | Mostly correct with minor omissions |
| 3 | Acceptable | Partially correct, missing some key information |
| 2 | Poor | Mostly incorrect or very incomplete |
| 1 | Fail | Completely wrong or no relevant answer |

### Streaming Evaluation

For large benchmark sets, stream results as they complete:

```python
evaluator = KnowledgeEvaluator(config)

async for result in evaluator.run_streaming():
    status = "PASS" if result.passed else "FAIL"
    print(f"{status} [{result.score}/5] {result.case.utterance[:60]}...")

# Get final metrics
metrics = evaluator.metrics
print(metrics.summary())
```

### Ablation Studies

Compare different configurations to understand the impact of each feature:

```python
from lasagna.sdk import (
    KnowledgeClient,
    config_for_speed,
    config_for_standard,
    config_for_balanced,
    config_for_max_quality,
)
from lasagna.knowledge_evaluation_sdk import KnowledgeEvaluator, EvalConfig

configs = {
    "speed": config_for_speed(),
    "standard": config_for_standard(),
    "balanced": config_for_balanced(),
    "max_quality": config_for_max_quality(),
}

results = {}
for name, config in configs.items():
    eval_config = EvalConfig(
        document_path="./report.pdf",
        ground_truth_path="./ground_truth.jsonl",
        output_dir=f"./eval_results/{name}",
    )
    evaluator = KnowledgeEvaluator(eval_config)
    metrics = await evaluator.run()
    results[name] = metrics
    await evaluator.cleanup()

# Compare results
for name, metrics in results.items():
    print(f"{name:15s} | Score: {metrics.avg_score:.2f} | "
          f"Pass: {metrics.pass_rate:.0%} | Latency: {metrics.avg_latency_ms:.0f}ms")
```

### Pipeline Serialization

Export and import the current pipeline configuration for reproducibility:

```python
# Export current config as a serializable dict
config_dict = client.get_config_dict()

# Save to file for reproducibility
import json
with open("pipeline_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

# Load and apply later
with open("pipeline_config.json") as f:
    saved_config = json.load(f)
await client.set_config(**saved_config)
```

---

## See Also

- [Features](features.md)
- [Extending](extending.md)
