---
title: Lasagna
hide:
  - navigation
  - toc
---

<div class="lg-hero" markdown>

# Develop a RAG technique and test it inside a full pipeline.

Lasagna splits a retrieval-augmented generation pipeline into eight stages. You
swap one component, hold the other seven fixed, and get a verdict on whether the
difference is real. The same pipeline also serves your agent over MCP and REST.

<div class="lg-actions" markdown>
[Get started](get-started/quickstart.md){ .lg-button .lg-button--primary }
[Compare a component](guides/compare-a-component.md){ .lg-button }
[View on GitHub](https://github.com/dolev31/lasagna){ .lg-button }
</div>

</div>

## Why it exists

RAG papers report one component at a time. A new reranker is measured against a
baseline with no query expansion and an old chunker. If a reranker claims +4 NDCG
and a query expander claims +3, you cannot tell from the papers whether stacking
them gives +7, +4 or -1.

The study behind Lasagna tested 450 configurations across eight benchmarks. The
three strongest single retrieval techniques were predicted to add 47% NDCG@10
together. They added 17%. Faithfulness gains did carry over. The
[paper study](concepts/the-paper-study.md) has the detail.

<div class="lg-stat" markdown>
<div markdown><strong>8</strong><span>swappable pipeline stages</span></div>
<div markdown><strong>55</strong><span>built-in components</span></div>
<div markdown><strong>31</strong><span>public benchmarks</span></div>
<div markdown><strong>0</strong><span>API keys for a retrieval comparison</span></div>
</div>

## Install

```bash
pip install lasagna-rag
python -m nltk.downloader punkt_tab
```

Lasagna runs on Python 3.10 to 3.13. The `speed` profile downloads one 92 MB
model and never calls an LLM. See [installation](get-started/installation.md) for
extras and platform notes.

## Your first comparison

Put a reranker in `my_ideas.py`:

```python
import lasagna

@lasagna.reranker
def keyword_overlap(query, text):
    return len(set(query.lower().split()) & set(text.lower().split()))
```

Save a sweep as `sweep.yaml`. It runs your reranker against a built-in one, with
everything else fixed:

```yaml
name: does keyword_overlap help
benchmarks: [{name: smoke}]
tracks: [retrieval]
pipeline:
  base_profile: speed
  rerankers: [cross_encoder, "my_ideas:keyword_overlap"]
n_runs: 1
```

Run it, then ask whether the gap is real:

```bash
lasagna benchmark experiment sweep.yaml
lasagna results significance results.json --all
```

[Your first comparison](get-started/first-comparison.md) walks through the
output, and shows the same sweep on a real benchmark.

## Where to go

<div class="grid cards" markdown>

-   **Compare a component**

    ---

    Write a component, sweep it against a built-in, and read the verdict.

    [:octicons-arrow-right-24: Compare a component](guides/compare-a-component.md)

-   **Give your agent RAG**

    ---

    Serve search over your documents through MCP, REST or Docker Compose.

    [:octicons-arrow-right-24: MCP server](serving/mcp.md)

-   **The catalog**

    ---

    The eight stages, all 55 components and 31 benchmarks, generated from the code.

    [:octicons-arrow-right-24: Stages](catalog/stages.md)

-   **The reference**

    ---

    Every CLI command, setting, metric and Python name.

    [:octicons-arrow-right-24: CLI reference](reference/cli.md)

</div>

## What it reproduces

On the full SciFact and NFCorpus test sets, the built-in dense models land within
0.002 NDCG@10 of their MTEB results. BM25 followed by a MiniLM cross-encoder lands
within 0.004 of the BEIR paper. Nothing was tuned. The
[reproduction guide](guides/reproduce-beir.md) shows the table and the command.

## Cite it

If you publish work that uses Lasagna, see [citing Lasagna](project/citing.md).
